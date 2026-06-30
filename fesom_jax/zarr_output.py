"""Sharded, gather-free model output to Zarr (Phase 8b B.3).

The FESOM2 → JAX model state lives **sharded** across the device mesh as folded
``[P*Lmax_kind, …]`` arrays (device ``d`` owns rows ``[d*Lmax : (d+1)*Lmax]``). Writing
output the C/Kokkos way — gather the global field to rank 0, then write one NetCDF — does
not scale: at NG5 (7.4 M × 70) that global gather is ~tens of GB on one rank (the OOM the
Kokkos ``SCALING_NG5`` notes, and the same single-device-materialization class as the input
placement fix). Instead, **each GPU writes its own shard in parallel** to a Zarr store:

* the on-disk array is the *folded* ``[P*Lmax_kind, …]`` shape, **chunked at ``Lmax_kind``
  along axis 0** so each device's shard is exactly one chunk ⇒ different processes/devices
  write disjoint chunk files ⇒ no locking, no gather, fully parallel;
* per-kind index maps ``gid`` (global entity id of each folded lane) and ``owned`` (the lane
  is its entity's unique owner) are written once, so :func:`reconstruct_global` can scatter
  the owned lanes back to a dense global ``[nod2D, …]`` / ``[elem2D, …]`` array on read.

This is the output analogue of :func:`fesom_jax.integrate_sharded._to_global_sharded` (input):
nothing global is ever materialized on a single device.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util
from jax.sharding import NamedSharding, PartitionSpec

# Key prognostic fields a typical run wants on disk (override via ``fields=``).
# ``tke`` (Phase 9b) is prognostic — written so a TKE run is restartable / analysable.
DEFAULT_FIELDS = ("T", "S", "uv", "eta_n", "w",
                  "a_ice", "m_ice", "m_snow", "u_ice", "v_ice", "tke")


def _kind_of(lead: int, P: int, Lmax: dict) -> str:
    """Map a folded leading dim ``P*Lmax_kind`` back to its entity kind."""
    for k in ("nod", "elem", "edge"):
        if lead == P * Lmax[k]:
            return k
    raise ValueError(f"folded leading dim {lead} matches no kind (P={P}, Lmax={Lmax})")


def _folded_gid_owned(part, sm, kind: str):
    """Folded ``[P*Lmax]`` global-id + owned-mask for ``kind`` (the reconstruction maps).

    ``gid[d*Lmax + lane]`` = the global entity id of device ``d``'s local lane (``-1`` on
    pad lanes); ``owned`` is ``sm.owned_mask[kind]`` folded — True only on the lane that is
    the entity's UNIQUE owner, so scattering owned lanes by ``gid`` reproduces the dense
    global field with no double-write."""
    P, Lmax = sm.P, sm.Lmax[kind]
    mylist = {"nod": part.myList_nod2D, "elem": part.myList_elem2D,
              "edge": part.myList_edge2D}[kind]
    gid = np.full((P * Lmax,), -1, dtype=np.int64)
    for d in range(P):
        ml = np.asarray(mylist[d])
        gid[d * Lmax: d * Lmax + ml.size] = ml
    owned = np.asarray(sm.owned_mask[kind]).reshape(-1)        # [P*Lmax] bool, same fold
    return gid, owned


def write_state_zarr(out_dir, state_folded, sm, part, *, fields=None, attrs=None, layout="folded"):
    """Write the *folded sharded* ``state_folded`` to a Zarr store at ``out_dir`` — each
    process writes ONLY its addressable shards, in parallel, no gather.

    ``state_folded`` is a :class:`~fesom_jax.state.State` of folded ``[P*Lmax_kind, …]``
    sharded ``jax.Array`` leaves (e.g. the output of
    ``run_steps_sharded(..., return_executable=True)`` then ``jfn(*args)``). Rank 0 creates
    the array metadata + the per-kind index maps; a multi-host barrier; then every process
    writes its local device shards (``arr.addressable_shards`` → one ``Lmax``-chunk each).

    ``layout``: ``"folded"`` (default) = the gather-free ``[P*Lmax]`` sharded write described above
    (partition-DEPENDENT on-disk shape, carries gid/owned maps to reload onto any device count).
    ``"global"`` = **partition-INDEPENDENT** canonical global-node order (see
    :func:`_write_state_global`): byte-identical at any device count, read back by global index
    directly; works single- AND multi-process (host-gather / all_gather). Returns the store path."""
    if layout == "global":
        return _write_state_global(out_dir, state_folded, sm, part, fields=fields, attrs=attrs)
    if layout != "folded":
        raise ValueError(f"layout must be 'folded' or 'global', got {layout!r}")
    import zarr
    from jax.experimental import multihost_utils

    out_dir = Path(out_dir)
    fields = tuple(fields) if fields is not None else DEFAULT_FIELDS
    leaves = {name: getattr(state_folded, name) for name in fields}
    P = sm.P
    is0 = jax.process_index() == 0

    # 1) rank 0 creates the dataset metadata + index maps (writing .zarray once, no race).
    if is0:
        root = zarr.open_group(str(out_dir), mode="a")
        kinds = set()
        for name, arr in leaves.items():
            kind = _kind_of(arr.shape[0], P, sm.Lmax)
            kinds.add(kind)
            shape = tuple(int(s) for s in arr.shape)
            chunks = (sm.Lmax[kind],) + shape[1:]
            z = root.require_dataset(name, shape=shape, chunks=chunks,
                                     dtype=np.dtype(arr.dtype), overwrite=True)
            z.attrs["kind"] = kind
        for kind in kinds:
            gid, owned = _folded_gid_owned(part, sm, kind)
            zg = root.require_dataset(f"gid_{kind}", shape=gid.shape, chunks=gid.shape,
                                      dtype=gid.dtype, overwrite=True); zg[:] = gid
            zo = root.require_dataset(f"owned_{kind}", shape=owned.shape, chunks=owned.shape,
                                      dtype=owned.dtype, overwrite=True); zo[:] = owned
        root.attrs.update(P=int(P), nod2D=int(sm.nod2D), elem2D=int(sm.elem2D),
                          edge2D=int(sm.edge2D), nl=int(sm.nl), **(attrs or {}))

    multihost_utils.sync_global_devices("fesom_zarr_meta")

    # 2) every process writes its addressable device shards (disjoint Lmax-chunks → parallel).
    root = zarr.open_group(str(out_dir), mode="a")
    for name, arr in leaves.items():
        z = root[name]
        for shard in arr.addressable_shards:
            z[shard.index] = np.asarray(shard.data)

    multihost_utils.sync_global_devices("fesom_zarr_data")
    return out_dir


# Entity-axis chunk for canonical restarts. P-INDEPENDENT (depends only on n_global) ⇒ a restart is
# byte-identical at any device/process count; large enough that small meshes are one chunk, small enough
# that a huge mesh (NG5) splits into a handful of chunks for parallel multi-process disjoint writes.
_RESTART_CHUNK = 1_000_000


def _write_state_global(out_dir, state_folded, sm, part, *, fields=None, attrs=None):
    """Canonical global-node-order (**partition-independent**) State write — the ``layout='global'``
    path of :func:`write_state_zarr` (mirrors the FESOM3 ``mod_io_zarr`` layout).

    Each leaf's UNIQUE-owner lanes are scattered by global id into a dense ``[n_global_kind, …]`` array
    in canonical order, full dtype (float64 state preserved), no folded lane axis / FILL / gid maps. The
    store is **byte-identical at any device/process count** (a P-independent chunk grid) and reads back
    by global index directly (:func:`reconstruct_global` short-circuits on the ``layout`` attr).

    Works **single- AND multi-process**, like the output writer's ``all_gather`` method:

    * **single-process** — host-gathers each leaf (``np.asarray``) and writes it (one writer);
    * **multi-process** — ``all_gather`` gives every process the full folded leaf (no single-node
      funnel), each scatters to canonical global and writes its DISJOINT chunk range. Handles BOTH
      ``nod`` and ``elem`` State kinds. Host peak ≈ one global leaf (streamed one leaf at a time)."""
    import zarr

    from . import halo
    out_dir = Path(out_dir)
    fields = tuple(fields) if fields is not None else DEFAULT_FIELDS
    leaves = {name: getattr(state_folded, name) for name in fields}
    P = int(sm.P)
    n_global = {"nod": int(sm.nod2D), "elem": int(sm.elem2D), "edge": int(sm.edge2D)}
    multiproc = jax.process_count() > 1
    pid, nproc = jax.process_index(), jax.process_count()
    maps: dict = {}                                          # gid/owned per kind, computed once

    def _go(kind):
        if kind not in maps:
            maps[kind] = _folded_gid_owned(part, sm, kind)
        return maps[kind]

    # rank 0 creates the store + every dataset (C-chunked, C independent of P → byte-identical bytes).
    if pid == 0:
        root = zarr.open_group(str(out_dir), mode="a")
        for name, arr in leaves.items():
            kind = _kind_of(arr.shape[0], P, sm.Lmax)
            ng = n_global[kind]
            shape = (ng,) + tuple(int(s) for s in arr.shape[1:])
            chunks = (min(ng, _RESTART_CHUNK),) + shape[1:]
            z = root.require_dataset(name, shape=shape, chunks=chunks,
                                     dtype=np.dtype(arr.dtype), overwrite=True)
            z.attrs["kind"] = kind
        root.attrs.update(P=P, nod2D=int(sm.nod2D), elem2D=int(sm.elem2D), edge2D=int(sm.edge2D),
                          nl=int(sm.nl), layout="canonical_global", **(attrs or {}))
    if multiproc:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("restart_global_meta")

    gather = None
    if multiproc:
        from . import canonical_redist as cr
        jmesh = halo.device_mesh("p", devices=jax.devices()[:P])
        gather = cr._allgather_callable(jmesh, "p")

    root = zarr.open_group(str(out_dir), mode="a")
    for name, arr in leaves.items():
        kind = _kind_of(arr.shape[0], P, sm.Lmax)
        gid, owned = _go(kind)
        ng = n_global[kind]
        folded_host = np.asarray(gather(arr)) if multiproc else np.asarray(arr)   # full folded leaf
        g = np.zeros((ng,) + folded_host.shape[1:], dtype=folded_host.dtype)
        g[gid[owned]] = folded_host[owned]                  # canonical [n_global, …]
        # this process writes its DISJOINT whole-chunk range (single-process ⇒ the whole leaf)
        C = min(ng, _RESTART_CHUNK)
        nchunks = (ng + C - 1) // C
        cpp = (nchunks + nproc - 1) // nproc
        r0 = pid * cpp * C
        r1 = min(min((pid + 1) * cpp, nchunks) * C, ng)
        if r1 > r0:
            root[name][r0:r1] = g[r0:r1]
    if multiproc:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("restart_global_data")
    return out_dir


def reconstruct_global(out_dir, field: str):
    """Read a Zarr restart field back to a dense global ``[n_global, …]`` array (host numpy).

    Auto-detects the on-disk layout: a ``layout='canonical_global'`` store (the partition-independent
    default) is ALREADY dense global ⇒ returned as-is; a folded store scatters the OWNED lanes by
    their global id. Inverse of :func:`write_state_zarr` for one field — for analysis / the restart
    reload, not the hot path."""
    import zarr
    root = zarr.open_group(str(Path(out_dir)), mode="r")
    z = root[field]
    if root.attrs.get("layout") == "canonical_global":
        return z[:]                                           # already dense global [n_global, …]
    kind = z.attrs["kind"]
    data = z[:]                                            # [P*Lmax, …]
    gid = root[f"gid_{kind}"][:]
    owned = root[f"owned_{kind}"][:]
    n_global = {"nod": int(root.attrs["nod2D"]), "elem": int(root.attrs["elem2D"]),
                "edge": int(root.attrs["edge2D"])}[kind]
    out = np.zeros((n_global,) + data.shape[1:], dtype=data.dtype)
    out[gid[owned]] = data[owned]
    return out


# --------------------------------------------------------------------------
# Device-count-portable restart (Task A1) — the FULL prognostic State, gid-keyed
# --------------------------------------------------------------------------
def _all_state_fields() -> tuple[str, ...]:
    """Every :class:`~fesom_jax.state.State` leaf name, in pytree order.

    Derived from the dataclass (the State pytree is flat — every field is one array
    leaf) so a newly-added leaf is captured automatically: this is the no-silent-drop
    guarantee the restart needs (NOT a ``*_old`` name-glob, which would miss
    ``uv_rhsAB`` / ``sigma*`` / ``tke``)."""
    from .state import State
    return tuple(f.name for f in dataclasses.fields(State))


def write_restart(out_dir, state_folded, sm, part, *, step, calendar_date, dt_stage,
                  fields=None, attrs=None, layout="global"):
    """Write the **FULL** prognostic ``State`` as a device-count-portable restart.

    ``layout`` defaults to ``"global"`` = **partition-independent** canonical node order: the restart
    is byte-identical regardless of the device count it was written from, and reads back by global
    index (no folded lane axis / gid maps). Works single- AND multi-process (host-gather / all_gather —
    see :func:`_write_state_global`). ``"folded"`` is the gather-free sharded write (fastest / huge-mesh
    OOM escape); :func:`read_restart` auto-detects either on-disk layout and reloads onto any device count.

    Unlike :func:`write_state_zarr` (which writes the ``DEFAULT_FIELDS`` snapshot
    subset), this writes **every** ``State`` leaf — all history/carry slots
    (``T_old``, ``S_old``, ``uv_rhsAB``, ``ssh_rhs_old``, ``hbar_old``,
    ``sigma11/12/22``, ``tke``, …) — gid-keyed and gather-free (each shard writes its
    owned entities to disjoint Zarr chunks), plus the restart metadata (``step``,
    ``calendar_date``, ``dt_stage``). The on-disk format is **partition-independent**
    (the gid/owned index maps), so :func:`read_restart` can reload it onto ANY device
    count.

    ``state_folded`` is a ``State`` of folded ``[P*Lmax_kind, …]`` sharded ``jax.Array``
    leaves (the output of a sharded run). ``dt_stage`` is the timestep ``dt`` this
    restart was written at — persisted so a resumed run knows whether it is mid-dt-ramp
    (a dt change across a restart invalidates the AB2 history; the driver re-bootstraps).
    Returns the store path."""
    fields = tuple(fields) if fields is not None else _all_state_fields()
    meta = {"is_restart": 1, "step": int(step),
            "calendar_date": str(calendar_date), "dt_stage": float(dt_stage)}
    if attrs:
        meta.update(attrs)
    return write_state_zarr(out_dir, state_folded, sm, part, fields=fields, attrs=meta, layout=layout)


def read_restart(out_dir, mesh, new_part, *, devices=None):
    """Reload a :func:`write_restart` store onto ``new_part`` (ANY device count).

    **Field-by-field streaming** so the host peak is ≈ ONE global field, never the whole
    ~40-leaf State co-resident (an NG5 3-D field is ≈ 4 GB; all leaves at once ≈ 80 GB):
    for each leaf, reconstruct it to a dense global ``[n_global, …]`` array by gid
    (:func:`reconstruct_global`, partition-**independent**), gather/pad it to ``new_part``
    (``[P', Lmax', …]``), fold to ``[P'*Lmax', …]`` and ``device_put``-shard it across the
    new device mesh (per-shard host→device copy, no global staged on one device), then
    free it before the next leaf.

    Re-gathering from the COMPLETE global field fills ``new_part``'s halo lanes correctly
    too (a halo lane is just another gid in ``myList``), so the result is byte-identical to
    partitioning the original global State directly onto ``new_part``.

    Returns ``(State, meta)`` where ``State`` has folded sharded leaves ready for
    ``run_steps_sharded`` and ``meta`` carries ``step`` / ``calendar_date`` / ``dt_stage``
    (+ the global counts). ``devices`` overrides the placement devices (default: the first
    ``new_part.npes`` of :func:`jax.devices`)."""
    import zarr
    from . import halo
    from .integrate_sharded import _fold
    from .shard_mesh import _default_pad, _shard_along_axis, local_sizes
    from .state import State

    out_dir = Path(out_dir)
    root = zarr.open_group(str(out_dir), mode="r")
    meta = {k: v for k, v in root.attrs.items()}

    # Guard: the restart's global counts must match the target mesh / partition.
    for nm, want in (("nod2D", new_part.nod2D), ("elem2D", new_part.elem2D),
                     ("edge2D", new_part.edge2D)):
        if nm in root.attrs and int(root.attrs[nm]) != int(want):
            raise ValueError(f"restart {nm}={int(root.attrs[nm])} != target partition "
                             f"{nm}={int(want)} (wrong mesh?)")
    if mesh is not None and (int(mesh.nod2D) != int(new_part.nod2D)
                             or int(mesh.elem2D) != int(new_part.elem2D)):
        raise ValueError("mesh and new_part disagree on global counts (nod2D/elem2D)")

    _, Lmax = local_sizes(new_part)
    devs = jax.devices()[:new_part.npes] if devices is None else list(devices)
    if len(devs) < new_part.npes:
        raise ValueError(f"need {new_part.npes} devices, have {len(devs)}")
    jmesh = halo.device_mesh("p", devices=devs)
    sharding = NamedSharding(jmesh, PartitionSpec("p"))

    out = {}
    for name in _all_state_fields():
        g = reconstruct_global(out_dir, name)            # [n_global, …] host numpy
        n0 = g.shape[0]
        if n0 == new_part.nod2D:
            ml, L = new_part.myList_nod2D, Lmax["nod"]
        elif n0 == new_part.elem2D:
            ml, L = new_part.myList_elem2D, Lmax["elem"]
        else:
            raise ValueError(f"restart field {name!r} global dim {n0} is neither nod2D "
                             f"({new_part.nod2D}) nor elem2D ({new_part.elem2D})")
        folded = _fold(_shard_along_axis(g, ml, L, 0, _default_pad(g.dtype)))   # [P'*L', …]
        out[name] = jax.make_array_from_callback(
            folded.shape, sharding, lambda idx, a=folded: a[idx])
        del g, folded                                    # bound host peak to ~one field
    return State(**out), meta


# --------------------------------------------------------------------------
# Streaming time-mean / variance output (Task A2) — online, no per-step storage
# --------------------------------------------------------------------------
@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class OnlineStats:
    """Running **mean + variance** (Welford, single-pass) over a dict of fields.

    Accumulates the per-grid-point time mean **and** variance of selected fields without
    storing every step — so a multi-year run yields a mean state AND the EKE map (from the
    velocity variance, :func:`eke_from_stats`) at the memory cost of two extra copies per
    tracked field. Each ``update`` folds in one time sample (a dict ``{name: array}`` of the
    SAME shapes given at :meth:`init`); the leaves are plain ``jax.Array`` so this is a
    **pytree** that flows through ``lax.scan`` (accumulate inside the sharded step loop) or a
    Python chunk loop, sharded or dense alike (every op is element-wise ⇒ sharding-preserving).

    The Welford recurrence keeps each masked/dry lane **finite** (the running mean of a
    constant-0 lane is 0, its variance 0; no ``0/0`` — the denominator ``count`` is ≥1 after
    the first sample), satisfying the AD/masked-lane finiteness rule.

    Population variance (``ddof=0``) is the default — it is the ⟨·'²⟩ the EKE definition wants
    and is what ``numpy.var`` returns by default; pass ``ddof=1`` to :meth:`variance` for the
    unbiased sample estimator."""

    count: jax.Array                 # scalar — number of time samples folded in so far
    mean: dict                       # {name: running mean array}
    M2: dict                         # {name: running sum of squared deviations}

    # -- pytree plumbing (count + the two dicts are the children) ----------
    def tree_flatten(self):
        return (self.count, self.mean, self.M2), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        count, mean, M2 = children
        return cls(count=count, mean=mean, M2=M2)

    # -- construction / update ---------------------------------------------
    @classmethod
    def init(cls, fields: dict) -> "OnlineStats":
        """Zero accumulators shaped like ``fields`` (a ``{name: array}`` snapshot)."""
        mean = {k: jnp.zeros_like(jnp.asarray(v)) for k, v in fields.items()}
        M2 = {k: jnp.zeros_like(jnp.asarray(v)) for k, v in fields.items()}
        return cls(count=jnp.asarray(0.0, jnp.float64), mean=mean, M2=M2)

    def update(self, fields: dict) -> "OnlineStats":
        """Fold in one time sample; returns a new :class:`OnlineStats` (functional ⇒ scan-safe).

        ``fields`` must supply every tracked key (extra keys are ignored), each the shape given
        at :meth:`init`."""
        count = self.count + 1.0
        mean, M2 = {}, {}
        for k in self.mean:
            x = jnp.asarray(fields[k])
            delta = x - self.mean[k]
            m = self.mean[k] + delta / count
            delta2 = x - m
            mean[k] = m
            M2[k] = self.M2[k] + delta * delta2
        return OnlineStats(count=count, mean=mean, M2=M2)

    # -- finalize ----------------------------------------------------------
    def nobs(self) -> int:
        return int(self.count)

    def mean_dict(self) -> dict:
        """The time-mean of each tracked field (``{name: array}``)."""
        return dict(self.mean)

    def variance(self, ddof: int = 0) -> dict:
        """Per-grid-point time variance, ``M2 / (count - ddof)`` (``ddof=0`` ⇒ population).

        The denominator is floored at 1 so a single-sample (or empty) accumulator returns 0
        rather than ``inf``/``nan`` — finite by construction."""
        denom = jnp.maximum(self.count - ddof, 1.0)
        return {k: v / denom for k, v in self.M2.items()}

    def std(self, ddof: int = 0) -> dict:
        """Per-grid-point time standard deviation (``sqrt`` of :meth:`variance`)."""
        return {k: jnp.sqrt(v) for k, v in self.variance(ddof).items()}


def eke_from_stats(stats: OnlineStats, uv_name: str = "uv") -> jax.Array:
    """Surface/3-D **eddy kinetic energy** ``½⟨u'² + v'²⟩`` from a velocity variance accumulator.

    ``stats`` must have tracked the ``[…, 2]`` velocity field ``uv_name`` (``uv`` on elements
    or ``uvnode`` on nodes); the last axis is ``(u, v)``. Returns the EKE field shaped like the
    velocity field minus its trailing component axis (``[elem2D, nl]`` / ``[nod2D, nl]``)."""
    var = stats.variance()[uv_name]
    return 0.5 * (var[..., 0] + var[..., 1])


def snapshot_due(step: int, every: int, *, start: int = 0) -> bool:
    """Whether a time-subsampled snapshot is due at integer ``step``.

    ``True`` iff ``every > 0`` and ``step >= start`` and ``(step - start)`` is a multiple of
    ``every`` — the cadence predicate the run driver checks each step (a pure host-int test, no
    device sync). ``every <= 0`` disables snapshots."""
    return every > 0 and step >= start and (step - start) % every == 0


def write_snapshot(out_dir, state_folded, sm, part, *, step, every, fields=None,
                   start=0, attrs=None):
    """Write a time-subsampled instantaneous snapshot to ``out_dir/snap_<step>`` IF due.

    A thin cadence wrapper over :func:`write_state_zarr` (the existing sharded, gather-free
    path) for the visual fields / KE-spectra / LKF-deformation snapshots: writes only the
    selectable ``fields`` (default :data:`DEFAULT_FIELDS`) at the configured ``every`` cadence,
    stamping the ``step`` into the store attrs. Returns the snapshot path when written, else
    ``None`` (so the driver can log it). The accumulated mean/variance (:class:`OnlineStats`)
    is written separately by the driver at run end."""
    if not snapshot_due(int(step), int(every), start=int(start)):
        return None
    d = Path(out_dir) / f"snap_{int(step):08d}"
    return write_state_zarr(d, state_folded, sm, part, fields=fields,
                            attrs={**(attrs or {}), "step": int(step)})
