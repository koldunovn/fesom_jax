"""The sharded chunk of the CORE2 adjoint drivers (the fast-adjoint plan, Task 5, docs/plans/20261022-core2-fast-adjoint.md).

:func:`build` gives a :class:`ShardedChunk` whose :meth:`~ShardedChunk.make_chunk_fn` ``chunk_fn(carry, xs, u)`` is
the dense drivers' chunk (``scripts/adjoint/core2_adj._make_chunk_fn``: ``K`` steps of :func:`fesom_jax.step.step`
with the derivative-only :class:`~fesom_jax.adjoint_stab.AdjointConfig`, per-step remat, per-node running sums)
under ``shard_map`` over ``P`` devices, for :func:`fesom_jax.adjoint.chunked_grad` / ``chunked_forward`` unchanged:

* ``carry`` = the FOLDED ``[P*Lmax, …]`` State (:func:`fesom_jax.integrate_sharded.folded_state`), or ``(State,
  acc)`` with ``acc`` a dict of folded ``[P*Lmax_nod]`` node fields (the running sums; a scalar accumulator such as a
  box mean would need a global reduction inside the step and is refused);
* ``xs`` = ``(the chunk's folded per-step forcing [K, P*Lmax_nod, …], the run constants)``: the folded mesh, SSH
  operator, halo maps, forcing statics and coastal mask, placed once with their shardings and marked
  :func:`fesom_jax.adjoint.keep` so the per-chunk freeing of ``xs`` leaves them alone (a closed-over array would be
  embedded in the HLO as a constant, without its sharding);
* ``u`` = the control pytree, replicated (scalar controls; a per-node control field would need folding).

The halo transport is the padded slot all_to_all (AD-correct, the production default) or the coloured one; the
ragged one is refused (its transpose is wrong, docs/JAX_RAGGED_A2A_BUG.md). The sea-ice coastal mask is the GLOBAL
one, partitioned (a local recompute treats partition cuts as coasts). Every step continues AB2
(``is_first_step=False``), as the dense chunk.

Host helpers move a global State / node field into the carry layout and back on the owned lanes
(``zarr_output._folded_gid_owned``): costs on the carry use node weights folded with zeros on the halo and pad
lanes, so each node counts once.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from . import adjoint, halo, ops, shard_mesh, ssh
from . import adjoint_costs as AC
from . import precision as PR
from . import integrate_sharded as ish
from . import step as stepmod
from .zarr_output import _folded_gid_owned

_P = PartitionSpec("p")
_R = PartitionSpec()
TRANSPORTS = ("padded", "coloured")
_PHYS = ("ice_cfg", "gm_cfg", "kpp_cfg", "tke_cfg", "ale_cfg", "visc_cfg", "tracer_cfg")


@dataclasses.dataclass(eq=False)
class ShardedChunk:
    """One CORE2 model on ``npes`` devices: the chunk function and the host layout maps (see the module)."""

    npes: int
    part: Any
    sm: Any
    jmesh: Any
    transport: str
    precision: str                   #: "double" or "single" (FESOM2 #1207, as the production sharded runner)
    se_wide: int                     #: the SE wide-halo width K (``FESOM_SE_WIDE``, gpu.se_wide), 0 = per substep
    consts: dict                     #: the placed run constants (``xs[1]``), kept
    specs: dict                      #: their PartitionSpec trees
    gid: dict                        #: kind -> folded global id ``[P*Lmax]`` (-1 on pad lanes)
    owned: dict                      #: kind -> folded owned mask ``[P*Lmax]``
    nod2D: int
    elem2D: int

    # -- host layout ---------------------------------------------------------------------------------------------
    def _kind(self, n0: int) -> str:
        for k in ("nod", "elem"):
            if n0 == self.gid[k].size:
                return k
        raise ValueError(f"leading dim {n0} is neither P*Lmax_nod ({self.gid['nod'].size}) nor P*Lmax_elem "
                         f"({self.gid['elem'].size})")

    def fold_state(self, state):
        """A global host State -> the folded host State of the carry (pad lanes as ``partition_state``); single
        precision: float32 but ``hnode``/``hnode_new``, with the rounding carry (``precision.cast_state``)."""
        fs = jax.tree.map(ish._fold, shard_mesh.partition_state(jax.device_get(state), self.part))
        return PR.cast_state(fs) if self.precision == "single" else fs

    def unfold_state(self, fs):
        """A folded State (host or device) -> the global host State from the owned lanes."""
        def one(a):
            a = np.asarray(a)
            if a.dtype == jax.dtypes.float0:               # a cotangent's integer leaf (the ice counter)
                return a
            k = self._kind(a.shape[0])
            n = self.nod2D if k == "nod" else self.elem2D
            out = np.zeros((n,) + a.shape[1:], a.dtype)
            o = self.owned[k]
            out[self.gid[k][o]] = a[o]
            return out
        return jax.tree.map(one, jax.device_get(fs))

    def fold_nodes(self, v):
        """A global ``[nod2D, …]`` node field -> folded ``[P*Lmax_nod, …]``, ZERO off the owned lanes (cost weights:
        each node counts once)."""
        v = np.asarray(v)
        out = np.zeros((self.gid["nod"].size,) + v.shape[1:], v.dtype)
        o = self.owned["nod"]
        out[o] = v[self.gid["nod"][o]]
        return out

    def unfold_nodes(self, vf):
        """A folded ``[P*Lmax_nod, …]`` node field (host or device) -> global ``[nod2D, …]`` from the owned lanes."""
        vf = np.asarray(jax.device_get(vf))
        out = np.zeros((self.nod2D,) + vf.shape[1:], vf.dtype)
        o = self.owned["nod"]
        out[self.gid["nod"][o]] = vf[o]
        return out

    def place(self, tree):
        """A folded host tree (a carry) onto the devices, every leaf sharded along its leading axis."""
        return ish._to_global_sharded(tree, jax.tree.map(lambda _: _P, tree), self.jmesh)

    # -- the chunk -----------------------------------------------------------------------------------------------
    def make_chunk_fn(self, *, dt: float, phys: Mapping, adjoint_cfg, to_params: Callable,
                      accumulators: Mapping | None = None, rematerialize: bool = True,
                      save_names: tuple = ()) -> Callable:
        """``chunk_fn(carry, xs, u)`` (see the module): ``phys`` the step's physics configs, ``accumulators`` ``name ->
        factory(mesh) -> fn(State)`` per-node running sums (the factory gets the device-local mesh inside the
        ``shard_map``), ``rematerialize``/``save_names`` as :func:`fesom_jax.adjoint.scan_steps`."""
        bad = sorted(set(phys) - set(_PHYS))
        if bad:
            raise ValueError(f"phys: unknown {bad}; known {_PHYS}")
        names = tuple(self.consts)
        specs, jmesh = self.specs, self.jmesh
        n_c = len(names)
        se_K = int(self.se_wide)
        use_padded, use_coloured = self.transport == "padded", self.transport == "coloured"
        col_meta = ish._col_meta(self.sm) if use_coloured else None
        n_global = int(self.sm.nod2D)
        acc_f = dict(accumulators or {})
        phys = dict(phys)
        save = tuple(save_names) if rematerialize else ()

        def body(*args):
            cs = dict(zip(names, args[:n_c]))
            carry, seq, u = args[n_c:]
            ctx = ish._make_halo_ctx(cs["ha"], n_global, False, use_padded, None, use_coloured, col_meta)
            if se_K:                               # gpu.se_wide: one SE exchange every K substeps (K+1-ring lanes)
                from . import se_wide as se_wide_mod
                ctx = dataclasses.replace(ctx, se_wide=se_wide_mod.local_se_wide(cs["sw"], se_K, ctx.axis_name))
            p = PR.wp_tree(to_params(u))          # single precision: WP Params (the controls stay float64)
            if "gid" in cs:                        # the global forcing controls cut to this device's lanes
                p = localize_forcing_adj(p, cs["gid"], cs["valid"])
            fm = cs["fm"]

            def step3(s, sf, month):
                pp = p if month is None else AC.month_params(p, month)
                return stepmod.step(s, fm, cs["fop"], cs["fstress"], pp, dt=dt, is_first_step=False, step_forcing=sf,
                                    forcing_static=cs["fsf"], halo_ctx=ctx, boundary_node=cs.get("bn"),
                                    adjoint_cfg=adjoint_cfg, **phys)

            accs = {k: f(fm) for k, f in acc_f.items()}
            if _scheduled(seq):                    # (step forcing, the per-step schedule {month, cost_w})
                stepf = AC.with_schedule(step3, accs)
            else:
                one_step = lambda s, sf: step3(s, sf, None)            # noqa: E731
                stepf = AC.with_running_costs(one_step, accs) if accs else one_step
            return adjoint.scan_steps(stepf, carry, seq, rematerialize=rematerialize, save_names=save)

        def chunk_fn(carry, xs, u):
            seq, cs = xs
            carry_spec = jax.tree.map(lambda _: _P, carry)
            if _scheduled(seq):
                seq_spec = (jax.tree.map(lambda _: PartitionSpec(None, "p"), seq[0]), jax.tree.map(lambda _: _R, seq[1]))
            else:
                seq_spec = jax.tree.map(lambda _: PartitionSpec(None, "p"), seq)
            in_specs = tuple(specs[k] for k in names) + (carry_spec, seq_spec, jax.tree.map(lambda _: _R, u))
            f = jax.shard_map(body, mesh=jmesh, in_specs=in_specs, out_specs=carry_spec, check_vma=False)
            return f(*(cs[k] for k in names), carry, seq, u)
        chunk_fn.precision = self.precision         # adjoint._jitted_forward/_backward: compile options and scope
        return chunk_fn

    def forcing_xs(self, step_forcing_stack, sched=None):
        """The chunk's ``xs``: a global host ``StepForcing`` stack ``[K, nod2D, …]`` partitioned, folded and placed,
        with the run constants; with ``sched`` (host ``{"month": int[K], "cost_w": float[K]}``) the forcing is paired
        with it (replicated)."""
        seq, spec = ish._fold_forcing_seq(shard_mesh.partition_step_forcing(step_forcing_stack, self.part))
        if self.precision == "single":
            seq = PR.cast_tree(seq)
        seq = ish._to_global_sharded(seq, spec, self.jmesh)
        if sched is not None:
            seq = (seq, jax.device_put({k: np.asarray(v) for k, v in sched.items()}))
        return seq, self.consts


def _scheduled(seq) -> bool:
    """A scheduled chunk's forcing ``(StepForcing, {month, cost_w})``: a plain tuple (a ``StepForcing`` is a NamedTuple,
    so ``isinstance(seq, tuple)`` holds for the plain forcing too)."""
    return type(seq) is tuple


def localize_forcing_adj(params, gid, valid):
    """``params`` with GLOBAL ``Params.forcing_adj`` maps (``{name: [..., nod2D]}``, replicated on every device; a
    leading window-month axis allowed) cut to this device's lanes: ``field[..., gid]`` on the owned and halo lanes, 0
    on the pad lanes (after the ice session's ``integrate_sharded.localize_forcing_adj``, e0381dc). The transpose
    scatters each lane's cotangent back to its global node, so the gradient that leaves the ``shard_map``
    (replicated input ⇒ summed over devices) is the global one, the halo copies' uses included. ``None`` ⇒ ``params``."""
    adj = getattr(params, "forcing_adj", None)
    if adj is None:
        return params
    g = jnp.where(gid >= 0, gid, 0)
    loc = {k: jnp.where(valid, jnp.take(jnp.asarray(v), g, axis=-1), jnp.zeros((), jnp.asarray(v).dtype))
           for k, v in adj.items()}
    return dataclasses.replace(params, forcing_adj=loc)


def build(mesh, part, op, forcing_static, *, phys: Mapping, transport: str = "padded",
          devices=None, precision: str = "double", node_ids: bool = False) -> ShardedChunk:
    """The layout and the placed run constants of one model on ``part.npes`` devices; :meth:`ShardedChunk.
    make_chunk_fn` then gives its chunk functions. ``op`` = the dense SSH operator (partitioned here),
    ``forcing_static`` the dense ``ForcingStatic``, ``phys`` the step's physics configs (it decides the SE tables and
    the coastal mask). ``precision="single"``: the constants cast as the production sharded runner casts them, the
    carry and forcing cast on the host (:meth:`ShardedChunk.fold_state`, :meth:`ShardedChunk.forcing_xs`), the
    chunk compiled and run in :func:`fesom_jax.precision.scope` (``chunk_fn.precision``)."""
    PR.check(precision)
    if transport not in TRANSPORTS:
        raise ValueError(f"transport must be one of {TRANSPORTS} (ragged: its transpose is wrong), got {transport!r}")
    bad = sorted(set(phys) - set(_PHYS))
    if bad:
        raise ValueError(f"phys: unknown {bad}; known {_PHYS}")
    npes = int(part.npes)
    devs = jax.devices()[:npes] if devices is None else list(devices)
    if len(devs) < npes:
        raise ValueError(f"the partition has {npes} parts, {len(devs)} devices")
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    if ops.ordered_sums_enabled():                          # gpu.ordered_sums, as run.py
        sm = shard_mesh.with_inverse_maps(sm)
    ale_cfg = phys.get("ale_cfg")
    if ale_cfg is not None and ale_cfg.split_explicit is not None:
        sm = shard_mesh.with_se_tables(sm, mesh, part)
    sop = ssh.partition_ssh_operator(op, part)
    if ssh.ssh_ell_enabled():
        sop = ssh.to_ell_sharded(sop)
    fs_p = shard_mesh.partition_forcing_static(forcing_static, part)
    stress_p = np.zeros((npes, sm.Lmax["elem"], 2))         # unused under forcing (the bulk recomputes it)
    bn_p = None
    if phys.get("ice_cfg") is not None:                     # the GLOBAL coastal mask, partitioned (run.py)
        from . import ice_evp
        bn_p = shard_mesh._shard_along_axis(np.asarray(ice_evp.boundary_node_mask(mesh)), part.myList_nod2D,
                                            sm.Lmax["nod"], 0, False)
    # gpu.se_wide (FESOM_SE_WIDE, read here as run.py reads it): the SE substeps on K+1-ring lanes, one exchange
    # every K substeps; its lane tables are run constants like the mesh. (gpu.mevp_wide is not taken: the
    # frozen-sweep ice backward runs its own per-subcycle exchanges.)
    from . import se_wide as se_wide_mod
    se_K = se_wide_mod.se_wide_from_env() if (ale_cfg is not None and ale_cfg.split_explicit is not None) else 0
    sw_host = se_wide_mod.build_se_wide(mesh, part, sm, se_K) if se_K else None
    se_K = se_K if sw_host is not None else 0
    c = ish._forced_consts(sm, sop, stress_p, fs_p, bn_p, None, None, None, False, transport == "padded",
                           transport == "coloured", precision, sw_host)
    jmesh = halo.device_mesh("p", devices=devs)
    names = (("fm", "fop", "fstress", "ha", "fsf") + (("bn",) if bn_p is not None else ())
             + (("sw",) if sw_host is not None else ()))
    if node_ids:                                     # the folded global node id and live mask of each node lane
        gid_n, _ = _folded_gid_owned(part, sm, "nod")
        c["gid"], c["gid_spec"] = gid_n.astype(np.int32), _P
        c["valid"], c["valid_spec"] = np.asarray(sm.valid_mask["nod"]).reshape(-1), _P
        names = names + ("gid", "valid")
    specs = {k: c[k + "_spec"] for k in names}
    consts = adjoint.keep({k: ish._to_global_sharded(c[k], specs[k], jmesh) for k in names})
    gid, owned = {}, {}
    for k in ("nod", "elem"):
        gid[k], owned[k] = _folded_gid_owned(part, sm, k)
    return ShardedChunk(npes=npes, part=part, sm=sm, jmesh=jmesh, transport=transport, precision=precision,
                        se_wide=se_K, consts=consts,
                        specs=specs, gid=gid, owned=owned, nod2D=int(mesh.nod2D), elem2D=int(mesh.elem2D))


def init_accumulators(chunk: ShardedChunk, shapes: Mapping[str, tuple]) -> dict:
    """Zero folded running sums ``[P*Lmax_nod]`` (float64), one per accumulator; ``shapes`` = each accumulator's value
    shape on the dense model. Only per-node fields ``(nod2D,)``: a scalar (a box mean) would need a global reduction
    inside the step, and is refused."""
    out = {}
    for k, shp in shapes.items():
        if tuple(shp) != (chunk.nod2D,):
            raise ValueError(f"accumulator {k!r}: value shape {tuple(shp)}, the sharded chunk sums per-node fields "
                             f"({chunk.nod2D},) only")
        out[k] = np.zeros((chunk.gid["nod"].size,), np.float64)
    return out
