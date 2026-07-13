"""Device-mesh placement + ``shard_map`` wrapper for the full step (Phase 8, Task S.7).

Turns the host-side :class:`~fesom_jax.shard_mesh.ShardedMesh` (S.2) + partitioned
:class:`~fesom_jax.state.State` (S.2b) + per-device :class:`~fesom_jax.ssh.ShardedSSHOperator`
(S.6) into the **device-placed** inputs of a single ``jax.shard_map`` over a 1-D device
mesh ``('p',)``, and reconstructs the per-device LOCAL :class:`~fesom_jax.mesh.Mesh` /
``State`` / ``SSHOperator`` *inside* the body so the existing :func:`fesom_jax.step.step`
runs **unchanged** on each device's ``[Lmax, …]`` shard.

Sharding convention (S.3): the device axis is folded INTO the leading dim — a global
``[P, Lmax_kind, …]`` array becomes ``[P*Lmax_kind, …]`` sharded ``PartitionSpec('p')``,
so each device sees ``[Lmax_kind, …]`` (the natural per-device shape, no stray size-1
axis). The reconstructed local ``Mesh``'s **static sizes are the local ``Lmax``** (so the
kernels' ``num_segments``/shape bounds are local), and the omitted node→elem CSR
(``nod_in_elem2D``, IC-only) is a step-unused dummy.

``halo`` exchanges + the fused-kernel splits + the reduction routing live in
:mod:`fesom_jax.step` behind a static-arg gate (the remainder of S.7); this module is the
placement scaffold + the ``npes==1`` no-op invariant (the whole step under ``shard_map`` on
one device == the dense step, byte-identical — the proof the plumbing is correct).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec, NamedSharding

from . import halo
from . import step as stepmod
from .halo import HaloCtx
from .mesh import Mesh
from .params import Params
from .shard_mesh import (CONN_FIELDS, EDGE_FIELDS, ELEM_FIELDS, NODE_FIELDS,
                         REPLICATED_FIELDS, ShardedMesh)
from .ssh import SSHHalo, SSHOperator, ShardedSSHOperator
from .state import State

_P = PartitionSpec("p")          # sharded on the device axis (folded leading dim)
_R = PartitionSpec()             # replicated (every device gets the full array)
_TP = PartitionSpec(None, "p")   # [n_steps, P*Lmax]: time replicated, the folded dim sharded (A4)


def _to_global_sharded(x, spec, mesh):
    """Place a host-numpy folded input ``x`` (+ its ``PartitionSpec`` tree) onto a
    ``NamedSharding`` over ``mesh`` by copying **each addressable shard directly from the host
    numpy to its device** (:func:`jax.make_array_from_callback`), so NO global-sized array is
    ever staged on a single device.

    **Why not ``device_put`` (the NG5 fix, Phase 8b B.3):** ``jax.device_put`` of a folded
    global array routed a ``jit__identity_fn`` that materialized the WHOLE folded global
    ``[P·Lmax, …]`` on one GPU before sharding — ~56 GB for dars (fit an 80 GB A100, so it
    worked) but **~125.81 GiB for NG5 dist_32 ⇒ OOM**. ``make_array_from_callback`` calls the
    per-shard callback with each addressable shard's index and pulls ONLY that slice from the
    host numpy, so the global never lands on a device. Single-process: every shard addressable.
    Multi-process: each process builds the full global host array and places only its local
    shards (the rows for its local devices)."""
    shardings = jax.tree.map(lambda sp: NamedSharding(mesh, sp), spec,
                             is_leaf=lambda v: isinstance(v, PartitionSpec))

    def _place(leaf, sharding):
        # A device-resident jax.Array (a resumed/multi-chunk State, multi-node) is ALREADY sharded —
        # reshard on device to the target (no-op if it matches). np.asarray-ing it would pull a
        # MULTI-PROCESS global array to host ("spans non-addressable devices"). Host numpy (the
        # cold-start build pipeline) takes the per-shard callback path (no global staged on a device).
        if isinstance(leaf, jax.Array):
            return leaf if leaf.sharding == sharding else jax.device_put(leaf, sharding)
        a = np.asarray(leaf)                      # already host numpy (the host-build pipeline)
        return jax.make_array_from_callback(a.shape, sharding, lambda idx, a=a: a[idx])

    return jax.tree.map(_place, x, shardings)

# entity-leading mesh fields (shard 'p'); the rest (zbar/Z + the CSR dummy) replicate.
_ENTITY_FIELDS = (set(NODE_FIELDS) | set(ELEM_FIELDS) | set(EDGE_FIELDS)
                  | {c[0] for c in CONN_FIELDS})


# --------------------------------------------------------------------------
# Local reconstruction (one device's [Lmax, …] Mesh)
# --------------------------------------------------------------------------
def local_mesh(sm: ShardedMesh, d: int) -> Mesh:
    """Reconstruct device ``d``'s local :class:`Mesh` from a :class:`ShardedMesh`
    (host helper / tests). Static sizes are the local ``Lmax``; the node→elem CSR is
    a step-unused dummy (S.2 omitted it — IC-only). The ``npes==1`` case is array-equal
    to the dense ``Mesh`` for every step-read field (verified)."""
    Ln, Le, Led = sm.Lmax["nod"], sm.Lmax["elem"], sm.Lmax["edge"]
    f = {name: jnp.asarray(arr if name in REPLICATED_FIELDS else arr[d])
         for name, arr in sm.fields.items()}
    return Mesh(
        **f,
        nod_in_elem2D_offsets=jnp.zeros(Ln + 1, jnp.int32),
        nod_in_elem2D=jnp.zeros(1, jnp.int32),
        nod2D=Ln, elem2D=Le, edge2D=Led, nl=sm.nl,
        edge2D_in=sm.edge2D_in, myDim_edge2D=Led, ocean_area=sm.ocean_area,
    )


# --------------------------------------------------------------------------
# Fold helpers ([P, Lmax_kind, …] → [P*Lmax_kind, …]) + matching spec-trees
# --------------------------------------------------------------------------
def _fold(arr):
    """``[P, X, …] → [P*X, …]`` (fold the device axis into the leading dim).

    **Polymorphic placement (Phase 8b B.3, the dars/NG5 setup-OOM fix):** a concrete
    ``np.ndarray`` input (the host-built data pipeline — ``partition_state`` etc.) stays on
    the HOST (numpy reshape) so the full global folded array is NEVER materialized on GPU 0
    before ``device_put`` shards it; a traced/jax input (grad-through-fold, e.g. the IC-field
    gradient gate folds a ``jax.grad`` tracer of ``state_p.T``) goes through ``jnp`` so
    autodiff still flows. The reshape is value-identical either way."""
    a = arr if isinstance(arr, np.ndarray) else jnp.asarray(arr)
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def _unfold(arr, P: int) -> jax.Array:
    """``[P*X, …] → [P, X, …]``."""
    a = jnp.asarray(arr)
    return a.reshape((P, a.shape[0] // P) + a.shape[1:])


def folded_mesh(sm: ShardedMesh) -> tuple[Mesh, Mesh]:
    """Build the ``shard_map`` input ``Mesh`` (folded ``[P*Lmax_kind, …]`` leaves, local
    ``Lmax`` static sizes) and its ``PartitionSpec`` tree. Entity-leading fields shard
    ``'p'``; ``zbar``/``Z`` + the CSR dummy replicate."""
    Ln, Le, Led = sm.Lmax["nod"], sm.Lmax["elem"], sm.Lmax["edge"]
    data: dict = {}
    spec: dict = {}
    for name, arr in sm.fields.items():
        if name in REPLICATED_FIELDS:
            data[name] = np.asarray(arr)          # host (replicated; device_put places it)
            spec[name] = _R
        else:
            data[name] = _fold(arr)               # host numpy (sm.fields are numpy)
            spec[name] = _P
    data["nod_in_elem2D_offsets"] = np.zeros(Ln + 1, np.int32)
    data["nod_in_elem2D"] = np.zeros(1, np.int32)
    spec["nod_in_elem2D_offsets"] = _R
    spec["nod_in_elem2D"] = _R
    meta = dict(nod2D=Ln, elem2D=Le, edge2D=Led, nl=sm.nl, edge2D_in=sm.edge2D_in,
                myDim_edge2D=Led, ocean_area=sm.ocean_area)
    return Mesh(**data, **meta), Mesh(**spec, **meta)


def folded_state(state_p: State) -> tuple[State, State]:
    """Fold a partitioned ``[P, Lmax, …]`` :class:`State` to ``[P*Lmax, …]`` + its
    all-``'p'`` spec-tree (every State field is entity-leading)."""
    fs = jax.tree.map(_fold, state_p)
    return fs, jax.tree.map(lambda _: _P, fs)


def _fold_forcing(f):
    """Fold a ``[P, Lmax_nod]``-leaf ``StepForcing``/``ForcingStatic`` NamedTuple (from
    :func:`shard_mesh.partition_step_forcing` / ``partition_forcing_static``) to a
    ``[P*Lmax_nod]``-leaf ``shard_map`` input + its spec-tree: node fields shard ``'p'``,
    the replicated scalar ``ocean_area`` (a 0-d leaf) stays ``PartitionSpec()`` (it becomes a
    ``psum`` over owned nodes in the reductions). Same-typed NamedTuples out, so the spec-tree
    matches the input pytree structure."""
    data, spec = {}, {}
    for name in f._fields:
        arr = getattr(f, name)
        if np.ndim(arr) == 0:
            data[name], spec[name] = np.asarray(arr), _R   # scalar (ocean_area) → replicated (host)
        else:
            data[name], spec[name] = _fold(arr), _P        # [P, Lmax_nod, …] → [P*Lmax_nod, …] (host)
    return type(f)(**data), type(f)(**spec)


def _fold_forcing_seq(sf_seq):
    """Fold a **stacked** ``[P, n_steps, Lmax_nod, …]``-leaf ``StepForcing`` (from
    :func:`shard_mesh.partition_step_forcing` of an ``[n_steps, nod2D]`` stack) into a per-step
    scan ``xs``: ``[n_steps, P*Lmax_nod, …]`` leaves + a ``PartitionSpec(None,'p')`` spec-tree
    (the ``n_steps`` axis replicated, the folded device axis sharded). Each ``shard_map`` device
    then sees its ``[n_steps, Lmax_nod, …]`` slice; ``lax.scan`` over axis 0 hands one step's
    local forcing to each step (Task A4 — vs the device-CONSTANT ``_fold_forcing`` the timing
    path uses). Mirrors :func:`_fold` (host numpy in ⇒ host numpy out, never staged on a device)."""
    data, spec = {}, {}
    for name in sf_seq._fields:
        arr = np.asarray(getattr(sf_seq, name))            # [P, T, Lmax, …]
        P, T = arr.shape[0], arr.shape[1]
        rest = arr.shape[2:]                                # (Lmax, …)
        moved = np.transpose(arr, (1, 0) + tuple(range(2, arr.ndim)))   # [T, P, Lmax, …]
        data[name] = moved.reshape((T, P * rest[0]) + rest[1:])         # [T, P*Lmax, …]
        spec[name] = _TP
    return type(sf_seq)(**data), type(sf_seq)(**spec)


def folded_operator(sop: ShardedSSHOperator) -> tuple[SSHOperator, SSHOperator]:
    """Fold the per-device :class:`ShardedSSHOperator` to a ``shard_map`` input
    :class:`SSHOperator` (``[P*nnz_max]`` rows/cols/vals, ``[P*Lmax_nod]`` diag,
    ``n_nodes=Lmax_nod`` static) + its all-``'p'`` spec."""
    op = SSHOperator(
        rows=_fold(sop.rows), cols=_fold(sop.cols),
        stiff_vals=_fold(sop.stiff_vals), precond_vals=_fold(sop.precond_vals),
        diag=_fold(sop.diag), n_nodes=sop.Lmax_nod,
    )
    return op, jax.tree.map(lambda _: _P, op)


def unfold_state(folded: State, P: int) -> State:
    """``[P*Lmax, …]`` State → ``[P, Lmax, …]`` (one row per device)."""
    return jax.tree.map(lambda a: _unfold(a, P), folded)


# --------------------------------------------------------------------------
# The shard_map wrapper (npes==1 no-op scaffold; halo/splits are the rest of S.7)
# --------------------------------------------------------------------------
_RKINDS = ("nod", "elem", "edge")


def _halo_arrays(sm: ShardedMesh, ragged: bool = False,
                 padded: bool = False) -> tuple[dict, dict]:
    """Fold the per-kind exchange maps + the node owned-mask to ``[P*Lmax_kind]``
    ``shard_map`` inputs + their all-``'p'`` spec dict. With ``ragged=True`` (Phase 8b)
    ALSO fold the :class:`~fesom_jax.shard_mesh.RaggedExchange` maps for the halo-only
    ``ragged_all_to_all`` path; with ``padded=True`` (Phase 8c) instead fold its
    ``pad_*`` maps for the slot-padded dense ``all_to_all`` (left out otherwise ⇒ the
    all_gather traces are byte-unchanged)."""
    ha: dict = {}
    for k in _RKINDS:
        src_dev, src_lane = sm.exchange[k]
        ha[f"sd_{k}"] = _fold(src_dev).astype(np.int32)       # mesh constants → host numpy
        ha[f"sl_{k}"] = _fold(src_lane).astype(np.int32)
        if ragged or padded:
            r = sm.exchange_ragged[k]
            ha[f"rhm_{k}"] = _fold(r.halo_mask)
        if ragged:
            ha[f"rsi_{k}"] = _fold(r.send_idx).astype(np.int32)
            ha[f"rss_{k}"] = _fold(r.send_sizes).astype(np.int32)
            ha[f"rso_{k}"] = _fold(r.send_offsets).astype(np.int32)
            ha[f"roo_{k}"] = _fold(r.out_offsets).astype(np.int32)
            ha[f"rrs_{k}"] = _fold(r.recv_sizes).astype(np.int32)
            ha[f"rrg_{k}"] = _fold(r.recv_gather).astype(np.int32)
        if padded:
            ha[f"pps_{k}"] = _fold(r.pad_src).astype(np.int32)
            ha[f"ppv_{k}"] = _fold(r.pad_valid)
            ha[f"ppp_{k}"] = _fold(r.pad_slotpos).astype(np.int32)
    ha["owned_nod"] = _fold(sm.owned_mask["nod"])
    return ha, {k: _P for k in ha}


def _ragged_ctx(h: dict) -> dict:
    """Re-assemble the per-device :class:`RaggedExchange` dict (inside ``shard_map``)
    from the folded halo-arrays ``h`` — the ``{kind: {...}}`` :class:`HaloCtx` consumes."""
    return {k: {"send_idx": h[f"rsi_{k}"], "send_sizes": h[f"rss_{k}"],
                "send_off": h[f"rso_{k}"], "out_off": h[f"roo_{k}"],
                "recv_sizes": h[f"rrs_{k}"], "recv_gather": h[f"rrg_{k}"],
                "halo_mask": h[f"rhm_{k}"]} for k in _RKINDS}


def _padded_ctx(h: dict) -> dict:
    """The Phase 8c analogue of :func:`_ragged_ctx`: the per-device padded dense-a2a
    maps, ``{kind: {pad_src, pad_valid, pad_slotpos, halo_mask}}``."""
    return {k: {"pad_src": h[f"pps_{k}"], "pad_valid": h[f"ppv_{k}"],
                "pad_slotpos": h[f"ppp_{k}"], "halo_mask": h[f"rhm_{k}"]}
            for k in _RKINDS}


def _recv_max(sm: ShardedMesh) -> dict:
    """Per-kind static recv-buffer extent for ``ragged_all_to_all`` (closed into the body)."""
    return {k: sm.exchange_ragged[k].recv_max for k in _RKINDS}


def _make_halo_ctx(h: dict, n_global: int, use_ragged: bool, use_padded: bool,
                   recv_max) -> HaloCtx:
    """Build the per-device :class:`HaloCtx` (with its :class:`SSHHalo`) inside
    ``shard_map`` from the folded halo-arrays ``h`` — the one construction every
    sharded body shares, routed by the transport flags (padded / ragged / all_gather)."""
    exch = {k: (h[f"sd_{k}"], h[f"sl_{k}"]) for k in _RKINDS}
    rmaps = (_ragged_ctx(h) if use_ragged
             else _padded_ctx(h) if use_padded else None)
    ssh_halo = SSHHalo(src_dev=h["sd_nod"], src_lane=h["sl_nod"],
                       owned_mask=h["owned_nod"], n_global=n_global, axis_name="p",
                       ragged=rmaps["nod"] if rmaps is not None else None,
                       recv_max=recv_max["nod"] if use_ragged else 0,
                       use_ragged=use_ragged, use_padded=use_padded)
    return HaloCtx(exch=exch, axis_name="p", ssh_halo=ssh_halo,
                   owned_mask={"nod": h["owned_nod"]},
                   exch_ragged=rmaps, recv_max=recv_max,
                   use_ragged=use_ragged, use_padded=use_padded)


def _check_halo_mode(use_ragged: bool, use_padded: bool):
    """The two point-to-point transports are mutually exclusive — refuse at entry."""
    if use_ragged and use_padded:
        raise ValueError("use_ragged and use_padded are mutually exclusive halo "
                         "transports (padded IS the ragged substitute; pick one)")


def run_gm_diag_sharded(sm: ShardedMesh, T_p, S_p, bvfreq_p, hnode_new_p, helem_p,
                        *, npes: int, gm_cfg, params=None):
    """Run :func:`fesom_jax.gm.gm_diagnostics` (WITH the S.7-part-3 GM halo exchanges)
    under ``shard_map`` over ``npes`` devices — the per-kernel GM-exchange gate (the S.4
    scatter-gate analogue, isolating the GM coefficient/bolus chain from the FCT). Inputs
    are ``[P, Lmax, …]`` partitioned (``T``/``S``/``bvfreq``/``hnode_new`` node, ``helem``
    elem). Returns the ``[P, Lmax, …]`` ``(fer_uv, slope_tapered, Ki)`` — the fields the
    bolus advection + Redi terms consume — so a caller can assert they match single-device
    on OWNED entities (≈ scatter floor), proving the ``fer_gamma``/``fer_uv``/
    ``slope_tapered``/``Ki`` exchanges are correct (hence any residual FCT-tracer N-vs-1
    spread is the upwind-flip floor, not a missing exchange)."""
    from . import gm
    if params is None:
        params = Params.defaults()
    fm, fm_spec = folded_mesh(sm)
    ha, ha_spec = _halo_arrays(sm)
    fT, fS, fbv, fhn = _fold(T_p), _fold(S_p), _fold(bvfreq_p), _fold(hnode_new_p)
    fhe = _fold(helem_p)
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])

    def body(m, T, S, bv, hn, he, h):
        exch = {k: (h[f"sd_{k}"], h[f"sl_{k}"]) for k in ("nod", "elem", "edge")}
        ctx = HaloCtx(exch=exch, axis_name="p", ssh_halo=None,
                      owned_mask={"nod": h["owned_nod"]})
        d = gm.gm_diagnostics(m, T, S, bv, hn, he, params, gm_cfg, exch=ctx.exchange)
        return d.fer_uv, d.slope_tapered, d.Ki

    out = jax.shard_map(body, mesh=jmesh,
                        in_specs=(fm_spec, _P, _P, _P, _P, _P, ha_spec),
                        out_specs=(_P, _P, _P), check_vma=False)(
                            fm, fT, fS, fbv, fhn, fhe, ha)
    return tuple(_unfold(o, npes) for o in out)


def run_step_sharded(sm: ShardedMesh, state_p: State, sop: ShardedSSHOperator,
                     stress_p, *, dt: float, is_first_step: bool, npes: int,
                     wire_halo: bool = True, params=None, step_forcing=None,
                     forcing_static=None, ice_cfg=None, gm_cfg=None,
                     kpp_cfg=None, tke_cfg=None, ale_cfg=None, visc_cfg=None,
                     tracer_cfg=None, boundary_node_p=None,
                     use_ragged: bool = False, use_padded: bool = False) -> State:
    """Run one :func:`fesom_jax.step.step` under ``shard_map`` over ``npes`` devices and
    return the ``[P, Lmax, …]`` next State.

    ``wire_halo`` (default True): build the per-device :class:`~fesom_jax.halo.HaloCtx`
    (the exchange maps + the S.6 :class:`~fesom_jax.ssh.SSHHalo`) and thread it into the
    step ⇒ the broadcast halo refreshes fire at the C's exchange points. ``wire_halo=False``
    passes ``halo_ctx=None`` (the dead-branch dense path — every exchange is the identity)
    for the ``npes==1`` byte-identity no-op. ``step_forcing``/``stress_p`` must already be
    partitioned to ``[P, Lmax, …]``.

    ``check_vma=False`` is required (the kernels' tridiagonal-solve / FCT ``lax.scan``s
    carry CONSTANT initial carries — non-"varying" under ``shard_map``'s varying-manual-axes
    typing — while their bodies produce per-device-varying outputs; relaxing it treats every
    value conservatively as per-device-varying, always correct here, so the kernels lower)."""
    fm, fm_spec = folded_mesh(sm)
    fs, fs_spec = folded_state(state_p)
    fop, fop_spec = folded_operator(sop)
    fstress = _fold(stress_p)
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    n_global = sm.nod2D

    # Optional CORE2 forcing + the GLOBAL ice boundary_node → sharded shard_map inputs
    # (varargs; () when absent ⇒ the no-forcing pi/GM path traces EXACTLY as before). The
    # forcing folds the StepForcing/ForcingStatic NamedTuples; boundary_node_p is the
    # PARTITIONED global coastal mask (the local-mesh recompute mis-flags partition-boundary
    # nodes as coastal — the EVP needs the global one). `have_*` flags are closed over so the
    # body can parse the positional varargs (shard_map inputs can't be None).
    extras, extras_spec = [], []
    have_forcing = step_forcing is not None
    if have_forcing:
        sff, sff_spec = _fold_forcing(step_forcing)
        fsf, fsf_spec = _fold_forcing(forcing_static)
        extras += [sff, fsf]
        extras_spec += [sff_spec, fsf_spec]
    have_bn = boundary_node_p is not None
    if have_bn:
        extras.append(_fold(boundary_node_p))
        extras_spec.append(_P)
    extras, extras_spec = tuple(extras), tuple(extras_spec)

    def _run(m, s, o, stress, ctx, ex):
        i = 0
        if have_forcing:
            sf_, fs_, i = ex[0], ex[1], 2
        else:
            sf_, fs_ = None, None
        bn_ = ex[i] if have_bn else None
        return stepmod.step(s, m, o, stress, params, dt=dt,
                            is_first_step=is_first_step, step_forcing=sf_,
                            forcing_static=fs_, ice_cfg=ice_cfg, gm_cfg=gm_cfg,
                            kpp_cfg=kpp_cfg, tke_cfg=tke_cfg, ale_cfg=ale_cfg,
                            visc_cfg=visc_cfg, tracer_cfg=tracer_cfg, halo_ctx=ctx,
                            boundary_node=bn_)

    if not wire_halo:
        def body0(m, s, o, stress, *ex):
            return _run(m, s, o, stress, None, ex)
        out = jax.shard_map(body0, mesh=jmesh,
                            in_specs=(fm_spec, fs_spec, fop_spec, _P) + extras_spec,
                            out_specs=fs_spec, check_vma=False)(fm, fs, fop, fstress, *extras)
        return unfold_state(out, npes)

    _check_halo_mode(use_ragged, use_padded)
    ha, ha_spec = _halo_arrays(sm, ragged=use_ragged, padded=use_padded)
    recv_max = _recv_max(sm) if use_ragged else None

    def body(m, s, o, stress, h, *ex):
        ctx = _make_halo_ctx(h, n_global, use_ragged, use_padded, recv_max)
        return _run(m, s, o, stress, ctx, ex)

    out = jax.shard_map(body, mesh=jmesh,
                        in_specs=(fm_spec, fs_spec, fop_spec, _P, ha_spec) + extras_spec,
                        out_specs=fs_spec, check_vma=False)(fm, fs, fop, fstress, ha, *extras)
    return unfold_state(out, npes)


def run_steps_sharded(sm: ShardedMesh, state_p: State, sop: ShardedSSHOperator,
                      stress_p, n_steps: int, *, dt: float, npes: int, params=None,
                      gm_cfg=None, kpp_cfg=None, tke_cfg=None, ale_cfg=None,
                      visc_cfg=None, tracer_cfg=None,
                      use_ragged: bool = False, use_padded: bool = False,
                      ice_cfg=None, step_forcing=None, forcing_static=None,
                      boundary_node_p=None, return_executable: bool = False,
                      return_grad_fn: bool = False):
    """Multi-step (S.7 part 3): step-1 eager (``is_first_step=True``) + ``lax.scan`` of steps
    2..N under ONE ``shard_map`` — the :func:`fesom_jax.integrate.integrate` pattern, sharded.
    Returns the ``[P, Lmax, …]`` final State after ``n_steps``.

    The scan body's halo exchanges (the ``OCEAN_SCHEDULE`` posts) + the CG are collectives
    (``all_gather``/``psum``) INSIDE ``lax.scan`` inside ``shard_map`` (``check_vma=False``) —
    the same "collective in a scan/loop under ``shard_map``" lowering the S.6 CG ``while_loop``
    and the ice EVP subcycle rely on. ``jax.checkpoint`` wraps the scan body (caps the backward
    memory for the S.8 AD gate; forward-transparent). The :class:`~fesom_jax.halo.HaloCtx` is
    built ONCE inside the body and closed over by every step. **No-forcing path** (pi/GM ocean,
    ``stress_p`` static); the forced multi-step (per-step forcing as the scan ``xs``, a
    ``[n_steps, P*Lmax]`` ``PartitionSpec(None,'p')`` fold) is the follow-up for the PRIMARY
    KPP+GM+ice few-step gate."""
    _check_halo_mode(use_ragged, use_padded)
    if return_grad_fn and use_ragged:
        # lax.ragged_all_to_all has a WRONG reverse-mode transpose (over-counts ~axis_size×,
        # GPU-only; docs/JAX_RAGGED_A2A_BUG.md + the test_halo.py xfail). The forward is
        # byte-exact but any gradient through it is silently wrong — refuse loudly, at entry,
        # before any setup work (guard added by the 2026-07-03 review). use_padded (Phase 8c)
        # is the sanctioned point-to-point GRADIENT path: its transpose is a plain
        # all_to_all, gated bit-exact vs the all_gather oracle.
        raise ValueError(
            "return_grad_fn=True with use_ragged=True would differentiate through "
            "lax.ragged_all_to_all, whose autodiff transpose is broken (silently wrong "
            "gradients; see docs/JAX_RAGGED_A2A_BUG.md). Use use_padded=True (slot-padded "
            "dense all_to_all, AD-correct) or use_ragged=False (all_gather halo) for the "
            "gradient path.")
    fm, fm_spec = folded_mesh(sm)
    fs, fs_spec = folded_state(state_p)
    fop, fop_spec = folded_operator(sop)
    fstress = _fold(stress_p)
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    n_global = sm.nod2D
    ha, ha_spec = _halo_arrays(sm, ragged=use_ragged, padded=use_padded)
    recv_max = _recv_max(sm) if use_ragged else None

    # Full forced+ice step (constant forcing across the scan — correct for TIMING / scaling, the
    # per-step cost is forcing-VALUE-independent; per-step forcing as the scan xs is the science
    # follow-up). Forcing + the global boundary_node folded as device-constant shard_map inputs,
    # mirroring run_step_sharded. All None ⇒ the no-forcing ocean path (byte-unchanged).
    extras, extras_spec = [], []
    have_forcing = step_forcing is not None
    if have_forcing:
        sff, sff_spec = _fold_forcing(step_forcing)
        fsf, fsf_spec = _fold_forcing(forcing_static)
        extras += [sff, fsf]; extras_spec += [sff_spec, fsf_spec]
    have_bn = boundary_node_p is not None
    if have_bn:
        extras.append(_fold(boundary_node_p)); extras_spec.append(_P)
    extras, extras_spec = tuple(extras), tuple(extras_spec)

    def body(m, s, o, stress, h, *ex):
        ctx = _make_halo_ctx(h, n_global, use_ragged, use_padded, recv_max)
        sf_, fs_ = (ex[0], ex[1]) if have_forcing else (None, None)
        bn_ = ex[2 if have_forcing else 0] if have_bn else None

        def one(carry, is_first):
            return stepmod.step(carry, m, o, stress, params, dt=dt, is_first_step=is_first,
                                step_forcing=sf_, forcing_static=fs_, ice_cfg=ice_cfg,
                                gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, tke_cfg=tke_cfg,
                                ale_cfg=ale_cfg, visc_cfg=visc_cfg, tracer_cfg=tracer_cfg,
                                halo_ctx=ctx, boundary_node=bn_)

        st = one(s, True)                            # step 1 eager (AB2 first-step branch)
        if n_steps > 1:
            st, _ = lax.scan(jax.checkpoint(lambda c, _: (one(c, False), None)),
                             st, xs=None, length=n_steps - 1)
        return st

    # GRAD path (S.8 multi-step, e.g. the §3 sharded NN twin): make `params` a REPLICATED shard_map
    # INPUT so jax.grad can differentiate w.r.t. it, and device-place the CONSTANT folded args ONCE
    # here (eager ⇒ _to_global_sharded's np.asarray sees concrete host numpy). The returned run(p)
    # only calls the jitted shard_map jfng(p, *placed) — so the grad trace never reaches np.asarray
    # (the bug when run_steps_sharded itself is wrapped in jax.grad). jit-around-shard_map kept (the
    # jax.checkpoint'd scan backward needs it). `params`/`p` REPLACES the closed-over `params`.
    if return_grad_fn:
        def bodyg(p, m, s, o, stress, h, *ex):
            ctx = _make_halo_ctx(h, n_global, use_ragged, use_padded, recv_max)
            sf_, fs_ = (ex[0], ex[1]) if have_forcing else (None, None)
            bn_ = ex[2 if have_forcing else 0] if have_bn else None

            def one(carry, is_first):
                return stepmod.step(carry, m, o, stress, p, dt=dt, is_first_step=is_first,
                                    step_forcing=sf_, forcing_static=fs_, ice_cfg=ice_cfg,
                                    gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, tke_cfg=tke_cfg,
                                    ale_cfg=ale_cfg, visc_cfg=visc_cfg, tracer_cfg=tracer_cfg,
                                    halo_ctx=ctx, boundary_node=bn_)

            st = one(s, True)
            if n_steps > 1:
                st, _ = lax.scan(jax.checkpoint(lambda c, _: (one(c, False), None)),
                                 st, xs=None, length=n_steps - 1)
            return st

        gspecs = (_R, fm_spec, fs_spec, fop_spec, _P, ha_spec) + extras_spec
        jfng = jax.jit(jax.shard_map(bodyg, mesh=jmesh, in_specs=gspecs, out_specs=fs_spec,
                                     check_vma=False))
        cargs = (fm, fs, fop, fstress, ha, *extras)
        cspecs = (fm_spec, fs_spec, fop_spec, _P, ha_spec) + extras_spec
        cargs = tuple(_to_global_sharded(a, sp, jmesh) for a, sp in zip(cargs, cspecs))

        def run(p):
            return unfold_state(jfng(p, *cargs), npes)
        return run

    # ⚠️ jax.jit AROUND the shard_map is REQUIRED for the BACKWARD (S.8): the scan body is
    # jax.checkpoint'd (a ``closed_call`` primitive), and grad of a closed_call inside a
    # shard_map raises "Eager evaluation of closed_call inside a shard_map isn't yet supported"
    # (JAX 0.10) unless the shard_map-decorated fn is jitted. Forward-transparent (the S.7p3
    # multistep forward gate + the npes==1 byte-identity are unaffected — jit is semantically
    # identity); ``run_step_sharded`` (no scan ⇒ no checkpoint ⇒ no closed_call) doesn't need it.
    body_sm = jax.shard_map(body, mesh=jmesh,
                            in_specs=(fm_spec, fs_spec, fop_spec, _P, ha_spec) + extras_spec,
                            out_specs=fs_spec, check_vma=False)
    jfn = jax.jit(body_sm)
    # ALWAYS place the (host-numpy) folded inputs onto their shards via device_put (Phase 8b
    # B.3): single-process to the local device mesh, multi-process to the global one. This is
    # what keeps the full global off GPU 0 — without it the folded arrays would be re-uploaded
    # whole to the default device (the dars/NG5 setup-OOM). Differentiated inputs are closed
    # over the body (params/kv), not in `args`, so device_put of these constants is grad-safe.
    args = (fm, fs, fop, fstress, ha, *extras)
    specs = (fm_spec, fs_spec, fop_spec, _P, ha_spec) + extras_spec
    args = tuple(_to_global_sharded(a, sp, jmesh) for a, sp in zip(args, specs))
    # return_executable: hand back the jitted fn + its inputs so a caller (the timing benchmark)
    # can compile ONCE then time a SECOND call that reuses the executable — otherwise every
    # run_steps_sharded call rebuilds the shard_map + re-jits (a fresh closure ⇒ cache miss ⇒
    # recompile), so a one-shot call's wall-time is dominated by XLA COMPILE, not stepping.
    if return_executable:
        return jfn, args, npes
    return unfold_state(jfn(*args), npes)


# Compiled-executable cache for run_steps_sharded_forced(reuse_executable=True) — keyed by the full
# set of compilation determinants (see the call site). Process-lifetime; clear between independent
# setups (different meshes/cfgs reusing freed object ids) via clear_forced_jit_cache().
_FORCED_JIT_CACHE: dict = {}


def clear_forced_jit_cache() -> None:
    """Drop all cached forced-step executables (call between independent run setups / in tests)."""
    _FORCED_JIT_CACHE.clear()


def run_steps_sharded_forced(sm: ShardedMesh, state_p: State, sop: ShardedSSHOperator,
                             stress_p, step_forcing_seq, forcing_static, n_steps: int, *,
                             dt: float, npes: int, params=None, gm_cfg=None, kpp_cfg=None,
                             tke_cfg=None, ale_cfg=None, ice_cfg=None, visc_cfg=None,
                             tracer_cfg=None, boundary_node_p=None, use_ragged: bool = False,
                             use_padded: bool = False,
                             bootstrap_ab2: bool = True, state_is_folded: bool = False,
                             return_folded: bool = False, return_executable: bool = False,
                             reuse_executable: bool = False, sample_fn=None):
    """Multi-step sharded forward with **per-step time-varying forcing** (Task A4).

    Like :func:`run_steps_sharded` but the surface forcing **changes every step** — the real
    seasonal cycle NG5/dars need — instead of being held constant across the scan (which is
    correct only for TIMING, where the per-step cost is forcing-value-independent). The
    mechanism: ``step_forcing_seq`` (a ``StepForcing`` pre-partitioned to ``[P, n_steps, Lmax]``
    by :func:`shard_mesh.partition_step_forcing`) is folded to a ``[n_steps, P*Lmax]`` scan
    ``xs`` with a ``PartitionSpec(None,'p')`` spec and threaded through the ``lax.scan`` so step
    ``i`` consumes ``step_forcing_seq[i]``. ``forcing_static`` is constant ⇒ a device-constant
    input (as before). ``stress_p`` is unused under forcing (the bulk recomputes the stress from
    the live SST/current), kept for signature parity with :func:`run_steps_sharded`.

    **Chunk/memory policy (the A6 driver's contract):** pre-stacking a whole NG5 year is
    infeasible (~0.6 GB/step × ~175 k steps/yr). ``step_forcing_seq`` must therefore cover a
    **fine time-chunk (≈ a few days)**; the A6 driver loops chunks in Python, re-stacks the
    forcing per chunk, and continues the State across chunks (forward-only). ``return_executable``
    hands back ``(jfn, const_args, npes)`` so the driver compiles ONCE and reuses the executable
    across chunks of equal length (a fresh forcing stack each call ⇒ same shapes ⇒ cache hit).

    Forward-only (no ``return_grad_fn``): the model paper is forward at scale; the differentiable
    path stays single-GPU dense (the sharded ragged-halo AD bug — lifted for ``use_padded``,
    the Phase 8c AD-correct transport, in :func:`run_steps_sharded`).

    **Folded I/O (multi-node, the A6 chunk/restart chaining).** ``state_is_folded`` ⇒ ``state_p`` is
    ALREADY a folded ``[P*Lmax]`` device-sharded State (a prior chunk's output / ``read_restart``);
    skip ``folded_state`` (whose ``[P,Lmax]→[P*Lmax]`` reshape MATERIALIZES the global array on one
    device under multi-node ⇒ OOM). ``return_folded`` ⇒ return the folded ``[P*Lmax]`` scan output
    directly (no ``unfold_state`` reshape). Cold start passes a HOST ``[P,Lmax]`` (``state_is_folded
    =False``); the driver chains the folded output between chunks and to ``write_restart``."""
    fm, fm_spec = folded_mesh(sm)
    if state_is_folded:
        fs = state_p                                      # already folded [P*Lmax] device-sharded
        fs_spec = jax.tree.map(lambda _: _P, fs)
    else:
        fs, fs_spec = folded_state(state_p)               # [P,Lmax] (host cold-start) → [P*Lmax]
    fop, fop_spec = folded_operator(sop)
    fstress = _fold(stress_p)
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    n_global = sm.nod2D
    _check_halo_mode(use_ragged, use_padded)
    ha, ha_spec = _halo_arrays(sm, ragged=use_ragged, padded=use_padded)
    recv_max = _recv_max(sm) if use_ragged else None

    if forcing_static is None:
        raise ValueError("run_steps_sharded_forced needs forcing_static (the CORE2 forced path)")
    seq_folded, seq_spec = _fold_forcing_seq(step_forcing_seq)
    # leaf 0's leading dim is n_steps — every forcing field must be stacked to n_steps.
    T = int(np.asarray(getattr(seq_folded, seq_folded._fields[0])).shape[0])
    if T != n_steps:
        raise ValueError(f"step_forcing_seq has {T} steps but n_steps={n_steps}")

    # forcing_static (device-constant) + the optional global boundary_node, as in run_steps_sharded.
    fsf, fsf_spec = _fold_forcing(forcing_static)
    extras, extras_spec = [fsf], [fsf_spec]
    have_bn = boundary_node_p is not None
    if have_bn:
        extras.append(_fold(boundary_node_p)); extras_spec.append(_P)
    extras, extras_spec = tuple(extras), tuple(extras_spec)

    def body(m, s, o, stress, h, seq, *ex):
        ctx = _make_halo_ctx(h, n_global, use_ragged, use_padded, recv_max)
        fs_ = ex[0]
        bn_ = ex[1] if have_bn else None

        def one(carry, is_first, sf_step):
            return stepmod.step(carry, m, o, stress, params, dt=dt, is_first_step=is_first,
                                step_forcing=sf_step, forcing_static=fs_, ice_cfg=ice_cfg,
                                gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, tke_cfg=tke_cfg,
                                ale_cfg=ale_cfg, visc_cfg=visc_cfg, tracer_cfg=tracer_cfg,
                                halo_ctx=ctx, boundary_node=bn_)

        sf0 = jax.tree.map(lambda x: x[0], seq)             # this chunk's first-step forcing
        # bootstrap_ab2: True ⇒ cold start / post-dt-ramp (AB2 first-step branch); False ⇒ a
        # restart-continuation chunk carries the AB2 history forward (is_first_step=False), so a
        # chained run is bit-identical to a continuous one (the A6 restart-seam invariant).
        st = one(s, bootstrap_ab2, sf0)
        # Optional per-step OUTPUT accumulation (the diurnal-cycle fix): sample_fn(state) extracts the
        # output-stream fields and we sum them over EVERY step ⇒ a TRUE time-mean (the driver ÷ by the
        # step count), dt-INDEPENDENT, matching Fortran's every-timestep mean. Without it the monthly
        # mean was one chunk-final snapshot/chunk; at day-aligned chunks that samples a FIXED time-of-day
        # and aliases the diurnal cycle into a wavenumber-1 SST pattern. sample_fn=None ⇒ byte-identical.
        if sample_fn is None:
            if n_steps > 1:
                rest = jax.tree.map(lambda x: x[1:], seq)   # [n_steps-1, Lmax] per leaf
                st, _ = lax.scan(jax.checkpoint(lambda c, sf: (one(c, False, sf), None)),
                                 st, xs=rest, length=n_steps - 1)
            return st
        acc = sample_fn(st)                                 # running sum, seeded by step-1's output
        if n_steps > 1:
            rest = jax.tree.map(lambda x: x[1:], seq)
            def _accum(carry, sf):
                c, a = carry
                c = one(c, False, sf)
                a = jax.tree.map(lambda u, v: u + v, a, sample_fn(c))
                return (c, a), None
            (st, acc), _ = lax.scan(jax.checkpoint(_accum), (st, acc),
                                    xs=rest, length=n_steps - 1)
        return st, acc

    in_specs = (fm_spec, fs_spec, fop_spec, _P, ha_spec, seq_spec) + extras_spec
    # out_specs: the final State (fs_spec) alone; + the per-step accumulation sum (acc_spec) when
    # sample_fn is given. acc leaves are node-folded [P*Lmax,…] ⇒ the same _P PartitionSpec as State.
    if sample_fn is None:
        out_specs = fs_spec
    else:
        acc_spec = jax.tree.map(lambda _: _P, jax.eval_shape(sample_fn, fs))
        out_specs = (fs_spec, acc_spec)
    # EXECUTABLE REUSE across chunks (the A6 driver's multi-chunk contract). `body` is a FRESH
    # closure every call, so a bare `jax.jit(shard_map(body))` is a NEW jitted object ⇒ XLA RE-
    # COMPILES on every chunk (~25 s at CORE2 all-on — the ~5× driver overhead the perf decomposition
    # found: a 96 ms/step model ran the hindcast at 520 ms/step). When `reuse_executable`, cache the
    # compiled `jfn` keyed by EVERY compilation determinant: the static config (n_steps, dt, the AB2-
    # bootstrap + folded-input flags, ragged/npes/have_bn) AND the IDENTITY of the structural inputs
    # (sm/sop/params/the physics cfgs are ONE persistent object per run.py invocation ⇒ id() is a
    # complete, collision-free key within a process-run; differing objects miss ⇒ safely recompile).
    # A hit reuses the executable; the args (fresh state + forcing VALUES, identical shapes) are
    # rebuilt every call and fed to it ⇒ bit-identical to a fresh compile (asserted on GPU).
    if reuse_executable:
        key = (id(sm), id(sop), id(params), int(n_steps), float(dt), bool(bootstrap_ab2),
               bool(state_is_folded), bool(use_ragged), bool(use_padded), int(npes), bool(have_bn),
               id(ice_cfg), id(gm_cfg), id(kpp_cfg), id(tke_cfg),
               id(ale_cfg), id(visc_cfg), id(tracer_cfg), id(sample_fn))
        jfn = _FORCED_JIT_CACHE.get(key)
        if jfn is None:
            jfn = jax.jit(jax.shard_map(body, mesh=jmesh, in_specs=in_specs,
                                        out_specs=out_specs, check_vma=False))
            _FORCED_JIT_CACHE[key] = jfn
    else:
        jfn = jax.jit(jax.shard_map(body, mesh=jmesh, in_specs=in_specs,
                                    out_specs=out_specs, check_vma=False))
    args = (fm, fs, fop, fstress, ha, seq_folded, *extras)
    args = tuple(_to_global_sharded(a, sp, jmesh) for a, sp in zip(args, in_specs))
    if return_executable:
        return jfn, args, npes
    out = jfn(*args)                                       # folded [P*Lmax] sharded scan output
    # sample_fn ⇒ out is (final_state, per-step-sum); the driver divides the sum by the step count
    # for a true time-mean. Otherwise out is just the final State (byte-identical to before).
    if sample_fn is not None:
        state_out, acc_out = out
        return (state_out, acc_out) if return_folded else (unfold_state(state_out, npes), acc_out)
    return out if return_folded else unfold_state(out, npes)
