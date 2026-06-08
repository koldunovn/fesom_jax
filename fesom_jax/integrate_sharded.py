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
from jax.sharding import PartitionSpec

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
def _fold(arr) -> jax.Array:
    """``[P, X, …] → [P*X, …]`` (fold the device axis into the leading dim)."""
    a = jnp.asarray(arr)
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
            data[name] = jnp.asarray(arr)
            spec[name] = _R
        else:
            data[name] = _fold(arr)
            spec[name] = _P
    data["nod_in_elem2D_offsets"] = jnp.zeros(Ln + 1, jnp.int32)
    data["nod_in_elem2D"] = jnp.zeros(1, jnp.int32)
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
        arr = jnp.asarray(getattr(f, name))
        if arr.ndim == 0:
            data[name], spec[name] = arr, _R          # scalar (ocean_area) → replicated
        else:
            data[name], spec[name] = _fold(arr), _P   # [P, Lmax_nod, …] → [P*Lmax_nod, …]
    return type(f)(**data), type(f)(**spec)


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


def _halo_arrays(sm: ShardedMesh, ragged: bool = False) -> tuple[dict, dict]:
    """Fold the per-kind exchange maps + the node owned-mask to ``[P*Lmax_kind]``
    ``shard_map`` inputs + their all-``'p'`` spec dict. With ``ragged=True`` (Phase 8b)
    ALSO fold the :class:`~fesom_jax.shard_mesh.RaggedExchange` maps for the halo-only
    ``ragged_all_to_all`` path (left out when ``ragged=False`` ⇒ the all_gather traces
    are byte-unchanged)."""
    ha: dict = {}
    for k in _RKINDS:
        src_dev, src_lane = sm.exchange[k]
        ha[f"sd_{k}"] = _fold(src_dev).astype(jnp.int32)
        ha[f"sl_{k}"] = _fold(src_lane).astype(jnp.int32)
        if ragged:
            r = sm.exchange_ragged[k]
            ha[f"rsi_{k}"] = _fold(r.send_idx).astype(jnp.int32)
            ha[f"rss_{k}"] = _fold(r.send_sizes).astype(jnp.int32)
            ha[f"rso_{k}"] = _fold(r.send_offsets).astype(jnp.int32)
            ha[f"roo_{k}"] = _fold(r.out_offsets).astype(jnp.int32)
            ha[f"rrs_{k}"] = _fold(r.recv_sizes).astype(jnp.int32)
            ha[f"rrg_{k}"] = _fold(r.recv_gather).astype(jnp.int32)
            ha[f"rhm_{k}"] = _fold(r.halo_mask)
    ha["owned_nod"] = _fold(sm.owned_mask["nod"])
    return ha, {k: _P for k in ha}


def _ragged_ctx(h: dict) -> dict:
    """Re-assemble the per-device :class:`RaggedExchange` dict (inside ``shard_map``)
    from the folded halo-arrays ``h`` — the ``{kind: {...}}`` :class:`HaloCtx` consumes."""
    return {k: {"send_idx": h[f"rsi_{k}"], "send_sizes": h[f"rss_{k}"],
                "send_off": h[f"rso_{k}"], "out_off": h[f"roo_{k}"],
                "recv_sizes": h[f"rrs_{k}"], "recv_gather": h[f"rrg_{k}"],
                "halo_mask": h[f"rhm_{k}"]} for k in _RKINDS}


def _recv_max(sm: ShardedMesh) -> dict:
    """Per-kind static recv-buffer extent for ``ragged_all_to_all`` (closed into the body)."""
    return {k: sm.exchange_ragged[k].recv_max for k in _RKINDS}


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
                     kpp_cfg=None, boundary_node_p=None, use_ragged: bool = False) -> State:
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
                            kpp_cfg=kpp_cfg, halo_ctx=ctx, boundary_node=bn_)

    if not wire_halo:
        def body0(m, s, o, stress, *ex):
            return _run(m, s, o, stress, None, ex)
        out = jax.shard_map(body0, mesh=jmesh,
                            in_specs=(fm_spec, fs_spec, fop_spec, _P) + extras_spec,
                            out_specs=fs_spec, check_vma=False)(fm, fs, fop, fstress, *extras)
        return unfold_state(out, npes)

    ha, ha_spec = _halo_arrays(sm, ragged=use_ragged)
    recv_max = _recv_max(sm) if use_ragged else None

    def body(m, s, o, stress, h, *ex):
        exch = {k: (h[f"sd_{k}"], h[f"sl_{k}"]) for k in ("nod", "elem", "edge")}
        ssh_halo = SSHHalo(src_dev=h["sd_nod"], src_lane=h["sl_nod"],
                           owned_mask=h["owned_nod"], n_global=n_global, axis_name="p",
                           ragged=_ragged_ctx(h)["nod"] if use_ragged else None,
                           recv_max=recv_max["nod"] if use_ragged else 0,
                           use_ragged=use_ragged)
        ctx = HaloCtx(exch=exch, axis_name="p", ssh_halo=ssh_halo,
                      owned_mask={"nod": h["owned_nod"]},
                      exch_ragged=_ragged_ctx(h) if use_ragged else None,
                      recv_max=recv_max, use_ragged=use_ragged)
        return _run(m, s, o, stress, ctx, ex)

    out = jax.shard_map(body, mesh=jmesh,
                        in_specs=(fm_spec, fs_spec, fop_spec, _P, ha_spec) + extras_spec,
                        out_specs=fs_spec, check_vma=False)(fm, fs, fop, fstress, ha, *extras)
    return unfold_state(out, npes)


def run_steps_sharded(sm: ShardedMesh, state_p: State, sop: ShardedSSHOperator,
                      stress_p, n_steps: int, *, dt: float, npes: int, params=None,
                      gm_cfg=None, kpp_cfg=None, use_ragged: bool = False) -> State:
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
    fm, fm_spec = folded_mesh(sm)
    fs, fs_spec = folded_state(state_p)
    fop, fop_spec = folded_operator(sop)
    fstress = _fold(stress_p)
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    n_global = sm.nod2D
    ha, ha_spec = _halo_arrays(sm, ragged=use_ragged)
    recv_max = _recv_max(sm) if use_ragged else None

    def body(m, s, o, stress, h):
        exch = {k: (h[f"sd_{k}"], h[f"sl_{k}"]) for k in ("nod", "elem", "edge")}
        ssh_halo = SSHHalo(src_dev=h["sd_nod"], src_lane=h["sl_nod"],
                           owned_mask=h["owned_nod"], n_global=n_global, axis_name="p",
                           ragged=_ragged_ctx(h)["nod"] if use_ragged else None,
                           recv_max=recv_max["nod"] if use_ragged else 0,
                           use_ragged=use_ragged)
        ctx = HaloCtx(exch=exch, axis_name="p", ssh_halo=ssh_halo,
                      owned_mask={"nod": h["owned_nod"]},
                      exch_ragged=_ragged_ctx(h) if use_ragged else None,
                      recv_max=recv_max, use_ragged=use_ragged)

        def one(carry, is_first):
            return stepmod.step(carry, m, o, stress, params, dt=dt, is_first_step=is_first,
                                gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, halo_ctx=ctx)

        st = one(s, True)                            # step 1 eager (AB2 first-step branch)
        if n_steps > 1:
            st, _ = lax.scan(jax.checkpoint(lambda c, _: (one(c, False), None)),
                             st, xs=None, length=n_steps - 1)
        return st

    # ⚠️ jax.jit AROUND the shard_map is REQUIRED for the BACKWARD (S.8): the scan body is
    # jax.checkpoint'd (a ``closed_call`` primitive), and grad of a closed_call inside a
    # shard_map raises "Eager evaluation of closed_call inside a shard_map isn't yet supported"
    # (JAX 0.10) unless the shard_map-decorated fn is jitted. Forward-transparent (the S.7p3
    # multistep forward gate + the npes==1 byte-identity are unaffected — jit is semantically
    # identity); ``run_step_sharded`` (no scan ⇒ no checkpoint ⇒ no closed_call) doesn't need it.
    body_sm = jax.shard_map(body, mesh=jmesh,
                            in_specs=(fm_spec, fs_spec, fop_spec, _P, ha_spec),
                            out_specs=fs_spec, check_vma=False)
    out = jax.jit(body_sm)(fm, fs, fop, fstress, ha)
    return unfold_state(out, npes)
