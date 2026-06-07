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
from jax.sharding import PartitionSpec

from . import halo
from . import step as stepmod
from .halo import HaloCtx
from .mesh import Mesh
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
def _halo_arrays(sm: ShardedMesh) -> tuple[dict, dict]:
    """Fold the per-kind exchange maps + the node owned-mask to ``[P*Lmax_kind]``
    ``shard_map`` inputs + their all-``'p'`` spec dict."""
    ha: dict = {}
    for k in ("nod", "elem", "edge"):
        src_dev, src_lane = sm.exchange[k]
        ha[f"sd_{k}"] = _fold(src_dev).astype(jnp.int32)
        ha[f"sl_{k}"] = _fold(src_lane).astype(jnp.int32)
    ha["owned_nod"] = _fold(sm.owned_mask["nod"])
    return ha, {k: _P for k in ha}


def run_step_sharded(sm: ShardedMesh, state_p: State, sop: ShardedSSHOperator,
                     stress_p, *, dt: float, is_first_step: bool, npes: int,
                     wire_halo: bool = True, params=None, step_forcing=None,
                     forcing_static=None, ice_cfg=None, gm_cfg=None,
                     kpp_cfg=None) -> State:
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

    if not wire_halo:
        def body0(m, s, o, stress):
            return stepmod.step(s, m, o, stress, params, dt=dt,
                                is_first_step=is_first_step, step_forcing=step_forcing,
                                forcing_static=forcing_static, ice_cfg=ice_cfg,
                                gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, halo_ctx=None)
        out = jax.shard_map(body0, mesh=jmesh, in_specs=(fm_spec, fs_spec, fop_spec, _P),
                            out_specs=fs_spec, check_vma=False)(fm, fs, fop, fstress)
        return unfold_state(out, npes)

    ha, ha_spec = _halo_arrays(sm)

    def body(m, s, o, stress, h):
        exch = {k: (h[f"sd_{k}"], h[f"sl_{k}"]) for k in ("nod", "elem", "edge")}
        ssh_halo = SSHHalo(src_dev=h["sd_nod"], src_lane=h["sl_nod"],
                           owned_mask=h["owned_nod"], n_global=n_global, axis_name="p")
        ctx = HaloCtx(exch=exch, axis_name="p", ssh_halo=ssh_halo,
                      owned_mask={"nod": h["owned_nod"]})
        return stepmod.step(s, m, o, stress, params, dt=dt,
                            is_first_step=is_first_step, step_forcing=step_forcing,
                            forcing_static=forcing_static, ice_cfg=ice_cfg,
                            gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, halo_ctx=ctx)

    out = jax.shard_map(body, mesh=jmesh,
                        in_specs=(fm_spec, fs_spec, fop_spec, _P, ha_spec),
                        out_specs=fs_spec, check_vma=False)(fm, fs, fop, fstress, ha)
    return unfold_state(out, npes)
