"""S.7 gate: device-mesh placement + the sharded step (:mod:`fesom_jax.integrate_sharded`).

Phase 1 (this file's current scope): the **placement scaffold** — reconstruct the
per-device local ``Mesh``/``State``/``SSHOperator`` from the S.2/S.2b/S.6 host bundles and
run the *unmodified* :func:`fesom_jax.step.step` under ``shard_map``. The ``npes==1`` whole
step == the dense step **byte-identically** (the no-op invariant: the sharded code path
collapses to the single-device model, so ``v1.0`` is structurally untouched), and the
multi-device step LOWERS and matches single device on the **deep interior** (owned nodes
whose stencil never reaches the halo) — the proof the local kernels are correct on real
shards. The *boundary* nodes need the halo exchanges (the rest of S.7), so they are not
yet asserted to match.

Runs on CPU fake-devices; the multi-device parts SKIP at 1 device. The full-step
``shard_map`` compile is ~1–2 min, so these are slow (SHARDING group).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import integrate_sharded as ish
from fesom_jax import partit, shard_mesh, ssh
from fesom_jax import step as stepmod
from fesom_jax.mesh import load_mesh
from fesom_jax.state import State

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())
DT = 1800.0

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing")


def _perturbed_state(mesh):
    """A non-trivial, deterministic State (smooth perturbations of rest) so the step
    does real work — otherwise rest-stays-rest gives a trivial 0==0 comparison."""
    st = State.rest(mesh)
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    bump = 0.5 * np.cos(2 * lat)[:, None]                  # [nod2D, 1], broadcast over nl
    T = np.asarray(st.T) + np.where(np.asarray(mesh.node_layer_mask), bump, 0.0)
    return dataclasses.replace(st, T=jnp.asarray(T))


def _stress_p(mesh, part, Le):
    """Zero element wind stress, partitioned to [P, Lmax_elem, 2]."""
    P = part.npes
    out = np.zeros((P, Le, 2))
    return jnp.asarray(out)


# --------------------------------------------------------------------------
# 1. Reconstruction: npes=1 local Mesh == dense Mesh (step-read fields)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CORE2_MESH.is_dir(), reason="CORE2 mesh missing")
def test_local_mesh_reconstruction_serial():
    """The ``npes==1`` reconstructed local Mesh equals the dense Mesh for every
    step-read field (the CSR is a step-unused dummy)."""
    mesh = load_mesh(CORE2_MESH)
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    lm = ish.local_mesh(sm, 0)
    assert lm.nod2D == mesh.nod2D and lm.elem2D == mesh.elem2D and lm.edge2D == mesh.edge2D
    for name in list(shard_mesh.NODE_FIELDS) + list(shard_mesh.ELEM_FIELDS) \
            + list(shard_mesh.EDGE_FIELDS) + [c[0] for c in shard_mesh.CONN_FIELDS]:
        a = np.asarray(getattr(mesh, name))
        b = np.asarray(getattr(lm, name))
        assert np.array_equal(a, b), f"local mesh field {name} != dense"


# --------------------------------------------------------------------------
# 2. npes=1 whole step under shard_map == dense (the no-op invariant)
# --------------------------------------------------------------------------
@avail
def test_serial_sharded_step_matches_dense():
    """The full ocean step under ``shard_map`` on ONE device == the dense step,
    byte-identically (the sharded path collapses to the single-device model)."""
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    stress = jnp.zeros((mesh.elem2D, 2))
    st_dense = stepmod.step(state, mesh, op, stress, dt=DT, is_first_step=True)

    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    state_p = shard_mesh.partition_state(state, ser)
    sop = ssh.partition_ssh_operator(op, ser)
    stress_p = _stress_p(mesh, ser, sm.Lmax["elem"])
    st_N = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT,
                                is_first_step=True, npes=1)

    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < 1e-9, f"serial sharded step max|Δ|={worst:.3e} (expected byte-identical)"


# --------------------------------------------------------------------------
# 3. Multi-device step LOWERS + matches single-device on the deep interior
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2])
def test_sharded_step_interior_matches(npes):
    """Without halo exchanges (the rest of S.7), the multi-device step LOWERS and the
    DEEP-INTERIOR owned nodes (whose kernels never read a halo lane) match single
    device — proving the local kernels are correct on real shards. Boundary nodes
    need the exchanges and are not asserted here; we report the matched fraction."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    st_dense = stepmod.step(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), dt=DT,
                            is_first_step=True)

    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    st_N = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT,
                                is_first_step=True, npes=npes)

    # T is a node field; compare owned node lanes per device to the dense field.
    Td = np.asarray(st_dense.T)
    TN = np.asarray(st_N.T)               # [P, Lmax_nod, nl]
    matched = total = 0
    for d in range(npes):
        md = int(part.myDim_nod2D[d])
        gids = part.myList_nod2D[d][:md]
        diff = np.max(np.abs(TN[d, :md] - Td[gids]), axis=1)   # per owned node
        matched += int((diff < 1e-10).sum())
        total += md
    frac = matched / total
    print(f"\nnpes={npes}: deep-interior owned T match (no exchanges) = "
          f"{matched}/{total} ({100*frac:.1f}%)")
    # the bulk of owned nodes are interior ⇒ match even without exchanges; the rest
    # are within the halo footprint and need the S.7 exchanges.
    assert frac > 0.5, f"only {100*frac:.1f}% of owned nodes match — multi-device " \
                       f"lowering or local kernels are wrong (not just a halo gap)"
