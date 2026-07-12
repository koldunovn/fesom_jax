"""S.2b gate: partition State + forcing (:mod:`fesom_jax.shard_mesh`).

Gathers a global :class:`~fesom_jax.state.State` and the CORE2 forcing pytrees to
per-device padded form and asserts:
  * the **serial ``npes==1``** partition is array-equal to the dense pytree (the
    no-op invariant — IC built on the host then partitioned);
  * a 2/4-device gather places each owned (+halo) field on the right device;
  * the state and mesh pad to the **same ``Lmax``** so padded lanes coincide with
    the mesh's invalid (masked) lanes ⇒ padded state is inert;
  * a scanned ``[n_steps, nod2D]`` forcing stack shards the node axis, keeping
    ``n_steps``.

SKIPs cleanly when the dense mesh / partitions are absent.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fesom_jax import partit, shard_mesh
from fesom_jax.surface_forcing import ForcingStatic, StepForcing
from fesom_jax.mesh import Mesh, load_mesh
from fesom_jax.state import State

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing",
)


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    return load_mesh(CORE2_MESH)


@pytest.fixture(scope="module")
def state(mesh) -> State:
    # State.rest populates T/S/hnode/helem (node + elem fields); the rest zero —
    # exercises every field of partition_state. (PHC / ice cold-start ICs follow
    # the same host-build → partition_state pattern.)
    return State.rest(mesh)


def _mk_static(mesh) -> ForcingStatic:
    return ForcingStatic(
        runoff_node=np.arange(mesh.nod2D, dtype=np.float64),
        areasvol_surf=np.asarray(mesh.areasvol[:, 0], dtype=np.float64),
        ocean_area=np.float64(mesh.ocean_area),
        open_water=np.asarray(mesh.ulevels_nod2D) <= 1,
        a_ice=np.zeros(mesh.nod2D, dtype=np.float64),
    )


def _mk_step(nod2D, n_steps=None) -> StepForcing:
    if n_steps is None:
        fields = [np.arange(nod2D, dtype=np.float64) + i for i in range(10)]
    else:
        fields = [np.arange(n_steps * nod2D, dtype=np.float64).reshape(n_steps, nod2D) + i
                  for i in range(10)]
    return StepForcing(*fields)


# --------------------------------------------------------------------------
# Serial no-op invariants
# --------------------------------------------------------------------------
@avail
def test_partition_state_serial_noop(mesh, state):
    p1 = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    ss = shard_mesh.partition_state(state, p1)
    for f in dataclasses.fields(State):
        dense = np.asarray(getattr(state, f.name))
        got = np.asarray(getattr(ss, f.name))
        assert got.shape == (1,) + dense.shape, f"{f.name}: {got.shape}"
        assert np.array_equal(got[0], dense), f"{f.name} differs from dense"


@avail
def test_partition_forcing_serial_noop(mesh):
    p1 = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    fs = _mk_static(mesh)
    fss = shard_mesh.partition_forcing_static(fs, p1)
    assert np.asarray(fss.ocean_area).shape == ()                  # scalar replicated
    assert np.asarray(fss.ocean_area) == np.asarray(fs.ocean_area)
    for name in ("runoff_node", "areasvol_surf", "open_water", "a_ice"):
        dense = np.asarray(getattr(fs, name))
        got = np.asarray(getattr(fss, name))
        assert got.shape == (1, mesh.nod2D)
        assert np.array_equal(got[0], dense)
    # step forcing: single + stacked
    for sf in (_mk_step(mesh.nod2D), _mk_step(mesh.nod2D, n_steps=3)):
        sff = shard_mesh.partition_step_forcing(sf, p1)
        for name in StepForcing._fields:
            dense = np.asarray(getattr(sf, name))
            got = np.asarray(getattr(sff, name))
            assert got.shape == (1,) + dense.shape
            assert np.array_equal(got[0], dense)


# --------------------------------------------------------------------------
# Multi-device placement
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_partition_state_placement(mesh, state, npes):
    part = partit.read_partition(CORE2_DIST, npes)
    _, Lmax = shard_mesh.local_sizes(part)
    ss = shard_mesh.partition_state(state, part)
    # node field T → [P, Lmax_nod, nl]; elem field uv → [P, Lmax_elem, nl, 2]
    T = np.asarray(ss.T)
    uv = np.asarray(ss.uv)
    assert T.shape == (npes, Lmax["nod"], mesh.nl)
    assert uv.shape == (npes, Lmax["elem"], mesh.nl, 2)
    Tg, uvg = np.asarray(state.T), np.asarray(state.uv)
    for d in range(npes):
        nl_n = part.myList_nod2D[d].size
        nl_e = part.myList_elem2D[d].size
        assert np.array_equal(T[d, :nl_n], Tg[part.myList_nod2D[d]])
        assert np.array_equal(uv[d, :nl_e], uvg[part.myList_elem2D[d]])


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_step_forcing_stacked_shards_node_axis(mesh, npes):
    part = partit.read_partition(CORE2_DIST, npes)
    _, Lmax = shard_mesh.local_sizes(part)
    n_steps = 3
    sf = _mk_step(mesh.nod2D, n_steps=n_steps)
    sff = shard_mesh.partition_step_forcing(sf, part)
    ua = np.asarray(sff.u_air)
    assert ua.shape == (npes, n_steps, Lmax["nod"])               # n_steps preserved
    uag = np.asarray(sf.u_air)
    for d in range(npes):
        nl_n = part.myList_nod2D[d].size
        assert np.array_equal(ua[d, :, :nl_n], uag[:, part.myList_nod2D[d]])


# --------------------------------------------------------------------------
# State / mesh pad alignment ⇒ padded state is inert
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_padded_lanes_inert(mesh, state, npes):
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    ss = shard_mesh.partition_state(state, part)
    # state pads to the SAME Lmax as the mesh
    assert np.asarray(ss.T).shape[1] == sm.Lmax["nod"]
    assert np.asarray(ss.uv).shape[1] == sm.Lmax["elem"]
    # node-field pad region [n_local:Lmax] coincides with the mesh's invalid lanes,
    # whose layer mask is all-False ⇒ the padded state contributes nothing.
    T = np.asarray(ss.T)
    for d in range(npes):
        n_loc = int(sm.counts["n_local_nod"][d])
        if n_loc < sm.Lmax["nod"]:
            assert not sm.valid_mask["nod"][d, n_loc:].any()
            assert np.isfinite(T[d, n_loc:]).all()               # finite (masked-NaN rule)
