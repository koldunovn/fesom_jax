"""Partial bottom cells on the sharded step (CPU fake devices): the partial-cell fields are partitioned with the
mesh, the 1-device sharded step equals the dense one, and the 2-device step matches single device on owned entities
(the same budgets as ``test_step_sharded``), for linfs and zstar.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 PY -m pytest fesom_jax/tests/test_partial_cells_sharded.py
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import integrate_sharded as ish
from fesom_jax import partial_cells as pc
from fesom_jax import partit, shard_mesh, ssh
from fesom_jax import step as stepmod
from fesom_jax.ale import AleConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.state import State
from fesom_jax.tests.test_step_sharded import (CORE2_DIST, CORE2_MESH, DT, NDEV, _BYTE_ID_ATOL, _owned_match,
                                               _perturbed_state, _stress_p, avail)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def core2_pc():
    return pc.apply(load_mesh(CORE2_MESH))


@avail
def test_sharded_mesh_carries_partial_cells(core2_pc):
    ser = partit.synth_serial(core2_pc.nod2D, core2_pc.elem2D, core2_pc.edge2D)
    lm = ish.local_mesh(shard_mesh.build_sharded_mesh(core2_pc, ser), 0)
    for name in ("zbar_e_bot", "zbar_n_bot", "zbar_3d_n"):
        assert np.array_equal(np.asarray(getattr(lm, name)), np.asarray(getattr(core2_pc, name))), name
    # a full-cell mesh shards exactly as before (no partial-cell fields)
    fc = load_mesh(CORE2_MESH)
    assert "zbar_e_bot" not in shard_mesh.build_sharded_mesh(fc, ser).fields


@avail
@pytest.mark.parametrize("zstar", [False, True])
@pytest.mark.parametrize("npes", [1, 2])
def test_sharded_partial_cell_step(core2_pc, zstar, npes):
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = core2_pc
    ale_cfg = AleConfig() if zstar else None
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    st_dense = stepmod.step(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), dt=DT, is_first_step=True,
                            ale_cfg=ale_cfg)
    part = (partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D) if npes == 1
            else partit.read_partition(CORE2_DIST, npes))
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    st_N = ish.run_step_sharded(sm, shard_mesh.partition_state(state, part), ssh.partition_ssh_operator(op, part),
                                _stress_p(mesh, part, sm.Lmax["elem"]), dt=DT, is_first_step=True, npes=npes,
                                ale_cfg=ale_cfg)
    if npes == 1:
        worst = 0.0
        for fld in dataclasses.fields(State):
            if getattr(st_dense, fld.name) is None:
                continue
            a = np.asarray(getattr(st_dense, fld.name))
            b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
            if a.size:
                worst = max(worst, float(np.max(np.abs(a - b))))
        assert worst < _BYTE_ID_ATOL, worst
    else:
        _owned_match(st_dense, st_N, mesh, part, npes, tag=f"partial cells {'zstar' if zstar else 'linfs'}")
