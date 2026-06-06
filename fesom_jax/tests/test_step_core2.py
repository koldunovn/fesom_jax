"""Task 5.1 (rest-state smoke) + Task 5.7 seed: the assembled ``step()`` runs on
the **CORE2** mesh and stays at rest for a constant T/S field with zero wind.

This is the first end-to-end run of the ocean step on the big mesh (126858 nod /
244659 elem). At rest with constant T/S the density is horizontally uniform per
layer ⇒ PGF=0 ⇒ no flow ⇒ eta/uv stay ~0 and tracers stay exactly constant — so
any spurious flow (a geometry/scatter asymmetry on the larger mesh) shows up here.
It confirms the design claim that CORE2 needs no kernel change (full-cell, global
zbar/Z), complementing the static all-CW orientation gate in ``test_mesh_core2``.

SKIPS until the CORE2 mesh is exported (``data/mesh_core2/``, Task 5.1 job). The
eager step is ~32 s/step on CPU (CORE2 is ~40x pi), so this uses a modest step
count; full multi-step stability + real PHC/JRA forcing is Task 5.7.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import mesh as meshmod, ssh, step as stepmod
from fesom_jax.state import State

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
DT = 500.0  # CORE2 dynamics timestep (FRESH_START §4/§15)

pytestmark = pytest.mark.skipif(
    not CORE2_MESH_DIR.is_dir(),
    reason=f"CORE2 mesh export missing: {CORE2_MESH_DIR} (run Task 5.1 export job)",
)


@pytest.fixture(scope="module")
def core2_mesh():
    return meshmod.load_mesh(CORE2_MESH_DIR)


@pytest.fixture(scope="module")
def core2_op(core2_mesh):
    return ssh.build_ssh_operator(core2_mesh, dt=DT)


def test_rest_state_stays_at_rest(core2_mesh, core2_op):
    """Constant T=10/S=35, zero wind → CORE2 stays at rest to machine precision
    and tracers are bit-exactly preserved (the no-spurious-flow gate)."""
    rest = State.rest(core2_mesh, T0=10.0, S0=35.0)
    zero_stress = jnp.zeros((core2_mesh.elem2D, 2))
    st = rest
    for i in range(3):  # eager ~32 s/step on CPU; 3 is enough to expose drift
        st = stepmod.step(st, core2_mesh, core2_op, zero_stress,
                          dt=DT, is_first_step=(i == 0))
    ml = np.asarray(core2_mesh.node_layer_mask)
    assert not np.isnan(np.asarray(st.uv)).any()
    assert np.max(np.abs(np.asarray(st.uv))) < 1e-12
    assert np.max(np.abs(np.asarray(st.eta_n))) < 1e-12
    assert np.max(np.abs(np.asarray(st.d_eta))) < 1e-12
    assert np.max(np.abs(np.asarray(st.T)[ml] - 10.0)) == 0.0
    assert np.max(np.abs(np.asarray(st.S)[ml] - 35.0)) == 0.0
