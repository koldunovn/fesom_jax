"""Sea-ice EVP dynamics gate — Phase 6, Task 6.4.

Verifies :mod:`fesom_jax.ice_evp` against the config-B C EVP dump
(``data/ice_evp_dump_core2/``, EVP-ON / FCT-OFF, written by the existing
``FESOM_EVP_DUMP_DIR`` harness). The dump records, per all-node/all-elem at step 1:
  Q (a_ice,m_ice,m_snow,elevation,srfoce_temp), F (stress_atmice_x/y,u_w,v_w),
  P node (inv_mass,inv_areamass,rhs_a,rhs_m) + P elem (ice_strength),
  1 (σ11/12/22 after subcycle 0), 2 (u_rhs,v_rhs after sub 0),
  3/4 (u_ice,v_ice after the velocity update / coastal BC, sub 0), END (after all 120).

JAX is fed the C Q+F inputs and matched at each stage. Step-0 fields are per-element/node
MAPS (bit-exact); the END is scatter-class (~1e-9, 120 subcycles of element→node
reassociation). At step 1 the SSH-tilt (rhs_a/rhs_m) ≈ 0 (elevation = hbar ≈ 0 at rest) —
gated as ~0 here; the tilt's gradient-contraction is the pgf/momentum pattern (already gated).

SKIPS cleanly if the mesh / EVP dump are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
EVPD = ROOT / "data" / "ice_evp_dump_core2"

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.exists() and (EVPD / "evp_dump_s1_END_node_rank0.txt").exists()),
    reason="CORE2 mesh / EVP dump missing (run port2/jobs/jax_ice_evp_dump_core2.sh)",
)


def _load(pt, cls):
    return np.loadtxt(EVPD / f"evp_dump_s1_{pt}_{cls}_rank0.txt")   # comments='#' skips header


@pytest.fixture(scope="module")
def evp():
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_evp as ie
    from fesom_jax.mesh import load_mesh

    mesh = load_mesh(MESH_DIR)
    cfg = IceConfig()
    Q = _load("Q", "node"); F = _load("F", "node")
    a_ice, m_ice, m_snow, elev = (jnp.asarray(Q[:, i]) for i in (1, 2, 3, 4))
    sax, say, u_w, v_w = (jnp.asarray(F[:, i]) for i in (1, 2, 3, 4))
    z = jnp.zeros(int(mesh.nod2D)); ze = jnp.zeros(int(mesh.elem2D))
    bn = ie.boundary_node_mask(mesh)

    st = ie.evp_setup(cfg, mesh, a_ice, m_ice, m_snow, elev)
    s11, s12, s22 = ie.stress_tensor(cfg, mesh, z, z, ze, ze, ze, st.ice_strength)
    ur, vr = ie.stress2rhs(cfg, mesh, s11, s12, s22, st.ice_strength,
                           st.inv_areamass, st.tilt_u, st.tilt_v)
    u4, v4 = ie.velocity_update(cfg, mesh, z, z, ur, vr, u_w, v_w, sax, say,
                                st.inv_mass, a_ice, bn)
    ue, ve, _, _, _ = ie.evp_dynamics(
        cfg, mesh, a_ice=a_ice, m_ice=m_ice, m_snow=m_snow, u_ice=z, v_ice=z,
        sigma11=ze, sigma12=ze, sigma22=ze, srfoce_u=u_w, srfoce_v=v_w,
        elevation=elev, stress_ax=sax, stress_ay=say, boundary_node=bn)
    return dict(mesh=mesh, cfg=cfg, st=st, s=(s11, s12, s22), urvr=(ur, vr),
                u4v4=(u4, v4), ueve=(ue, ve), bn=bn,
                inputs=(a_ice, m_ice, m_snow, elev, sax, say, u_w, v_w))


def _mx(j, c):
    return float(np.abs(np.asarray(j) - c).max())


# --------------------------------------------------------------------------
# Setup (P) — bit-exact per-node/elem
# --------------------------------------------------------------------------
def test_setup_matches_dump(evp):
    st = evp["st"]
    Pn = _load("P", "node"); Pe = _load("P", "elem")
    assert _mx(st.inv_mass, Pn[:, 1]) < 1e-12
    assert _mx(st.inv_areamass, Pn[:, 2]) < 1e-18
    assert _mx(st.tilt_u, Pn[:, 3]) < 1e-12 and _mx(st.tilt_v, Pn[:, 4]) < 1e-12
    assert _mx(st.ice_strength, Pe[:, 1]) < 1e-9      # max ~4060; bit-exact map
    assert np.abs(Pe[:, 1]).max() > 100              # ice strength is non-trivial


# --------------------------------------------------------------------------
# One subcycle: σ (1), rhs (2), velocity (4) — bit-exact (maps; sub-0 scatter trivial)
# --------------------------------------------------------------------------
def test_stress_tensor_matches_dump(evp):
    s11, s12, s22 = evp["s"]
    S1 = _load("1", "elem")
    assert _mx(s11, S1[:, 1]) < 1e-10 and _mx(s12, S1[:, 2]) < 1e-10 and _mx(s22, S1[:, 3]) < 1e-10
    assert np.abs(S1[:, 1]).max() > 1.0              # σ is non-trivial


def test_stress2rhs_matches_dump(evp):
    ur, vr = evp["urvr"]
    S2 = _load("2", "node")
    assert _mx(ur, S2[:, 1]) < 1e-12 and _mx(vr, S2[:, 2]) < 1e-12


def test_velocity_update_matches_dump(evp):
    u4, v4 = evp["u4v4"]
    S4 = _load("4", "node")
    assert _mx(u4, S4[:, 1]) < 1e-12 and _mx(v4, S4[:, 2]) < 1e-12


# --------------------------------------------------------------------------
# END (120 subcycles) — scatter-class (accumulated element→node reassociation)
# --------------------------------------------------------------------------
def test_evp_end_matches_dump(evp):
    ue, ve = evp["ueve"]
    SE = _load("END", "node")
    du = _mx(ue, SE[:, 1]); dv = _mx(ve, SE[:, 2])
    mx = max(np.abs(SE[:, 1]).max(), np.abs(SE[:, 2]).max())
    assert mx > 0.05                                  # ice actually moves (gate meaningful)
    assert du < 1e-6 and dv < 1e-6, f"END |Δ| u={du:.2e} v={dv:.2e} (max|u|={mx:.3f})"


def test_boundary_mask(evp):
    """Coastal nodes are masked and the EVP zeros them."""
    bn = np.asarray(evp["bn"])
    assert bn.sum() > 1000
    ue, ve = evp["ueve"]
    assert np.all(np.asarray(ue)[bn] == 0.0) and np.all(np.asarray(ve)[bn] == 0.0)


# --------------------------------------------------------------------------
# AD — the delta singularity + the subcycle scan
# --------------------------------------------------------------------------
def test_ad_delta_singularity(evp):
    """``d(Σσ²)/d(u_ice)`` is finite AT u_ice=0 (where the strain-rate invariant Δ→0 and the
    raw EVP would give a 1/√0 gradient) — the double-`where` safe-sqrt + the delta_min clamp."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_evp as ie
    mesh = evp["mesh"]; cfg = evp["cfg"]; st = evp["st"]
    ze = jnp.zeros(int(mesh.elem2D))

    def loss(u):
        s11, s12, s22 = ie.stress_tensor(cfg, mesh, u, u, ze, ze, ze, st.ice_strength)
        return jnp.sum(s11 ** 2 + s12 ** 2 + s22 ** 2)

    g = np.asarray(jax.grad(loss)(jnp.zeros(int(mesh.nod2D))))
    assert np.all(np.isfinite(g)), f"{int((~np.isfinite(g)).sum())} non-finite at Δ=0"


def test_ad_scan_finite(evp):
    """Gradient through a SHORT EVP subcycle scan is finite everywhere (incl. ice-free lanes)
    — the masked-NaN probe for the 120-subcycle pipeline (run short for CPU speed)."""
    import jax
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_evp as ie
    mesh = evp["mesh"]
    cfg = IceConfig(evp_rheol_steps=4)               # short scan for the CPU AD probe
    a_ice, m_ice, m_snow, elev, sax, say, u_w, v_w = evp["inputs"]
    z = jnp.zeros(int(mesh.nod2D)); ze = jnp.zeros(int(mesh.elem2D))
    bn = evp["bn"]

    def loss(stress_ax):
        ue, ve, *_ = ie.evp_dynamics(
            cfg, mesh, a_ice=a_ice, m_ice=m_ice, m_snow=m_snow, u_ice=z, v_ice=z,
            sigma11=ze, sigma12=ze, sigma22=ze, srfoce_u=u_w, srfoce_v=v_w,
            elevation=elev, stress_ax=stress_ax, stress_ay=say, boundary_node=bn)
        return jnp.sum(ue ** 2 + ve ** 2)

    g = np.asarray(jax.grad(loss)(sax))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)
