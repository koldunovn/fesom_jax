"""Sea-ice cold-start IC + IceConfig gate — Phase 6, Task 6.1.

Verifies the cold-start ice initial condition (:func:`fesom_jax.ice.ice_initial_state`,
a port of ``fesom_ice_initial_state`` / ``fesom_ice.c:246-277``) and the State ice-field
plumbing. The C cold-start is a pure threshold rule of the IC surface temperature, and the
JAX PHC SST already matches the C to ~1e-14 (Task 5.2 gate), so the JAX IC matches the C's
transitively — the only sensitivity is nodes whose SST sits within FP noise of 0 (counted
below; expected 0). The rule itself is gated against an independent per-node loop reference.

SKIPS cleanly if the CORE2 mesh / PHC IC cache are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fesom_jax import surface_forcing, ice
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.state import State

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.exists() and (IC_DIR / "T_ic.npy").exists()),
    reason="CORE2 mesh / PHC IC cache missing (run Task 5.1 / 5.2)",
)


@pytest.fixture(scope="module")
def core2():
    mesh = load_mesh(MESH_DIR)
    state = phc_initial_state(mesh, IC_DIR)
    sst = np.asarray(state.T[:, 0])
    a, m, ms = ice.ice_initial_state(mesh, sst)
    return mesh, state, sst, np.asarray(a), np.asarray(m), np.asarray(ms)


def _loop_reference(mesh, sst):
    """Independent per-node loop port of fesom_ice_initial_state (fesom_ice.c:260-276)."""
    N = int(mesh.nod2D)
    ul = np.asarray(mesh.ulevels_nod2D)
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    a = np.zeros(N); m = np.zeros(N); ms = np.zeros(N)
    for n in range(N):
        if ul[n] > 1:                      # cavity skip
            continue
        if sst[n] < 0.0:
            if lat[n] > 0.0:               # Northern hemisphere
                m[n] = 1.0; ms[n] = 0.1
            else:                          # Southern hemisphere
                m[n] = 2.0; ms[n] = 0.5
            a[n] = 0.9
    return a, m, ms


# --------------------------------------------------------------------------
# Cold-start IC
# --------------------------------------------------------------------------
def test_ic_matches_loop_reference(core2):
    """Vectorized ice_initial_state == an independent per-node loop port — bit-for-bit."""
    mesh, _state, sst, a, m, ms = core2
    a_ref, m_ref, ms_ref = _loop_reference(mesh, sst)
    assert np.array_equal(a, a_ref)
    assert np.array_equal(m, m_ref)
    assert np.array_equal(ms, ms_ref)


def test_ic_structure(core2):
    """a_ice ∈ {0, 0.9}; ice-covered nodes carry the hemisphere-split m_ice/m_snow."""
    mesh, _state, sst, a, m, ms = core2
    # a_ice is exactly 0 or 0.9
    assert set(np.unique(a)).issubset({0.0, 0.9})
    iced = a > 0.0
    # the three tracers turn on together
    assert np.array_equal(iced, m > 0.0)
    assert np.array_equal(iced, ms > 0.0)
    # m_ice ∈ {1,2}, m_snow ∈ {0.1,0.5} on iced nodes
    assert set(np.unique(m[iced])).issubset({1.0, 2.0})
    assert set(np.unique(ms[iced])).issubset({0.1, 0.5})
    # hemisphere split is exact: NH→(1,0.1), SH→(2,0.5)
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    nh = iced & (lat > 0.0)
    sh = iced & (lat <= 0.0)
    assert np.all(m[nh] == 1.0) and np.all(ms[nh] == 0.1)
    assert np.all(m[sh] == 2.0) and np.all(ms[sh] == 0.5)
    # there IS ice (the gate is meaningful) and it is cold + non-cavity only
    assert iced.sum() > 1000
    assert np.all(sst[iced] < 0.0)
    assert np.all(np.asarray(mesh.ulevels_nod2D)[iced] <= 1)


def test_ic_host_build_matches_device(core2):
    """``ice_initial_state(xp=np)`` is value-identical to the default device build and returns
    HOST numpy — the big-mesh (dars/NG5) host-build path the model-paper run driver now uses to
    seed the cold-start ice (so the prognostic ice starts at the C ``a_ice=0.9`` where SST<0, in
    BOTH hemispheres, not at 0)."""
    from fesom_jax.ice import ice_initial_state
    mesh, _state, sst, a, m, ms = core2
    a_h, m_h, ms_h = ice_initial_state(mesh, sst, xp=np)
    assert isinstance(a_h, np.ndarray) and isinstance(m_h, np.ndarray)
    np.testing.assert_array_equal(a_h, np.asarray(a))
    np.testing.assert_array_equal(m_h, np.asarray(m))
    np.testing.assert_array_equal(ms_h, np.asarray(ms))
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    assert (a_h > 0).sum() > 1000 and ((a_h > 0) & (lat < 0)).sum() > 0   # ice in BOTH hemispheres


def test_cold_start_state_equals_manual_seed(core2):
    """``cold_start_state`` == ``ice.seed_ice(phc_initial_state(...))`` **byte-for-byte** — the ONE
    canonical cold-start builder that ``run_from_config`` + the run scripts now share, so the IC +
    ice-seed sequence can't drift (the bug that left run_from_config's prognostic ice at 0).
    ``seed_sea_ice=False`` returns the bare PHC IC (ice State at 0)."""
    import dataclasses

    from fesom_jax import ice
    from fesom_jax.phc_ic import cold_start_state
    mesh = core2[0]
    base = phc_initial_state(mesh, IC_DIR)
    manual = ice.seed_ice(base, mesh, base.T[:, 0])          # the pattern every run script uses
    helper = cold_start_state(mesh, IC_DIR)
    for f in dataclasses.fields(type(helper)):
        np.testing.assert_array_equal(
            np.asarray(getattr(helper, f.name)), np.asarray(getattr(manual, f.name)),
            err_msg=f"cold_start_state differs from the manual seed in {f.name}")
    assert (np.asarray(helper.a_ice) > 0).sum() > 1000      # the helper DID seed ice
    bare = cold_start_state(mesh, IC_DIR, seed_sea_ice=False)
    assert np.all(np.asarray(bare.a_ice) == 0.0)            # opt-out ⇒ bare PHC IC, ice at 0


def test_ic_consistent_with_phase5_mask(core2):
    """The a_ice component equals the Phase-5 static mask (surface_forcing.ice_ic_aice)."""
    mesh, _state, sst, a, _m, _ms = core2
    a5 = np.asarray(surface_forcing.ice_ic_aice(mesh, sst))
    assert np.array_equal(a, a5)


def test_ic_threshold_not_fp_fragile(core2):
    """No node sits within FP noise of the SST<0 threshold ⇒ the JAX↔C IC match (which is
    transitive through the ~1e-14 PHC SST gate) is not threshold-fragile."""
    mesh, _state, sst, _a, _m, _ms = core2
    non_cavity = np.asarray(mesh.ulevels_nod2D) <= 1
    borderline = np.abs(sst[non_cavity]) < 1e-12
    assert borderline.sum() == 0, f"{borderline.sum()} nodes within 1e-12 of SST=0"


# --------------------------------------------------------------------------
# State plumbing
# --------------------------------------------------------------------------
def test_seed_ice_state(core2):
    """seed_ice writes the ice tracers and leaves everything else untouched."""
    mesh, state, sst, a, m, ms = core2
    seeded = ice.seed_ice(state, mesh, sst)
    assert np.array_equal(np.asarray(seeded.a_ice), a)
    assert np.array_equal(np.asarray(seeded.m_ice), m)
    assert np.array_equal(np.asarray(seeded.m_snow), ms)
    # velocity / skin / stress stay zero (cold start)
    for f in ("u_ice", "v_ice", "t_skin", "sigma11", "sigma12", "sigma22"):
        assert np.all(np.asarray(getattr(seeded, f)) == 0.0)
    # ocean fields untouched (T/S/uv/eta identical objects-by-value)
    assert np.array_equal(np.asarray(seeded.T), np.asarray(state.T))
    assert np.array_equal(np.asarray(seeded.S), np.asarray(state.S))
    assert np.array_equal(np.asarray(seeded.eta_n), np.asarray(state.eta_n))


def test_state_ice_fields_zero_by_default(core2):
    """State.zeros / rest seed the ice fields to zero (the ocean-only path is inert)."""
    mesh, state, *_ = core2
    z = State.zeros(mesh)
    n, e = int(mesh.nod2D), int(mesh.elem2D)
    for f, sz in (("a_ice", n), ("m_ice", n), ("m_snow", n), ("u_ice", n),
                  ("v_ice", n), ("t_skin", n), ("sigma11", e), ("sigma12", e),
                  ("sigma22", e)):
        arr = np.asarray(getattr(z, f))
        assert arr.shape == (sz,)
        assert np.all(arr == 0.0)
    # the rest/IC ocean state also has zero ice (seeding is explicit)
    assert np.all(np.asarray(state.a_ice) == 0.0)


def test_state_pytree_roundtrip(core2):
    """The 9 ice fields are registered pytree leaves (flatten/unflatten round-trips)."""
    import jax
    mesh, state, sst, *_ = core2
    seeded = ice.seed_ice(state, mesh, sst)
    leaves, treedef = jax.tree_util.tree_flatten(seeded)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.array_equal(np.asarray(rebuilt.a_ice), np.asarray(seeded.a_ice))
    assert np.array_equal(np.asarray(rebuilt.sigma11), np.asarray(seeded.sigma11))


# --------------------------------------------------------------------------
# IceConfig
# --------------------------------------------------------------------------
def test_ice_config_constants():
    """IceConfig matches the CORE2 namelist.ice (fesom_ice.c:53-111) + derived values."""
    c = ice.IceConfig()
    assert c.evp_rheol_steps == 120
    assert c.ellipse == 2.0 and c.vale == 0.25
    assert c.pstar == 30000.0 and c.c_pressure == 20.0
    assert c.delta_min == 1.0e-11
    assert c.cd_oce_ice == 5.5e-3 and c.cd_atm_ice == 1.2e-3
    assert c.ch_atm_ice == 1.75e-3 and c.ce_atm_ice == 1.75e-3
    assert c.rhoice == 910.0 and c.rhosno == 290.0 and c.rhowat == 1025.0
    assert c.albw == 0.1 and c.Sice == 4.0 and c.iclasses == 7
    assert c.use_virt_salt == 1 and c.ref_sss_local == 1
    # derived (ice_dt=500 default = CORE2 ocean dt)
    assert c.ice_dt == 500.0
    assert c.cc == 1025.0 * 4190.0
    assert c.cl == 910.0 * 3.34e5
    assert c.Tevp_inv == 3.0 / 500.0           # fesom_ice.c:233 (NOT evp_rheol_steps/ice_dt)
    assert abs(c.dte - 500.0 / 120.0) < 1e-12
    # rebuildable for a different ocean dt
    assert ice.IceConfig(ice_dt=1800.0).Tevp_inv == 3.0 / 1800.0
