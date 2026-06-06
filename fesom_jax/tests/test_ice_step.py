"""Assembled prognostic-ice step gate — Phase 6, Task 6.6.

Runs the full assembled CORE2 step with PROGNOSTIC sea ice (ocean2ice → EVP → FCT → cut_off
→ thermo → oce_fluxes → stress blend → shortwave → the ocean substeps) on the PHC IC +
cold-start ice, and verifies it against the config-C full-ice C dump
(``data/ice_full_dump_core2/core2_cdump.00000``):

* the ice-mediated surface forcing (``water_flux``/``virtual_salt``/``relax_salt``) at sub 0,
* the post-step ``T``/``S`` (substep 15) — the comprehensive integration gate.

Both are climate-close (~1e-6): the 120-subcycle EVP scatter reassociation (~1e-9, Task 6.4)
propagates through ``u_ice`` → ustar/stress → the whole step, so this is NOT bit-exact (the
per-kernel gates are; the assembled multi-kernel step accumulates the EVP floor). Also checks
the pi / Phase-5 no-ice path is untouched (``ice_cfg=None``).

SKIPS cleanly if the CORE2 mesh / IC / config-C dump are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DUMP = ROOT / "data" / "ice_full_dump_core2" / "core2_cdump.00000"
DT = 500.0
PROBES = [1001, 33778, 43828, 61202, 66921, 79663, 94122]

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.exists() and (IC_DIR / "T_ic.npy").exists() and DUMP.exists()),
    reason="CORE2 mesh / IC / config-C full-ice dump missing (Task 6.1/6.2 + the config-C job)",
)


@pytest.fixture(scope="module")
def assembled():
    import jax.numpy as jnp
    from fesom_jax import core2_forcing, ice, io_dump, ssh
    from fesom_jax import step as stepmod
    from fesom_jax.ice import IceConfig
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import core2_initial_state

    mesh = load_mesh(MESH_DIR)
    sst = np.asarray(core2_initial_state(mesh, IC_DIR).T[:, 0])
    state0 = ice.seed_ice(core2_initial_state(mesh, IC_DIR), mesh, sst)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, 1958, sst_ic=sst)
    sf = cf.step_forcing(1958, 1, 0.0, 1)               # step 1: 1958-01-01 00:00, January
    cfg = IceConfig()
    stress0 = jnp.zeros((int(mesh.elem2D), 2))
    new = stepmod.step(state0, mesh, op, stress0, dt=DT, is_first_step=True,
                       step_forcing=sf, forcing_static=cf.static, ice_cfg=cfg)
    recs = io_dump.load_records(DUMP)
    return mesh, state0, new, recs, cf, sf


def _probe(recs, sub, field, gid):
    for r in recs:
        if r.step == 1 and r.substep == sub and r.probe_gid == gid and r.field.strip() == field:
            return float(r.values[0]) if field in ("water_flux", "virtual_salt", "relax_salt",
                                                    "heat_flux") else r.values
    raise KeyError(f"{field}@sub{sub}/{gid}")


# --------------------------------------------------------------------------
# Post-step T/S — the comprehensive integration gate
# --------------------------------------------------------------------------
def test_post_step_TS(assembled):
    """Post-step surface T/S match the config-C C dump (climate-close — the EVP floor
    propagates, so ~1e-6 not bit-exact)."""
    mesh, _s0, new, recs, _cf, _sf = assembled
    T = np.asarray(new.T); S = np.asarray(new.S)
    for gid in PROBES:
        cT = _probe(recs, 15, "T", gid)
        cS = _probe(recs, 15, "S", gid)
        nlev = len(cT)
        dT = np.abs(T[gid - 1, :nlev] - np.asarray(cT)).max()
        dS = np.abs(S[gid - 1, :nlev] - np.asarray(cS)).max()
        assert dT < 1e-5, f"T@{gid}: max|Δ|={dT:.3e}"
        assert dS < 1e-5, f"S@{gid}: max|Δ|={dS:.3e}"


def test_surface_forcing(assembled):
    """The ice-mediated surface forcing (water_flux/virtual_salt/relax_salt) matches the C."""
    from fesom_jax import ice_step
    from fesom_jax.ice import IceConfig
    mesh, s0, _new, recs, cf, sf = assembled
    out = ice_step.ice_surface_step(IceConfig(), mesh, s0, sf, cf.static, dt=DT)
    for field, j in (("water_flux", out.water_flux), ("virtual_salt", out.virtual_salt),
                     ("relax_salt", out.relax_salt)):
        jj = np.asarray(j)
        for gid in PROBES:
            c = _probe(recs, 0, field, gid)
            assert abs(jj[gid - 1] - c) < 1e-6, f"{field}@{gid}: {jj[gid-1]:.3e} vs {c:.3e}"


def test_ice_state_advances(assembled):
    """The prognostic ice state actually evolves (EVP moved u_ice, thermo changed a/m)."""
    _mesh, s0, new, _recs, _cf, _sf = assembled
    assert np.abs(np.asarray(new.u_ice)).max() > 0.01          # EVP drove ice motion
    assert not np.array_equal(np.asarray(new.a_ice), np.asarray(s0.a_ice))   # thermo/adv moved a
    assert np.all(np.isfinite(np.asarray(new.m_ice)))
    # sigma is now nonzero (EVP elastic memory built up)
    assert np.abs(np.asarray(new.sigma11)).max() > 0.0


# --------------------------------------------------------------------------
# The pi / no-ice path is untouched
# --------------------------------------------------------------------------
def test_no_ice_path_unchanged(assembled):
    """``ice_cfg=None`` ⇒ the Phase-5 static-ice surface-flux path (the prognostic ice fields
    stay at their seeded values — the ice step did not run)."""
    import jax.numpy as jnp
    from fesom_jax import core2_forcing, ice, ssh
    from fesom_jax import step as stepmod
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import core2_initial_state
    mesh, s0, _new, _recs, cf, sf = assembled
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress0 = jnp.zeros((int(mesh.elem2D), 2))
    no_ice = stepmod.step(s0, mesh, op, stress0, dt=DT, is_first_step=True,
                          step_forcing=sf, forcing_static=cf.static, ice_cfg=None)
    # the ice fields are carried through unchanged (no ice step ran)
    assert np.array_equal(np.asarray(no_ice.a_ice), np.asarray(s0.a_ice))
    assert np.array_equal(np.asarray(no_ice.u_ice), np.asarray(s0.u_ice))
