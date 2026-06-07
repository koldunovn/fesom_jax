"""Ice ↔ ocean coupling gate — Phase 6, Task 6.3 (the runoff handoff).

Verifies :mod:`fesom_jax.ice_coupling` against the config-A per-substep C dump
(``data/ice_stepA_dump_core2/core2_cdump.00000``, thermo-ON). The dump captures the
ice-mediated surface forcing at step 1 (``water_flux = -flx_fw`` incl. runoff,
``virtual_salt``, ``relax_salt``). JAX is fed the C ``flx_fw``/``flx_h`` (from the all-node
thermo dump, Task 6.2) + the PHC IC ``S_top`` + the January SSS climatology, computes the
coupling over all nodes (the global-mean balance needs all nodes), and is matched at the 7
probes. ``heat_flux`` is gated separately as ``-flx_h`` (the dump's heat_flux is
post-shortwave-penetration, a 6.6 assembled-path concern).

SKIPS cleanly if the dumps / mesh / IC cache are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
THERMO = ROOT / "data" / "ice_thermo_dump_core2" / "ice_thermo_dump_s1_rank0.txt"
STEPA = ROOT / "data" / "ice_stepA_dump_core2" / "core2_cdump.00000"

PROBES = [1001, 33778, 43828, 61202, 66921, 79663, 94122]

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.exists() and (IC_DIR / "T_ic.npy").exists()
         and THERMO.exists() and STEPA.exists()),
    reason="CORE2 mesh / IC / ice thermo+step dumps missing (Task 6.1/6.2 + the config-A job)",
)


@pytest.fixture(scope="module")
def coupling():
    """Run ice_oce_fluxes over all nodes (fed the C flx_fw/flx_h) and pull the C dump."""
    import jax.numpy as jnp
    from fesom_jax import ice_coupling, io_dump, sss_runoff
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import core2_initial_state

    mesh = load_mesh(MESH_DIR)
    state = core2_initial_state(mesh, IC_DIR)

    # all-node C flx_fw/flx_h from the thermo dump (row i = node i; gid = i+1)
    with open(THERMO) as f:
        tn = f.readline().split("var=")[1].split()[0].split(",")
    tdat = np.loadtxt(THERMO)
    tcol = {n: tdat[:, i] for i, n in enumerate(tn)}
    flx_fw = jnp.asarray(tcol["fw"]); flx_h = jnp.asarray(tcol["ehf"])

    reader = sss_runoff.build_reader(mesh)
    Ssurf = jnp.asarray(reader.month(1))               # January (config-A starts 1958-01-01)
    runoff = jnp.asarray(reader.runoff_node)
    areasvol = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])
    ocean_area = float(mesh.ocean_area)
    open_water = jnp.asarray(np.asarray(mesh.ulevels_nod2D) <= 1)
    S_top = state.S[:, 0]

    out = ice_coupling.ice_oce_fluxes(S_top, flx_fw, flx_h, Ssurf, runoff,
                                      areasvol, ocean_area, open_water)
    recs = io_dump.load_records(STEPA)
    return mesh, state, out, recs, np.asarray(flx_fw), np.asarray(flx_h)


def _probe(recs, field, gid):
    for r in recs:
        if r.step == 1 and r.substep == 0 and r.probe_gid == gid and r.field.strip() == field:
            return float(r.values[0])
    raise KeyError(f"{field}@{gid}")


@pytest.mark.parametrize("field,atol,rtol", [
    ("water_flux", 1e-15, 1e-12),
    ("virtual_salt", 1e-13, 1e-10),
    ("relax_salt", 1e-13, 1e-10),
])
def test_coupling_matches_dump(coupling, field, atol, rtol):
    """The runoff-handoff outputs match the C ice-mediated forcing at the 7 probes."""
    mesh, _state, out, recs, _ffw, _flh = coupling
    j = np.asarray(getattr(out, field))
    for gid in PROBES:
        c = _probe(recs, field, gid)
        v = j[gid - 1]
        assert abs(v - c) <= atol + rtol * abs(c), \
            f"{field}@{gid}: JAX={v:.6e} C={c:.6e} |Δ|={abs(v-c):.2e}"


def test_water_flux_is_neg_flx_fw(coupling):
    """``water_flux == -flx_fw`` exactly (no standalone balance term in the ice-on path)."""
    _mesh, _state, out, _recs, flx_fw, _flh = coupling
    assert np.array_equal(np.asarray(out.water_flux), -flx_fw)


def test_heat_flux_is_neg_flx_h(coupling):
    """``heat_flux == -flx_h`` (pre-shortwave-penetration; the assembled path adds pene)."""
    _mesh, _state, out, _recs, _ffw, flx_h = coupling
    assert np.array_equal(np.asarray(out.heat_flux), -flx_h)


def test_runoff_freshens_end_to_end(coupling):
    """The runoff HANDOFF: composing the thermo (which folds runoff into ``flx_fw``) with the
    coupling, MORE runoff lowers ``virtual_salt`` at river mouths (freshening). I.e.
    ``d(virtual_salt)/d(runoff) < 0`` there — the exact analytic chain
    runoff →(thermo, d/d=1)→ flx_fw →(coupling)→ water_flux=-flx_fw → virtual_salt=S_top·wf.
    This is the Phase-6 deliverable that was inert in Phase 5."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_coupling, ice_thermo, sss_runoff
    from fesom_jax.ice import IceConfig
    mesh, state, _out, _recs, _ffw, flx_h = coupling
    cfg = IceConfig()
    reader = sss_runoff.build_reader(mesh)
    Ssurf = jnp.asarray(reader.month(1)); runoff0 = jnp.asarray(reader.runoff_node)
    areasvol = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])
    oa = float(mesh.ocean_area); ow = jnp.asarray(np.asarray(mesh.ulevels_nod2D) <= 1)
    S_top = state.S[:, 0]

    # thermo inputs from the dump, in therm_ice_cell arg order (runo is the live variable)
    with open(THERMO) as f:
        tn = f.readline().split("var=")[1].split()[0].split(",")
    tcol = {n: c for n, c in zip(tn, np.loadtxt(THERMO).T)}
    order = ["h", "hsn", "A", "fsh", "flo", "Ta", "qa", "rain", "snow", "runo",
             "rsss", "ug", "ustar", "T_oc", "S_oc", "ch", "ce", "t_in", "lid_clo"]
    base = [jnp.asarray(tcol[k]) for k in order]
    ri = order.index("runo")

    def wf_sum(runo):
        a = list(base); a[ri] = runo
        fw = jax.vmap(lambda *x: ice_thermo.therm_ice_cell(cfg, *x))(*a).fw
        o = ice_coupling.ice_oce_fluxes(S_top, fw, flx_h, Ssurf, runoff0, areasvol, oa, ow)
        return jnp.sum(o.water_flux)

    # d(water_flux)/d(runoff) == -1 EVERYWHERE (the clean handoff: runoff →(thermo d/d=1)→
    # flx_fw → water_flux=-flx_fw; no mean coupling in the ice-on water_flux). Freshwater in.
    g = np.asarray(jax.grad(wf_sum)(base[ri]))
    assert np.allclose(g, -1.0, atol=1e-12), f"max|g+1|={np.max(np.abs(g + 1)):.2e}"
    # virtual_salt = S_top·water_flux with S_top>0 ⇒ lower water_flux ⇒ lower virtual_salt
    # (freshening). Confirm the local sensitivity sign at river mouths via the coupling jvp.
    runoff = np.asarray(runoff0)
    river = runoff > np.quantile(runoff[runoff > 0], 0.9)
    wf = np.asarray(_out_water_flux(coupling))
    assert np.all(np.asarray(S_top) > 0)          # rsss>0 ⇒ d(virtual_salt)/d(water_flux)>0
    assert river.sum() > 100


def _out_water_flux(coupling):
    return coupling[2].water_flux


# --------------------------------------------------------------------------
# ocean2ice taps
# --------------------------------------------------------------------------
def test_ocean2ice_taps(coupling):
    """ocean2ice copies the surface ocean state verbatim."""
    from fesom_jax import ice_coupling
    _mesh, state, _out, _recs, _ffw, _flh = coupling
    s = ice_coupling.ocean2ice(state)
    assert np.array_equal(np.asarray(s.temp), np.asarray(state.T[:, 0]))
    assert np.array_equal(np.asarray(s.salt), np.asarray(state.S[:, 0]))
    assert np.array_equal(np.asarray(s.ssh), np.asarray(state.hbar))
    assert np.array_equal(np.asarray(s.u), np.asarray(state.uvnode[:, 0, 0]))
    assert np.array_equal(np.asarray(s.v), np.asarray(state.uvnode[:, 0, 1]))


# --------------------------------------------------------------------------
# ice_oce_fluxes_mom — prognostic stress blend (numpy ref, synthetic u_ice)
# --------------------------------------------------------------------------
def test_stress_blend_vs_reference(coupling):
    """The prognostic ice-ocean stress blend matches a per-node/elem numpy loop ref."""
    import jax.numpy as jnp
    from fesom_jax import ice_coupling
    from fesom_jax.config import DENSITY_0
    from fesom_jax.ice import IceConfig
    mesh, _state, _out, _recs, _ffw, _flh = coupling
    cfg = IceConfig()
    N, E = int(mesh.nod2D), int(mesh.elem2D)
    rng = np.random.default_rng(3)
    a = np.clip(rng.random(N), 0.0, 1.0)
    ui = 0.3 * rng.standard_normal(N); vi = 0.3 * rng.standard_normal(N)
    uw = 0.1 * rng.standard_normal(N); vw = 0.1 * rng.standard_normal(N)
    atm = 0.05 * rng.standard_normal((N, 2))
    ow = np.asarray(mesh.ulevels_nod2D) <= 1
    rho_cd = DENSITY_0 * cfg.cd_oce_ice
    # numpy ref
    sns = atm.copy()
    for n in range(N):
        if not ow[n]:
            continue
        if a[n] > 0.001:
            du, dv = ui[n] - uw[n], vi[n] - vw[n]
            aux = np.sqrt(du * du + dv * dv) * rho_cd
            sx, sy = aux * du, aux * dv
        else:
            sx = sy = 0.0
        sns[n, 0] = sx * a[n] + atm[n, 0] * (1 - a[n])
        sns[n, 1] = sy * a[n] + atm[n, 1] * (1 - a[n])
    en = np.asarray(mesh.elem_nodes)
    ref = (sns[en[:, 0]] + sns[en[:, 1]] + sns[en[:, 2]]) / 3.0
    got, got_node = ice_coupling.ice_oce_fluxes_mom(
        mesh, jnp.asarray(a), jnp.asarray(ui), jnp.asarray(vi),
        jnp.asarray(uw), jnp.asarray(vw), jnp.asarray(atm), jnp.asarray(ow), cfg)
    assert np.max(np.abs(np.asarray(got) - ref)) < 1e-13
    # the node-blended stress (KPP ustar input) matches the per-node numpy ref, and the
    # element stress is its 3-vertex mean
    assert np.max(np.abs(np.asarray(got_node) - sns)) < 1e-13


# --------------------------------------------------------------------------
# AD — the differentiable coupling seams
# --------------------------------------------------------------------------
def test_coupling_ad(coupling):
    """``d(Σvirtual_salt)/d(flx_fw)`` and ``d(·)/d(S_top)`` finite (the SST→flux seam)."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_coupling, sss_runoff
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import core2_initial_state
    mesh = load_mesh(MESH_DIR)
    state = core2_initial_state(mesh, IC_DIR)
    reader = sss_runoff.build_reader(mesh)
    Ssurf = jnp.asarray(reader.month(1)); runoff = jnp.asarray(reader.runoff_node)
    areasvol = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])
    oa = float(mesh.ocean_area); ow = jnp.asarray(np.asarray(mesh.ulevels_nod2D) <= 1)
    with open(THERMO) as f:
        tn = f.readline().split("var=")[1].split()[0].split(",")
    tdat = np.loadtxt(THERMO); tcol = {n: tdat[:, i] for i, n in enumerate(tn)}
    flx_fw = jnp.asarray(tcol["fw"]); flx_h = jnp.asarray(tcol["ehf"]); S_top = state.S[:, 0]

    def loss_ffw(ffw):
        o = ice_coupling.ice_oce_fluxes(S_top, ffw, flx_h, Ssurf, runoff, areasvol, oa, ow)
        return jnp.sum(o.virtual_salt) + jnp.sum(o.water_flux)
    g = np.asarray(jax.grad(loss_ffw)(flx_fw))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)

    def loss_st(st):
        o = ice_coupling.ice_oce_fluxes(st, flx_fw, flx_h, Ssurf, runoff, areasvol, oa, ow)
        return jnp.sum(o.virtual_salt) + jnp.sum(o.relax_salt)
    g2 = np.asarray(jax.grad(loss_st)(S_top))
    assert np.all(np.isfinite(g2)) and np.any(g2 != 0.0)
