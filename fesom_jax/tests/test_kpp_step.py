"""Phase 6C Task K.8 gate — KPP wired into the assembled CORE2 step.

The mixing seam (:func:`fesom_jax.step.step` substep 4) now dispatches to KPP — the
real FESOM2 CORE2 default — behind ``kpp_cfg`` (the ``gm_cfg``/``ice_cfg`` precedent),
via :func:`fesom_jax.kpp.mixing_kpp` (compute_vel_nodes → ri_iwmix → bldepth → blmix →
enhance → assemble → mo_convect), emitting the same ``(Kv, Av)`` PP does.

Validated by running one JAX KPP CORE2 step (PHC IC + JRA55 1958 + static-ice forcing —
the ``jax_kpp_dump_core2.sh`` config: KPP / GM OFF / ice OFF, dt=500) and comparing to
the C KPP dumps. The JAX surface forcing is a **validated 1:1 port of the C forcing**
(Phase 5, ``test_core2_step.py``), so there is NO JAX↔C step-1 forcing transient (the
~52 % diff documented in the plan is the *C↔Fortran* transient) — the assembled KPP step
is therefore **bit-faithful** to the C, not merely a sanity match:

* ``stress_node_surf`` (the NEW KPP ``ustar`` input, ice-blended node stress) vs the C
  ``iceforce`` dump cols 8–9 — ~1e-15;
* ``Kv`` (node probes) / ``Av`` (elem probes), post-mo_convect, vs the C step dump
  (substep 4) — ~1e-12;
* the assembled chain all-nodes (pre-mo_convect) vs the C final dump
  (``diffKt``/``viscA``/``viscAE``) — ~1e-12;
* ``hbl`` / ``ustar`` all-nodes vs the C ``bldepth``/``prestep`` dumps — ~1e-9 / ~1e-15;
* ``d(ΣKv+ΣAv)/dT`` through :func:`kpp.mixing_kpp` finite + nonzero (the driver-level AD;
  the full assembled-step masked-NaN grad gate is K.10 / SLURM);
* KPP actually changes the mixing (``Kv`` distinct from the PP path) and the pi-path
  guard (``kpp_cfg`` without forcing raises).

SKIPS cleanly if the CORE2 mesh / PHC IC cache / KPP dumps are absent.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import core2_forcing, eos, io_dump, kpp, ssh
from fesom_jax import step as stepmod
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
KPP_DUMP = ROOT / "data" / "kpp_dump_core2"
STEP_DUMP = ROOT / "data" / "kpp_step_dump_core2" / "core2_cdump.00000"
DT = 500.0
YEAR = 1958
# the C dump's node/element probes (jax_kpp_dump_core2.sh FESOM_DUMP_PROBES + the
# incident-element gids the C pairs with each, from test_core2_step.py)
PROBES = [1001, 33778, 43828, 61202, 66921, 79663, 94122]
ELEM_PROBES = [307, 747, 25954, 61526, 99096, 110065, 154575]

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file() and STEP_DUMP.is_file()
         and (KPP_DUMP / "kpp_dump_s1_iceforce_rank0.txt").is_file()),
    reason="CORE2 mesh / PHC IC / KPP step+iceforce dumps missing "
           "(run port2/jobs/jax_kpp_dump_core2.sh)",
)


@pytest.fixture(scope="module")
def run1():
    """Build the CORE2 model + forcing, run one eager KPP step, and recompute the
    KPP chain internals (for the all-nodes pre-mo_convect gates). Eager ~30 s — built
    once for the module."""
    mesh = load_mesh(MESH_DIR)
    state = core2_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    fs = cf.static
    cfg = kpp.KppConfig()
    sf0 = cf.step_forcing(*core2_forcing.dates_for_steps(YEAR, DT, 1)[0])
    sfx = core2_forcing.compute_surface_fluxes(mesh, state, sf0, fs, dt=DT)

    # one assembled KPP step (post-mo_convect Kv/Av — the substep-4 dump fields)
    st1 = stepmod.step(state, mesh, op, None, dt=DT, is_first_step=True,
                       step_forcing=sf0, forcing_static=fs, kpp_cfg=cfg)

    # recompute the chain internals (mirror kpp.mixing_kpp MINUS mo_convect) for the
    # all-nodes pre-mo_convect gate + the hbl/ustar intermediates, driven by the real
    # JAX forcing
    bvfreq = eos.compute_pressure_bv(mesh, state.T, state.S, state.hnode)[2]
    sw_alpha, sw_beta = eos.compute_sw_alpha_beta(mesh, state.T, state.S)
    dbsfc = eos.compute_dbsfc(mesh, state.T, state.S)
    uvnode = stepmod.pp.compute_vel_nodes(mesh, state.uv)
    wmt, wst = kpp.build_wscale_tables(cfg)
    viscA, diffKt, diffKs = kpp.ri_iwmix(mesh, uvnode, bvfreq, cfg)
    dVsq, ustar, Bo = kpp.prestep(mesh, uvnode, sfx.stress_node_surf, sfx.heat_flux,
                                  sfx.water_flux, sw_alpha, sw_beta, state.S, cfg)
    hbl, kbl, bfsfc, stable, caseA = kpp.bldepth(
        mesh, dVsq, ustar, Bo, bvfreq, dbsfc, sfx.sw_3d, sw_alpha, wmt, wst, cfg)
    bM, bT, bS, gh, dk = kpp.blmix(mesh, state.hnode, viscA, diffKt, diffKs, hbl, bfsfc,
                                   stable, caseA, kbl, ustar, wmt, wst, cfg)
    bM, bT, bS, gh = kpp.enhance(mesh, bM, bT, bS, gh, dk, viscA, diffKt, diffKs, hbl,
                                 caseA, kbl, cfg)
    _, Av_pre, vA, dT, dS, _ = kpp.assemble_mixing(mesh, bM, bT, bS, gh, viscA, diffKt,
                                                   diffKs, kbl, cfg)
    return dict(mesh=mesh, state=state, op=op, fs=fs, cfg=cfg, sf0=sf0, sfx=sfx, st1=st1,
                hbl=np.asarray(hbl), ustar=np.asarray(ustar),
                diffKt_pre=np.asarray(dT), viscA_pre=np.asarray(vA),
                viscAE_pre=np.asarray(Av_pre))


# ---------------------------------------------------------------------------
# 1) stress_node_surf — the NEW KPP forcing input (ice-blended node stress)
# ---------------------------------------------------------------------------
def test_stress_node_surf_matches_c(run1):
    """The threaded ``stress_node_surf`` (KPP ``ustar`` input) reproduces the C's blended
    ``forcing->stress_node_surf`` (iceforce dump cols 8–9, written by oce_fluxes_mom)."""
    mesh, sfx = run1["mesh"], run1["sfx"]
    icf, _ = io_dump.load_kpp_dump(KPP_DUMP, ["iceforce"])
    sns_c = icf["iceforce"][:, 8:10]
    sns_j = np.asarray(sfx.stress_node_surf)
    ow = np.asarray(mesh.ulevels_nod2D) <= 1            # the C blend's open-water scope
    d = float(np.max(np.abs(sns_j[ow] - sns_c[ow])))
    print(f"\nstress_node_surf max|Δ|={d:.3e}  (scale {np.abs(sns_c[ow]).max():.3e})")
    assert d < 1e-12


# ---------------------------------------------------------------------------
# 2) assembled step output — Kv (node) / Av (elem) at the C probes (post-mo_convect)
# ---------------------------------------------------------------------------
def test_step1_Kv_Av_probes_match_c(run1):
    """Post-step ``Kv``/``Av`` at the C probe columns match the C step dump (substep 4,
    post-mo_convect) — the full assembled KPP step output."""
    st1 = run1["st1"]
    recs = io_dump.load_records(STEP_DUMP)
    Kv, Av = np.asarray(st1.Kv), np.asarray(st1.Av)
    worst_kv = 0.0
    for g in PROBES:
        r = io_dump.find_record(recs, step=1, substep=4, probe_gid=g, field="Kv")
        worst_kv = max(worst_kv, float(np.max(np.abs(Kv[g - 1, :len(r.values)] - r.values))))
    worst_av = 0.0
    for g in ELEM_PROBES:
        r = io_dump.find_record(recs, step=1, substep=4, probe_gid=g, field="Av")
        worst_av = max(worst_av, float(np.max(np.abs(Av[g - 1, :len(r.values)] - r.values))))
    print(f"\nKv@probes max|Δ|={worst_kv:.3e}   Av@probes max|Δ|={worst_av:.3e}")
    assert worst_kv < 1e-12
    assert worst_av < 1e-12


# ---------------------------------------------------------------------------
# 3) the assembled chain ALL-NODES (pre-mo_convect) vs the C final dump
# ---------------------------------------------------------------------------
def test_step1_assembled_allnodes_match_c(run1):
    """The KPP driver output (pre-mo_convect ``diffKt``=Kv / ``viscA`` node, ``viscAE``=Av
    element), driven by the real JAX forcing, reproduces the C final dump on EVERY node —
    bit-faithful (libm-ULP class), the strongest forward gate on the wired chain."""
    mesh = run1["mesh"]
    fin, _ = io_dump.load_kpp_dump(KPP_DUMP, ["diffKt", "viscA", "viscAE"])
    mn = np.asarray(mesh.node_iface_mask)
    me = np.asarray(mesh.elem_iface_mask)
    d_kv = float(np.max(np.abs(run1["diffKt_pre"][mn] - fin["diffKt"][mn])))
    d_va = float(np.max(np.abs(run1["viscA_pre"][mn] - fin["viscA"][mn])))
    d_ae = float(np.max(np.abs(run1["viscAE_pre"][me] - fin["viscAE"][me])))
    print(f"\nall-nodes diffKt={d_kv:.3e}  viscA={d_va:.3e}  viscAE={d_ae:.3e}")
    assert d_kv < 1e-9 and d_va < 1e-9 and d_ae < 1e-9


def test_step1_hbl_ustar_allnodes_match_c(run1):
    """``hbl`` / ``ustar`` (the OBL depth + friction velocity) all-nodes vs the C
    bldepth/prestep dumps — the forcing-driven intermediates of the wired chain."""
    f, _ = io_dump.load_kpp_dump(KPP_DUMP, ["bldepth", "prestep"])
    d_hbl = float(np.max(np.abs(run1["hbl"] - f["bldepth"][:, 0])))
    d_us = float(np.max(np.abs(run1["ustar"] - f["prestep"][:, 0])))
    print(f"\nhbl max|Δ|={d_hbl:.3e} (scale {np.abs(f['bldepth'][:,0]).max():.3e})  "
          f"ustar max|Δ|={d_us:.3e}")
    assert d_hbl < 1e-7
    assert d_us < 1e-12


# ---------------------------------------------------------------------------
# 4) KPP is actually engaged (distinct from PP) + the pi-path guard
# ---------------------------------------------------------------------------
def test_kpp_distinct_from_pp(run1):
    """A KPP step and a PP step (``kpp_cfg=None``) from the same IC give materially
    different ``Kv`` — the genuine scheme difference, confirming KPP is wired in (and
    that ``kpp_cfg=None`` is the PP branch)."""
    mesh, state, op, fs, sf0 = (run1[k] for k in ("mesh", "state", "op", "fs", "sf0"))
    st_pp = stepmod.step(state, mesh, op, None, dt=DT, is_first_step=True,
                         step_forcing=sf0, forcing_static=fs, kpp_cfg=None)
    mn = np.asarray(mesh.node_iface_mask)
    kpp_kv = np.asarray(run1["st1"].Kv)[mn]
    pp_kv = np.asarray(st_pp.Kv)[mn]
    rel = float(np.max(np.abs(kpp_kv - pp_kv)) / np.abs(pp_kv).max())
    print(f"\nmax|Kv_kpp − Kv_pp| / scale = {rel:.3f}")
    assert rel > 0.1                                     # the schemes genuinely differ


def test_kpp_requires_forcing(run1):
    """KPP on the pi path (no ``step_forcing``) raises — it needs surface forcing for
    ``ustar``/``Bo`` (locked decision 7: the pi path keeps PP)."""
    mesh, state, op, cfg = (run1[k] for k in ("mesh", "state", "op", "cfg"))
    with pytest.raises(ValueError, match="KPP .* requires CORE2 surface forcing"):
        stepmod.step(state, mesh, op, None, dt=DT, is_first_step=True, kpp_cfg=cfg)


def test_kpp_two_jitted_steps_no_leak(run1):
    """Two **jitted** KPP steps (``is_first_step`` True→False = two compiled traces) run
    finite — the regression for the ``build_wscale_tables`` lru_cache tracer leak: caching
    the *jnp* tables bound the first trace's arrays into the cache, which then leaked into
    the second trace (``UnexpectedTracerError``); the fix caches the numpy build and casts
    to jnp fresh per trace. (The eager single-step tests above don't exercise this — it is
    jit/scan-trace specific, which is why the K.9 multi-step run first surfaced it.)"""
    mesh, state, op, fs, sf0, cfg = (run1[k] for k in
                                     ("mesh", "state", "op", "fs", "sf0", "cfg"))
    s1 = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=True,
                          step_forcing=sf0, forcing_static=fs, kpp_cfg=cfg)
    s2 = stepmod.step_jit(s1, mesh, op, None, dt=DT, is_first_step=False,
                          step_forcing=sf0, forcing_static=fs, kpp_cfg=cfg)
    assert bool(jnp.isfinite(s2.T).all() and jnp.isfinite(s2.Kv).all())


# ---------------------------------------------------------------------------
# 5) driver-level AD (the full assembled-step masked-NaN grad gate is K.10/SLURM)
# ---------------------------------------------------------------------------
def test_mixing_kpp_ad_finite(run1):
    """``d(ΣKv+ΣAv)/dT`` through :func:`kpp.mixing_kpp` finite everywhere + nonzero (T
    enters via N²/dbsfc/α/β) — the assembled-driver AD (safe-sqrt ustar, stop-grad kbl,
    physical floors all compose cleanly)."""
    mesh, state, sfx, cfg = (run1[k] for k in ("mesh", "state", "sfx", "cfg"))

    def loss(T):
        sa, sb = eos.compute_sw_alpha_beta(mesh, T, state.S)
        db = eos.compute_dbsfc(mesh, T, state.S)
        bv = eos.compute_pressure_bv(mesh, T, state.S, state.hnode)[2]
        Kv, Av, _ = kpp.mixing_kpp(mesh, state.uv, bv, db, sa, sb, state.S,
                                   sfx.heat_flux, sfx.water_flux, sfx.stress_node_surf,
                                   sfx.sw_3d, state.hnode, cfg)
        return jnp.sum(Kv) + jnp.sum(Av)

    g = jax.grad(loss)(state.T)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.sum(jnp.abs(g))) > 0.0
