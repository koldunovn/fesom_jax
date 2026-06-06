"""Task 5.4 gate: the JAX L&Y09 open-water bulk (``forcing.bulk_surface_fluxes`` +
``ncar_ocean_fluxes_mode`` + ``obudget``) reproduces the C ``fesom_bulk_compute`` on
CORE2, verified against the C ``bulk_dump_*`` all-node dumps at three configs:

  * ``d1z`` — JRA (day 1, sec 0),     surface current = 0 (IC state).
  * ``inz`` — JRA (day 100, 12:00),   surface current = 0.
  * ``ins`` — JRA (day 100, 12:00),   **synthetic** surface current (an 8-entry
    table) — exercises the relative-wind path in the coefficients/stress while
    ``obudget``'s ``ug`` stays ABSOLUTE (the deliberate Fortran mismatch) and the
    ``current→stress`` feedback.

The comparison **isolates the bulk**: JAX is fed the C dump's exact ``T_oc`` (col 10)
and ``u_w``/``v_w`` (cols 8/9) plus the bit-exact JRA fields (the Task-5.3 reader), so
the only difference is the bulk's own FP reassociation (transcendentals
``exp``/``log``/``atan``/``sqrt`` in libm vs XLA, through the 5-iteration loop).
Achieved over all 126858 nodes (essentially bit-exact, MAP-class): **cd/ce/ch ~1e-17,
heat_flux ~6e-13 W/m², water_flux ~2e-22 m/s, stress(node+elem) ~5e-16 N/m²**.

The C dump runs a **fixed 5-iteration** L&Y09 loop (``FESOM_BULK_FIXED_ITERS``), matching
the AD-safe JAX unrolled loop — so ``cd``/``ce``/``ch`` compare apples-to-apples. The dump
also carries the **early-break** ``cd_eb``/``ce_eb``/``ch_eb`` (cols 11-13); we confirm
dropping the data-dependent break is physically tiny (``test_fixed_vs_earlybreak``).

SKIPS unless the CORE2 mesh export, the C bulk dump, and the JRA55 NetCDF all exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
BULK_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "bulk_dump_core2"
JRA_DIR = Path("/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0")
YEAR = 1958
INTERIOR_DAY = 100
INTERIOR_SEC = 43200.0

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and BULK_DUMP_DIR.is_dir()
         and (JRA_DIR / f"uas.{YEAR}.nc").is_file()),
    reason="needs CORE2 mesh export + C bulk dump + JRA55 NetCDF (Task 5.4 artifacts)",
)

# Dump node columns (0-based): gid cd ce ch heat_flux water_flux sns_x sns_y
#                              u_w v_w T_oc cd_eb ce_eb ch_eb
C_GID, C_CD, C_CE, C_CH, C_HF, C_WF, C_SX, C_SY, C_UW, C_VW, C_TOC, C_CDE, C_CEE, C_CHE = range(14)

# Per-field absolute tolerances. The fields are MAP-class (per-node pure functions,
# no scatter), so XLA-vs-libm transcendentals through the 5-iter loop land within a
# few ULP. Tolerances are set well above the achieved error (reported by the test)
# yet far below any real-bug signal (a wrong constant/sign/index → O(1e-3..100)).
TOL = {
    "cd":         (C_CD, 1e-12),    # ~1e-3
    "ce":         (C_CE, 1e-12),    # ~1e-3
    "ch":         (C_CH, 1e-12),    # ~1e-3
    "heat_flux":  (C_HF, 1e-9),     # ~O(100) W/m²
    "water_flux": (C_WF, 1e-15),    # ~O(1e-8) m/s
    "sns_x":      (C_SX, 1e-12),    # ~O(0.1) N/m²
    "sns_y":      (C_SY, 1e-12),
}


def _load(tag):
    d = np.loadtxt(BULK_DUMP_DIR / f"bulk_dump_{tag}_rank0.txt")
    return d[np.argsort(d[:, 0])]          # order by 1-based gid → index = gid-1


def _load_elem(tag):
    d = np.loadtxt(BULK_DUMP_DIR / f"bulk_dump_{tag}_elem_rank0.txt")
    return d[np.argsort(d[:, 0])]          # eid 1-based → index = eid-1


@pytest.fixture(scope="module")
def setup():
    from fesom_jax import mesh as meshmod, jra55
    m = meshmod.load_mesh(CORE2_MESH_DIR)
    reader = jra55.JRA55Reader(m, YEAR, JRA_DIR)
    jra_d1 = reader.step(YEAR, 1, 0.0)
    jra_in = reader.step(YEAR, INTERIOR_DAY, INTERIOR_SEC)
    reader.close()
    cdump = {t: _load(t) for t in ("d1z", "inz", "ins")}
    edump = {t: _load_elem(t) for t in ("d1z", "inz", "ins")}
    jra = {"d1z": jra_d1, "inz": jra_in, "ins": jra_in}
    return dict(mesh=m, jra=jra, cdump=cdump, edump=edump)


def _run_jax(mesh, jraf, c):
    """Run the JAX bulk fed the C dump's exact T_oc / u_w / v_w (cols 8,9,10) + the
    JRA fields. Returns a BulkFluxes."""
    import jax.numpy as jnp
    from fesom_jax import forcing
    return forcing.bulk_surface_fluxes(
        mesh,
        jnp.asarray(jraf.u_wind), jnp.asarray(jraf.v_wind), jnp.asarray(jraf.shum),
        jnp.asarray(jraf.shortwave), jnp.asarray(jraf.longwave), jnp.asarray(jraf.Tair),
        jnp.asarray(jraf.prec_rain), jnp.asarray(jraf.prec_snow),
        T_surf=jnp.asarray(c[:, C_TOC]),
        u_w=jnp.asarray(c[:, C_UW]), v_w=jnp.asarray(c[:, C_VW]),
    )


def test_dumps_are_full_mesh_in_order(setup):
    n = setup["mesh"].nod2D
    e = setup["mesh"].elem2D
    for t in ("d1z", "inz", "ins"):
        assert np.array_equal(setup["cdump"][t][:, C_GID].astype(np.int64),
                              np.arange(1, n + 1))
        assert np.array_equal(setup["edump"][t][:, 0].astype(np.int64),
                              np.arange(1, e + 1))


@pytest.mark.parametrize("tag", ["d1z", "inz", "ins"])
def test_bulk_matches_c(setup, tag):
    mesh = setup["mesh"]
    c = setup["cdump"][tag]
    out = _run_jax(mesh, setup["jra"][tag], c)
    got = {"cd": out.cd, "ce": out.ce, "ch": out.ch,
           "heat_flux": out.heat_flux, "water_flux": out.water_flux,
           "sns_x": out.stress_node_surf[:, 0], "sns_y": out.stress_node_surf[:, 1]}
    worst, bad = {}, {}
    for name, (col, atol) in TOL.items():
        d = float(np.max(np.abs(np.asarray(got[name], np.float64) - c[:, col])))
        worst[name] = d
        if not d < atol:
            bad[name] = (d, atol)
    assert not bad, f"[{tag}] fields exceeding atol: {bad}\n all worst: {worst}"


@pytest.mark.parametrize("tag", ["d1z", "inz", "ins"])
def test_elem_stress_matches_c(setup, tag):
    """node→elem mean-of-3 stress (stress_surf)."""
    mesh = setup["mesh"]
    out = _run_jax(mesh, setup["jra"][tag], setup["cdump"][tag])
    ce = setup["edump"][tag]
    sx = float(np.max(np.abs(np.asarray(out.stress_surf[:, 0], np.float64) - ce[:, 1])))
    sy = float(np.max(np.abs(np.asarray(out.stress_surf[:, 1], np.float64) - ce[:, 2])))
    assert sx < 1e-12 and sy < 1e-12, f"[{tag}] elem stress: sx={sx:.2e} sy={sy:.2e}"


def test_synthetic_current_is_active(setup):
    """The 'ins' dump must carry a genuinely nonzero current (else the relative-wind /
    current→stress path is untested) and differ from the zero-current 'inz' run."""
    cz = setup["cdump"]["inz"]
    cs = setup["cdump"]["ins"]
    cur = np.hypot(cs[:, C_UW], cs[:, C_VW])
    assert cur.max() > 0.4 and (cur > 0.05).mean() > 0.5   # table spans 0..0.5 m/s
    # the synthetic current visibly moves the stress (relative-wind coupling)
    dsx = np.abs(cs[:, C_SX] - cz[:, C_SX])
    assert dsx.max() > 1e-3


def test_earlybreak_drop_is_physically_bounded(setup):
    """⚠️ FINDING (corrects the sub-plan's "post-convergence iters are no-ops → identical"
    claim): dropping the data-dependent early break is NOT a no-op. The L&Y09 Monin-Obukhov
    loop does not robustly converge at near-calm nodes, so fixed-5 vs early-break diverge —
    the dump's fixed-5 ``ch`` (col 3) vs early-break ``ch_eb`` (col 13) differ by up to ~88%
    at the calmest tropical nodes (``cd``/``ce`` up to ~4.5%).

    BUT that coefficient divergence is **physically bounded**: ``ch``/``ce`` enter only the
    ``ug``-scaled sensible/latent terms, so the heat_flux impact is ≤~10 W/m² at a handful of
    nodes (mean ~2e-4 W/m²) and the stress impact ≤~4e-3 N/m². JAX runs fixed-5 (AD-safe) and
    is verified against the **fixed-5** C dump (``test_bulk_matches_c``); the residual vs the C
    *production* early-break config is this bounded, documented effect — acceptable (the M-O
    iteration is a 5-iter-capped approximation either way), and re-matched exactly in Task 5.7
    by running the reference dump with ``FESOM_BULK_FIXED_ITERS``."""
    from fesom_jax import forcing
    import jax.numpy as jnp
    # coefficient-level divergence (reported — the honest large number)
    ch_rel = 0.0
    for t in ("d1z", "inz", "ins"):
        c = setup["cdump"][t]
        ch_rel = max(ch_rel, float(np.max(np.abs(c[:, C_CH] - c[:, C_CHE]) / np.abs(c[:, C_CH]))))
    assert ch_rel > 0.1, "expected a genuine (non-trivial) ch divergence at calm nodes"

    # physical-flux impact: heat_flux with fixed-5 vs early-break coeffs (same JRA + T_oc).
    worst_hf, worst_n = 0.0, 0
    for t in ("inz", "ins"):
        c, jraf = setup["cdump"][t], setup["jra"][t]
        ug = jnp.hypot(jnp.asarray(jraf.u_wind), jnp.asarray(jraf.v_wind))
        common = (jnp.asarray(jraf.shum), jnp.asarray(jraf.shortwave),
                  jnp.asarray(jraf.longwave), jnp.asarray(c[:, C_TOC]), ug, jnp.asarray(jraf.Tair))
        qsr5, qns5, _ = forcing.obudget(*common, jnp.asarray(c[:, C_CH]), jnp.asarray(c[:, C_CE]))
        qsre, qnse, _ = forcing.obudget(*common, jnp.asarray(c[:, C_CHE]), jnp.asarray(c[:, C_CEE]))
        dhf = np.abs(np.asarray((qns5 - qsr5) - (qnse - qsre)))
        worst_hf = max(worst_hf, float(dhf.max()))
        worst_n = max(worst_n, int((dhf > 0.1).sum()))
    # bounded: ≤~10 W/m² at a few nodes (observed max ~7.2 W/m², ~4 nodes >0.5 W/m²).
    assert worst_hf < 12.0, f"early-break drop heat_flux impact too large: {worst_hf:.2f} W/m²"
    assert worst_n < 50, f"early-break drop affects too many nodes: {worst_n} > 0.1 W/m²"


def test_ad_finite_sst_and_current(setup):
    """The differentiable seams: d(Σheat_flux)/d(SST) and d(Σstress)/d(current) are
    FINITE at every CORE2 node — including any Δu≈0 lane (the safe-sqrt probe). Uses
    the C dump's T_oc / current as the linearization point (the 'ins' synthetic
    current gives genuinely nonzero Δu coupling)."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import forcing
    mesh = setup["mesh"]
    jraf = setup["jra"]["ins"]
    c = setup["cdump"]["ins"]
    args = (jnp.asarray(jraf.u_wind), jnp.asarray(jraf.v_wind), jnp.asarray(jraf.shum),
            jnp.asarray(jraf.shortwave), jnp.asarray(jraf.longwave), jnp.asarray(jraf.Tair),
            jnp.asarray(jraf.prec_rain), jnp.asarray(jraf.prec_snow))
    T0 = jnp.asarray(c[:, C_TOC])
    uw0 = jnp.asarray(c[:, C_UW])
    vw0 = jnp.asarray(c[:, C_VW])

    def heat_of_sst(T):
        return jnp.sum(forcing.bulk_surface_fluxes(mesh, *args, T_surf=T,
                                                   u_w=uw0, v_w=vw0).heat_flux)

    def stress_of_cur(uw):
        return jnp.sum(forcing.bulk_surface_fluxes(mesh, *args, T_surf=T0,
                                                   u_w=uw, v_w=vw0).stress_node_surf)

    g_sst = jax.grad(heat_of_sst)(T0)
    g_cur = jax.grad(stress_of_cur)(uw0)
    assert bool(jnp.isfinite(g_sst).all()), "d(heat_flux)/d(SST) has non-finite lanes"
    assert bool(jnp.isfinite(g_cur).all()), "d(stress)/d(current) has non-finite lanes"
    # both feedbacks are genuinely active (not all-zero)
    assert float(jnp.max(jnp.abs(g_sst))) > 1.0     # ~ -d(net heat)/dSST, O(10) W/m²/K
    assert float(jnp.max(jnp.abs(g_cur))) > 1e-3
