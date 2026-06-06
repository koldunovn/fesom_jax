"""Task 5.5 gate: the JAX SSS-restoring + CORE2-runoff port (``sss_runoff.py``)
reproduces the C ``fesom_sss_runoff_step`` on CORE2, verified against the C
``sss_dump_*`` all-node dumps at two months:

  * ``m1`` — January  (jra@day1,            month 1, first-step state).
  * ``m4`` — a month crossing (jra@day100 ≈ Apr 10, month 4).

Two independent gates, mirroring the bulk (Task 5.4) split:

  1. **Readers** (host numpy) — ``reader.month(m)`` vs the C ``Ssurf`` column and
     ``reader.runoff_node`` vs the C ``runoff`` column. The bilinear ``interp_2d_field``
     is MAP-class (bit-for-bit for ocean-bracket nodes); the SSS 30-cell missing fill is a
     reduction (~1e-13 at coastal/land-bracket nodes — the ``/count`` crushes the box-sum
     reassociation). Runoff has no fill (``check_dummy=0``) ⇒ bit-for-bit.
  2. **Flux math** (AD-safe JAX) — ``sss_runoff_fluxes`` is fed the C dump's exact
     ``(S_top, water_flux_in, Ssurf, runoff)`` so the gate isolates the salt/water balance
     from the reader: the multiplies are MAP (bit-exact, same f64 inputs), the only
     difference is the area-weighted global-mean reductions (~1e-21, well inside 1e-12).

Plus an AD-finiteness probe (``d/d(water_flux)`` = the SST→flux seam via the bulk;
``d/d(S_top)`` = the restoring seam).

SKIPS unless the CORE2 mesh export, the C sss dump, and the SSS/runoff NetCDF all exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
SSS_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "sss_dump_core2"
SSS_PATH = "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/PHC2_salx.nc"
RUNOFF_PATH = "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/CORE2_runoff.nc"

# month tag → 1-based SSS month it dumped (must match jax_sss_dump_core2.sh).
TAG_MONTH = {"m1": 1, "m4": 4}

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and SSS_DUMP_DIR.is_dir()
         and Path(SSS_PATH).is_file() and Path(RUNOFF_PATH).is_file()),
    reason="needs CORE2 mesh export + C sss dump + SSS/runoff NetCDF (Task 5.5 artifacts)",
)

# Dump node columns (0-based): gid Ssurf runoff S_top water_flux_in
#                              virtual_salt relax_salt water_flux_out
C_GID, C_SSURF, C_RUNOFF, C_STOP, C_WFIN, C_VS, C_RS, C_WFOUT = range(8)


def _load(tag):
    d = np.loadtxt(SSS_DUMP_DIR / f"sss_dump_{tag}_rank0.txt")
    return d[np.argsort(d[:, 0])]              # order by 1-based gid → index = gid-1


@pytest.fixture(scope="module")
def setup():
    from fesom_jax import mesh as meshmod, sss_runoff
    m = meshmod.load_mesh(CORE2_MESH_DIR)
    reader = sss_runoff.build_reader(m, SSS_PATH, RUNOFF_PATH)
    cdump = {t: _load(t) for t in TAG_MONTH}
    return dict(mesh=m, reader=reader, cdump=cdump)


def test_dumps_are_full_mesh_in_order(setup):
    n = setup["mesh"].nod2D
    for t in TAG_MONTH:
        assert np.array_equal(setup["cdump"][t][:, C_GID].astype(np.int64),
                              np.arange(1, n + 1))


@pytest.mark.parametrize("tag", list(TAG_MONTH))
def test_runoff_reader_matches_c(setup, tag):
    """Runoff has no missing-fill (``check_dummy=0``) ⇒ pure bilinear ⇒ bit-for-bit."""
    runoff = np.asarray(setup["reader"].runoff_node, np.float64)
    c = setup["cdump"][tag][:, C_RUNOFF]
    worst = float(np.max(np.abs(runoff - c)))
    assert worst < 1e-15, f"[{tag}] runoff reader max|Δ|={worst:.3e}"


@pytest.mark.parametrize("tag", list(TAG_MONTH))
def test_sss_reader_matches_c(setup, tag):
    """SSS bilinear is MAP-class (bit-for-bit for ocean-bracket nodes); the 30-cell
    expanding fill is a reduction at coastal/land-bracket nodes. Gate at a reduction
    tolerance and report how tight the bulk of nodes is."""
    month = TAG_MONTH[tag]
    ssurf = np.asarray(setup["reader"].month(month), np.float64)
    c = setup["cdump"][tag][:, C_SSURF]
    d = np.abs(ssurf - c)
    worst = float(d.max())
    n_loose = int((d > 1e-12).sum())
    n_exact = int((d == 0.0).sum())
    # MAP-class for the (majority) ocean-bracket nodes; reduction-class where the
    # bilinear bracket includes a land-extrapolated (filled) cell.
    assert worst < 1e-9, (f"[{tag}] SSS reader max|Δ|={worst:.3e} "
                          f"(#>1e-12={n_loose}, #bit-exact={n_exact}/{len(c)})")


@pytest.mark.parametrize("tag", list(TAG_MONTH))
def test_flux_math_matches_c(setup, tag):
    """Feed the JAX flux math the C dump's exact (S_top, water_flux_in, Ssurf, runoff);
    the only JAX↔C difference is the area-weighted global-mean reductions. Gate the three
    outputs (virtual_salt, relax_salt, water_flux_out) at reduction tolerance."""
    import jax.numpy as jnp
    from fesom_jax import sss_runoff
    mesh = setup["mesh"]
    c = setup["cdump"][tag]
    areasvol_surf = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])
    ocean_area = float(mesh.ocean_area)
    open_water = jnp.asarray(np.asarray(mesh.ulevels_nod2D) <= 1)

    out = sss_runoff.sss_runoff_fluxes(
        S_top=jnp.asarray(c[:, C_STOP]),
        water_flux=jnp.asarray(c[:, C_WFIN]),
        Ssurf_month=jnp.asarray(c[:, C_SSURF]),
        runoff_node=jnp.asarray(c[:, C_RUNOFF]),
        areasvol_surf=areasvol_surf, ocean_area=ocean_area, open_water=open_water)

    got = {"virtual_salt": (out.virtual_salt, C_VS, 1e-12),
           "relax_salt":   (out.relax_salt,   C_RS, 1e-12),
           "water_flux":   (out.water_flux,   C_WFOUT, 1e-15)}
    worst, bad = {}, {}
    for name, (arr, col, atol) in got.items():
        dd = float(np.max(np.abs(np.asarray(arr, np.float64) - c[:, col])))
        worst[name] = dd
        if not dd < atol:
            bad[name] = (dd, atol)
    assert not bad, f"[{tag}] flux fields exceeding atol: {bad}\n all worst: {worst}"


def test_global_means_are_zero(setup):
    """The virtual_salt / relax_salt area-weighted global means are subtracted to ~0
    (the no-net-salt-flux constraint); confirm on the C dump's own output columns."""
    mesh = setup["mesh"]
    areasvol = np.asarray(mesh.areasvol)[:, 0]
    oa = float(mesh.ocean_area)
    for tag in TAG_MONTH:
        c = setup["cdump"][tag]
        vs_mean = float(np.sum(c[:, C_VS] * areasvol) / oa)
        rs_mean = float(np.sum(c[:, C_RS] * areasvol) / oa)
        scale = float(np.max(np.abs(c[:, C_VS]))) + 1e-30
        assert abs(vs_mean) < 1e-9 * scale, f"[{tag}] virtual_salt mean {vs_mean:.3e}"
        assert abs(rs_mean) < 1e-9 * scale, f"[{tag}] relax_salt mean {rs_mean:.3e}"


def test_ad_finiteness(setup):
    """The flux math is differentiable w.r.t. the bulk ``water_flux`` (the SST→flux seam
    via the bulk) and ``S_top`` (the restoring seam), finite everywhere (no sqrt/÷0)."""
    import jax, jax.numpy as jnp
    from fesom_jax import sss_runoff
    mesh = setup["mesh"]
    c = setup["cdump"]["m1"]
    areasvol_surf = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])
    ocean_area = float(mesh.ocean_area)
    Ssurf = jnp.asarray(c[:, C_SSURF])
    runoff = jnp.asarray(c[:, C_RUNOFF])
    wf = jnp.asarray(c[:, C_WFIN])
    st = jnp.asarray(c[:, C_STOP])

    def loss(water_flux, S_top):
        o = sss_runoff.sss_runoff_fluxes(S_top, water_flux, Ssurf, runoff,
                                         areasvol_surf, ocean_area)
        return jnp.sum(o.virtual_salt**2) + jnp.sum(o.relax_salt**2) + jnp.sum(o.water_flux**2)

    g_wf, g_st = jax.grad(loss, argnums=(0, 1))(wf, st)
    assert bool(jnp.all(jnp.isfinite(g_wf))), "d/d(water_flux) not finite"
    assert bool(jnp.all(jnp.isfinite(g_st))), "d/d(S_top) not finite"
