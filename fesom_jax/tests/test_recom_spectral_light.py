"""Lane L of the spectral port (docs/plans/20261028-recom-spectral-port.md): the fork's spectral light model
(:mod:`fesom_jax.recom.spectral`) against the spectral twin's dumps (run ``pi_30d_spectral_adv01`` on Albedo,
``install_rel/bin/fesom.x``, 37 tracers with CDOM, OASIM, RECOM_RADTRANS, RECOM_CALC_ACDOM, RECOM_CALC_APART,
RECOM_BMASS).

From ``recom_in`` of each dumped step (the BGC tracers at the start of REcoM, ``hnode``, ``a_ice``, the calendar) and
the OASIM fields the step read (``recom_out.oasim_ed2D/es2D``), the JAX :func:`spectral_step` must reproduce the five
light planes ``Edz3D, Esz3D, Euz3D, Eutop3D, Estop3D`` (``Light_watercolumn``, recom_forcing.F90:812-816) per band
and layer, and ``PAR3D`` (the ``PARave`` of ``REcoM_sms``, recom_sms.F90:1458-1465) on the wet layers.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from fesom_jax.io_dump import load_recom_dump
from fesom_jax.recom import layout as LY
from fesom_jax.recom import params as P
from fesom_jax.recom import spectral as SP
from fesom_jax.recom.forcing import cobeta

REF = Path(os.environ.get("FESOM_RECOM_FORK_REF", "/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/recom_ref"))
OPTICS = Path(os.environ.get("FESOM_RECOM_OPTICS_DIR",
                             "/albedo/work/projects/p_specbic/sloza/fesom_jax_data/recom_pi/optics"))
REPO = Path(__file__).resolve().parents[2]
NML = REPO / "scripts" / "recom" / "fortran" / "namelists_albedo_spectral" / "namelist.recom"
RUN = os.environ.get("FESOM_RECOM_SPECTRAL_RUN", "pi_30d_spectral_adv01")
STEPS = [s for s in (1, 2, 3, 17, 224, 960) if (REF / RUN / "dumps" / f"step_{s:06d}").is_dir()] or [1]
TOL = 1e-11
PLANES = ("Edz3D", "Esz3D", "Euz3D", "Eutop3D", "Estop3D")

pytestmark = pytest.mark.skipif(not (REF / RUN / "dumps").is_dir() or not OPTICS.is_dir(),
                                reason=f"the spectral twin's dumps or the optics files are absent ({REF / RUN}, {OPTICS})")


@lru_cache(maxsize=1)
def _setup():
    sw, par = P.load_namelist(NML, check=False)
    lay = LY.layout_of(sw)
    tab = SP.load_tables(sw, par, OPTICS)

    @jax.jit
    def run(bgc, hnode, cos_ai, Edsf, Essf, params):
        return SP.spectral_step(bgc, hnode, cos_ai, Edsf, Essf, tab, params, sw, lay)
    return run, par, sw, lay, tab


def _inputs(step):
    run, par, sw, lay, tab = _setup()
    names = [f"recom_in.tr{i}" for i in lay.ids] + ["recom_in.hnode", "recom_in.a_ice", "recom_in.calendar",
                                                     "mesh.geo_coord_nod2D", "mesh.nlevels_nod2D",
                                                     "recom_out.oasim_ed2D", "recom_out.oasim_es2D", "recom_out.Ed0_2D",
                                                     "recom_out.PAR3D"] + [f"recom_out.{n}" for n in PLANES]
    f, meta = load_recom_dump(REF / RUN / "dumps", step, names=names)
    bgc = np.stack([f[f"recom_in.tr{i}"] for i in lay.ids], axis=-1)
    hnode = np.asarray(f["recom_in.hnode"], np.float64)
    nlev = f["mesh.nlevels_nod2D"].astype(np.int32) - 1
    lat_rad = np.asarray(f["mesh.geo_coord_nod2D"])[:, 1]
    daynew, ndpyr = int(meta["daynew"]), int(meta["ndpyr"])
    cos_ai = np.asarray(cobeta(daynew, ndpyr, lat_rad))
    ed, es = np.asarray(f["recom_out.oasim_ed2D"]).T, np.asarray(f["recom_out.oasim_es2D"]).T     # (N, 13)
    Edsf, Essf = SP.surface_irradiance(ed, es, f["recom_in.a_ice"], None, tab, sw)
    assert np.abs(np.asarray(Edsf) - np.asarray(f["recom_out.Ed0_2D"]).T).max() < 1e-14
    want = {n: np.moveaxis(np.asarray(f[f"recom_out.{n}"]), 0, -1) for n in PLANES}             # (N, nl1, 13)
    want["PAR3D"] = np.asarray(f["recom_out.PAR3D"])
    return run, bgc, hnode, cos_ai, np.asarray(Edsf), np.asarray(Essf), par, want, nlev


def light_errors(step) -> dict:
    """``{plane: (rel. error, max |fortran|)}`` over every layer (the planes) / the wet layers (PAR3D)."""
    run, bgc, hnode, cos_ai, Edsf, Essf, par, want, nlev = _inputs(step)
    out = run(bgc, hnode, cos_ai, Edsf, Essf, par)
    got = {"Edz3D": out.rt.Edbot, "Esz3D": out.rt.Esbot, "Euz3D": out.rt.Eubot, "Eutop3D": out.rt.Eutop,
           "Estop3D": out.rt.Estop, "PAR3D": out.light.PARave}
    nl1 = hnode.shape[1]
    wet = np.arange(nl1)[None, :] < nlev[:, None]
    err = {}
    for n, b in want.items():
        a = np.asarray(got[n])
        if n == "PAR3D":
            a, b = a[wet], b[wet]
        den = float(np.max(np.abs(b)))
        err[n] = (float(np.max(np.abs(a - b))) / den if den > 0 else float(np.max(np.abs(a))), den)
    return err


@pytest.mark.parametrize("step", STEPS)
def test_spectral_light_vs_fortran(step):
    err = light_errors(step)
    worst = max(err.items(), key=lambda kv: kv[1][0])
    print(f"SPECTRAL LIGHT {RUN} {step}: worst {worst[0]} {worst[1][0]:.2e}")
    bad = {k: v for k, v in err.items() if v[0] > TOL}
    assert not bad, bad


def test_band_tables():
    """The fixed tables of WAVEBANDS_INIT_FIXED: 13 bands 400..700, widths 12.5/25/12.5 summing to 300 nm."""
    assert SP.PWAVES == tuple(range(400, 701, 25))
    assert np.allclose(SP.WB_WIDTH, [12.5] + [25.0] * 11 + [12.5]) and SP.WB_TOTAL_WIDTH == 300.0
    assert SP.WTOUEINS.shape == (13,) and 3.3 < SP.WTOUEINS[0] < 3.4             # 400 nm: 3.343 uEin s-1 per W


if __name__ == "__main__":
    for step in STEPS:
        err = light_errors(step)
        print(f"== {RUN} step {step}:", {k: f"{v[0]:.2e}" for k, v in err.items()})
