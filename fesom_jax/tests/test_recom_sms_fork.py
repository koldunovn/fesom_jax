"""Lane E gate (docs/plans/completed/20261028-recom-spectral-port.md): stage 1 of the spectral fork's REcoM
(``recom_sources_sinks_fork``, 4 phytoplankton + 3 zooplankton + 2 detritus, broadband light) from each dumped
``recom_in`` of the fork's NON-spectral twin (``install_nowb_rel``, run ``pi_30d_nowb_adv01`` on Albedo, namelist.tra weights 0., 1.) against its
``recom_out``: the 36 updated tracers, the fork's 3-D and 2-D diagnostics, PAR, the benthos, ``GlodecayBenthos``,
the surface fluxes and the carbonate cache. Error = ``max |jax - fortran| / max |fortran|`` over the wet cells,
``TOL = 1e-11`` unless listed in ``LOOSE``. Skips without the dumps. ``python -m fesom_jax.tests.test_recom_sms_fork``
prints the table.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from fesom_jax.io_dump import load_recom_dump                                      # noqa: E402
from fesom_jax.recom import layout as LY                                           # noqa: E402
from fesom_jax.recom import params as P                                            # noqa: E402
from fesom_jax.recom import spectral as SP                                         # noqa: E402
from fesom_jax.recom.forcing import cobeta                                         # noqa: E402
from fesom_jax.recom.sms import CARB_FIELDS, CarbCache                             # noqa: E402
from fesom_jax.recom.sms_fork import DIAG2D_BASE, DIAG2D_COCCO, DIAG3D_BASE, DIAG3D_COCCO   # noqa: E402
from fesom_jax.recom.stage1 import BgcStepInputs, RecomGeometry, recom_sources_sinks_fork  # noqa: E402

REF = Path(os.environ.get("FESOM_RECOM_FORK_REF", "/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/recom_ref"))
REPO = Path(__file__).resolve().parents[2]
RUN = os.environ.get("FESOM_RECOM_FORK_RUN", "pi_30d_nowb_adv01")
SPECTRAL = "spectral" in RUN                      # the spectral twin: 37 tracers (CDOM), the spectral light
NML = REPO / "scripts" / "recom" / "fortran" / ("namelists_albedo_spectral" if SPECTRAL else
                                                "namelists_albedo_spectral_nowb") / "namelist.recom"
STEPS = (1, 2, 3, 17, 224, 960)
OPTICS = Path(os.environ.get("FESOM_RECOM_OPTICS_DIR", "/albedo/work/projects/p_specbic/sloza/fesom_jax_data/recom_pi/optics"))
TOL = 1e-11
LOOSE: dict = {}

pytestmark = pytest.mark.skipif(not (REF / RUN / "dumps").is_dir(),
                                reason=f"the fork's reference dumps are absent ({REF / RUN})")


@lru_cache(maxsize=1)
def _setup():
    sw, par = P.load_namelist(NML, check=False)
    lay = LY.layout_of(sw)
    tab = SP.load_tables(sw, par, OPTICS) if SPECTRAL else None

    @jax.jit
    def run(bgc, carb, benthos, alk_surf, temp, salt, geom, inputs, params, spectral):
        # the spectral build: the N:C clamp, the fork's own (no) CO2 limitation in the spectral growth
        return recom_sources_sinks_fork(bgc, carb, benthos, alk_surf, temp, salt, geom, inputs, 2700.0, params,
                                        sw, lay, nc_clamp=SPECTRAL, spectral=spectral, spectral_co2lim=False)

    @jax.jit
    def light(bgc, hnode, cos_ai, Edsf, Essf, params):
        return SP.spectral_step(bgc, hnode, cos_ai, Edsf, Essf, tab, params, sw, lay).light
    return run, par, sw, lay, (light if SPECTRAL else None)


def _inputs(f, lay):
    tr = lambda st: np.stack([f[f"{st}.tr{i}"] for i in lay.ids], axis=-1)       # noqa: E731
    cal = f["recom_in.calendar"]
    nlev = f["mesh.nlevels_nod2D"].astype(np.int32) - 1
    assert (f["mesh.ulevels_nod2D"] == 1).all()
    geom = RecomGeometry(hnode=f["recom_in.hnode"], zbar=f["recom_in.zbar_3d_n"], nlev=nlev,
                         lat_rad=f["mesh.geo_coord_nod2D"][:, 1], area_top=f["mesh.areasvol"][:, 0],
                         ocean_area=f["mesh.ocean_area"][0])
    inp = BgcStepInputs(daynew=int(cal[0]), ndpyr=int(cal[3]), mstep=int(cal[2]),
                        fe_dust=f["recom_in.GloFeDust"], n_dust=f["recom_in.GloNDust"],
                        co2_atm=f["recom_in.AtmCO2"][int(cal[1]) - 1], press_air=f["recom_in.press_air"],
                        u_wind=f["recom_in.u_wind"], v_wind=f["recom_in.v_wind"],
                        shortwave=f["recom_in.shortwave"], a_ice=f["recom_in.a_ice"])
    carb = CarbCache(*(f[f"recom_in.{n}3D"] for n in CARB_FIELDS))
    return dict(bgc=tr("recom_in"), carb=carb, benthos=f["recom_in.Benthos"].T, alk_surf=f["recom_in.Alk_surf"],
                temp=f["recom_in.tr1"], salt=f["recom_in.tr2"], geom=geom, inputs=inp,
                dt=float(f["recom_in.dt"][0])), tr("recom_out")


def _err(a, b, mask):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[mask], b[mask]
    den = np.max(np.abs(b)) if b.size else 0.0
    num = np.max(np.abs(a - b)) if b.size else 0.0
    return num / den if den > 0 else num, den


def gate_errors(step) -> dict:
    run, par, sw, lay, light = _setup()
    f, meta = load_recom_dump(REF / RUN / "dumps", step)
    x, out_tr = _inputs(f, lay)
    assert x["dt"] == 2700.0
    spec = None
    if SPECTRAL:          # the light of this step from recom_in and the OASIM fields the step read (lane L's gate)
        _, tab = _setup()[2], SP.load_tables(sw, par, OPTICS)
        cos_ai = cobeta(x["inputs"].daynew, x["inputs"].ndpyr, x["geom"].lat_rad)
        Edsf, Essf = SP.surface_irradiance(np.asarray(f["recom_out.oasim_ed2D"]).T,
                                           np.asarray(f["recom_out.oasim_es2D"]).T, x["inputs"].a_ice, None, tab, sw)
        spec = light(x["bgc"], x["geom"].hnode, cos_ai, Edsf, Essf, par)
    r = jax.device_get(run(x["bgc"], x["carb"], x["benthos"], x["alk_surf"], x["temp"], x["salt"], x["geom"],
                           x["inputs"], par, spec))
    nl1 = x["bgc"].shape[1]
    wet = np.arange(nl1)[None, :] < x["geom"].nlev[:, None]
    allnodes = np.ones(wet.shape[0], bool)
    e = {}
    for s, tid in enumerate(lay.ids):
        e[f"tr{tid}_{lay.names[s]}"] = _err(r.bgc[..., s], out_tr[..., s], wet)
    d3 = DIAG3D_BASE + (DIAG3D_COCCO if sw.enable_coccos else ())
    d2 = DIAG2D_BASE + (DIAG2D_COCCO if sw.enable_coccos else ())
    for n in d3:
        if SPECTRAL and n.startswith("TCphotLigLim"):      # unset by the fork in a spectral build
            continue
        if f"recom_out.{n}" in f:
            e[n] = _err(r.sms.diag3d[n], f[f"recom_out.{n}"], wet)
    for n in d2:
        if f"recom_out.{n}" in f:
            e[n] = _err(r.sms.diag2d[n], f[f"recom_out.{n}"], allnodes)
    e["PAR3D"] = _err(r.sms.par, f["recom_out.PAR3D"], wet)
    for i in range(4):
        e[f"Benthos{i + 1}"] = _err(r.benthos[:, i], f["recom_out.Benthos"][i], allnodes)
        e[f"GlodecayBenthos{i + 1}"] = _err(r.sfc.GlodecayBenthos[:, i], f["recom_out.GlodecayBenthos"][i], allnodes)
    for n in ("GloCO2flux_seaicemask", "GloO2flux_seaicemask", "AtmFeInput", "AtmNInput", "relax_alk",
              "GloCO2flux", "GloO2flux", "GloPCO2surf", "GlodPCO2surf", "GloHplus"):
        if f"recom_out.{n}" in f:
            e[n] = _err(getattr(r.sfc, n), f[f"recom_out.{n}"], allnodes)
    for i, n in enumerate(CARB_FIELDS):
        e[f"{n}3D"] = _err(r.carb[i], f[f"recom_out.{n}3D"], wet)
    changed = np.zeros_like(wet)
    for n in CARB_FIELDS:
        changed |= f[f"recom_out.{n}3D"] != f[f"recom_in.{n}3D"]
    refr = np.asarray(r.sms.refreshed) & wet
    e["_refresh_mismatch_cells"] = (float(np.sum(changed & wet & ~refr)), float(np.sum(refr & ~changed)))
    return e


def _available_steps():
    return [s for s in STEPS if (REF / RUN / "dumps" / f"step_{s:06d}").is_dir()]


@pytest.mark.parametrize("step", _available_steps() or [1])
def test_g2_fork(step):
    if not (REF / RUN / "dumps" / f"step_{step:06d}").is_dir():
        pytest.skip(f"no dump for step {step}")
    e = gate_errors(step)
    bad = {k: v for k, v in e.items() if not k.startswith("_") and v[0] > LOOSE.get(k, (TOL,))[0]}
    assert not bad, f"step {step}: {bad}"
    assert e["_refresh_mismatch_cells"] == (0.0, 0.0), e["_refresh_mismatch_cells"]


if __name__ == "__main__":
    for step in _available_steps():
        e = gate_errors(step)
        print(f"== step {step}")
        for k, v in sorted(e.items(), key=lambda kv: -kv[1][0] if not kv[0].startswith("_") else 0):
            flag = "" if k.startswith("_") or v[0] <= LOOSE.get(k, (TOL,))[0] else "   <-- FAIL"
            print(f"  {k:28s} {v[0]:10.3e}  (max|f| {v[1]:10.3e}){flag}")
