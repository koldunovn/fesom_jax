"""Gate G2 (parent plan Tasks 9-10): stage 1 (``recom_sources_sinks``) from each dumped ``recom_in`` against the
Fortran ``recom_out`` of the same step: the 31 updated tracers, the 3-D and 2-D diagnostics, PAR, the benthos,
``GlodecayBenthos``, the surface fluxes, ``GloHplus``, pCO2surf and the carbonate cache (refresh steps 1, 224,
960; untouched elsewhere).

Error measure: ``max |jax - fortran| / max |fortran|`` over the wet cells (``k < nlevels_nod2D - 1``) of each
field, tolerance ``TOL = 1e-11`` unless a field is listed in ``LOOSE`` with its reason. Skips when the reference
dumps are absent. ``python -m fesom_jax.tests.test_recom_sms`` prints the full table.
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

from fesom_jax.io_dump import load_recom_dump                                    # noqa: E402
from fesom_jax.recom import params as P                                          # noqa: E402
from fesom_jax.recom import tracers as T                                         # noqa: E402
from fesom_jax.recom.sms import CARB_FIELDS, DIAG2D, DIAG3D, CarbCache           # noqa: E402
from fesom_jax.recom.stage1 import BgcStepInputs, RecomGeometry, recom_sources_sinks  # noqa: E402

REF = Path(os.environ.get("FESOM_RECOM_REF", "/work/ab0995/a270088/recom_ref"))
REPO = Path(__file__).resolve().parents[2]
NML = REPO / "scripts" / "recom" / "fortran" / "namelists" / "namelist.recom"
STEPS = [("pi_30d_jan", s) for s in (1, 2, 17, 224, 225, 960)] + [("pi_1d_jul", s) for s in (1, 17)]
TOL = 1e-11
LOOSE: dict = {}        # field -> (tolerance, reason)

pytestmark = pytest.mark.skipif(not (REF / "pi_30d_jan" / "dumps").is_dir(),
                                reason=f"REcoM Fortran reference dumps absent ({REF})")


def _load(run, step):
    f, meta = load_recom_dump(REF / run / "dumps", step)
    return f, meta


def _inputs(f, meta):
    tr = lambda st: np.stack([f[f"{st}.tr{i}"] for i in T.BGC_IDS], axis=-1)
    cal = f["recom_in.calendar"]                    # daynew, month, mstep, ndpyr, yearold
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


@lru_cache(maxsize=1)
def _stage1():
    _, params = P.load_namelist(NML)

    @jax.jit
    def run(bgc, carb, benthos, alk_surf, temp, salt, geom, inputs, params):
        return recom_sources_sinks(bgc, carb, benthos, alk_surf, temp, salt, geom, inputs, 2700.0, params)
    return run, params


def _err(a, b, mask):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[mask], b[mask]
    den = np.max(np.abs(b)) if b.size else 0.0
    num = np.max(np.abs(a - b)) if b.size else 0.0
    return num / den if den > 0 else num, den


def g2_errors(run, step) -> dict:
    """``{field: (rel. error, max |fortran|)}`` for one dumped step."""
    f, meta = _load(run, step)
    x, out_tr = _inputs(f, meta)
    assert x["dt"] == 2700.0
    stage1, params = _stage1()
    r = stage1(x["bgc"], x["carb"], x["benthos"], x["alk_surf"], x["temp"], x["salt"], x["geom"], x["inputs"],
               params)
    r = jax.device_get(r)
    nl1 = x["bgc"].shape[1]
    wet = np.arange(nl1)[None, :] < x["geom"].nlev[:, None]
    allnodes = np.ones(wet.shape[0], bool)
    e = {}
    for s, tid in enumerate(T.BGC_IDS):
        e[f"tr{tid}_{T.NAMES[s]}"] = _err(r.bgc[..., s], out_tr[..., s], wet)
    for n in DIAG3D:
        e[n] = _err(r.sms.diag3d[n], f[f"recom_out.{n}"], wet)
    for n in DIAG2D:
        e[n] = _err(r.sms.diag2d[n], f[f"recom_out.{n}"], allnodes)
    e["PAR3D"] = _err(r.sms.par, f["recom_out.PAR3D"], wet)
    for i in range(4):
        e[f"Benthos{i + 1}"] = _err(r.benthos[:, i], f["recom_out.Benthos"][i], allnodes)
        e[f"GlodecayBenthos{i + 1}"] = _err(r.sfc.GlodecayBenthos[:, i], f["recom_out.GlodecayBenthos"][i], allnodes)
    for n in ("GloCO2flux_seaicemask", "GloO2flux_seaicemask", "AtmFeInput", "AtmNInput", "relax_alk",
              "GloCO2flux", "GloO2flux", "GloPCO2surf", "GlodPCO2surf", "GloHplus", "PistonVelocity", "alphaCO2",
              "x_co2atm"):
        e[n] = _err(getattr(r.sfc, n), f[f"recom_out.{n}"], allnodes)
    for i, n in enumerate(CARB_FIELDS):
        e[f"{n}3D"] = _err(r.carb[i], f[f"recom_out.{n}3D"], wet)
    # refresh mask: where Fortran's cache changed vs where we refreshed
    changed = np.zeros_like(wet)
    for n in CARB_FIELDS:
        changed |= f[f"recom_out.{n}3D"] != f[f"recom_in.{n}3D"]
    refr = np.asarray(r.sms.refreshed) & wet
    e["_refresh_mismatch_cells"] = (float(np.sum(changed & wet & ~refr)), float(np.sum(refr & ~changed)))
    e["_refresh_counts"] = (float(np.sum(np.asarray(r.sms.refreshed) & wet)), float(np.sum(changed & wet)))
    return e


@pytest.mark.parametrize("run,step", STEPS)
def test_g2(run, step):
    e = g2_errors(run, step)
    nref, nf = e.pop("_refresh_counts")
    e.pop("_refresh_mismatch_cells")
    assert nref >= nf, f"fewer refreshed cells ({nref}) than Fortran changed ({nf})"
    bad = {k: v for k, v in e.items() if v[0] > LOOSE.get(k, (TOL,))[0]}
    assert not bad, f"G2 {run} step {step}: " + ", ".join(f"{k} {v[0]:.2e}" for k, v in sorted(bad.items()))


if __name__ == "__main__":
    import sys
    steps = STEPS if len(sys.argv) == 1 else [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
    for run, step in steps:
        e = g2_errors(run, step)
        print(f"== {run} step {step}: refreshed/changed cells {e.pop('_refresh_counts')}, "
              f"{e.pop('_refresh_mismatch_cells')}")
        for k, (err, mx) in e.items():
            flag = "  <-- FAIL" if err > LOOSE.get(k, (TOL,))[0] else ""
            print(f"  {k:28s} {err:10.2e}  (max {mx:10.3e}){flag}")
