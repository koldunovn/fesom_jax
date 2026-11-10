#!/usr/bin/env python
"""JAX-Fortran sea-ice parity of one year: monthly means of a JAX forward against a Fortran run.

    python scripts/icecal/parity.py --jax <dir with fields.npz> --fortran <run dir> --year 2013 \
                                    --out <new dir> [--mesh-dir data/mesh_core2]

Both models are read with the readers of ``score.py``: the JAX ``fields.npz`` of
``icecal.py fwd``/``eval`` (``C_mean``, ``H_mean`` [12, nod2D], ``n_mon``, ``year``) and the
Fortran ``a_ice.fesom.<YYYY>.nc``/``m_ice.fesom.<YYYY>.nc`` (record i = calendar month i + 1, the
fill value at ice-free nodes read as 0 ice). Only months present in both are compared.

Per month and hemisphere (NH lat > 0, SH lat < 0, wet surface nodes), JAX, Fortran and JAX minus
Fortran of

* ice area  sum(clip(C, 0, 1) * node area)              [10^6 km^2]
* extent    sum(node area where C > 15 %)               [10^6 km^2]
* volume    sum(H * node area), H = m_ice (effective)   [10^3 km^3]

and the node-area-weighted spatial RMS and mean of the concentration and effective-thickness
differences over the ice domain (nodes where either model has C > 0), with that domain's area.
The year block gives, per hemisphere and quantity, the mean and the largest absolute monthly
difference and the RMS over the months. Writes ``<out>/parity.json``; ``<out>`` must not exist.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as SC  # noqa: E402

MKM2 = 1e12                    # m^2 per 10^6 km^2
KKM3 = 1e12                    # m^3 per 10^3 km^3
QUANTITIES = ("area", "extent", "volume")


def hemisphere_totals(C, H, area, hem) -> dict:
    a = np.clip(C, 0.0, 1.0)
    return {"area": float(np.sum(a[hem] * area[hem])) / MKM2,
            "extent": float(np.sum(area[hem & (C > SC.THRESH)])) / MKM2,
            "volume": float(np.sum(H[hem] * area[hem])) / KKM3}


def field_differences(Cj, Hj, Cf, Hf, area, hem) -> dict:
    """Area-weighted RMS and mean of the C and H differences over the ice domain of ``hem``."""
    dom = hem & ((Cj > 0) | (Cf > 0))
    out = {"ice_domain_area": float(np.sum(area[dom])) / MKM2}
    if not dom.any():
        out.update(rms_C=None, mean_C=None, rms_H=None, mean_H=None)
        return out
    w = area[dom] / np.sum(area[dom])
    dC = (Cj - Cf)[dom]
    dH = (Hj - Hf)[dom]
    out.update(rms_C=float(np.sqrt(np.sum(w * dC * dC))), mean_C=float(np.sum(w * dC)),
               rms_H=float(np.sqrt(np.sum(w * dH * dH))), mean_H=float(np.sum(w * dH)))
    return out


def parity(jax: tuple, fortran: tuple, area, lat, wet) -> dict:
    """``jax``/``fortran`` = ``(C, H, present)`` [12, nod2D] of one year (``score`` readers)."""
    Cj, Hj, pj = jax
    Cf, Hf, pf = fortran
    hems = {"NH": wet & (lat > 0), "SH": wet & (lat < 0)}
    months = [m for m in range(12) if pj[m] and pf[m]]
    res = {"months": [m + 1 for m in months], "monthly": {}, "year": {}}
    for name, hem in hems.items():
        rows = []
        for m in months:
            tj = hemisphere_totals(Cj[m], Hj[m], area, hem)
            tf = hemisphere_totals(Cf[m], Hf[m], area, hem)
            row = {"month": m + 1}
            for q in QUANTITIES:
                row[f"{q}_jax"] = tj[q]
                row[f"{q}_fortran"] = tf[q]
                row[f"{q}_diff"] = tj[q] - tf[q]
            row.update(field_differences(Cj[m], Hj[m], Cf[m], Hf[m], area, hem))
            rows.append(row)
        res["monthly"][name] = rows
        yr = {}
        for q in [f"{q}_diff" for q in QUANTITIES] + ["rms_C", "rms_H", "mean_C", "mean_H"]:
            v = np.array([np.nan if r[q] is None else r[q] for r in rows], dtype=np.float64)
            ok = np.isfinite(v)
            yr[q] = None if not ok.any() else {
                "mean": float(np.mean(v[ok])), "max_abs": float(np.max(np.abs(v[ok]))),
                "rms": float(np.sqrt(np.mean(v[ok] ** 2)))}
        res["year"][name] = yr
    return res


def print_table(res: dict) -> None:
    for name, rows in res["monthly"].items():
        print(f"{name}  month  area J/F/diff [1e6 km2]   extent diff  volume J/F/diff [1e3 km3]"
              f"   rmsC   rmsH[m]")
        for r in rows:
            rc = "   -  " if r["rms_C"] is None else f"{r['rms_C']:.3f}"
            rh = "   -  " if r["rms_H"] is None else f"{r['rms_H']:.3f}"
            print(f"     {r['month']:5d}  {r['area_jax']:6.3f} {r['area_fortran']:6.3f} "
                  f"{r['area_diff']:+7.3f}   {r['extent_diff']:+7.3f}     {r['volume_jax']:6.2f} "
                  f"{r['volume_fortran']:6.2f} {r['volume_diff']:+7.2f}   {rc}  {rh}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jax", required=True, help="directory with fields.npz / fields_<YYYY>.npz")
    ap.add_argument("--fortran", required=True, help="Fortran run directory")
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mesh-dir", type=Path, default=ROOT / "data" / "mesh_core2")
    a = ap.parse_args(argv)

    if a.out.exists():
        print(f"refusing to write into an existing directory: {a.out}", file=sys.stderr)
        return 3
    _, lat, wet, area = SC.mesh_nodes(a.mesh_dir)
    jx = SC.load_model(a.jax, "jax", [a.year], area.size)
    ft = SC.load_model(a.fortran, "fortran", [a.year], area.size)
    if a.year not in jx or a.year not in ft:
        print(f"{a.year}: JAX {'found' if a.year in jx else 'missing'}, Fortran "
              f"{'found' if a.year in ft else 'missing'}", file=sys.stderr)
        return 2
    res = parity(jx[a.year], ft[a.year], area, lat, wet)
    if not res["months"]:
        print(f"{a.year}: no month present in both", file=sys.stderr)
        return 2
    res["provenance"] = {"jax": str(a.jax), "fortran": str(a.fortran), "year": a.year,
                         "mesh_dir": str(a.mesh_dir), "argv": sys.argv if argv is None else argv,
                         "created": datetime.datetime.now().isoformat(timespec="seconds"),
                         "extent_threshold": SC.THRESH,
                         "units": {"area": "1e6 km^2", "extent": "1e6 km^2",
                                   "volume": "1e3 km^3", "rms_H": "m"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(a.out)
    with open(a.out / "parity.json", "x") as f:
        json.dump(res, f, indent=1)
    print_table(res)
    print(f"wrote {a.out}/parity.json ({len(res['months'])} months)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
