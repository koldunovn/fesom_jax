"""Day-by-day agreement of fesom-jax runs with a Fortran FESOM2 run (the converter's validation).

    python scripts/fortran_restart/compare_daily.py --fortran <dir with sst.fesom.YYYY.nc ...> \
        --jax conv=<daily-out dir> --jax cold=<daily-out dir> --mesh-dir data/mesh_core2 \
        --year 1958 --doy0 32 --days 28 --out <md>

For every day and field (sst, sss, ssh, a_ice, m_ice) the area-weighted RMS difference of each JAX run
against the Fortran daily mean over the surface wet nodes, and the hemispheric ice area and volume.
A run started from the converted Fortran restart should sit far closer to Fortran on the first days
than a run that spun up on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = ("sst", "sss", "ssh", "a_ice", "m_ice")


def fortran_day(d: Path, var: str, year: int, k: int, n: int) -> np.ndarray:
    import netCDF4
    with netCDF4.Dataset(d / f"{var}.fesom.{year}.nc") as ds:
        v = ds.variables[var]
        v.set_auto_mask(False)
        a = np.asarray(v[k], np.float64)
    if a.shape[-1] != n:
        a = np.moveaxis(a, [i for i, s in enumerate(a.shape) if s == n][0], -1)
    return a.reshape(n)


def jax_day(d: Path, var: str, year: int, doy: int, n: int) -> np.ndarray | None:
    import zarr
    p = d / f"day_{year:04d}_{doy:03d}"
    if not p.exists():
        return None
    a = np.asarray(zarr.open(str(p), mode="r")[var][...], np.float64)
    return a.reshape(-1)[:n] if a.size >= n else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fortran", required=True)
    ap.add_argument("--jax", action="append", required=True, help="name=<daily-out dir>")
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--doy0", type=int, required=True, help="day of year of the first day")
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    md = Path(args.mesh_dir)
    area = np.load(md / "area.npy")
    area = area[:, 0] if area.ndim == 2 else area
    n = area.size
    lat = np.load(md / "geo_coord_nod2D.npy")
    lat = lat[1] if lat.shape[0] == 2 else lat[:, 1]
    lat = np.degrees(lat) if np.nanmax(np.abs(lat)) < 4 else lat
    runs = dict(s.split("=", 1) for s in args.jax)
    F = Path(args.fortran)
    rows, doc = [], {"days": []}
    for k in range(args.days):
        doy = args.doy0 + k
        day = {"doy": doy, "rms": {}, "ice": {}}
        f = {v: fortran_day(F, v, args.year, k, n) for v in FIELDS}
        wet = np.isfinite(f["sst"]) & (area > 0)
        for name, d in runs.items():
            j = {v: jax_day(Path(d), v, args.year, doy, n) for v in FIELDS}
            if any(x is None for x in j.values()):
                continue
            day["rms"][name] = {v: float(np.sqrt(np.sum(area[wet] * (j[v][wet] - f[v][wet]) ** 2)
                                                 / np.sum(area[wet]))) for v in FIELDS}
            for hemi, sel in (("NH", lat > 0), ("SH", lat < 0)):
                day["ice"].setdefault(hemi, {})[name] = {
                    "area_Mkm2": float(np.sum((j["a_ice"] * area)[sel & wet]) / 1e12),
                    "vol_kkm3": float(np.sum((j["m_ice"] * area)[sel & wet]) / 1e12)}
        for hemi, sel in (("NH", lat > 0), ("SH", lat < 0)):
            day["ice"].setdefault(hemi, {})["fortran"] = {
                "area_Mkm2": float(np.sum((f["a_ice"] * area)[sel & wet]) / 1e12),
                "vol_kkm3": float(np.sum((f["m_ice"] * area)[sel & wet]) / 1e12)}
        doc["days"].append(day)
    names = list(runs)
    L = ["# Day-by-day RMS difference to Fortran (area-weighted, surface wet nodes)\n",
         "| day | " + " | ".join(f"{v} {nm}" for v in FIELDS for nm in names) + " |",
         "|---|" + "---|" * (len(FIELDS) * len(names))]
    for day in doc["days"]:
        cells = [f"{day['rms'].get(nm, {}).get(v, float('nan')):.3g}" for v in FIELDS for nm in names]
        L.append(f"| {day['doy']} | " + " | ".join(cells) + " |")
    L += ["\n# Ice area (10^6 km^2) / volume (10^3 km^3)\n",
          "| day | " + " | ".join(f"{h} {nm}" for h in ("NH", "SH") for nm in ["fortran"] + names) + " |",
          "|---|" + "---|" * (2 * (len(names) + 1))]
    for day in doc["days"]:
        cells = []
        for h in ("NH", "SH"):
            for nm in ["fortran"] + names:
                x = day["ice"].get(h, {}).get(nm)
                cells.append(f"{x['area_Mkm2']:.3f}/{x['vol_kkm3']:.2f}" if x else "—")
        L.append(f"| {day['doy']} | " + " | ".join(cells) + " |")
    Path(args.out).write_text("\n".join(L) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(doc, indent=1))
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
