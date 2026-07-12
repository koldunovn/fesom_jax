#!/usr/bin/env python
"""Spatial JAX-vs-C-port-KPP climate-bias map — diagnose where the ~0.15 °C SST bias lives.

Computes the annual-mean (mean of the 12 monthly means, ice fields nan->0 first like
m32_climate_compare) JAX − C-port-KPP difference per node for sst/sss/ssh/a_ice/m_ice + a
3-D temp difference (vertical structure ⇒ mixing-vs-forcing tell), writes a CF-1.8 NetCDF
ushow reads (`ushow bias_map_<yr>.nc`), and prints global + per-latitude-band stats.

Usage:  python scripts/debug/kpp_bias_map.py --year 1958
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
CREF = "/work/ab0995/a270088/port/kpp_5yr_fix"
ICE = {"a_ice", "m_ice"}


def annual(path, var):
    d = Dataset(path); x = np.asarray(d[var][:], dtype=np.float64); d.close()
    if var in ICE:
        x = np.nan_to_num(x, nan=0.0)
    return np.nanmean(x, axis=0)            # (nod2,) or (nz_1, nod2)


def stats(diff, valid):
    d = diff[valid]
    return d.mean(), np.sqrt(np.mean(d * d)), np.abs(d).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--jax-dir", default=str(ROOT / "data" / "kpp_climate_2yr"))
    ap.add_argument("--cref", default=CREF)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    yr = args.year
    out = args.out or str(Path(args.jax_dir) / f"bias_map_{yr}.nc")

    # coords from the JAX file (degrees, embedded)
    d = Dataset(f"{args.jax_dir}/sst.fesom.{yr}.monthly.nc")
    lon = np.asarray(d["lon"][:]); lat = np.asarray(d["lat"][:]); d.close()
    dz = Dataset(f"{args.jax_dir}/temp.fesom.{yr}.monthly.nc"); z = np.asarray(dz["z"][:]); dz.close()

    fields2d = ["sst", "sss", "ssh", "a_ice", "m_ice"]
    res = {}
    print(f"\n=== JAX − C-port-KPP annual-mean bias, {yr} ===")
    print(f"{'field':6s} | {'bias':>12s} {'RMS':>12s} {'|d|max':>10s}  (wet nodes)")
    for v in fields2d:
        j = annual(f"{args.jax_dir}/{v}.fesom.{yr}.monthly.nc", v)
        c = annual(f"{args.cref}/{v}.fesom.{yr}.monthly.nc", v)
        valid = np.isfinite(j) & np.isfinite(c)
        diff = np.where(valid, j - c, np.nan)
        res[v] = (j, c, diff, valid)
        b, r, m = stats(diff, valid)
        print(f"{v:6s} | {b:+12.4e} {r:12.4e} {m:10.3e}")

    # latitude-band breakdown of the SST bias (localize: polar/ice vs tropics vs midlat)
    js, cs, ds, vs = res["sst"]
    print(f"\n  SST bias by latitude band:")
    print(f"  {'band':>14s} | {'bias':>10s} {'RMS':>10s}  {'nodes':>7s}")
    for lo, hi in [(-90, -60), (-60, -45), (-45, -15), (-15, 15), (15, 45), (45, 66), (66, 90)]:
        m = vs & (lat >= lo) & (lat < hi)
        if m.sum() == 0:
            continue
        b, r, _ = stats(ds, m)
        print(f"  [{lo:+3d},{hi:+3d})°  | {b:+10.3e} {r:10.3e}  {int(m.sum()):7d}")

    # top-bias hotspots (where is JAX most different?)
    order = np.argsort(-np.abs(np.nan_to_num(ds)))
    print(f"\n  Top-10 |SST bias| hotspots (lon, lat, JAX, C, Δ):")
    for i in order[:10]:
        print(f"    ({lon[i]:+7.2f},{lat[i]:+6.2f})  JAX={js[i]:+6.2f}  C={cs[i]:+6.2f}  Δ={ds[i]:+6.2f}")

    # 3-D temp difference (vertical structure)
    jt = annual(f"{args.jax_dir}/temp.fesom.{yr}.monthly.nc", "temp")   # (nz_1, nod2)
    ct = annual(f"{args.cref}/temp.fesom.{yr}.monthly.nc", "temp")
    tdiff = jt - ct
    vt = np.isfinite(jt) & np.isfinite(ct)
    print(f"\n  temp Δ by depth (global mean over wet nodes):")
    print(f"  {'z[m]':>7s} | {'bias':>10s} {'RMS':>10s}")
    for k in [0, 2, 5, 9, 14, 19, 24, 29]:
        if k >= tdiff.shape[0]:
            break
        mk = vt[k]
        if mk.sum() == 0:
            continue
        dk = tdiff[k][mk]
        print(f"  {z[k]:7.1f} | {dk.mean():+10.3e} {np.sqrt(np.mean(dk*dk)):10.3e}")

    # write the ushow-readable bias map
    ds_out = Dataset(out, "w", format="NETCDF4_CLASSIC")
    ds_out.createDimension("time", 1); ds_out.createDimension("nod2", len(lon))
    ds_out.createDimension("nz_1", tdiff.shape[0])
    tv = ds_out.createVariable("time", "f8", ("time",)); tv.units = f"days since {yr}-07-01"; tv[0] = 0.0
    lo = ds_out.createVariable("lon", "f8", ("nod2",)); lo.units = "degrees_east"; lo[:] = lon
    la = ds_out.createVariable("lat", "f8", ("nod2",)); la.units = "degrees_north"; la[:] = lat
    zz = ds_out.createVariable("z", "f8", ("nz_1",)); zz.units = "m"; zz.positive = "down"; zz[:] = z
    for v in fields2d:
        j, c, diff, valid = res[v]
        for nm, arr in [(f"{v}_jax", j), (f"{v}_cref", c), (f"{v}_diff", diff)]:
            vh = ds_out.createVariable(nm, "f8", ("time", "nod2"), fill_value=np.nan)
            vh.coordinates = "lon lat"; vh[0, :] = np.where(np.isfinite(arr), arr, np.nan)
    vh = ds_out.createVariable("temp_diff", "f8", ("time", "nz_1", "nod2"), fill_value=np.nan)
    vh.coordinates = "lon lat"; vh.long_name = "JAX - C-port temperature"
    vh[0, :, :] = np.where(vt, tdiff, np.nan)
    ds_out.Conventions = "CF-1.8"; ds_out.title = f"JAX - C-port-KPP {yr} annual-mean bias"
    ds_out.close()
    print(f"\n  → wrote {out}  (ushow it: sst_diff / sss_diff / temp_diff ...)")


if __name__ == "__main__":
    raise SystemExit(main())
