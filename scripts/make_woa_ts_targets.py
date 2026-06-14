"""Build the WOA upper-ocean/thermocline T/S obs target for Task D2b (GM→T/S via EKI).

The §2 slow-target half calibrates the GM thickness-diffusivity ``k_gm`` so the model's
**upper-ocean / thermocline stratification** matches WOA. The EKI observable is a small set of
**basin-mean T/S profiles** on a handful of standard depth levels (see
``scripts/core2_paper_calib_gm_eki.py``); this script just stages the raw WOA18 **annual** T/S
on those levels (annual = the right target for the slow, GM-controlled equilibrium stratification,
unlike the per-month MLD of D2a). The basin reduction itself lives in the driver so the SAME
reduction is applied to model and obs.

Saves ``scripts/woa_ts_targets.npz``:
  T[n_depth, lat, lon], S[n_depth, lat, lon]   (°C, psu; NaN→0, validity in the mask)
  valid[n_depth, lat, lon]                      (bool: WOA has data at this cell+level)
  lat[n_lat], lon[n_lon], depth[n_depth]        (depth = positive metres, the chosen levels)

Runs on the login node (CPU, just netCDF reads). Usage:  python scripts/make_woa_ts_targets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

OBS = Path("/work/ab0995/a270088/port_jax/obs/woa18")
OUT = Path(__file__).resolve().parent / "woa_ts_targets.npz"
# upper-ocean / thermocline standard levels (m) — where GM redistributes heat/strat and the
# few-year adjustment is meaningful. All map EXACTLY onto WOA18 standard depths.
WANT_DEPTHS = [0.0, 100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0]


def load(path, var):
    with Dataset(path) as d:
        a = np.ma.filled(d[var][0], np.nan)                  # [depth, lat, lon]
        depth = np.asarray(d["depth"][:], np.float64)
        lat = np.asarray(d["lat"][:], np.float64)
        lon = np.asarray(d["lon"][:], np.float64)
    return a, depth, lat, lon


def main():
    Tfull, depth, lat, lon = load(OBS / "woa18_decav_t00_01.nc", "t_an")   # annual
    Sfull, _, _, _ = load(OBS / "woa18_decav_s00_01.nc", "s_an")
    idx = [int(np.argmin(np.abs(depth - w))) for w in WANT_DEPTHS]
    sel_depth = depth[idx]
    print(f"[woa-ts] grid {lat.size}x{lon.size}; chosen levels (m): "
          f"{sel_depth.astype(int).tolist()}", flush=True)

    T = Tfull[idx]                                            # [n_depth, lat, lon]
    S = Sfull[idx]
    valid = np.isfinite(T) & np.isfinite(S)
    # AD masked-NaN rule: zero the NaN land cells (validity carried separately); even at
    # weight 0, a stray 0·NaN would poison a misfit.
    T = np.where(valid, T, 0.0)
    S = np.where(valid, S, 0.0)

    w_area = np.cos(np.radians(lat))[:, None]                # [lat,1] for area-mean reports
    print("[woa-ts] per-level global area-mean (valid cells):", flush=True)
    for k, dlev in enumerate(sel_depth):
        v = valid[k]
        wT = w_area * v
        mt = float(np.sum(wT * T[k]) / np.sum(wT)) if np.sum(wT) > 0 else np.nan
        ms = float(np.sum(wT * S[k]) / np.sum(wT)) if np.sum(wT) > 0 else np.nan
        print(f"    {int(dlev):5d} m: valid {int(v.sum()):6d}  T {mt:6.2f} °C  S {ms:6.2f} psu",
              flush=True)

    np.savez(OUT, T=T, S=S, valid=valid, lat=lat, lon=lon, depth=sel_depth)
    print(f"[woa-ts] wrote {OUT}  (T/S {T.shape}, levels {sel_depth.astype(int).tolist()})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
