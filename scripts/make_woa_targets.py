"""Build the WOA obs targets for Task D2a — per-month MLD (averaged → annual) + SST.

de Boyer Montégut is IFREMER-blocked, so we derive the MLD climatology from **monthly** WOA18
T/S — the right way: compute the density-threshold MLD (0.03 kg/m³ from 10 m, the dBM DR003
criterion) **per month** with the model's own diagnostic (:func:`fesom_jax.obs_compare.mld_density_
threshold`), then average the 12 monthly MLD fields. This avoids the Jensen bias of computing MLD
from the annual-mean profile (which collapses to ~20 m everywhere because the deep winter
convection averages out — verified 2026-06-14). SST comes from the annual + monthly surface T.

Saves ``scripts/woa_targets.npz``:
  mld_monthly[12, lat, lon], mld_annual[lat, lon]   (positive metres; annual = mean of valid months)
  sst_monthly[12, lat, lon], sst_annual[lat, lon]   (°C, surface)
  lat, lon, mld_valid_monthly[12], mld_valid_annual, sst_valid

Same MLD diagnostic on both sides (model + obs) ⇒ a consistent comparison by construction. Runs on
the login node (CPU). Usage:  python scripts/make_woa_targets.py
"""

from __future__ import annotations

from pathlib import Path

import fesom_jax  # noqa: F401  (x64)
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from fesom_jax import obs_compare

OBS = Path("/work/ab0995/a270088/port_jax/obs/woa18")
OUT = Path(__file__).resolve().parent / "woa_targets.npz"
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def load(path, var):
    with Dataset(path) as d:
        a = np.ma.filled(d[var][0], np.nan)                 # [depth, lat, lon]
        depth = np.asarray(d["depth"][:], np.float64)
        lat = np.asarray(d["lat"][:], np.float64)
        lon = np.asarray(d["lon"][:], np.float64)
    return a, depth, lat, lon


def column_mld(T, S, depth):
    """MLD [lat, lon] from a [depth, lat, lon] T/S grid via the model's exact diagnostic."""
    nd, nlat, nlon = T.shape
    ncols = nlat * nlon
    Tc = np.transpose(T, (1, 2, 0)).reshape(ncols, nd)
    Sc = np.transpose(S, (1, 2, 0)).reshape(ncols, nd)
    vc = np.isfinite(Tc) & np.isfinite(Sc)
    Z = np.broadcast_to(-np.abs(depth)[None, :], (ncols, nd))
    mld, valid = obs_compare.mld_density_threshold(
        jnp.asarray(np.where(vc, Tc, 0.0)), jnp.asarray(np.where(vc, Sc, 35.0)),
        jnp.asarray(Z.copy()), jnp.asarray(vc))
    return np.asarray(mld).reshape(nlat, nlon), np.asarray(valid).reshape(nlat, nlon)


def main():
    # annual file (for the annual SST + lat/lon reference)
    Ta, _, lat, lon = load(OBS / "woa18_decav_t00_01.nc", "t_an")
    sst_annual = Ta[0]                                       # surface level, °C
    sst_valid = np.isfinite(sst_annual)
    nlat, nlon = lat.size, lon.size
    print(f"[woa] grid {nlat}x{nlon}; annual SST mean "
          f"{np.nanmean(np.where(sst_valid, sst_annual, np.nan)):.2f} °C", flush=True)

    mlds, mvalids, ssts = [], [], []
    for m in MONTHS:
        tnc, snc = OBS / f"woa18_decav_t{m}_01.nc", OBS / f"woa18_decav_s{m}_01.nc"
        if not (tnc.exists() and snc.exists()):
            print(f"  ⚠️ month {m} missing ({tnc.name}/{snc.name}) — skipping", flush=True)
            continue
        T, depth, _, _ = load(tnc, "t_an")
        S, _, _, _ = load(snc, "s_an")
        mld, valid = column_mld(T, S, depth)
        mlds.append(mld); mvalids.append(valid); ssts.append(T[0])
        w = np.cos(np.radians(lat))[:, None] * valid
        print(f"  month {m}: depth to {depth[-1]:.0f} m  valid {int(valid.sum())}  "
              f"area-mean MLD {float(np.sum(w*mld)/np.sum(w)):.1f} m", flush=True)

    if len(mlds) < 12:
        print(f"  ⚠️ only {len(mlds)}/12 months present — annual mean uses available months", flush=True)
    mld_monthly = np.stack(mlds)                              # [nm, lat, lon]
    mvalid_monthly = np.stack(mvalids)
    sst_monthly = np.stack(ssts)
    # annual = mean over months valid at each cell (the dBM-style per-month-then-average)
    cnt = mvalid_monthly.sum(0)
    mld_annual = np.where(cnt > 0, np.nansum(np.where(mvalid_monthly, mld_monthly, 0.0), 0)
                          / np.maximum(cnt, 1), 0.0)
    mld_valid_annual = cnt >= max(1, len(mlds) // 2)          # ≥ half the months valid

    w = np.cos(np.radians(lat))[:, None] * mld_valid_annual
    print(f"\n[mld] annual (per-month avg): valid {int(mld_valid_annual.sum())}  "
          f"area-mean {float(np.sum(w*mld_annual)/np.sum(w)):.1f} m  "
          f"max {mld_annual[mld_valid_annual].max():.0f} m", flush=True)
    for name, la, lo in [("Labrador", 57, -53), ("Weddell", -65, -30),
                         ("N.Atlantic", 50, -30), ("Equator", 0, -140)]:
        i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon % 360 - lo % 360)))
        m = mld_annual[i, j] if mld_valid_annual[i, j] else np.nan
        print(f"      {name:11s} ({la:+d},{lo:+d}): annual MLD {m:.0f} m", flush=True)

    np.savez(OUT, mld_monthly=mld_monthly, mld_annual=mld_annual,
             sst_monthly=sst_monthly, sst_annual=sst_annual, lat=lat, lon=lon,
             mld_valid_monthly=mvalid_monthly, mld_valid_annual=mld_valid_annual,
             sst_valid=sst_valid, dsigma=0.03, ref_depth=10.0, months=np.array(MONTHS))
    print(f"[woa] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
