#!/usr/bin/env python
"""Long-window Task A1 — Foundation-run multi-year DRIFT VERDICT.

Consumes the §0 foundation all-on (zstar+TKE+mEVP+GM) multi-year climate (SLURM job
25740424; outputs on ``$WORK/foundation_{climate,snaps}`` + ``foundation_final.pkl``) and
decides — with NUMERIC, falsifiable bars, not a judgement call — whether the developed
climate is a usable base for the 5-yr spin-up (Task A2) and the 10-yr reference (Task B1).

The verdict gates on the drift *slope* (bounded / leveling), NOT on any field's *level*:
a properly-equilibrating upper ocean still has a finite drift after only 3 yr, and Arctic
multi-year ice ~5-6 m is physical (central-Arctic ice exceeded 10 m in the 1950s-60s). A
runaway — a slope that is NOT flattening, or non-physical T/S/SSH/|vel|/hnode — is what
fails the gate (a bad base invalidates every downstream long-time-mean).

Metrics (monthly-mean netCDF is the drift source — snapshot sampling noise would swamp the
signal; instantaneous snapshots give the hnode>0 / |vel| physicality time series):
  * upper-ocean (0-700 m) VOLUME-weighted basin-mean T and S  -> the OHC-style drift bar
  * global surface (area-weighted) SST / SSS
  * total sea-ice area = Σ a_ice·area_surf, volume = Σ m_ice·area_surf
  * Arctic (lat>60°N) area-weighted m_ice  -> the ice-slope-flattening bar
  * per-snapshot: min wet-layer hnode (>0), max |uv| (bounded), all-finite

Bars (FOUNDATION_MULTIYEAR_OK):
  (a) every snapshot finite; SST/SSS/|SSH|/|vel| physical & bounded; min wet hnode > 0
  (b) upper-ocean 0-700 m T drift DECELERATING (|Δ_yr(y2→y3)| ≤ |Δ_yr(y1→y2)|) — leveling;
      the tail |dT/dt| is reported and flagged if > ~0.1 °C/yr (still adjusting at 3 yr is
      expected — deep equilibration is out of scope — so the gate is deceleration, not the level)
  (c) Arctic m_ice drift flattening (|Δ_yr(y2→y3)| ≤ |Δ_yr(y1→y2)|)

Heavy reads from ``$WORK`` only; writes a single small ``scripts/lw_foundation_check.jsonl``.

Usage:
  JAX_PLATFORMS=cpu python scripts/paper/core2_lw_foundation_check.py            # full (all snapshots)
  JAX_PLATFORMS=cpu python scripts/paper/core2_lw_foundation_check.py --snap-stride 0   # final pkl only (smoke)
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
WORK_DEFAULT = "/work/ab0995/a270088/port_jax"
RESULTS = ROOT / "scripts" / "lw_foundation_check.jsonl"

# physicality envelopes (monthly-mean ranges; the run's own per-step monitor used the
# instantaneous SSH/|vel|/m_ice caps 5/3/20 — these mirror that, plus a freezing-point floor)
SST_LO, SST_HI = -2.2, 40.0
SSS_LO, SSS_HI = 1.0, 45.0
SSH_ABSMAX, VEL_ABSMAX = 5.0, 3.0
UPPER_DEPTH = 700.0          # 0-UPPER_DEPTH m = the "upper ocean" drift band
ARCTIC_LAT = 60.0            # °N for the Arctic m_ice area-mean
DTDT_BAR = 0.1              # °C/yr tail-drift advisory bar


# ----------------------------------------------------------------------------
# The DRIFT-METRIC REDUCER (pure numpy, NaN/mask-safe, area/volume-weighted).
# This is the unit-tested seam (LW_DRIFT_SEAM_OK): a finite, weighted reduction that
# treats non-finite values as masked and guards the empty-weight denominator.
# ----------------------------------------------------------------------------
def weighted_mean(values, weights) -> float:
    """Area/volume-weighted mean of ``values`` with non-negative ``weights``; non-finite
    values (NaN below-bottom / land) and non-positive weights are excluded. Returns a
    **finite** scalar (0.0 if no valid entry) — the project masked-reduction contract."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    num = np.sum(np.where(m, w * v, 0.0))
    den = np.sum(np.where(m, w, 0.0))
    return float(num / den) if den > 0.0 else 0.0


def weighted_total(values, weights) -> float:
    """Σ weights·values over finite entries (a TOTAL, not a mean — e.g. sea-ice area/volume)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(w)
    return float(np.sum(np.where(m, w * v, 0.0)))


def annual_means(monthly: np.ndarray, months_per_year: int = 12) -> np.ndarray:
    """Reduce a monthly series ``[n_months]`` to annual means ``[n_years]`` (drops a ragged
    trailing partial year)."""
    x = np.asarray(monthly, dtype=np.float64)
    n_yr = x.size // months_per_year
    return x[: n_yr * months_per_year].reshape(n_yr, months_per_year).mean(axis=1)


def lin_slope_per_year(monthly: np.ndarray, months_per_year: int = 12) -> float:
    """Least-squares linear slope of a monthly series in units / YEAR (finite; 0 if <2 pts)."""
    y = np.asarray(monthly, dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() < 2:
        return 0.0
    t = np.arange(y.size, dtype=np.float64)[m] / months_per_year   # time in years
    yy = y[m]
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, yy, rcond=None)[0]
    return float(slope)


def decel(series_monthly: np.ndarray) -> dict:
    """Year-over-year drift of an annual-mean series + the deceleration verdict (is the
    |year2→year3| step ≤ the |year1→year2| step ⇒ leveling)."""
    ann = annual_means(series_monthly)
    out = {"annual": [round(float(a), 5) for a in ann]}
    if ann.size >= 3:
        d_early = float(ann[1] - ann[0])      # y1->y2 drift (units/yr)
        d_late = float(ann[2] - ann[1])       # y2->y3 drift (units/yr)
        out.update(drift_y1y2=round(d_early, 5), drift_y2y3=round(d_late, 5),
                   decelerating=bool(abs(d_late) <= abs(d_early) + 1e-12),
                   lin_slope_per_year=round(lin_slope_per_year(series_monthly), 5))
    else:
        out.update(drift_y1y2=None, drift_y2y3=None, decelerating=None,
                   lin_slope_per_year=round(lin_slope_per_year(series_monthly), 5))
    return out


# ----------------------------------------------------------------------------
# Mesh-derived static weights (CPU; numpy only)
# ----------------------------------------------------------------------------
def load_weights(mesh_dir: Path):
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(mesh_dir)
    area_surf = np.asarray(mesh.area[:, 0], dtype=np.float64)         # [nod2D] surface CV area
    lat_deg = np.degrees(np.asarray(mesh.geo_coord_nod2D)[:, 1])      # [nod2D]
    layer_mask = np.asarray(mesh.node_layer_mask).astype(bool)        # [nod2D, nl]
    Z = np.asarray(mesh.Z, dtype=np.float64)                          # [nl-1] mid-depths ≤ 0
    zbar = np.asarray(mesh.zbar, dtype=np.float64)                    # [nl] interfaces ≤ 0
    nz1 = Z.size
    dz = (zbar[:nz1] - zbar[1 : nz1 + 1])                             # [nz1] layer thickness (>0)
    areasvol = np.asarray(mesh.areasvol[:, :nz1], dtype=np.float64)   # [nod2D, nz1] mid CV area
    in_upper = (np.abs(Z) <= UPPER_DEPTH)                             # [nz1] 0-700 m levels
    # volume weight (nz1, nod2D) restricted to the upper band & wet layers
    vol_w = (dz * in_upper)[:, None] * (areasvol * layer_mask[:, :nz1]).T   # [nz1, nod2D]
    arctic_w = area_surf * (lat_deg >= ARCTIC_LAT)                    # [nod2D]
    return dict(mesh=mesh, area_surf=area_surf, surf_mask=layer_mask[:, 0],
                layer_mask=layer_mask, vol_w=vol_w, arctic_w=arctic_w,
                n_upper_levels=int(in_upper.sum()), nz1=nz1)


# ----------------------------------------------------------------------------
# Read the monthly-mean netCDF climate -> drift time series
# ----------------------------------------------------------------------------
def read_climate(climate_dir: Path, W, start_year: int, years: int):
    from netCDF4 import Dataset
    series = {k: [] for k in
              ("sst", "sss", "ssh_absmax", "t0700", "s0700", "ice_area", "ice_vol", "arctic_mice")}
    months = []
    area_surf, surf_mask, vol_w, arctic_w = (W["area_surf"], W["surf_mask"], W["vol_w"], W["arctic_w"])
    surf_w = area_surf * surf_mask
    for yi in range(years):
        yr = start_year + yi
        nc = {v: Dataset(climate_dir / f"{v}.fesom.{yr}.monthly.nc")
              for v in ("sst", "sss", "ssh", "a_ice", "m_ice", "temp", "salt")}
        n_mon = nc["sst"].dimensions["time"].size
        for mi in range(n_mon):
            sst = np.asarray(nc["sst"].variables["sst"][mi])
            sss = np.asarray(nc["sss"].variables["sss"][mi])
            ssh = np.asarray(nc["ssh"].variables["ssh"][mi])
            a_ice = np.asarray(nc["a_ice"].variables["a_ice"][mi])
            m_ice = np.asarray(nc["m_ice"].variables["m_ice"][mi])
            temp = np.asarray(nc["temp"].variables["temp"][mi])      # [nz1, nod2]
            salt = np.asarray(nc["salt"].variables["salt"][mi])
            series["sst"].append(weighted_mean(sst, surf_w))
            series["sss"].append(weighted_mean(sss, surf_w))
            series["ssh_absmax"].append(float(np.nanmax(np.abs(np.where(surf_mask, ssh, np.nan)))))
            series["t0700"].append(weighted_mean(temp, vol_w))
            series["s0700"].append(weighted_mean(salt, vol_w))
            series["ice_area"].append(weighted_total(a_ice, area_surf))
            series["ice_vol"].append(weighted_total(m_ice, area_surf))
            series["arctic_mice"].append(weighted_mean(m_ice, arctic_w))
            months.append(f"{yr}-{mi+1:02d}")
        for ds in nc.values():
            ds.close()
    return {k: np.asarray(v) for k, v in series.items()}, months


# ----------------------------------------------------------------------------
# Read instantaneous snapshots -> hnode>0 / |vel| / finite physicality time series
# ----------------------------------------------------------------------------
def read_snapshots(snap_dir: Path, final_pkl: Path, W, stride: int):
    layer_mask = W["layer_mask"]
    surf = W["surf_mask"]
    files = []
    if stride and stride > 0:
        files = sorted(glob.glob(str(snap_dir / "snap_step*.pkl")))[:: stride]
    if final_pkl.exists():
        files = files + [str(final_pkl)]
    phys = []
    for f in files:
        with open(f, "rb") as fh:
            st = pickle.load(fh)
        hn = np.asarray(st.hnode)
        uv = np.asarray(st.uv)
        T0 = np.asarray(st.T)[:, 0]
        S0 = np.asarray(st.S)[:, 0]
        finite = bool(np.isfinite(hn).all() and np.isfinite(uv).all()
                      and np.isfinite(np.asarray(st.T)).all()
                      and np.isfinite(np.asarray(st.S)).all()
                      and np.isfinite(np.asarray(st.eta_n)).all())
        wet_hnode_min = float(np.min(hn[layer_mask]))
        phys.append(dict(file=Path(f).name, finite=finite,
                         min_wet_hnode=wet_hnode_min, max_abs_uv=float(np.max(np.abs(uv))),
                         sst_min=float(np.min(np.where(surf, T0, np.inf))),
                         sst_max=float(np.max(np.where(surf, T0, -np.inf))),
                         sss_min=float(np.min(np.where(surf, S0, np.inf))),
                         sss_max=float(np.max(np.where(surf, S0, -np.inf)))))
        del st, hn, uv, T0, S0
    return phys


# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
def make_verdict(series, phys):
    t_decel = decel(series["t0700"])
    s_decel = decel(series["s0700"])
    ice_decel = decel(series["arctic_mice"])

    all_finite = all(p["finite"] for p in phys) if phys else False
    min_hnode = min((p["min_wet_hnode"] for p in phys), default=float("nan"))
    max_vel = max((p["max_abs_uv"] for p in phys), default=float("nan"))
    ssh_absmax = float(series["ssh_absmax"].max())
    # LOCAL (instantaneous, per-snapshot) surface extrema — a genuine no-supercooling /
    # no-salinity-blowup check, stronger than the global monthly-mean range
    sst_lo = min((p["sst_min"] for p in phys), default=float("nan"))
    sst_hi = max((p["sst_max"] for p in phys), default=float("nan"))
    sss_lo = min((p["sss_min"] for p in phys), default=float("nan"))
    sss_hi = max((p["sss_max"] for p in phys), default=float("nan"))

    physical = bool(all_finite and (min_hnode > 0.0) and (max_vel < VEL_ABSMAX)
                    and (ssh_absmax < SSH_ABSMAX)
                    and (SST_LO <= sst_lo) and (sst_hi <= SST_HI)
                    and (SSS_LO <= sss_lo) and (sss_hi <= SSS_HI))

    t_leveling = bool(t_decel.get("decelerating"))
    ice_flattening = bool(ice_decel.get("decelerating"))
    tail_dtdt = t_decel.get("drift_y2y3")
    tail_within_bar = (tail_dtdt is not None) and (abs(tail_dtdt) < DTDT_BAR)

    ok = bool(physical and t_leveling and ice_flattening)

    return dict(
        FOUNDATION_MULTIYEAR_OK=ok,
        physical=physical,
        t0700_leveling=t_leveling, arctic_mice_flattening=ice_flattening,
        tail_dTdt_within_bar=bool(tail_within_bar), tail_dTdt=tail_dtdt, dTdt_bar=DTDT_BAR,
        checks=dict(all_finite=all_finite, min_wet_hnode=min_hnode, max_abs_uv=max_vel,
                    ssh_absmax=ssh_absmax, sst_range=[sst_lo, sst_hi], sss_range=[sss_lo, sss_hi]),
        t0700=t_decel, s0700=s_decel, arctic_mice=ice_decel,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=str, default=WORK_DEFAULT)
    ap.add_argument("--start-year", type=int, default=1958)
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--snap-stride", type=int, default=1,
                    help="read every Nth foundation snapshot for the hnode/|vel| series "
                         "(0 ⇒ only foundation_final.pkl — a quick smoke)")
    ap.add_argument("--results", type=str, default=str(RESULTS))
    args = ap.parse_args()

    work = Path(args.work)
    climate_dir = work / "foundation_climate"
    snap_dir = work / "foundation_snaps"
    final_pkl = work / "foundation_final.pkl"

    t0 = time.time()
    print(f"[A1] mesh + static weights from {MESH_DIR}", flush=True)
    W = load_weights(MESH_DIR)
    print(f"[A1] upper-ocean band 0-{UPPER_DEPTH:.0f} m = {W['n_upper_levels']} levels; "
          f"Arctic = lat≥{ARCTIC_LAT:.0f}°N", flush=True)

    print(f"[A1] reading monthly climate {climate_dir} ({args.years} yr)", flush=True)
    series, months = read_climate(climate_dir, W, args.start_year, args.years)

    print(f"[A1] reading snapshots {snap_dir} (stride={args.snap_stride}) + {final_pkl.name}", flush=True)
    phys = read_snapshots(snap_dir, final_pkl, W, args.snap_stride)

    verdict = make_verdict(series, phys)

    # ----- report -----
    print("\n=== A1 FOUNDATION DRIFT VERDICT (3-yr all-on climate) ===", flush=True)
    print(f"{'month':>8} {'SST':>7} {'SSS':>7} {'T0700':>8} {'S0700':>8} "
          f"{'iceA(e12)':>10} {'iceV(e12)':>10} {'arcMice':>8}", flush=True)
    for i, mo in enumerate(months):
        print(f"{mo:>8} {series['sst'][i]:7.3f} {series['sss'][i]:7.3f} "
              f"{series['t0700'][i]:8.4f} {series['s0700'][i]:8.4f} "
              f"{series['ice_area'][i]/1e12:10.3f} {series['ice_vol'][i]/1e12:10.4f} "
              f"{series['arctic_mice'][i]:8.4f}", flush=True)
    print("\n-- drift (annual means, year-over-year) --", flush=True)
    for name, key in (("upper-ocean T (0-700m)", "t0700"), ("upper-ocean S (0-700m)", "s0700"),
                      ("Arctic m_ice", "arctic_mice")):
        d = verdict[key]
        print(f"  {name:24s} annual={d['annual']}  Δy1y2={d['drift_y1y2']} "
              f"Δy2y3={d['drift_y2y3']}  decel={d['decelerating']}  "
              f"lin={d['lin_slope_per_year']}/yr", flush=True)
    c = verdict["checks"]
    print(f"\n-- physicality -- finite={c['all_finite']} min_wet_hnode={c['min_wet_hnode']:.4g} "
          f"max|vel|={c['max_abs_uv']:.3f} |SSH|max={c['ssh_absmax']:.3f} "
          f"SST{c['sst_range']} SSS{c['sss_range']}", flush=True)
    print(f"\n  physical={verdict['physical']}  t0700_leveling={verdict['t0700_leveling']}  "
          f"arctic_mice_flattening={verdict['arctic_mice_flattening']}  "
          f"tail|dT/dt|={verdict['tail_dTdt']} (<{DTDT_BAR}? {verdict['tail_dTdt_within_bar']})",
          flush=True)

    rec = dict(task="A1", t_sec=round(time.time() - t0, 1), years=args.years,
               start_year=args.start_year, snap_stride=args.snap_stride,
               n_snapshots=len(phys), months=months, series={k: [round(float(x), 6) for x in v]
                                                              for k, v in series.items()},
               phys=phys, verdict=verdict)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n[A1] wrote {args.results}", flush=True)

    if verdict["FOUNDATION_MULTIYEAR_OK"]:
        print("\nFOUNDATION_MULTIYEAR_OK", flush=True)
        return 0
    print("\nFOUNDATION_MULTIYEAR_NOT_OK — drift not leveling or unphysical; "
          "document + decide mitigation BEFORE the 5-yr spin-up", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
