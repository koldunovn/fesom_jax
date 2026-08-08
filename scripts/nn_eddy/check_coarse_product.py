#!/usr/bin/env python
"""Check a coarse-grained product file before anything trains on it.

This exists because of what happened on 2026-08-08: the month-01 product completed, reported
sensible-looking surface statistics, and carried FESOM's ``9.96921e+36`` fill value in 12.5% of
its column from level 3 down. It was found by reading the file by hand. A check that lives in a
script gets re-run; one that lives in somebody's terminal history does not.

Every assertion here is one the contaminated file would have failed:

* **Physical bounds.** Not "finite" — finite is satisfied by 7.6e+36. Temperature in [-3, 40] °C,
  salinity in [0, 46], |velocity| < 5 m/s, |eddy flux| < 1e3. Wide enough that real data never
  trips them, narrow enough that a fill value cannot pass.
* **NO-DATA cells are exactly zero and are not silently plausible.** ``cell_area == 0`` is the
  marker; a value there must be 0.0 so it cannot be mistaken for a measurement.
* **The no-data fraction is stable across records.** Bathymetry does not change in a month, so a
  record whose dry fraction differs from its neighbours means a mask was built per record from
  something that varies — which is exactly how a per-level mask goes wrong.
* **The time axis matches the source snapshots**, read from the teacher file rather than
  recomputed, because records sit one time step short of the round mark.
* **The eddy fluxes are not all zero.** A pipeline whose covariance cancelled everywhere would
  produce a clean, conservative, entirely useless file.

Usage:  python scripts/nn_eddy/check_coarse_product.py FILE [--teacher DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import netCDF4
import numpy as np

TEACHER = pathlib.Path("/work/ba1583/a270088/nn_eddy/ng5_teacher")

# (variable, low, high). Deliberately generous — the point is to exclude fill values and
# arithmetic accidents, not to police oceanography.
BOUNDS = {
    "temp": (-3.0, 40.0), "salt": (0.0, 46.0),
    "u": (-5.0, 5.0), "v": (-5.0, 5.0),
    "uT_eddy": (-1e3, 1e3), "vT_eddy": (-1e3, 1e3),
    "uS_eddy": (-1e3, 1e3), "vS_eddy": (-1e3, 1e3),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=pathlib.Path)
    ap.add_argument("--teacher", type=pathlib.Path, default=TEACHER)
    ap.add_argument("--month", default=None, help="default: read source_month from the file")
    args = ap.parse_args()

    problems = []
    with netCDF4.Dataset(args.file) as ds:
        month = args.month or getattr(ds, "source_month", None)
        area = np.asarray(ds.variables["cell_area"][:])
        times = np.asarray(ds.variables["time"][:])
        n_rec = times.size
        have = area > 0
        print(f"{args.file.name}: {n_rec} records, "
              f"{area.shape[1]} coarse nodes x {area.shape[2]} levels, month {month}")
        print(f"cells with data: {have.sum()} of {have.size} ({100 * have.mean():.1f}%)\n")

        print(f"{'field':10s} {'min|data':>12s} {'max|data':>12s} {'max|abs|':>11s} "
              f"{'nonzero where dry':>18s}")
        for name, (lo, hi) in BOUNDS.items():
            if name not in ds.variables:
                problems.append(f"{name}: missing from the file")
                continue
            v = np.asarray(ds.variables[name][:])
            d = v[have]
            n_dry_nonzero = int((v[~have] != 0.0).sum())
            print(f"{name:10s} {d.min():12.4g} {d.max():12.4g} {np.abs(d).max():11.4g} "
                  f"{n_dry_nonzero:18d}")
            if not np.isfinite(d).all():
                problems.append(f"{name}: {int((~np.isfinite(d)).sum())} non-finite values "
                                f"where cell_area > 0")
            n_out = int(((d < lo) | (d > hi)).sum())
            if n_out:
                problems.append(
                    f"{name}: {n_out} values outside [{lo:g}, {hi:g}] where cell_area > 0 "
                    f"(max |{np.abs(d).max():.4g}|) — the classic cause is FESOM's "
                    f"9.96921e+36 fill leaking through an unmasked element average")
            if n_dry_nonzero:
                problems.append(f"{name}: {n_dry_nonzero} nonzero values where cell_area == 0; "
                                f"no-data must read exactly 0.0")

        # --- the dry fraction must not vary between records ------------------------------
        frac = [float((area[i] == 0).mean()) for i in range(n_rec)]
        print(f"\ndry fraction per record: {[round(f, 6) for f in frac]}")
        if max(frac) - min(frac) > 1e-9:
            problems.append(f"the dry fraction varies across records ({min(frac):.6f} .. "
                            f"{max(frac):.6f}); bathymetry does not change within a month, so a "
                            f"mask is being built from something that does")

        # --- the fluxes must not be uniformly zero ---------------------------------------
        for name in ("uT_eddy", "vT_eddy", "uS_eddy", "vS_eddy"):
            if name in ds.variables:
                # float64: a contaminated file holds ~7e+36, and squaring that overflows
                # float32 into a warning and an inf rms, which obscures the real complaint
                v = np.asarray(ds.variables[name][:], dtype=np.float64)[have]
                if not np.any(v != 0.0):
                    problems.append(f"{name} is identically zero everywhere — a conservative, "
                                    f"well-formed, useless file")
                else:
                    print(f"{name}: rms {np.sqrt((v ** 2).mean()):.4g}, "
                          f"{100 * np.mean(v != 0.0):.1f}% nonzero")

    # --- the time axis, against the source ----------------------------------------------
    if month:
        src = args.teacher / f"temp_snap.fesom.{month}.nc"
        if src.exists():
            with netCDF4.Dataset(src) as ds:
                t_src = np.asarray(ds.variables["time"][:])
            if t_src.size != times.size or not np.allclose(t_src, times, atol=1e-6):
                problems.append(f"time axis differs from {src.name}: {times.tolist()} vs "
                                f"{t_src.tolist()}")
            else:
                print(f"\ntime axis matches {src.name} ({times.size} records, "
                      f"first {times[0]:.0f} s)")
        else:
            print(f"\n(source {src.name} not found; time axis not cross-checked)")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("coarse product OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
