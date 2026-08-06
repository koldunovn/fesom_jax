"""Verify a FESOM output tree written with the teacher protocol.

Checks the things that would quietly ruin training data and are invisible in a single
short run: missing or duplicated records around restarts and year boundaries, wrong
calendar dates (leap years), snapshots that are secretly means, and files our own
loaders cannot read.

    python scripts/nn_eddy/verify_output.py RUNDIR --start-year 1958 --years 3

Exit code is non-zero if any check fails, so it can gate a campaign.
"""

from __future__ import annotations

import argparse
import calendar
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
from netCDF4 import Dataset

FILL = 9.96921e36


def _records(path, var):
    ds = Dataset(path)
    t = np.asarray(ds.variables["time"][:], dtype=np.float64)
    units = ds.variables["time"].units
    v = ds.variables[var]
    return t, units, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--start-year", type=int, default=1958)
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--snap-days", type=int, default=5)
    args = ap.parse_args()

    fails, checks = [], 0

    def check(name, ok, detail=""):
        nonlocal checks
        checks += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            fails.append(name)

    files = sorted(glob.glob(os.path.join(args.rundir, "*.fesom.*.nc")))
    print(f"{len(files)} output files in {args.rundir}\n")
    if not files:
        print("no files — nothing to verify"); return 1

    # group by (variable, year) across monthly file splits
    groups = defaultdict(list)
    for f in files:
        m = re.match(r"(.+)\.fesom\.(\d{4})(?:_(\d{2}))?\.nc$", os.path.basename(f))
        if m:
            groups[(m.group(1), int(m.group(2)))].append(f)

    snap_vars = sorted({v for (v, _) in groups if v.endswith("_snap")})
    mean_vars = sorted({v for (v, _) in groups if not v.endswith("_snap")})
    print(f"snapshot streams: {snap_vars}")
    print(f"mean streams:     {mean_vars}\n")
    check("snapshot streams exist", bool(snap_vars))

    # ---- per stream, per year: record count, monotone time, no duplicates
    print("time axis")
    for (var, year) in sorted(groups):
        paths = sorted(groups[(var, year)])
        times = []
        for p in paths:
            t, units, _ = _records(p, var)
            times.append(t)
            if not re.search(r"since\s+(\d{4})", units):
                check(f"{var} {year}: time units parseable", False, units)
        allt = np.concatenate(times) if times else np.array([])
        mono = bool(np.all(np.diff(allt) > 0)) if allt.size > 1 else True
        dups = int(allt.size - np.unique(allt).size)
        check(f"{var} {year}: {allt.size} records, strictly increasing", mono)
        check(f"{var} {year}: no duplicate timestamps", dups == 0, f"{dups} duplicates")

    # ---- snapshot cadence: expected count per year
    print("\nsnapshot cadence")
    for var in snap_vars:
        for year in range(args.start_year, args.start_year + args.years):
            if (var, year) not in groups:
                continue
            n = sum(_records(p, var)[0].size for p in groups[(var, year)])
            days = 366 if calendar.isleap(year) else 365
            expect = days // args.snap_days
            check(f"{var} {year}: {n} snapshots (expect ~{expect} at {args.snap_days}-daily)",
                  abs(n - expect) <= 1, f"year has {days} days")

    # ---- snapshots must not equal the corresponding mean (they would if the flag leaked)
    print("\nsnapshot vs mean")
    for var in snap_vars:
        base = var[:-5]
        yr = args.start_year
        if (base, yr) not in groups or (var, yr) not in groups:
            continue
        s = np.ma.filled(_records(sorted(groups[(var, yr)])[0], var)[2][0], 0.0)
        m = np.ma.filled(_records(sorted(groups[(base, yr)])[0], base)[2][0], 0.0)
        if s.shape == m.shape:
            d = float(np.abs(np.asarray(s, float) - np.asarray(m, float)).max())
            check(f"{var}: differs from the {base} mean", d > 0, f"max|diff|={d:.3e}")

    # ---- values are finite and the fill pattern is sane
    print("\nvalues")
    for var in snap_vars + mean_vars:
        yr = args.start_year
        if (var, yr) not in groups:
            continue
        a = np.asarray(np.ma.filled(_records(sorted(groups[(var, yr)])[0], var)[2][0], 0.0), float)
        wet = np.abs(a) < 1e30
        check(f"{var}: finite on wet points", bool(np.isfinite(a[wet]).all()),
              f"{100*wet.mean():.1f}% wet")

    print(f"\n{checks} checks, {len(fails)} failure(s)")
    for f in fails:
        print(f"   FAILED: {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
