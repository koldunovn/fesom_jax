#!/usr/bin/env python
"""Split the NG5 teacher's wall time into COMPUTE and OUTPUT, from the sampler's CSV.

This is the measurement the compute-budget gate turns on. The chain's log stamps no wallclock
on anything, so ``sample_teacher_rate.sh`` records (wallclock, last model step, archive size)
every 60 s and this reads it back.

⚠️ **The split rests on an assumption that stopped being true.** At ``compression_level=1``
(months 01-02) writing a snapshot set stalled the model completely: zero steps for ~12 minutes
while the archive grew at 0.6-1.1 GB/min. That made the two regimes trivially separable. At
``compression_level=0`` (months 03+) the writes are fast enough that the model **keeps stepping
through them** — measured 2026-08-08: 3.97 steps/s in the "archive growing" samples against
3.19 in the quiet ones, i.e. no stall at all, and the naive split reported "-19% of the wall time
is output".

So this script now reports the OVERALL rate as the primary number, which is assumption-free, and
attempts the split only as a diagnostic. If the writing bucket contains a material share of the
steps, the split is declared INVALID and the reason is printed rather than a nonsense percentage.
That the split *fails* is itself the result worth having: output no longer blocks compute.

⚠️ Instantaneous step rate is NOT reliable per sample. FESOM prints its progress marker only
every 960 steps, so a one-minute sample sees either 0 or 960 and neither is the rate. Always
difference over a span of several samples, which is what this does.

Usage:  python scripts/nn_eddy/analyse_teacher_rate.py [--csv PATH] [--set-gb 9.18]
"""

from __future__ import annotations

import argparse
import csv
import pathlib

CSV = pathlib.Path("/work/ab0995/a270088/nn_eddy/monitoring/teacher_rate.csv")
# Split point between the two regimes. The observed distribution is sharply bimodal: while a
# snapshot set is being written the archive grows at 600-1100 MB/min, and while the model steps
# it grows at 0-150 MB/min (the daily 2-D means, which are small and continuous). 50 MB/min put
# ordinary compute intervals in the writing bucket and made the split meaningless -- the giveaway
# was 1920 steps appearing inside "archive growing", where there should be none.
GROWTH_MB_PER_MIN = 250.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=pathlib.Path, default=CSV)
    ap.add_argument("--set-gb", type=float, default=9.18,
                    help="measured size of one snapshot set ON DISK. 9.18 GB at deflate level 1 "
                         "(months 01-02); pass ~14.7 for compression_level=0 (months 03+) or the "
                         "size actually measured there — the per-month output cost is computed "
                         "from this and the observed write rate, so a stale value skews it.")
    ap.add_argument("--sets-per-month", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=64)
    ap.add_argument("--spd", type=int, default=None,
                    help="analyse only samples at this step_per_day (default: the latest in the "
                         "CSV). Rates from different dt are not comparable.")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.csv.open()) if r.get("mstep")]
    if len(rows) < 3:
        print(f"only {len(rows)} usable samples in {args.csv} — let the sampler run longer")
        return 1

    # per-interval deltas
    ivals = []
    for a, b in zip(rows, rows[1:]):
        dt = int(b["epoch"]) - int(a["epoch"])
        if dt <= 0:
            continue
        dms = int(b["mstep"]) - int(a["mstep"])
        dkb = int(b["archive_kb"] or 0) - int(a["archive_kb"] or 0)
        if dms < 0:                       # a new job restarted the month; step counter resets
            continue
        ivals.append(dict(dt=dt, dms=dms, mb_min=dkb / 1024.0 / (dt / 60.0),
                          spd=int(b["spd"] or 0), iso=b["iso"]))

    # ⚠️ Restrict to ONE time step before computing any rate. The chain switches step_per_day
    # from 720 to 360 at a month boundary, so a CSV that spans the switch holds two populations
    # whose steps/s are not comparable, and averaging them produces a number that describes
    # neither. Default to the LATEST step_per_day present, which is the configuration anyone
    # asking is asking about; --spd picks another.
    present = sorted({i["spd"] for i in ivals if i["spd"]})
    spd = args.spd or (ivals[-1]["spd"] if ivals[-1]["spd"] else 720)
    if len(present) > 1:
        print(f"⚠️ CSV spans step_per_day {present}; analysing spd={spd} only "
              f"(pass --spd to choose). Rates across a dt switch are not comparable.\n")
    ivals = [i for i in ivals if i["spd"] == spd]
    if len(ivals) < 2:
        print(f"only {len(ivals)} intervals at spd={spd} — let the sampler run longer")
        return 1

    writing = [i for i in ivals if i["mb_min"] > GROWTH_MB_PER_MIN]
    quiet = [i for i in ivals if i["mb_min"] <= GROWTH_MB_PER_MIN]

    def rate(group):
        t = sum(i["dt"] for i in group)
        s = sum(i["dms"] for i in group)
        return (s / t if t else 0.0), t, s

    r_all, t_all, s_all = rate(ivals)
    r_quiet, t_quiet, s_quiet = rate(quiet)
    r_writ, t_writ, s_writ = rate(writing)

    print(f"samples {len(rows)}, span {t_all / 60:.0f} min, step_per_day={spd} "
          f"(dt={86400 / spd:.0f} s)\n")
    print(f"{'regime':22s} {'wall min':>9s} {'steps':>8s} {'steps/s':>8s} {'GB/min':>8s}")
    for name, g, (r, t, s) in (("archive growing", writing, (r_writ, t_writ, s_writ)),
                               ("archive quiet", quiet, (r_quiet, t_quiet, s_quiet)),
                               ("both", ivals, (r_all, t_all, s_all))):
        mb = sum(i["mb_min"] * i["dt"] / 60.0 for i in g) / max(t / 60.0, 1e-9)
        print(f"{name:22s} {t / 60:9.1f} {s:8d} {r:8.2f} {mb / 1024:8.2f}")

    if r_quiet <= 0:
        print("\nno step progress in the quiet samples yet — cannot split the two regimes")
        return 1

    def sypd(steps_per_s):
        return steps_per_s / spd * 86400 / 365.0

    # PRIMARY, assumption-free: what the run actually achieved over the whole window.
    print(f"\nACHIEVED      {r_all:.2f} steps/s => {sypd(r_all):.3f} SYPD at {args.nodes} nodes")
    print(f"              => 10 SY in {10 / sypd(r_all):.1f} wall days = "
          f"{10 / sypd(r_all) * 24 * args.nodes:,.0f} node-h")

    # DIAGNOSTIC: only meaningful while writing stalls the model completely.
    steps_while_writing = sum(i["dms"] for i in writing)
    share = steps_while_writing / max(s_all, 1)
    if share > 0.1:
        print(f"\ncompute/output split: NOT APPLICABLE. {100 * share:.0f}% of the steps were "
              f"taken while the archive was growing\n  ({r_writ:.2f} steps/s writing vs "
              f"{r_quiet:.2f} quiet), so output is OVERLAPPING compute rather than blocking it "
              f"and there is no stall to measure.\n  That is the good outcome: at "
              f"compression_level=1 the model took ZERO steps for ~12 min per snapshot set.")
        if t_writ > 0:
            mbs = sum(i["mb_min"] * i["dt"] / 60.0 for i in writing) / (t_writ / 60.0) / 60.0
            print(f"  write throughput while growing: {mbs:.0f} MB/s "
                  f"(deflate level 1 measured 13 MB/s)")
        return 0
    print(f"compute-only  {r_quiet:.2f} steps/s => {sypd(r_quiet):.3f} SYPD")
    print(f"as observed   {r_all:.2f} steps/s => {sypd(r_all):.3f} SYPD  "
          f"({100 * (1 - r_all / r_quiet):.0f}% of the wall time is output)")

    # per-month accounting, and the same at the production time step
    # GB actually written during the output-dominated intervals, over their wall time
    gb_written = sum(i["mb_min"] * i["dt"] / 60.0 for i in writing) / 1024.0
    write_min_total = sum(i["dt"] for i in writing) / 60.0
    write_gb_per_min = gb_written / max(write_min_total, 1e-9)
    write_min_per_set = args.set_gb / max(write_gb_per_min, 1e-9)
    write_min = write_min_per_set * args.sets_per_month
    for label, this_spd in (("as running", spd), ("at dt=240 s", 360)):
        comp_min = 30.0 * this_spd / r_quiet / 60.0
        tot = comp_min + write_min
        print(f"\n30-day month, {label} (spd={this_spd}):")
        print(f"  compute {comp_min:6.1f} min + output {write_min:6.1f} min = {tot / 60:.2f} h"
              f"   ({100 * write_min / tot:.0f}% output)")
        print(f"  => {365.0 / 30 * tot / 60:.1f} h per model year = "
              f"{24 / (365.0 / 30 * tot / 60):.2f} SYPD, "
              f"10 SY = {10 * 365.0 / 30 * tot / 60 * args.nodes:,.0f} node-h")

    print(f"\nmeasured output rate {write_gb_per_min * 1024 / 60:.0f} MB/s "
          f"=> {write_min_per_set:.1f} min for a {args.set_gb:.2f} GB set "
          f"({gb_written:.1f} GB seen written in {write_min_total:.0f} min of sampling).")
    print("⚠️ Output cost is INDEPENDENT of dt: halving the steps halves the compute and leaves"
          "\n   the writes untouched. That is why the dt=240 s column is less than 2x better,"
          "\n   and why compression_level=0 from month 03 is the lever that matters.")

    # --- what month 03 is expected to change -------------------------------------------
    # Months 01-02 ran compression_level=1; 03 onward runs 0. Deflate is the reason the write
    # rate above is ~13 MB/s: the same filesystem took the 8 GB mesh diagnostic at 422 MB/s
    # uncompressed. Uncompressed sets are larger (14.7 GB against 9.18) and should still be far
    # cheaper. This is a PROJECTION, printed so month 03 can be checked against it rather than
    # explained after the fact -- do not quote it as a measurement.
    print("\nPROJECTION for compression_level=0 (months 03+), to be checked against month 03:")
    for mbs in (200.0, 422.0):
        w = 14.7 * 1024 / mbs / 60.0 * args.sets_per_month     # min/month, uncompressed sets
        for this_spd, lbl in ((360, "dt=240 s"),):
            comp = 30.0 * this_spd / r_quiet / 60.0
            tot = comp + w
            print(f"  at {mbs:5.0f} MB/s, {lbl}: compute {comp:5.1f} + output {w:5.1f} min "
                  f"= {tot / 60:.2f} h/month => {24 / (365.0 / 30 * tot / 60):.2f} SYPD, "
                  f"10 SY = {10 * 365.0 / 30 * tot / 60 * args.nodes:,.0f} node-h")
    print(f"  (measured now: {write_gb_per_min * 1024 / 60:.0f} MB/s at deflate level 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
