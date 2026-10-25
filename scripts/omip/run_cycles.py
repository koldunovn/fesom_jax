"""Forcing cycles (OMIP style: JRA55-do 1958-2019 again and again) for the chain job that scripts/omip/submit.py
writes: what it runs next (fesom3-jax's tools/run_cycles.py, adapted to fesom-jax's archival restarts).

    run_cycles.py next ROOT [--budget-s S] [--s-per-year Y] [--tag T]
    run_cycles.py env ROOT        prints CONFIG=... DT=... PART=... CYCLES=... RYEARS=... for the job

`next` prints one line: `run CYCLE RESTART_IN STEPS OUT_DIR ARCHIVE CYCLE_START`, `done` (every cycle complete),
`wait` (no whole year fits the budget) or `stop REASON` (something a person must look at).
Layout: ROOT/cycleN/restart_archive (the archival restarts with their restart.latest pointer) and
ROOT/cycleN/yearly/<year>_<tag> (one output directory per job segment, named by its first year, so a segment re-run
after a killed job never meets the stores of the first try). A cycle is complete when its newest restart stands at
1 January of last + 1.

Cycle 1 starts cold (RESTART_IN `-`) with a one-year segment (a canary: the job's end-of-segment diagnostics stop the
chain on a non-finite state before a whole cycle runs on it). A later cycle starts from the previous cycle's last restart with the clock at
1 January of `first` (run_from_config.py --cycle-start: the state and the AB2 history go on, the step counts from 0);
nothing is copied. A segment runs whole years only, as many as fit the budget, or to the cycle's end. The config's dt
must divide a day (whole-day steps).

ROOT/experiment.toml is the run's copy of experiments/<name>/experiment.toml; the config path in it is relative to
the working directory (the job runs in its frozen snapshot).
"""

import argparse
import datetime
import json
import math
import re
import sys
import tomllib
from pathlib import Path


def experiment(root):
    return tomllib.loads((Path(root) / "experiment.toml").read_text())


def config_dt(path):
    """The `dt:` of a run YAML (a top-level scalar; read without importing fesom_jax)."""
    m = re.search(r"^dt:\s*([0-9.eE+-]+)", Path(path).read_text(), re.M)
    if m is None:
        raise SystemExit(f"run_cycles: no top-level dt in {path}")
    dt = float(m.group(1))
    if 86400.0 % dt:
        raise SystemExit(f"run_cycles: dt {dt} does not divide a day")
    return dt


def latest(archive):
    """(path, step) of the archive's newest restart (restart.latest), None when there is none yet. Refuses an
    archive holding a restart newer than the pointer (a write the pointer never reached: a killed job)."""
    ptr = archive / "restart.latest"
    if not ptr.is_file():
        tags = sorted(p.name for p in archive.glob("fesom.*") if p.is_dir()) if archive.is_dir() else []
        if tags:
            return "stop", f"{archive} holds {tags[-1]} but no restart.latest"
        return None
    tag = ptr.read_text().strip()
    newer = sorted(p.name for p in archive.glob("fesom.*") if p.is_dir() and p.name > tag)
    if newer:
        return "stop", f"{archive} holds {newer[-1]}, newer than restart.latest {tag} (a killed write?)"
    path = archive / tag
    step = int(json.loads((path / ".zattrs").read_text())["step"])
    return path, step


def plan(root, budget_s, s_per_year, tag):
    e = experiment(root)
    first, last, cycles = int(e["first_year"]), int(e["last_year"]), int(e["cycles"])
    spd = int(86400.0 / config_dt(e["config"]))
    t0, end = datetime.date(first, 1, 1), datetime.date(last + 1, 1, 1)
    total = (end - t0).days * spd
    for c in range(1, cycles + 1):
        cur = latest(root / f"cycle{c}" / "restart_archive")
        if cur is not None and cur[0] == "stop":
            return f"stop {cur[1]}"
        cycle_start = 0
        if cur is not None:
            rin, step = cur
            if step >= total:
                continue                                   # this cycle is complete
        elif c == 1:
            rin, step = "-", 0                             # the cold start
        else:
            prev = latest(root / f"cycle{c - 1}" / "restart_archive")
            if prev is None or prev[0] == "stop" or prev[1] != total:
                return f"stop cycle {c - 1} is not complete ({prev})"
            rin, step, cycle_start = prev[0], 0, 1
        days, rem = divmod(step, spd)
        now = t0 + datetime.timedelta(days=days)
        if rem or (now.month, now.day) != (1, 1):
            return f"stop cycle {c} stands at step {step} ({now}), not at the start of a year"
        years = math.floor(budget_s / s_per_year) if s_per_year > 0 else end.year - now.year
        if rin == "-":
            years = min(years, 1)                          # the cold start: a one-year canary segment first
        target = min(end, datetime.date(now.year + max(years, 0), 1, 1))
        if target <= now:
            return "wait"
        out = root / f"cycle{c}" / "yearly" / f"{now.year}_{tag}"
        return f"run {c} {rin} {(target - now).days * spd} {out} {root / f'cycle{c}' / 'restart_archive'} {cycle_start}"
    return "done"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("next")
    n.add_argument("root", type=Path)
    n.add_argument("--budget-s", type=float, default=float("inf"), help="wall seconds left for running")
    n.add_argument("--s-per-year", type=float, default=0.0, help="wall seconds a model year takes (0: no budget)")
    n.add_argument("--tag", default="seg")
    v = sub.add_parser("env")
    v.add_argument("root", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "env":
        e = experiment(a.root)
        opts = " ".join(e.get("run_options", []))
        print(f"CONFIG={e['config']} DT={int(config_dt(e['config']))} PART={e['partition']} CYCLES={int(e['cycles'])} "
              f"RYEARS={int(e['restart_years'])} RUN_OPTIONS='{opts}'")
    else:
        print(plan(a.root.resolve(), a.budget_s, a.s_per_year, a.tag))


if __name__ == "__main__":
    sys.exit(main())
