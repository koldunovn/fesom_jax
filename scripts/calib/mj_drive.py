#!/usr/bin/env python
"""Drive multi-job TKE-constant calibrations (scripts/calib/tke_lbfgs.py mjstep / winjob) in
shared multi-node jobs.

    scripts/calib/mj_drive.py <spec.json> [--max-nodes-per-job 24] [--max-total-nodes 72] [--dry-run]

One pass: for every run of the spec, in priority order, advance its coordinator
(``tke_lbfgs.py mjstep``, no model) when its current evaluation is complete, and collect the
windows that have neither a result file nor an active job; then submit them as multi-node
jobs (``scripts/calib/mj_batch.sbatch``, one window per node), whole runs per job, at most
``--max-nodes-per-job`` windows per job and ``--max-total-nodes`` windows in flight. One job
counts once against the per-user running-job limit, however many runs it serves. Call it
periodically (e.g. every 5 minutes).

``spec.json``::

    {"runs": [{"name": "all4", "dir": "/work/.../cpu24_all4", "months": "1,2,...,12",
               "args": "--mode real --window-steps 24 ...", "priority": 1}, ...]}

``--dry-run`` reports what a pass would do and changes nothing (no coordinator step, no task
file, no job). ``args`` are the calibration's tke_lbfgs options without --months/--out/--eval; every
coordinator step and every window of a run gets the same ones. Job ids go to
``<dir>/eval_<k>/job_<MM>.txt``; a line ``#reset`` restarts a window's failure count; a window
whose jobs ended twice without its result file stops that run (reported, nothing resubmitted).
Nothing is deleted or overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
PY = "/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python"
TASK_DIR = Path("/work/ab0995/a270088/fesom_jax_dev/runs/calib/logs/tasks")
ACTIVE = {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "REQUEUED", "SUSPENDED"}


def job_state(job: str) -> str:
    out = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "-o", "State"],
                         capture_output=True, text=True).stdout.split()
    st = out[0] if out else "PENDING"             # just submitted: not in sacct yet
    if st == "PENDING":
        r = subprocess.run(["squeue", "-j", job, "-h", "-o", "%r"], capture_output=True,
                           text=True).stdout.strip()
        return "HELD" if r == "JobHeldUser" else "PENDING"
    return st


def jobs_after_reset(jf: Path) -> list:
    """The job ids of a window's job file after its last ``#reset`` line."""
    if not jf.exists():
        return []
    toks = jf.read_text().split()
    cut = max([i for i, t in enumerate(toks) if t == "#reset"], default=-1)
    return [t for t in toks[cut + 1:] if t.isdigit()]


def stopped(run_dir: Path):
    log = run_dir / "iterations.jsonl"
    if not log.exists():
        return None
    for line in reversed([x for x in log.read_text().splitlines() if x.strip()]):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        return rec if rec.get("type") == "stop" else None
    return None


def mjstep(run: dict) -> dict:
    p = subprocess.run([PY, str(WT / "scripts/calib/tke_lbfgs.py"), "mjstep", "--months",
                        run["months"], "--out", run["dir"]] + run["args"].split(),
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(WT), JAX_PLATFORMS="cpu"))
    lines = [x for x in p.stdout.splitlines() if x.strip().startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"{run['name']}: mjstep failed: {p.stderr[-500:]}")
    return json.loads(lines[-1])


def window_status(run_dir: Path, k: int, months, state=job_state) -> dict:
    """``{"have", "active", "idle", "failed"}`` month lists of evaluation ``k``."""
    d = run_dir / f"eval_{k:03d}"
    out = {"have": [], "active": [], "idle": [], "failed": []}
    for m in months:
        if (d / f"win_{m:02d}.json").exists():
            out["have"].append(m)
            continue
        jobs = jobs_after_reset(d / f"job_{m:02d}.txt")
        states = [state(j) for j in jobs]
        if any(s in ACTIVE for s in states):
            out["active"].append(m)
        elif sum(s not in ACTIVE and s != "HELD" for s in states) >= 2:
            out["failed"].append(m)
        else:
            out["idle"].append(m)
    return out


def plan_batches(ready, max_per_job: int, room: int) -> list:
    """Pack ``ready`` = [(name, k, [months]), ...] (priority order) into jobs of whole runs:
    a run joins the current job if it fits in ``max_per_job``, else starts the next one (a
    run larger than ``max_per_job`` is split); nothing beyond ``room`` windows in total."""
    jobs, cur = [], []
    for name, k, months in ready:
        months = list(months)
        while months:
            if room <= 0:
                break
            take = months[:min(len(months), max_per_job, room)]
            if cur and len(cur) + len(take) > max_per_job:
                jobs.append(cur)
                cur = []
            cur += [(name, k, m) for m in take]
            room -= len(take)
            months = months[len(take):]
        if room <= 0:
            break
    if cur:
        jobs.append(cur)
    return jobs


def submit(batch, runs: dict, dry: bool, index: int = 0) -> str:
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(["--months", str(m), "--eval", str(k), "--out", runs[n]["dir"]]
                      + runs[n]["args"].split()) for n, k, m in batch]
    if dry:
        return "DRY"
    tf = (TASK_DIR / f"tasks_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{index}_"
                     f"{len(batch)}.txt")        # unique per batch of this pass
    with open(tf, "x") as f:
        f.write("\n".join(lines) + "\n")
    job = subprocess.run(["sbatch", "--parsable", f"--nodes={len(batch)}",
                          f"--export=ALL,WT={WT}", "--job-name=mjb", str(WT /
                          "scripts/calib/mj_batch.sbatch"), str(tf)],
                         capture_output=True, text=True, check=True).stdout.strip()
    for n, k, m in batch:
        jf = Path(runs[n]["dir"]) / f"eval_{k:03d}" / f"job_{m:02d}.txt"
        with open(jf, "a") as f:
            f.write(job + "\n")
    return job


def one_pass(spec: dict, max_per_job: int, max_total: int, dry: bool = False) -> list:
    runs = {r["name"]: r for r in spec["runs"]}
    order = sorted(spec["runs"], key=lambda r: (r.get("priority", 99), r["name"]))
    report, ready, in_flight = [], [], 0
    for r in order:
        d = Path(r["dir"])
        months = [int(m) for m in r["months"].split(",")]
        st = stopped(d)
        if st is not None:
            report.append(f"{r['name']}: STOPPED {st.get('status')} n_eval {st.get('n_eval')}")
            continue
        evs = sorted(d.glob("eval_*")) if d.exists() else []
        k = int(evs[-1].name.split("_")[1]) if evs else 0
        ws = window_status(d, k, months) if k else {"have": [], "active": [], "idle": [],
                                                     "failed": []}
        if k and ws["failed"]:
            report.append(f"{r['name']}: eval {k} FAILED windows {ws['failed']}")
            continue
        if k == 0 or len(ws["have"]) == len(months):
            if dry:                               # a dry run changes nothing, not even the log
                report.append(f"{r['name']}: eval {k} complete (would run the coordinator)")
                continue
            res = mjstep(r)                       # logs the evaluation, posts the next point
            if res.get("status") != "need":
                report.append(f"{r['name']}: STOPPED {res.get('status')}")
                continue
            k = int(res["eval"])
            ws = window_status(d, k, months)
        in_flight += len(ws["active"])
        if ws["idle"]:
            ready.append((r["name"], k, ws["idle"]))
        report.append(f"{r['name']}: eval {k} have {len(ws['have'])} active "
                      f"{len(ws['active'])} idle {len(ws['idle'])}")
    for i, batch in enumerate(plan_batches(ready, max_per_job, max_total - in_flight)):
        job = submit(batch, runs, dry, index=i)
        names = sorted({n for n, _, _ in batch})
        report.append(f"submitted {job}: {len(batch)} nodes for {names}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--max-nodes-per-job", type=int, default=24)
    ap.add_argument("--max-total-nodes", type=int, default=72)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    for line in one_pass(spec, a.max_nodes_per_job, a.max_total_nodes, a.dry_run):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
