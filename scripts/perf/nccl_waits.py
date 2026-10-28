#!/usr/bin/env python
"""CORE2 GPU speed plan, Task 1: where the halo exchanges' and reductions' time goes, from an nsys
profile of a sharded run (one process driving P GPUs; adapted from fesom3-jax's scripts/nccl_waits.py).

    nccl_waits.py <prof.sqlite> [--steps S] [--top 25]

Every NCCL kernel is matched across the GPUs (the k-th collective on each GPU is the same
collective). The shortest of its P durations is its cost (the last GPU to arrive waits for nobody);
the rest of each GPU's duration is waiting for the slowest GPU. nsys inflates waits (the profiled
GPUs idle more than unprofiled ones), so the robust communication number stays (unprofiled steady
step) - (compute kernels); this script says which collectives the waits sit in.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import statistics


def analyse(db_path, steps) -> dict:
    db = sqlite3.connect(db_path)
    names = dict(db.execute("select id, value from StringIds"))
    rows = db.execute("select start, end, deviceId, shortName from CUPTI_ACTIVITY_KIND_KERNEL "
                      "order by start").fetchall()
    devs = sorted({r[2] for r in rows})
    per_dev = {d: [] for d in devs}
    for s, e, d, n in rows:
        per_dev[d].append((s, e, names[n]))
    nccl = {d: [i for i, k in enumerate(ks) if k[2].startswith("ncclDevKernel")] for d, ks in per_dev.items()}
    counts = {d: len(v) for d, v in nccl.items()}
    n = min(counts.values()) if counts else 0
    per = n // steps
    cost, wait, by_pos = [], [], collections.defaultdict(list)
    for k in range(n):
        ks = [per_dev[d][nccl[d][k]] for d in devs]
        durs = [e - s for s, e, _ in ks]
        last = max(range(len(devs)), key=lambda i: ks[i][0])         # the GPU that started it last
        c = min(durs)
        cost.append(c)
        wait.append(sum(x - c for x in durs) / len(devs))
        i = nccl[devs[last]][k]
        before = next((per_dev[devs[last]][j][2] for j in range(i - 1, -1, -1)
                       if not per_dev[devs[last]][j][2].startswith("ncclDevKernel")), "?")
        by_pos[k % per if per else k].append((c, wait[-1], max(durs), devs[last], before))
    positions = []
    for k in sorted(by_pos):
        r = by_pos[k]
        positions.append({"pos": k, "cost_us": statistics.mean(x[0] for x in r) / 1e3,
                          "wait_us": statistics.mean(x[1] for x in r) / 1e3,
                          "max_us": max(x[2] for x in r) / 1e3,
                          "last_gpu": collections.Counter(x[3] for x in r).most_common(1)[0],
                          "before": collections.Counter(x[4] for x in r).most_common(1)[0][0]})
    return {"gpus": len(devs), "nccl_per_gpu": counts, "collectives_per_step": per,
            "kernels_per_step_gpu": sum(len(v) for v in per_dev.values()) / max(len(devs), 1) / steps,
            "cost_ms_per_step": sum(cost) / 1e6 / steps, "wait_ms_per_step": sum(wait) / 1e6 / steps,
            "median_cost_us": statistics.median(cost) / 1e3 if cost else 0.0, "positions": positions}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("db")
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()
    r = analyse(a.db, a.steps)
    if len(set(r["nccl_per_gpu"].values())) > 1:
        print(f"NCCL kernels per GPU differ: {r['nccl_per_gpu']}; matched the first "
              f"{min(r['nccl_per_gpu'].values())}")
    print(f"{r['gpus']} GPUs, {r['collectives_per_step']} collectives per step, "
          f"{r['kernels_per_step_gpu']:.0f} kernels per step per GPU")
    print(f"per step per GPU: NCCL cost {r['cost_ms_per_step']:.2f} ms (median {r['median_cost_us']:.1f} us "
          f"each), waiting for the slowest GPU {r['wait_ms_per_step']:.2f} ms")
    print(f"{'pos':>4} {'cost us':>8} {'wait us':>8} {'max us':>8}  last GPU  before")
    for x in r["positions"]:
        print(f"{x['pos']:4d} {x['cost_us']:8.1f} {x['wait_us']:8.1f} {x['max_us']:8.0f}  "
              f"{x['last_gpu'][0]} ({x['last_gpu'][1]})  {x['before'][:60]}")
    worst = sorted(r["positions"], key=lambda x: -x["wait_us"])[:a.top]
    print("largest waits (pos, wait us): " + ", ".join(f"{x['pos']}: {x['wait_us']:.0f}" for x in worst))


if __name__ == "__main__":
    main()
