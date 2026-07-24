"""Reduce an nsys sqlite export (profile_ng5_128.sbatch) to the numbers that matter for
the ng5 x P=128 cliff: WHERE does step time go — NCCL kernels, compute kernels, or idle
(host/launch/sync gaps)?

Analyzes DEVICE 0 only (all four local GPUs behave alike).  Segments the GPU timeline at
>200 ms idle gaps (compile / first run / timed run / finiteness separate naturally), then
per segment reports the busy union split nccl vs compute.  NCCL device-kernel duration
INCLUDES remote-peer wait, so "nccl" here is wire time + desync stalls — the quantity
that distinguishes serialized-latency execution from overlapped execution.

    python reduce_nsys.py prod_p64.sqlite prod_p128.sqlite ...
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np


def union_len(iv: np.ndarray) -> float:
    """Total covered length of [start,end) intervals, ns. iv sorted by start."""
    if len(iv) == 0:
        return 0.0
    ends = np.maximum.accumulate(iv[:, 1])
    new_cover = np.minimum(iv[:, 1], np.maximum(iv[:, 0], np.concatenate([[iv[0, 0]], ends[:-1]])))
    return float(np.sum(iv[:, 1] - new_cover))


def reduce_one(path):
    db = sqlite3.connect(path)
    dev = db.execute("SELECT MIN(deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    rows = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName WHERE k.deviceId = ? ORDER BY k.start",
        (dev,)).fetchall()
    db.close()
    if not rows:
        print(f"== {path}: no kernel rows"); return
    start = np.array([r[0] for r in rows], dtype=np.int64)
    end = np.array([r[1] for r in rows], dtype=np.int64)
    isnccl = np.array([r[2].startswith("nccl") for r in rows])
    names = np.array([r[2] for r in rows])

    # segment at >200 ms gaps in device-0 activity
    run_max_end = np.maximum.accumulate(end)
    gap = start[1:] - run_max_end[:-1]
    cuts = np.where(gap > 200_000_000)[0] + 1
    bounds = np.concatenate([[0], cuts, [len(start)]])

    print(f"\n== {path}  (device {dev}, kernels={len(rows)}, segments={len(bounds)-1})")
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        wall = (end[a:b].max() - start[a]) / 1e9
        if wall < 0.05 and b - a < 500:
            continue
        seg_iv = np.stack([start[a:b], end[a:b]], axis=1)
        m = isnccl[a:b]
        busy = union_len(seg_iv) / 1e9
        nccl = union_len(seg_iv[m]) / 1e9
        comp = union_len(seg_iv[~m]) / 1e9
        print(f"  seg{i}: wall={wall:7.2f}s busy={busy:6.2f} idle={wall-busy:6.2f} | "
              f"nccl={nccl:6.2f}s ({int(m.sum())}x) comp={comp:6.2f}s ({int((~m).sum())}x)")
        if wall > 0.4:
            seg_names = names[a:b]
            dur = (end[a:b] - start[a:b])
            top = {}
            for n, d in zip(seg_names, dur):
                t = top.get(n, [0, 0]); t[0] += int(d); t[1] += 1; top[n] = t
            for n, (dd, cc) in sorted(top.items(), key=lambda kv: -kv[1][0])[:6]:
                print(f"        {dd/1e6:9.1f} ms {cc:7d}x  mean={dd/cc/1e3:8.1f} us  {n[:66]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        reduce_one(p)
