#!/usr/bin/env python
"""The standalone (file-forced) CORE2 path of this branch against fesom_jax main: the same run_from_config segment
(configs/core2_cpl.yaml, from its restart, --steps N) from a main snapshot, from this branch's snapshot, and from the
main snapshot again (the determinism control), each in its own process on the same GPU; every restart leaf compared
bit for bit."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np


def run(snap, cfg, out, steps, mesh):
    env = dict(os.environ, PYTHONPATH=snap)
    cmd = [sys.executable, f"{snap}/scripts/run_from_config.py", cfg, "--mesh-dir", mesh, "--steps", str(steps),
           "--chunk-steps", "48", "--restart-out", out, "--year", "1958"]
    print("[bitwise] run:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=snap, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:] if r.returncode else "", flush=True)
    if r.returncode:
        raise SystemExit(f"run failed in {snap}")


def compare(a, b):
    import zarr
    ga, gb = zarr.open_group(a, mode="r"), zarr.open_group(b, mode="r")
    na, nb = sorted(ga.array_keys()), sorted(gb.array_keys())
    bad = []
    if na != nb:
        bad.append(("field sets differ", sorted(set(na) ^ set(nb))))
    for k in sorted(set(na) & set(nb)):
        x, y = np.asarray(ga[k]), np.asarray(gb[k])
        if x.shape != y.shape or not np.array_equal(x.view(np.uint8), y.view(np.uint8)):
            bad.append((k, float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.shape == y.shape else -1))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-snap", required=True)
    ap.add_argument("--branch-snap", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--mesh", default="/work/ab0995/a270088/port_jax/data/mesh_core2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = f"{a.out}/config.yaml"
    shutil.copy(a.config, cfg)
    run(a.main_snap, cfg, f"{a.out}/main", a.steps, a.mesh)
    run(a.branch_snap, cfg, f"{a.out}/branch", a.steps, a.mesh)
    run(a.main_snap, cfg, f"{a.out}/main2", a.steps, a.mesh)
    b1 = compare(f"{a.out}/main", f"{a.out}/branch")
    b2 = compare(f"{a.out}/main", f"{a.out}/main2")
    print(f"[bitwise] branch vs main: {'BITWISE' if not b1 else b1}", flush=True)
    print(f"[bitwise] control main vs main: {'BITWISE' if not b2 else b2}", flush=True)


if __name__ == "__main__":
    main()
