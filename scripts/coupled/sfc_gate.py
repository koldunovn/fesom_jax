#!/usr/bin/env python
"""Lane L-SFC gate on one GPU (plan ~/awicm_jax/docs/plans/20261103-jaxcpl-core3.md), each run in its own process:

1. the standalone CORE2 path (scripts/run_from_config.py, configs/core2_cpl.yaml, ``--steps``): base snapshot, branch
   snapshot, base again (determinism control); every restart leaf bit for bit (scripts/coupled/bitwise_vs_main.py);
2. the coupled mode F path (scripts/coupled/check_mode_f.py, configs/core2_cpl.yaml, ``--days-default`` days):
   base and branch, the mode F restarts bit for bit;
3. jaxcpl's settings (configs/core2_cpl_jaxcpl.yaml) in mode F on the branch for ``--days-jaxcpl`` days, and the
   melt-pond / albedo / mEVP fields of its end state against the default run's.

    python scripts/coupled/sfc_gate.py --base-snap S0 --branch-snap S1 --out DIR
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bitwise_vs_main as BW   # noqa: E402


def mode_f(snap, cfg, out, days):
    env = dict(os.environ, PYTHONPATH=snap)
    cmd = [sys.executable, f"{snap}/scripts/coupled/check_mode_f.py", "--config", cfg, "--out", out, "--days",
           str(days), "--time-intervals", "4", "--ref-restart", "/nonexistent"]
    print("[gate] run:", " ".join(cmd), flush=True)
    os.makedirs(out, exist_ok=True)
    r = subprocess.run(cmd, env=env, cwd=snap, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[checkF]")]
    print("\n".join(lines), flush=True)
    if r.returncode:
        print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"mode F run failed in {snap}")


def summary(path, tag):
    from fesom_jax import zarr_output
    out = {}
    for k in ("a_ice", "m_ice", "cpl_apnd", "cpl_hpnd", "cpl_ipnd", "cpl_ice_alb", "sigma11", "T"):
        try:
            x = np.asarray(zarr_output.reconstruct_global(path, k), np.float64)
        except Exception as e:      # noqa: BLE001
            out[k] = f"n/a ({type(e).__name__})"
            continue
        x = x[:, 0] if x.ndim == 2 else x
        out[k] = (float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmean(x)), int(np.count_nonzero(x)))
    print(f"[gate] {tag} end state (min, max, mean, nonzero): {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-snap", required=True)
    ap.add_argument("--branch-snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--days-default", type=int, default=2)
    ap.add_argument("--days-jaxcpl", type=int, default=10)
    ap.add_argument("--mesh", default="/work/ab0995/a270088/port_jax/data/mesh_core2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    verdict = {}

    # 1. standalone path, default config
    cfg = f"{a.out}/core2_cpl.yaml"
    with open(f"{a.branch_snap}/configs/core2_cpl.yaml") as f, open(cfg, "w") as g:
        g.write(f.read().replace("output_dir: /work/ba1583/a270088/awicm/ocn/runs", f"output_dir: {a.out}/runs"))
    BW.run(a.base_snap, cfg, f"{a.out}/sa_base", a.steps, a.mesh)
    BW.run(a.branch_snap, cfg, f"{a.out}/sa_branch", a.steps, a.mesh)
    BW.run(a.base_snap, cfg, f"{a.out}/sa_base2", a.steps, a.mesh)
    b1 = BW.compare(f"{a.out}/sa_base", f"{a.out}/sa_branch")
    b2 = BW.compare(f"{a.out}/sa_base", f"{a.out}/sa_base2")
    verdict["standalone branch vs base"] = "BITWISE" if not b1 else b1
    verdict["standalone control base vs base"] = "BITWISE" if not b2 else b2
    print(f"[gate] standalone {a.steps} steps: branch vs base {verdict['standalone branch vs base']}; control "
          f"{verdict['standalone control base vs base']}", flush=True)

    # 2. mode F, default config
    mode_f(a.base_snap, cfg, f"{a.out}/f_base", a.days_default)
    mode_f(a.branch_snap, cfg, f"{a.out}/f_branch", a.days_default)
    b3 = BW.compare(f"{a.out}/f_base/restart_modeF", f"{a.out}/f_branch/restart_modeF")
    verdict["mode F branch vs base"] = "BITWISE" if not b3 else b3
    print(f"[gate] mode F {a.days_default} d (+ 8 timing intervals): branch vs base {verdict['mode F branch vs base']}",
          flush=True)

    # 3. jaxcpl settings, mode F, branch
    cfgj = f"{a.out}/core2_cpl_jaxcpl.yaml"
    with open(f"{a.branch_snap}/configs/core2_cpl_jaxcpl.yaml") as f, open(cfgj, "w") as g:
        g.write(f.read())
    mode_f(a.branch_snap, cfgj, f"{a.out}/f_jaxcpl", a.days_jaxcpl)
    sys.path.insert(0, a.branch_snap)
    summary(f"{a.out}/f_branch/restart_modeF", "default")
    summary(f"{a.out}/f_jaxcpl/restart_modeF", "jaxcpl")
    print(f"[gate] VERDICTS {verdict}", flush=True)
    ok = all(v == "BITWISE" for k, v in verdict.items())
    print("SFC GATE BITWISE" if ok else "SFC GATE NOT BITWISE", flush=True)


if __name__ == "__main__":
    main()
