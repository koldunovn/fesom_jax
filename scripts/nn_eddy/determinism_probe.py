"""Is this model bit-reproducible, and what does it cost to make it so?

Runs a short forward twice in the SAME process, with the same compiled executable and the
same inputs, and reports whether the two states are bit-for-bit identical. Nothing else
varies, so any difference is the model's own numerical nondeterminism.

The expected source is the finite-element assembly. Every operator that scatters element
contributions onto nodes (`ops.scatter_add`) becomes a GPU atomic add, and atomics complete
in whatever order the thread blocks finish. Floating-point addition is not associative, so
the same sum computed twice can differ in the last bits. On CPU the reduction order is
fixed and this cannot happen.

Run it under different XLA settings to find the cheapest one that gives bit-reproducibility:

    XLA_FLAGS=--xla_gpu_enable_command_buffer=                                  # default
    XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_reductions=true"
    XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true"
    JAX_PLATFORMS=cpu                                                          # reference

Reports, per field, the first step at which the two runs diverge and by how much, so a
partial fix (some reductions deterministic, others not) is visible rather than looking like
a total failure. Deliberately short (a few steps) — this is about bit-identity, not about
the gradient, and a short window keeps the compile fast even under the strict flags.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fesom_jax  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

if Path(fesom_jax.__file__).resolve().parents[1] != _ROOT:
    raise SystemExit(f"fesom_jax resolved to {fesom_jax.__file__}, not {_ROOT}")

from fesom_jax import ssh, surface_forcing  # noqa: E402
from fesom_jax.integrate import _run_steps  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.phc_ic import phc_initial_state  # noqa: E402
from fesom_jax.step import step  # noqa: E402
from horizon_scan import (ALE_CFG, DT, GM_CFG, ICE_FROZEN, IC_DIR, MESH_DIR,  # noqa: E402
                          STEPS_PER_DAY, TKE_CFG, seed_date)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--steps", type=int, default=6, help="forward steps per run (default 6 = 3 h)")
    ap.add_argument("--runs", type=int, default=3, help="identical runs to compare (default 3)")
    ap.add_argument("--run-start-year", type=int, default=1958)
    args = ap.parse_args()

    print(f"[setup] backend={jax.default_backend()}  XLA_FLAGS={os.environ.get('XLA_FLAGS','')!r}",
          flush=True)
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    with open(args.snap, "rb") as f:
        st0 = jax.device_put(pickle.load(f))
    yr, mo, dy = seed_date(args.snap, args.run_start_year)
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, yr, sst_ic=sst0)
    dates = surface_forcing.dates_for_steps(yr, DT, STEPS_PER_DAY, start_month=mo, start_day=dy)
    sfs = cf.stack(dates[:args.steps])
    p = Params.defaults()

    @jax.jit
    def run(state, forcing):
        def body(carry, sf):
            return step(carry, mesh, op, None, p, dt=DT, is_first_step=False, step_forcing=sf,
                        forcing_static=cf.static, ice_cfg=ICE_FROZEN, gm_cfg=GM_CFG,
                        tke_cfg=TKE_CFG, ale_cfg=ALE_CFG, remat_blocks=True), None
        # ⚠️ DELIBERATELY still `_run_steps`, and specifically `segments=0` — per-step
        # rematerialization with NO two-level scheme. `fesom_jax.adjoint.scan_steps` always
        # derives √N, which is a different graph, and this probe's entire question is whether a
        # SPECIFIC compiled executable reproduces itself bit for bit. Rewriting it onto a public
        # API that builds a different one would change the number the probe exists to report, and
        # ADJOINT_HORIZON.md cites that number. Private is the correct coupling here: this file
        # stays in fesom_jax, so the import never crosses a repository boundary.
        return _run_steps(body, state, xs=forcing, checkpoint=True, segments=0)

    outs, secs = [], []
    for r in range(args.runs):
        t0 = time.time()
        outs.append(jax.device_get(run(st0, sfs)))
        secs.append(time.time() - t0)
        print(f"[run {r}] {secs[-1]:.1f}s" + ("  (includes compile)" if r == 0 else ""), flush=True)

    ref = outs[0]
    names = [f for f in ref._fields] if hasattr(ref, "_fields") else None
    leaves_ref, treedef = jax.tree.flatten(ref)
    labels = names if names and len(names) == len(leaves_ref) else [f"leaf{i}" for i in range(len(leaves_ref))]

    n_identical = 0
    worst = []
    for r in range(1, args.runs):
        leaves = jax.tree.leaves(outs[r])
        for lab, a, b in zip(labels, leaves_ref, leaves):
            a, b = np.asarray(a), np.asarray(b)
            if a.dtype.kind != "f":
                continue
            if np.array_equal(a, b):
                n_identical += 1
                continue
            d = np.abs(a - b)
            scale = np.maximum(np.abs(a), 1e-300)
            worst.append((float((d / scale).max()), float(d.max()), lab, r))

    total = sum(1 for lab, a in zip(labels, leaves_ref) if np.asarray(a).dtype.kind == "f") * (args.runs - 1)
    print(f"\n  {n_identical}/{total} float fields bit-identical across "
          f"{args.runs} runs of {args.steps} steps", flush=True)
    if not worst:
        print("  BIT-REPRODUCIBLE — every field matched exactly", flush=True)
    else:
        worst.sort(reverse=True)
        print("  NOT bit-reproducible. Largest relative differences:", flush=True)
        for rel, absd, lab, r in worst[:8]:
            print(f"      {lab:12s} run0 vs run{r}: rel {rel:.3e}  abs {absd:.3e}")
    steady = np.mean(secs[1:]) if len(secs) > 1 else secs[0]
    print(f"  time per run after compile: {steady:.1f}s   (first run {secs[0]:.1f}s)", flush=True)
    print(f"DETPROBE_{'REPRODUCIBLE' if not worst else 'NONDETERMINISTIC'}"
          f"_{args.steps}steps_{steady:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
