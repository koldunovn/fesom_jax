#!/usr/bin/env python
"""Fast-adjoint plan, the GPU-options step (docs/plans/20261022-core2-fast-adjoint.md): time and profile ONE chunk of
the year-window adjoint (``scripts/adjoint/core2_adj.py``'s model, the sharded chunk of :mod:`fesom_jax.adjoint_sharded`)
— its forward program and its reverse program (the VJP: the per-step remat recompute plus the transpose), as
``fesom_jax.adjoint.chunked_grad`` calls them.

    adj_profile.py time OUT [--fast ice2+se+dt2700+sp] [--devices 4] [--chunk-steps 32] [--reps 3]
    adj_profile.py prof OUT [same]        (under nsys --capture-range=cudaProfilerApi: one forward + one reverse)

The model: the Y4raw7 derivative-only settings, exact sea ice with the frozen-sweep backward and the year-window ice
settings, the window-mean MLD over natl_subpolar, the five scalar controls, from the dt-2700 spin restart of
2013-01-02 (``--restart``). Both programs are compiled and run once before the clock starts. The reverse program gets
the cotangent the cost gives the chunk's end carry (the real seed of a window's last chunk). Written: OUT/time.json
(``fwd_ms_per_step``, ``bwd_ms_per_step``, the reps, peak device memory, the compile times, the chunk's control
gradient and the ``FESOM_*`` environment, for an A/B of the GPU options: ``adj_ab_gh200.sbatch``).
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))

Y4RAW7 = ("tke_pr_cut=1e-6+tke_pr_ratio=100+tke_pr_raw_cut=-1e-6+tke_ml_cut=1e-8+freeze_fct_limiter+tke_ml_ratio=100"
          "+tke_ml_raw_cut=-1e-7+tke_pr_lat_cut=70+tke_pr_ice_cut=0.15")
ICE = ["grad", "--ice-damping", "frozen_sweep", "--ice-alpha-kappa", "1", "--ice-theta", "0.97", "--ice-sweep-guard",
       "10", "--ice-thin-mass", "100", "--ice-cuts", "fct_flux+strength_conc"]
SPIN = "/work/ab0995/a270088/fesom_jax_dev/runs/adjfast/spin/fesom.2013.002.00000"


def _cudart():
    import nvidia.cuda_runtime
    lib = glob.glob(os.path.join(list(nvidia.cuda_runtime.__path__)[0], "lib", "libcudart.so*"))
    return ctypes.CDLL(sorted(lib)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("time", "prof"))
    ap.add_argument("out")
    ap.add_argument("--fast", default="ice2+se+dt2700+sp")
    ap.add_argument("--devices", type=int, default=4)
    ap.add_argument("--chunk-steps", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--start", default="2013-01-02")
    ap.add_argument("--restart", default=SPIN)
    ap.add_argument("--ice-mode", default="exact",
                    help="IceConfig.adjoint_mode (exact: the frozen-sweep backward; frozen: no ice derivative, the cost of "
                         "the ice adjoint by difference)")
    a = ap.parse_args(argv)
    import jax
    import jax.numpy as jnp
    import core2_adj as C
    import tke_cbar as T
    import tke_diag as D
    from fesom_jax import adjoint
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    K = int(a.chunk_steps)
    m = C.build_model(datetime.datetime.fromisoformat(a.start), K, K, a.restart, D.parse_setting(Y4RAW7), a.ice_mode,
                      ice_adjoint=C._ice_adjoint_args(C._parse(ICE)) if a.ice_mode == "exact" else {},
                      accumulators=T.accumulators if a.devices == 1 else T.mld_accumulators, devices=a.devices,
                      fast=a.fast)
    loss = T.cost_fn("MLD", K, m.mesh, fold=None if m.shard is None else m.shard.fold_nodes)
    fwd, bwd = adjoint._jitted_forward(m.chunk_fn), adjoint._jitted_backward(m.chunk_fn)
    u = jax.device_put(m.params0)
    c0, xs = m.init_carry(), m.xs_fn()(0)
    t0 = time.time()
    c1 = jax.block_until_ready(fwd(c0, xs, u))
    t_fc = time.time() - t0
    _, vjp = jax.vjp(loss, c1)                   # (a jax.grad would refuse the carry's integer ice counter)
    cot = jax.block_until_ready(vjp(jnp.ones((), jnp.float64))[0])
    t0 = time.time()
    g = jax.block_until_ready(bwd(c0, xs, u, cot))
    t_bc = time.time() - t0
    grad = {k: float(v) for k, v in jax.device_get(g[1]).items()}    # the chunk's control gradient (A/B check)
    rt = _cudart() if a.action == "prof" else None
    if rt is not None:
        rt.cudaProfilerStart()
    reps = 1 if a.action == "prof" else int(a.reps)
    tf, tb = [], []
    for _ in range(reps):
        t0 = time.time()
        jax.block_until_ready(fwd(c0, xs, u))
        tf.append(time.time() - t0)
        t0 = time.time()
        jax.block_until_ready(bwd(c0, xs, u, cot))
        tb.append(time.time() - t0)
    if rt is not None:
        rt.cudaProfilerStop()
    peak = [round((d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1024 ** 3, 2) for d in jax.local_devices()]
    rec = {"what": a.action, "fast": list(m.fast), "devices": m.devices, "chunk_steps": K, "dt": m.dt,
           "ice_mode": a.ice_mode,
           "fwd_ms_per_step": 1e3 * min(tf) / K, "bwd_ms_per_step": 1e3 * min(tb) / K,
           "fwd_s": tf, "bwd_s": tb, "first_fwd_s": t_fc, "first_bwd_s": t_bc, "peak_gb": peak,
           "job": os.environ.get("SLURM_JOB_ID"), "xla_flags": os.environ.get("XLA_FLAGS", ""),
           "loss": float(loss(c1)), "finite": bool(jnp.isfinite(loss(c1))), "grad": grad,
           "env": {k: v for k, v in os.environ.items() if k.startswith("FESOM_")}}
    (out / f"{a.action}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
