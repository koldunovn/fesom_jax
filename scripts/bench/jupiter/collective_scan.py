#!/usr/bin/env python
"""Which collective, at which size, is pathological at 128 devices?

    srun -N<n> -n<n> --ntasks-per-node=1 python scripts/bench/jupiter/collective_scan.py

Targets the ng5 x P=128 cliff.  The model's step uses two very different collective patterns,
and the earlier fabric_probe.py measured only the first, and only up to 8 nodes / 32 GPUs —
never the 32-node / 128-GPU configuration where the anomaly lives:

  * psum / all-reduce   -- the SSH CG's global dot products, a few per CG iteration.
                           TRANSPORT-INDEPENDENT: every halo transport pays these.
  * ppermute            -- the coloured halo transport's K rounds (point-to-point shifts).
  * all-gather          -- stands in for the padded transport's bulk pattern.

Sizes are chosen to bracket the model's REAL per-exchange message sizes, which are what a
NCCL algorithm/protocol switch would key on:
    ng5 halo/GPU x nl x 8B  ->  P=64 754 KB | P=128 574 KB | P=256 428 KB
    dars                    ->  P=128 290 KB
So the 128-256 KB .. 1 MB band is the one that matters; the scan spans it generously.

Everything is measured INSIDE one jitted function over `iters` repetitions, so the Python
dispatch floor that made fabric_probe.py's small-message rows meaningless (0.34 ms at 4 KB,
identical on 1 node) is amortised away and the LATENCY regime is actually visible.
"""
from __future__ import annotations

import os
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec

GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "4"))
if os.environ.get("JDIST"):
    jax.distributed.initialize(local_device_ids=list(range(GPUS_PER_NODE)))

devs = jax.devices()
P = len(devs)
proc0 = jax.process_index() == 0
mesh = Mesh(np.array(devs), ("p",))

KB = 1024
SIZES_KB = [16, 64, 128, 256, 512, 1024, 4096]
ITERS = 20          # collectives per jitted call — amortises dispatch
REPS = 5            # timed calls; report the MIN (least contaminated)


def bench(kind: str, n_elem: int):
    """Time `ITERS` back-to-back collectives of `kind` inside ONE jitted function."""
    def body(x):
        def one(carry, _):
            if kind == "allreduce":
                out = lax.psum(carry, "p")
            elif kind == "ppermute":
                # ring shift by 1 — the primitive the coloured transport issues K times
                perm = [(i, (i + 1) % P) for i in range(P)]
                out = lax.ppermute(carry, "p", perm=perm)
            elif kind == "allgather":
                g = lax.all_gather(carry, "p", tiled=True)
                out = lax.dynamic_slice(g, (0,) * g.ndim, carry.shape)
            else:
                raise ValueError(kind)
            return out * 1.0000001, None       # keep a data dependency; prevent CSE
        c, _ = lax.scan(one, x, None, length=ITERS)
        return c

    # check_vma=False: psum makes the scan carry REPLICATED along 'p', which otherwise trips
    # shard_map's varying-manual-axes tracking ("input carry has type f32[..]{V:p} but the
    # output carry component has type f32[..]").  We only want wire time here, not vma
    # inference.  NB in jax 0.10.1 the kwarg is `check_vma`; older docs call it `check_rep`.
    f = jax.jit(jax.shard_map(body, mesh=mesh,
                              in_specs=(PartitionSpec("p"),),
                              out_specs=PartitionSpec("p"), check_vma=False))
    x = jnp.ones((P, n_elem), dtype=jnp.float32)
    jax.block_until_ready(f(x))                       # compile + warm
    best = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(f(x))
        best = min(best, (time.perf_counter() - t0) / ITERS)
    return best


if __name__ == "__main__":
    if proc0:
        print(f"[cscan] devices={P} processes={jax.process_count()} "
              f"nodes={P // GPUS_PER_NODE}", flush=True)
        print(f"[cscan] {'KB/dev':>8s} {'allreduce us':>13s} {'ppermute us':>12s} "
              f"{'allgather us':>13s}", flush=True)
    for kb in SIZES_KB:
        n = kb * KB // 4                               # float32 elements per device
        row = {}
        for kind in ("allreduce", "ppermute", "allgather"):
            try:
                row[kind] = bench(kind, n) * 1e6
            except Exception as e:                     # allgather at P=256 x 4MB is huge
                row[kind] = float("nan")
                if proc0:
                    print(f"[cscan]   {kind} @{kb}KB failed: {type(e).__name__}", flush=True)
        if proc0:
            print(f"[cscan] {kb:8d} {row['allreduce']:13.1f} {row['ppermute']:12.1f} "
                  f"{row['allgather']:13.1f}", flush=True)
    if proc0:
        print("[cscan] done", flush=True)
