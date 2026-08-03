"""Raw collective cost on the CPU backend — the number needed to judge whether the SSH CG
solver is latency-bound.

The solver does ~1 halo exchange + 2 dot-product all-reduces per CG iteration, so per-step cost
carries (iterations x collectives) latencies. If a small psum costs ~1 ms at 32 ranks, ~127
iterations alone account for several hundred ms of a ~990 ms step, and CORE2-on-CPU is latency-
bound rather than compute-bound.

Small psum = latency floor; 1 MB psum = bandwidth. Launch one process per device, as the
campaign does.
"""
from __future__ import annotations

import os
import statistics
import time

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_cpu_collectives_implementation", os.environ.get("FESOM_CPU_COLL", "gloo"))
jax.distributed.initialize()

n = jax.process_count()
mesh = jax.make_mesh((n,), ("d",))
proc0 = jax.process_index() == 0


def time_psum(elems_per_rank: int, reps: int = 50) -> tuple[float, float]:
    """Median and min wall time (ms) of one psum over `elems_per_rank` float64 per rank."""
    x = jnp.ones((n, elems_per_rank), dtype=jnp.float64)
    gx = jax.device_put(x, NamedSharding(mesh, P("d")))
    f = jax.jit(jax.shard_map(lambda a: jax.lax.psum(a, "d"), mesh=mesh,
                              in_specs=P("d"), out_specs=P()))
    jax.block_until_ready(f(gx))                      # compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(gx))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


for elems, label in ((1, "scalar (latency floor)"),
                     (1 << 17, "1 MiB/rank (bandwidth)")):
    med, lo = time_psum(elems)
    if proc0:
        print(f"psum {label:<24} ranks={n}  median={med:8.3f} ms  min={lo:8.3f} ms", flush=True)

if proc0:
    # What that implies for a CG solve of the typical length.
    med, _ = time_psum(1)
    for iters in (127, 42):
        print(f"=> {iters} CG iterations x 3 collectives = {3 * iters * med:8.1f} ms of pure "
              f"collective latency per step", flush=True)
