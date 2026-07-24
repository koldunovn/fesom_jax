#!/usr/bin/env python
"""Measure JUPITER's inter-node collective bandwidth/latency directly, with the SAME stack the
model uses (jax.distributed + NCCL), so a fabric claim stops being a hypothesis.

    srun -N<n> -n<n> --ntasks-per-node=1 python scripts/bench/jupiter/fabric_probe.py

Why: the campaign found the `padded` halo transport ~4x slower than `coloured` at 8 GPUs from
~23:00 on 2026-07-23, while an earlier run measured them equal (46.2 vs 51.1 ms).  Padded is
bandwidth-bound and coloured is not, so "inter-node bandwidth dropped" explains it — but that
was inferred from model timings, never measured.  This measures it.

Reports, per message size, the ALGORITHM bandwidth of an all-reduce over the full device set
(and, for reference, the small-message time = latency floor).  Compare against JUPITER's
per-node injection: 4 x NDR200 NICs = 4 x 200 Gb/s = 100 GB/s/node.  A healthy fabric should
reach a substantial fraction of that at large messages; a routing-degraded one will not.
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

# float32 keeps the byte count trivial to reason about; x64 is irrelevant for a wire test.
SIZES_MB = [0.004, 0.064, 1, 4, 16, 64, 256]
REPS = 20

if proc0:
    print(f"[fabric] devices={P} processes={jax.process_count()} "
          f"({P // max(1, jax.process_count())}/proc)", flush=True)
    print(f"[fabric] {'MB':>9s} {'allreduce ms':>13s} {'alg GB/s':>9s} {'bus GB/s':>9s}",
          flush=True)


def make_allreduce(n_elem: int):
    """all-reduce of an [P, n] array over the device axis, under shard_map."""
    def body(x):
        return lax.psum(x, "p")
    f = jax.shard_map(body, mesh=mesh, in_specs=(PartitionSpec("p"),),
                      out_specs=PartitionSpec("p"))
    return jax.jit(f)


for mb in SIZES_MB:
    n = max(1, int(mb * 1e6 / 4))                      # float32 elements PER DEVICE
    x = jnp.ones((P, n), dtype=jnp.float32)            # sharded on the leading axis
    f = make_allreduce(n)
    y = jax.block_until_ready(f(x))                    # compile + warm
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = f(x)
    jax.block_until_ready(y)
    dt = (time.perf_counter() - t0) / REPS
    nbytes = n * 4                                     # per device
    # algorithm bw = bytes moved per device / time; bus bw uses the ring factor 2(P-1)/P
    alg = nbytes / dt / 1e9
    bus = alg * 2 * (P - 1) / P
    if proc0:
        print(f"[fabric] {nbytes/1e6:9.3f} {dt*1e3:13.4f} {alg:9.2f} {bus:9.2f}", flush=True)

if proc0:
    print("[fabric] reference: JUPITER node injection = 4 x NDR200 = 100 GB/s/node "
          "(4 GPUs/node, so ~25 GB/s per GPU at full rate)", flush=True)
    print("[fabric] done", flush=True)
