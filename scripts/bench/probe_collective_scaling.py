"""How does a JAX-CPU collective cost scale with rank count, and what does it cost IN-GRAPH?

Two questions the campaign left open.

1. My first probe timed ONE psum per Python call, so 0.877 ms at 32 ranks included dispatch and
   cross-process synchronisation. The CG pays neither — its ~254 collectives per step run inside
   one compiled ``while_loop``. Here a chain of N dependent psums runs inside a SINGLE jit and
   the cost is taken as a difference between chain lengths, so dispatch cancels exactly (the
   same subtraction the step benchmark uses).

2. MPI all-reduces a scalar in O(log P) hops; a naive ring is O(P). If gloo is the latter, the
   per-collective cost grows linearly with ranks — which would explain why CORE2 stops
   improving at 32 ranks while Fortran runs 512 happily. Run this at several rank counts and
   look at the slope.

    mpirun -n <P> ... python probe_collective_scaling.py
"""
from __future__ import annotations

import os
import statistics
import time

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS, NamedSharding

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_cpu_collectives_implementation", os.environ.get("FESOM_CPU_COLL", "gloo"))
jax.distributed.initialize()

P = jax.process_count()
mesh = jax.make_mesh((P,), ("d",))
proc0 = jax.process_index() == 0
INV = 1.0 / P


def chain(n_chain: int, elems: int, kind: str):
    """jit'd chain of `n_chain` DEPENDENT collectives (dependence prevents CSE/reordering).

    UNROLLED rather than a fori_loop: under ``shard_map`` a ``psum`` result no longer varies
    over the device axis, so it cannot be a loop carry of the same type as its input ("scan
    body carry input and output must have equal types"). Straight-line code sidesteps that,
    and 101 ops compile in seconds.
    """
    perm = tuple((i, (i + 1) % P) for i in range(P))

    def step(v):
        if kind == "psum":
            return jax.lax.psum(v, "d") * INV
        return jax.lax.ppermute(v, "d", perm) * 1.0

    def f(x):
        for _ in range(n_chain):
            x = step(x)
        return x

    fn = jax.jit(jax.shard_map(f, mesh=mesh, in_specs=PS("d"), out_specs=PS("d")))
    x = jax.device_put(jnp.ones((P, elems), dtype=jnp.float64),
                       NamedSharding(mesh, PS("d")))
    jax.block_until_ready(fn(x))             # compile
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def per_collective(elems: int, kind: str, short: int = 1, long: int = 101) -> float:
    """ms per in-graph collective, dispatch cancelled by subtraction."""
    t_s, t_l = chain(short, elems, kind), chain(long, elems, kind)
    return (t_l - t_s) / (long - short) * 1e3


for kind, elems, label in (("psum", 1, "psum scalar (CG dot-product)"),
                           ("psum", 4096, "psum 4k f64 = 32 KB"),
                           ("ppermute", 4096, "ppermute 32 KB (one coloured round)")):
    ms = per_collective(elems, kind)
    if proc0:
        print(f"[coll] ranks={P:>3}  {label:<34} {ms:8.4f} ms per in-graph collective",
              flush=True)

if proc0:
    # What the CG actually pays: 127 iterations x (2 psums + 2 halo exchanges).
    ps = per_collective(1, "psum")
    pp = per_collective(4096, "ppermute")
    print(f"[coll] ranks={P}: 127 iters x 2 psums          = {127 * 2 * ps:7.1f} ms/step",
          flush=True)
    print(f"[coll] ranks={P}: 127 iters x 2 halos x 6 rounds = {127 * 2 * 6 * pp:7.1f} ms/step "
          f"(coloured, if K=6)", flush=True)
