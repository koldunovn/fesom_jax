#!/usr/bin/env python
"""Micro-benchmark ONE halo exchange per transport — the diagnostic the full-model A/B cannot give.

The full model mixes two very different exchange populations:

  * the CG's **2-D** ssh exchange — tiny payload, ~2 per iteration x ~127 iterations, so the
    DOMINANT exchange by COUNT. Cost here is per-collective latency, not bytes.
  * the **3-D** tracer/momentum exchanges — [Lmax, nl], far fewer, so cost is bytes.

The coloured transport trades ONE fused collective for K (= max #neighbours) ppermute rounds in
exchange for 10-40x less wire volume. Which way that trade goes therefore depends entirely on
which population you are looking at — and a single per-step number cannot tell you. This script
times the exchange alone, at both ranks, so the latency/bandwidth crossover is visible directly:

    python bench_halo_micro.py --mesh-dir ... --dist-dir ... --npes 16 --halo padded,coloured,ragged

Reports us/exchange for a [Lmax] (2-D) and a [Lmax, nl] (3-D) field, plus the wire volume each
transport actually ships (from the maps, not a guess).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import PartitionSpec                         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--dist-dir", required=True)
    ap.add_argument("--npes", type=int, required=True)
    ap.add_argument("--halo", default="padded,coloured,ragged,allgather")
    ap.add_argument("--kind", default="nod", choices=["nod", "elem", "edge"])
    ap.add_argument("--reps", type=int, default=200, help="exchanges per timed call")
    ap.add_argument("--iters", type=int, default=5, help="timed calls (min is reported)")
    args = ap.parse_args()

    import os
    jax.config.update("jax_enable_x64", True)
    if os.environ.get("JDIST"):
        jax.distributed.initialize(
            local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))))

    from fesom_jax import halo, partit, shard_mesh                        # noqa: E402
    from fesom_jax.mesh import load_mesh                                  # noqa: E402

    proc0 = jax.process_index() == 0
    if len(jax.devices()) < args.npes:
        if proc0:
            print(f"[micro] SKIP: need {args.npes} devices, have {len(jax.devices())}")
        return

    part = partit.read_partition(Path(args.dist_dir), args.npes)
    sm = shard_mesh.build_sharded_mesh(load_mesh(args.mesh_dir), part)
    k, P, nl = args.kind, args.npes, sm.nl
    Lmax = sm.Lmax[k]
    jmesh = halo.device_mesh(devices=jax.devices()[:P])
    spec = PartitionSpec("p")
    rex, cex = sm.exchange_ragged[k], sm.exchange_coloured[k]
    sd, sl = sm.exchange[k]

    # wire volume per device per exchange, in LANES (the probe_pad_factor.py quantities)
    lanes = {"allgather": P * Lmax,                    # every device receives the whole global field
             "ragged": rex.send_max,
             "padded": P * rex.pad_slot,
             "coloured": cex.total}
    K = len(cex.perms)

    def fold(a):
        a = jnp.asarray(np.asarray(a))
        return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])

    def make(mode, ndim):
        """A shard_map'd chain of `reps` exchanges (chained so XLA cannot elide them)."""
        rest = () if ndim == 2 else (nl,)
        f0 = jnp.asarray(np.random.default_rng(0).standard_normal((P, Lmax) + rest))
        args_ = [fold(f0)]
        if mode == "allgather":
            args_ += [fold(sd).astype(jnp.int32), fold(sl).astype(jnp.int32)]

            def body(f, sdv, slv):
                for _ in range(args.reps):
                    f = halo.halo_exchange(f, sdv, slv, "p")
                return f
        elif mode == "ragged":
            rm = {n: fold(getattr(rex, a)).astype(t) for n, a, t in [
                ("send_idx", "send_idx", jnp.int32), ("send_sizes", "send_sizes", jnp.int32),
                ("send_off", "send_offsets", jnp.int32), ("out_off", "out_offsets", jnp.int32),
                ("recv_sizes", "recv_sizes", jnp.int32), ("recv_gather", "recv_gather", jnp.int32)]}
            rm["halo_mask"] = fold(rex.halo_mask)
            keys = list(rm)
            args_ += [rm[n] for n in keys]
            rmax = int(rex.recv_max)

            def body(f, *vals):
                r = dict(zip(keys, vals))
                for _ in range(args.reps):
                    f = halo.halo_exchange_ragged(f, r, rmax, "p")
                return f
        elif mode == "padded":
            keys = ["pad_src", "pad_valid", "pad_slotpos", "halo_mask"]
            vals = [fold(rex.pad_src).astype(jnp.int32), fold(rex.pad_valid),
                    fold(rex.pad_slotpos).astype(jnp.int32), fold(rex.halo_mask)]
            args_ += vals

            def body(f, *vs):
                r = dict(zip(keys, vs))
                for _ in range(args.reps):
                    f = halo.halo_exchange_padded(f, r, "p")
                return f
        else:                                            # coloured
            keys = ["send_idx", "send_valid", "colpos", "halo_mask"]
            vals = [fold(cex.send_idx).astype(jnp.int32), fold(cex.send_valid),
                    fold(cex.colpos).astype(jnp.int32), fold(rex.halo_mask)]
            args_ += vals
            meta = (cex.perms, cex.slots, cex.offs)

            def body(f, *vs):
                c = dict(zip(keys, vs))
                for _ in range(args.reps):
                    f = halo.halo_exchange_coloured(f, c, meta, "p")
                return f

        fn = jax.jit(jax.shard_map(body, mesh=jmesh, in_specs=(spec,) * len(args_),
                                   out_specs=spec))
        return fn, args_

    if proc0:
        print(f"[micro] {Path(args.mesh_dir).name} kind={k} npes={P} Lmax={Lmax:,} nl={nl} "
              f"K={K} rounds  ({args.reps} exchanges/call, min of {args.iters})", flush=True)
        print(f"{'transport':>10} {'lanes/exch':>11} {'2-D us/exch':>12} {'3-D us/exch':>12} "
              f"{'3-D MB/exch':>12}")
    for mode in args.halo.split(","):
        row = {}
        for ndim in (2, 3):
            fn, a = make(mode, ndim)
            jax.block_until_ready(fn(*a))                       # compile
            best = min(_time(fn, a) for _ in range(args.iters))
            row[ndim] = best / args.reps * 1e6                  # us per exchange
        if proc0:
            mb = lanes[mode] * nl * 8 / 1e6
            print(f"{mode:>10} {lanes[mode]:>11,} {row[2]:>12.1f} {row[3]:>12.1f} {mb:>12.2f}",
                  flush=True)


def _time(fn, a) -> float:
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*a))
    return time.perf_counter() - t0


if __name__ == "__main__":
    main()
