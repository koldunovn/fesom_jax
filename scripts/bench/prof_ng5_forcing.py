#!/usr/bin/env python
"""Profile the NG5 per-chunk HOST forcing build (Task B2 efficiency diagnosis).

The R0 timing showed ~110 s/chunk (48 steps) of HOST work — half the wall-clock, GPUs idle.
This isolates WHERE that goes (pure numpy ⇒ runs on a CPU compute node, no GPUs) so the R1
fix is measured, not guessed: ``cf.stack`` (JRA interp to the 7.4 M global, redundant on every
node) vs ``partition_step_forcing`` (the [n_steps, nod2D] → [P, n_steps, Lmax] sharding gather).
"""
import time
from pathlib import Path

import numpy as np

from fesom_jax import surface_forcing, partit, shard_mesh
from fesom_jax.mesh import load_mesh

MESH = "/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5"
POOL = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5"
NPES = 64
CHUNK = 48


def lap(label, t0):
    print(f"  {label}: {time.perf_counter() - t0:.2f}s", flush=True)
    return time.perf_counter()


print(f"=== NG5 forcing host-build profile (npes={NPES}, chunk={CHUNK}) ===", flush=True)
t = time.perf_counter()
mesh = load_mesh(MESH); t = lap("load_mesh", t)
part = partit.read_partition(Path(POOL), NPES); t = lap(f"read_partition dist_{NPES}", t)
cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=None); t = lap("build_surface_forcing", t)

dates = surface_forcing.dates_for_steps(1958, 180.0, CHUNK)

# single step (the JRA interp + np.asarray for ~10 fields over 7.4 M nodes)
t = time.perf_counter()
_ = cf.step_forcing(*dates[0], xp=np)
print(f"  single step_forcing: {time.perf_counter() - t:.3f}s  (x{CHUNK} = "
      f"{CHUNK * (time.perf_counter() - t):.1f}s if linear)", flush=True)

# the two halves of the per-chunk HOST cost
t = time.perf_counter()
seq = cf.stack(dates, xp=np)
t_stack = time.perf_counter() - t
print(f"  cf.stack({CHUNK}) [global JRA interp, host]: {t_stack:.1f}s", flush=True)

t = time.perf_counter()
seq_p = shard_mesh.partition_step_forcing(seq, part)
t_part = time.perf_counter() - t
print(f"  partition_step_forcing [sharding gather]: {t_part:.1f}s", flush=True)

print(f"=== HOST per-chunk total = {t_stack + t_part:.1f}s "
      f"(stack {100 * t_stack / (t_stack + t_part):.0f}% / partition "
      f"{100 * t_part / (t_stack + t_part):.0f}%) ===", flush=True)
print(f"  stack leaf shape {np.asarray(seq.Tair).shape}, "
      f"partitioned {np.asarray(seq_p.Tair).shape}", flush=True)
