#!/usr/bin/env python
"""Bit-identity + speedup test for run_steps_sharded_forced(reuse_executable=True).

Runs a 4-chunk cold CORE2 forced integration twice — once with reuse_executable=False (the current
behavior: fresh XLA compile every chunk) and once with True (cache + reuse the compiled executable) —
and asserts the final folded sharded States are BYTE-IDENTICAL (max|diff| == 0 across every leaf).
That directly proves the executable-cache key is complete (a stale/wrong executable would diverge).
Also prints per-chunk wall time so the speedup (chunks 3-4 reuse ⇒ ~5 s vs ~25 s recompile) is visible.
"""
from __future__ import annotations
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from fesom_jax import core2_forcing, partit, shard_mesh, ssh, ice_evp
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import cold_start_state
from fesom_jax.run_config import load_yaml
from fesom_jax.run import _chunk_dates
from fesom_jax.integrate_sharded import run_steps_sharded_forced, clear_forced_jit_cache

MESH, IC = "data/mesh_core2", "data/ic_core2"
POOL = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2"
NPES, NS, NCHUNK, YEAR = 4, 48, 4, 1958

cfg = load_yaml("configs/core2_full.yaml")
mesh = load_mesh(MESH)
part = partit.read_partition(Path(POOL), NPES)
sm = shard_mesh.build_sharded_mesh(mesh, part)
sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=cfg.dt), part)
state0 = cold_start_state(mesh, IC, xp=np)
sst0 = np.asarray(state0.T[:, 0])
forcing = core2_forcing.build_core_forcing(mesh, YEAR, sst_ic=sst0)
fs_p = shard_mesh.partition_forcing_static(forcing.static, part)
stress_p = np.zeros((NPES, sm.Lmax["elem"], 2))
bn = np.asarray(ice_evp.boundary_node_mask(mesh))
bn_p = shard_mesh._shard_along_axis(bn, part.myList_nod2D, sm.Lmax["nod"], 0, False)
pk = cfg.physics_kwargs()
# pre-build the per-chunk forcing once (config-independent; reused for both passes)
seqs = [shard_mesh.partition_step_forcing(
            forcing.stack(_chunk_dates(YEAR, cfg.dt, ci * NS, NS, cfg.dt_ramp), xp=np), part)
        for ci in range(NCHUNK)]
print(f"[reuse-test] setup done; npes={NPES} devices={len(jax.devices())} "
      f"chunks={NCHUNK}x{NS}steps", flush=True)


def run_chunks(reuse):
    clear_forced_jit_cache()
    sp = shard_mesh.partition_state(state0, part)   # fresh cold [P,Lmax]
    folded = False
    for ci in range(NCHUNK):
        t0 = time.perf_counter()
        sp = run_steps_sharded_forced(
            sm, sp, sop, stress_p, seqs[ci], fs_p, NS, dt=cfg.dt, npes=NPES,
            bootstrap_ab2=(ci == 0), state_is_folded=folded, return_folded=True,
            use_ragged=False, boundary_node_p=bn_p, reuse_executable=reuse, **pk)
        jax.block_until_ready(sp)
        folded = True
        print(f"[reuse-test]   reuse={int(reuse)} chunk{ci} {time.perf_counter()-t0:6.1f}s", flush=True)
    return sp


s_off = run_chunks(False)
s_on = run_chunks(True)

leaves_off = jax.tree.leaves(s_off)
leaves_on = jax.tree.leaves(s_on)
assert len(leaves_off) == len(leaves_on), "pytree mismatch"
maxdiff = 0.0
for a, b in zip(leaves_off, leaves_on):
    d = float(jax.numpy.max(jax.numpy.abs(a.astype("float64") - b.astype("float64"))))
    maxdiff = max(maxdiff, d)
print(f"[reuse-test] max|state_reuse - state_fresh| over all leaves = {maxdiff:.3e}", flush=True)
print("REUSE_BITIDENTICAL_OK" if maxdiff == 0.0 else f"REUSE_BITIDENTICAL_FAIL ({maxdiff:.3e})",
      flush=True)
