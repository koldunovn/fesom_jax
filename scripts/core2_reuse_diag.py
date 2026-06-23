#!/usr/bin/env python
"""Diagnose the executable-reuse bit-identity failure + measure the persistent-compile-cache fallback.

(A) Per-leaf divergence: run 4 chunks reuse=False vs reuse=True, print which State field diverges and
    by how much (points at the missing cache-key determinant).
(B) Persistent XLA compilation cache: re-run 4 chunks reuse=False with jax_compilation_cache_dir set to
    a fresh dir; chunks with a repeated HLO signature should hit the disk cache (faster) — and are
    bit-identical by construction (XLA owns the cache). Reports per-chunk wall time.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

MODE = sys.argv[1] if len(sys.argv) > 1 else "diag"     # "diag" or "pcache"
if MODE == "pcache":
    CACHE_DIR = "/scratch/a/a270088/tmp/jax_pcache_test"
    import shutil, os
    shutil.rmtree(CACHE_DIR, ignore_errors=True); os.makedirs(CACHE_DIR, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    print(f"[diag] persistent compile cache ON -> {CACHE_DIR}", flush=True)

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
print(f"[diag] setup done; npes={NPES} chunks={NCHUNK}x{NS}", flush=True)


def _seq(ci):  # build a FRESH per-chunk forcing stack (matches run.py; rules out shared-buffer artifacts)
    return shard_mesh.partition_step_forcing(
        forcing.stack(_chunk_dates(YEAR, cfg.dt, ci * NS, NS, cfg.dt_ramp), xp=np), part)


def run_chunks(reuse, label):
    clear_forced_jit_cache()
    sp = shard_mesh.partition_state(state0, part)
    folded = False
    for ci in range(NCHUNK):
        t0 = time.perf_counter()
        sp = run_steps_sharded_forced(
            sm, sp, sop, stress_p, _seq(ci), fs_p, NS, dt=cfg.dt, npes=NPES,
            bootstrap_ab2=(ci == 0), state_is_folded=folded, return_folded=True,
            use_ragged=False, boundary_node_p=bn_p, reuse_executable=reuse, **pk)
        jax.block_until_ready(sp)
        folded = True
        print(f"[diag]   {label} chunk{ci} {time.perf_counter()-t0:6.1f}s", flush=True)
    return sp


def _per_leaf(sa, sb, tag):
    pa = jax.tree_util.tree_flatten_with_path(sa)[0]
    pb = jax.tree_util.tree_flatten_with_path(sb)[0]
    worst = []
    for (kp, a), (_, b) in zip(pa, pb):
        d = float(jax.numpy.max(jax.numpy.abs(a.astype("float64") - b.astype("float64"))))
        worst.append((d, jax.tree_util.keystr(kp)))
    worst.sort(reverse=True)
    print(f"[diag] per-leaf max|{tag}| (top 12):", flush=True)
    for d, name in worst[:12]:
        print(f"[diag]   {d:12.4e}  {name}", flush=True)
    return worst[0][0]


if MODE == "det":
    # DETERMINISM: run the SAME fresh-compile path twice. 0 ⇒ deterministic (so reuse-divergence is a
    # real cache bug); >0 ⇒ the multi-GPU forced path is non-deterministic (bit-identity is the wrong test).
    a = run_chunks(False, "detA")
    b = run_chunks(False, "detB")
    mx = _per_leaf(a, b, "freshA - freshB")
    print("DET_IDENTICAL_OK" if mx == 0.0 else f"DET_NONDETERMINISTIC ({mx:.3e})", flush=True)
    print("DET_DONE", flush=True)
elif MODE == "pcache":
    run_chunks(False, "pcache")     # reuse=False but persistent disk cache on
    print("PCACHE_DONE", flush=True)
else:
    s_off = run_chunks(False, "fresh")
    s_on = run_chunks(True, "reuse")
    paths_off = jax.tree_util.tree_flatten_with_path(s_off)[0]
    paths_on = jax.tree_util.tree_flatten_with_path(s_on)[0]
    worst = []
    for (kp, a), (_, b) in zip(paths_off, paths_on):
        d = float(jax.numpy.max(jax.numpy.abs(a.astype("float64") - b.astype("float64"))))
        name = jax.tree_util.keystr(kp)
        worst.append((d, name))
    worst.sort(reverse=True)
    print("[diag] per-leaf max|reuse - fresh| (top 12):", flush=True)
    for d, name in worst[:12]:
        print(f"[diag]   {d:12.4e}  {name}", flush=True)
    print("DIAG_DONE", flush=True)
