#!/usr/bin/env python
"""Measure the CORE2 per-chunk forcing cost: interpolation vs device_put, and whether splitting the
interpolation across CPU cores (threads) actually parallelizes (the multi-core host-forcing probe).

(1) time forcing.stack (JRA bilinear interp, host numpy) vs partition_step_forcing (scatter + device_put).
(2) build M sub-readers on node subsets (the LocalForcing mechanism) and time serial-M vs threaded-M
    vs the global single reader ⇒ does the interp release the GIL enough for thread parallelism?
"""
from __future__ import annotations
import os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from fesom_jax import core2_forcing, partit, shard_mesh, jra55, sss_runoff
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import cold_start_state
from fesom_jax.run_config import load_yaml
from fesom_jax.run import _chunk_dates
from fesom_jax.forcing_local import _SubMesh

POOL = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2"
NS, YEAR, REPS = 48, 1958, 4

# numpy / BLAS threading visibility
try:
    from threadpoolctl import threadpool_info
    print("[meas] threadpool:", [(d["internal_api"], d["num_threads"]) for d in threadpool_info()], flush=True)
except Exception as e:
    print(f"[meas] threadpoolctl n/a ({e}); OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
print(f"[meas] os.cpu_count()={os.cpu_count()}", flush=True)

cfg = load_yaml("configs/core2_full.yaml")
mesh = load_mesh("data/mesh_core2")
part = partit.read_partition(Path(POOL), 4)
state0 = cold_start_state(mesh, "data/ic_core2", xp=np)
sst0 = np.asarray(state0.T[:, 0])
forcing = core2_forcing.build_core_forcing(mesh, YEAR, sst_ic=sst0)
dates = _chunk_dates(YEAR, cfg.dt, 0, NS, cfg.dt_ramp)
print(f"[meas] nod2D={mesh.nod2D} npes=4 NS={NS}", flush=True)


def _t(fn, reps=REPS, warm=1):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


# (1) interp vs device_put split
seq = forcing.stack(dates, xp=np)
t_stack = _t(lambda: forcing.stack(dates, xp=np))


def _part():
    sp = shard_mesh.partition_step_forcing(seq, part)
    jax.block_until_ready(getattr(sp, sp._fields[0]))
    return sp


t_part = _t(_part)
print(f"\n[meas] === per-chunk forcing split (48 steps) ===", flush=True)
print(f"[meas]   forcing.stack (interp, host)      = {t_stack*1e3:7.0f} ms  ({t_stack/(t_stack+t_part)*100:.0f}%)", flush=True)
print(f"[meas]   partition_step_forcing (scatter+dput) = {t_part*1e3:7.0f} ms  ({t_part/(t_stack+t_part)*100:.0f}%)", flush=True)
print(f"[meas]   total host build                  = {(t_stack+t_part)*1e3:7.0f} ms = {(t_stack+t_part)/NS*1e3:.0f} ms/step", flush=True)

# (2) multi-core interpolation probe: split nodes across M sub-readers
for M in (4, 8):
    subsets = np.array_split(np.arange(mesh.nod2D), M)
    subcfs = []
    for s in subsets:
        sub = _SubMesh(mesh, s)
        chl = sss_runoff.build_chl_clim(sub, sss_runoff.DEFAULT_CHL_PATH)
        cf = core2_forcing.CoreForcing(
            jra=jra55.JRA55Reader(sub, YEAR, jra55.DEFAULT_JRA_DIR),
            sss=sss_runoff.build_reader(sub, sss_runoff.DEFAULT_SSS_PATH, sss_runoff.DEFAULT_RUNOFF_PATH),
            chl_clim=chl, static=forcing.static)
        subcfs.append(cf)

    def _serial():
        return [cf.stack(dates, xp=np) for cf in subcfs]

    def _threaded():
        with ThreadPoolExecutor(max_workers=M) as ex:
            return list(ex.map(lambda cf: cf.stack(dates, xp=np), subcfs))

    t_ser = _t(_serial, reps=3)
    t_thr = _t(_threaded, reps=3)
    print(f"\n[meas] === M={M} sub-readers (interp only) ===", flush=True)
    print(f"[meas]   serial   {M} subsets = {t_ser*1e3:7.0f} ms  (≈ the global stack {t_stack*1e3:.0f} ms)", flush=True)
    print(f"[meas]   threaded {M} subsets = {t_thr*1e3:7.0f} ms  (speedup {t_ser/t_thr:.1f}x; ideal {M}x)", flush=True)

print("\nMEASURE_DONE", flush=True)
