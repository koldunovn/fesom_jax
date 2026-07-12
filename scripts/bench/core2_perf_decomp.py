#!/usr/bin/env python
"""CORE2 all-on per-step COST DECOMPOSITION (model-paper perf investigation).

The smoke measured the production all-on config (zstar+TKE+mEVP+GM) at ~520 ms/step on dist_4/1node
— 6x the 87 ms scaling-benchmark, which is base+ice+KPP+GM WITHOUT TKE/zstar. This isolates WHERE
the cost lives by timing the PRODUCTION kernel (run_steps_sharded_forced + cfg.physics_kwargs()) with
one component toggled off at a time (compile-once, time the reused 2nd call ⇒ compile excluded).

Run as a 1-node 4-GPU job (dist_4), single process (NO JDIST). ~25 steps/variant warm timing.
"""
from __future__ import annotations
import dataclasses
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from fesom_jax import surface_forcing, partit, shard_mesh, ssh, ice_evp
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import cold_start_state
from fesom_jax.run_config import load_yaml
from fesom_jax.run import _chunk_dates
from fesom_jax.integrate_sharded import run_steps_sharded_forced

MESH = "data/mesh_core2"
IC = "data/ic_core2"
POOL = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2"
NPES = 4
NSTEPS = 24
YEAR = 1958

cfg = load_yaml("configs/core2_full.yaml")
print(f"[decomp] loaded core2_full.yaml dt={cfg.dt} "
      f"ale={cfg.ale is not None} tke={cfg.tke is not None} gm={cfg.gm is not None} "
      f"ice={cfg.ice is not None} kpp={cfg.kpp is not None}", flush=True)

mesh = load_mesh(MESH)
part = partit.read_partition(Path(POOL), NPES)
sm = shard_mesh.build_sharded_mesh(mesh, part)
sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=cfg.dt), part)
state0 = cold_start_state(mesh, IC, xp=np)
sst0 = np.asarray(state0.T[:, 0])
state_p = shard_mesh.partition_state(state0, part)
forcing = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=sst0)
seq = forcing.stack(_chunk_dates(YEAR, cfg.dt, 0, NSTEPS, cfg.dt_ramp), xp=np)
seq_p = shard_mesh.partition_step_forcing(seq, part)
fs_p = shard_mesh.partition_forcing_static(forcing.static, part)
stress_p = np.zeros((NPES, sm.Lmax["elem"], 2))
bn = np.asarray(ice_evp.boundary_node_mask(mesh))
boundary_node_p = shard_mesh._shard_along_axis(bn, part.myList_nod2D, sm.Lmax["nod"], 0, False)
print(f"[decomp] setup done; npes={NPES} devices={len(jax.devices())} nsteps={NSTEPS}", flush=True)


def time_variant(name, cfgv):
    jfn, jargs, _ = run_steps_sharded_forced(
        sm, state_p, sop, stress_p, seq_p, fs_p, NSTEPS, dt=cfgv.dt, npes=NPES,
        use_ragged=False, boundary_node_p=boundary_node_p, return_executable=True,
        **cfgv.physics_kwargs())
    tc = time.perf_counter()
    jax.block_until_ready(jfn(*jargs))          # 1st call: compile (excluded)
    comp = time.perf_counter() - tc
    t0 = time.perf_counter()
    jax.block_until_ready(jfn(*jargs))          # 2nd call: warm run
    warm = time.perf_counter() - t0
    ms = warm / NSTEPS * 1e3
    print(f"[decomp] {name:22s} per_step={ms:8.2f} ms   (warm {warm:5.2f}s, compile {comp:5.1f}s)",
          flush=True)
    return ms


# Variants: all-on baseline, then one component removed at a time, then a bare baseline.
R = dataclasses.replace
variants = [
    ("all-on (baseline)",   cfg),
    ("minus TKE (->PP)",    R(cfg, tke=None)),
    ("minus zstar(->linfs)", R(cfg, ale=None)),
    ("minus GM",            R(cfg, gm=None)),
    ("minus ice (mEVP)",    R(cfg, ice=None)),
    ("minus TKE+zstar",     R(cfg, tke=None, ale=None)),
    ("bare (all phys off)", R(cfg, tke=None, ale=None, gm=None, ice=None)),
]
res = {}
for name, cfgv in variants:
    cfgv.validate()
    res[name] = time_variant(name, cfgv)

base = res["all-on (baseline)"]
print("\n[decomp] === component cost (all-on minus variant; +ve = that component is expensive) ===",
      flush=True)
for name in ["minus TKE (->PP)", "minus zstar(->linfs)", "minus GM", "minus ice (mEVP)"]:
    print(f"[decomp]   {name:22s} delta = {base - res[name]:8.2f} ms", flush=True)
print(f"[decomp]   bare baseline           = {res['bare (all phys off)']:8.2f} ms "
      f"(cf. 58 ms bench oce base)", flush=True)
print("DECOMP_DONE", flush=True)
