"""Phase 8b forward-scaling benchmark — time ``run_steps_sharded`` FORWARD (no grad) for
the halo-only ``ragged_all_to_all`` vs the ``all_gather`` exchange, at a given device count.

Reports ms/step + throughput (M node-levels / s). ``ragged_all_to_all`` is GPU-only. One
(npes, halo) combo per process (the OOM/cache-collision lesson) — the sbatch loops combos.

    <env-py> scripts/bench_forward_scaling.py --npes 4 --steps 100 --ragged 1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
DT = 1800.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--ragged", type=int, default=0)
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)

    from fesom_jax import partit, shard_mesh, ssh
    from fesom_jax import integrate_sharded as ish
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import core2_initial_state

    plat = jax.devices()[0].platform
    ndev = len(jax.devices())
    use_ragged = bool(args.ragged)
    if ndev < args.npes:
        print(f"[bench] SKIP npes={args.npes}: only {ndev} devices"); return
    if use_ragged and plat == "cpu":
        print(f"[bench] SKIP ragged on CPU (ragged_all_to_all unimplemented)"); return

    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = core2_initial_state(mesh, IC_DIR)
    part = partit.read_partition(CORE2_DIST, args.npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = jnp.zeros((args.npes, sm.Lmax["elem"], 2))

    def run(n):
        return ish.run_steps_sharded(sm, state_p, sop, stress_p, n, dt=DT,
                                     npes=args.npes, use_ragged=use_ragged)

    jax.block_until_ready(run(2))                 # warm up / compile (not timed)
    t0 = time.perf_counter()
    jax.block_until_ready(run(args.steps))
    elapsed = time.perf_counter() - t0

    per_step_ms = elapsed / args.steps * 1e3
    tput = mesh.nod2D * mesh.nl / (elapsed / args.steps) / 1e6   # M node-levels / s
    tag = "ragged" if use_ragged else "allgather"
    print(f"[bench] mesh=core2 nod2D={mesh.nod2D} nl={mesh.nl} npes={args.npes} "
          f"halo={tag:9s} steps={args.steps}  per_step={per_step_ms:8.2f} ms  "
          f"throughput={tput:8.1f} Mnodlev/s  plat={plat}")


if __name__ == "__main__":
    main()
