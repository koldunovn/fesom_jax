"""Phase 8b forward-scaling benchmark — time ``run_steps_sharded`` FORWARD (no grad) for
the halo-only ``ragged_all_to_all`` vs the ``all_gather`` exchange, on any exported mesh.

Reports ms/step + throughput (M node-levels / s). ``ragged_all_to_all`` is GPU-only. One
(mesh, npes, halo) combo per process (the OOM/cache-collision lesson) — the sbatch loops them.
Uses a generic perturbed rest-state (``State.rest`` + a smooth lat bump) so the step does real
work (representative CG iteration count) without needing a per-mesh physical IC.

    <env-py> scripts/bench_forward_scaling.py --mesh-dir <dir> --dist-dir <dir> --npes 4 \
        --steps 100 --ragged 1 --name farc
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT = 1800.0


def perturbed_state(mesh):
    """Generic non-trivial State: rest + a smooth lat perturbation of T (so the SSH CG does
    representative work). Mirrors the test helper, mesh-agnostic."""
    from fesom_jax.state import State
    st = State.rest(mesh)
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    bump = 0.5 * np.cos(2 * lat)[:, None]
    T = np.asarray(st.T) + np.where(np.asarray(mesh.node_layer_mask), bump, 0.0)
    return dataclasses.replace(st, T=jnp.asarray(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", default=str(ROOT / "data" / "mesh_core2"))
    ap.add_argument("--dist-dir", default="/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
    ap.add_argument("--name", default="core2")
    ap.add_argument("--npes", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--ragged", type=int, default=0)
    ap.add_argument("--dt", type=float, default=1800.0,
                    help="timestep (s). Kokkos: CORE2 1800, farc 900, dars/NG5 180 (cold-start CFL).")
    args = ap.parse_args()
    DT = args.dt
    jax.config.update("jax_enable_x64", True)

    from fesom_jax import partit, shard_mesh, ssh
    from fesom_jax import integrate_sharded as ish
    from fesom_jax.mesh import load_mesh

    plat = jax.devices()[0].platform
    ndev = len(jax.devices())
    use_ragged = bool(args.ragged)
    if ndev < args.npes:
        print(f"[bench] SKIP {args.name} npes={args.npes}: only {ndev} devices"); return
    if use_ragged and plat == "cpu":
        print(f"[bench] SKIP ragged on CPU (ragged_all_to_all unimplemented)"); return

    mesh = load_mesh(args.mesh_dir)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = perturbed_state(mesh)
    part = partit.read_partition(Path(args.dist_dir), args.npes)
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
    print(f"[bench] mesh={args.name:6s} nod2D={mesh.nod2D} nl={mesh.nl} npes={args.npes} "
          f"halo={tag:9s} steps={args.steps}  per_step={per_step_ms:8.2f} ms  "
          f"throughput={tput:8.1f} Mnodlev/s  plat={plat}")


if __name__ == "__main__":
    main()
