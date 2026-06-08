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


def phc_state(mesh, ic_dir=None):
    """REAL PHC3.0 winter IC (the Kokkos IC) as a rest State, with base T_old/S_old (the C
    step-1 AB2 history, per core2_initial_state). Loads a cached T_ic/S_ic.npy from ``ic_dir``
    if present, else interpolates the global PHC nc onto the mesh live (slow for big meshes —
    pre-cache with build_and_cache_ic → /work)."""
    import dataclasses as _dc
    from fesom_jax.state import State
    from fesom_jax import phc_ic
    if ic_dir and (Path(ic_dir) / "T_ic.npy").exists():
        T = jnp.asarray(np.load(Path(ic_dir) / "T_ic.npy"))
        S = jnp.asarray(np.load(Path(ic_dir) / "S_ic.npy"))
    else:
        res = phc_ic.load_phc_ic(mesh)
        T, S = jnp.asarray(res.T), jnp.asarray(res.S)
    mask = mesh.node_layer_mask
    st = State.rest(mesh, T0=10.0, S0=35.0)
    return _dc.replace(st, T=T, S=S, T_old=jnp.where(mask, 10.0, 0.0),
                       S_old=jnp.where(mask, 35.0, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", default=str(ROOT / "data" / "mesh_core2"))
    ap.add_argument("--dist-dir", default="/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
    ap.add_argument("--name", default="core2")
    ap.add_argument("--npes", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5,
                    help="steps excluded via the subtraction method (Kokkos omitted 5).")
    ap.add_argument("--ragged", type=int, default=0)
    ap.add_argument("--dt", type=float, default=1800.0,
                    help="timestep (s). Kokkos: CORE2 1800, farc 900, dars/NG5 180 (cold-start CFL).")
    ap.add_argument("--full", type=int, default=0,
                    help="1 = COMPLETE model (KPP+GM/Redi+prognostic sea-ice + REAL JRA55 forcing + "
                         "PHC IC), like Kokkos; 0 = ocean-only (no ice/forcing).")
    ap.add_argument("--ic-dir", default=None,
                    help="dir with cached T_ic.npy/S_ic.npy (PHC IC); else interpolate live.")
    ap.add_argument("--year", type=int, default=1958, help="JRA55 forcing year (Kokkos used 1958).")
    # component toggles (default on under --full) — to DECOMPOSE the per-step cost
    ap.add_argument("--ice", type=int, default=1)
    ap.add_argument("--kpp", type=int, default=1)
    ap.add_argument("--gm", type=int, default=1)
    args = ap.parse_args()
    DT = args.dt
    import os
    jax.config.update("jax_enable_x64", True)
    if os.environ.get("JDIST"):               # multi-node: 1 process/node, all local GPUs
        jax.distributed.initialize(
            local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))))

    from fesom_jax import partit, shard_mesh, ssh
    from fesom_jax import integrate_sharded as ish
    from fesom_jax.mesh import load_mesh

    proc0 = jax.process_index() == 0
    plat = jax.devices()[0].platform
    ndev = len(jax.devices())
    use_ragged = bool(args.ragged)
    if ndev < args.npes:
        print(f"[bench] SKIP {args.name} npes={args.npes}: only {ndev} devices"); return
    if use_ragged and plat == "cpu":
        print(f"[bench] SKIP ragged on CPU (ragged_all_to_all unimplemented)"); return

    mesh = load_mesh(args.mesh_dir)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    part = partit.read_partition(Path(args.dist_dir), args.npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)

    full_kw = {}
    if args.full:
        from fesom_jax import ice, ice_evp, core2_forcing
        from fesom_jax.ice import IceConfig
        from fesom_jax.kpp import KppConfig
        from fesom_jax.gm import GMConfig
        from fesom_jax.shard_mesh import _shard_along_axis, local_sizes
        # REAL Kokkos setup: PHC winter IC + JRA55 `year` forcing + prognostic ice.
        state = phc_state(mesh, args.ic_dir)
        sst0 = np.asarray(state.T[:, 0])
        state = ice.seed_ice(state, mesh, sst0)                        # seed BEFORE partition
        cf = core2_forcing.build_core_forcing(mesh, args.year, sst_ic=sst0)
        y, d, s, mo = core2_forcing.dates_for_steps(args.year, DT, 1)[0]
        sf, fs = cf.step_forcing(y, d, s, mo), cf.static               # real JRA, run start date
        bn = np.asarray(ice_evp.boundary_node_mask(mesh))
        _, Lmax = local_sizes(part)
        full_kw = dict(ice_cfg=IceConfig() if args.ice else None,
                       kpp_cfg=KppConfig() if args.kpp else None,
                       gm_cfg=GMConfig() if args.gm else None,
                       step_forcing=shard_mesh.partition_step_forcing(sf, part),
                       forcing_static=shard_mesh.partition_forcing_static(fs, part),
                       boundary_node_p=_shard_along_axis(bn, part.myList_nod2D,
                                                         Lmax["nod"], 0, False))
    else:
        state = perturbed_state(mesh)

    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = jnp.zeros((args.npes, sm.Lmax["elem"], 2))

    # Per-step time via the SUBTRACTION method (the JAX analog of Kokkos "omit the first W
    # steps"): time a warm N-step run and a warm W-step run, both with XLA compile EXCLUDED
    # (compile once, time the reused 2nd call), then per_step = (t_N - t_W)/(N - W) — this
    # cancels compile, the fixed per-call dispatch overhead, AND the first-W-step transient
    # (the AB2 is_first_step branch + early spin-up).
    def warm_time(n):
        jfn, jargs, _ = ish.run_steps_sharded(sm, state_p, sop, stress_p, n, dt=DT,
                                              npes=args.npes, use_ragged=use_ragged,
                                              return_executable=True, **full_kw)
        tc = time.perf_counter()
        jax.block_until_ready(jfn(*jargs))        # 1st call: trace + compile + run (excluded)
        comp = time.perf_counter() - tc
        t0 = time.perf_counter()
        jax.block_until_ready(jfn(*jargs))        # 2nd call: reuse executable → pure run
        return time.perf_counter() - t0, comp

    W = max(1, args.warmup)
    t_N, compile_s = warm_time(args.steps)
    t_W, _ = warm_time(W)
    per_step = (t_N - t_W) / (args.steps - W)
    per_step_ms = per_step * 1e3
    tput = mesh.nod2D * mesh.nl / per_step / 1e6   # M node-levels / s
    tag = "ragged" if use_ragged else "allgather"
    if args.full:
        comp = "+".join(c for c, on in [("ice", args.ice), ("kpp", args.kpp), ("gm", args.gm)] if on)
        model = f"FULL({comp or 'oce'}+JRA{args.year})"
    else:
        model = "ocean-only"
    if proc0:
        print(f"[bench] mesh={args.name:6s} nod2D={mesh.nod2D} nl={mesh.nl} npes={args.npes} "
              f"halo={tag:9s} model={model:20s} steps={args.steps}  per_step={per_step_ms:8.2f} ms  "
              f"throughput={tput:8.1f} Mnodlev/s  compile={compile_s:6.1f}s  plat={plat}")


if __name__ == "__main__":
    main()
