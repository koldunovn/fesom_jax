"""Phase 8b forward-scaling benchmark — time ``run_steps_sharded`` FORWARD (no grad) for
the halo-only ``ragged_all_to_all`` vs the ``all_gather`` exchange, on any exported mesh.

Reports ms/step + throughput (M node-levels / s). ``ragged_all_to_all`` is GPU-only. One
(mesh, npes, halo) combo per process (the OOM/cache-collision lesson) — the sbatch loops them.
Uses a generic perturbed rest-state (``State.rest`` + a smooth lat bump) so the step does real
work (representative CG iteration count) without needing a per-mesh physical IC.

    <env-py> scripts/bench/bench_forward_scaling.py --mesh-dir <dir> --dist-dir <dir> --npes 4 \
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

ROOT = Path(__file__).resolve().parents[2]
DT = 1800.0


def perturbed_state(mesh):
    """Generic non-trivial State: rest + a smooth lat perturbation of T (so the SSH CG does
    representative work). Mirrors the test helper, mesh-agnostic. **Built entirely on the HOST
    (numpy)** (Phase 8b B.3) — the global IC for dars/NG5 never materializes on GPU 0;
    ``partition_state`` + ``device_put`` place only each device's shard."""
    from fesom_jax.state import State
    st = State.rest(mesh, xp=np)
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    bump = 0.5 * np.cos(2 * lat)[:, None]
    T = np.asarray(st.T) + np.where(np.asarray(mesh.node_layer_mask), bump, 0.0)
    return dataclasses.replace(st, T=T)


def phc_state(mesh, ic_dir=None):
    """REAL PHC3.0 winter IC (the Kokkos IC) as a rest State, with base T_old/S_old (the C
    step-1 AB2 history, per phc_initial_state). Loads a cached T_ic/S_ic.npy from ``ic_dir``
    if present, else interpolates the global PHC nc onto the mesh live (slow for big meshes —
    pre-cache with build_and_cache_ic → /work). **Built entirely on the HOST (numpy)** (Phase
    8b B.3) so the global 3-D IC never lands on GPU 0 before sharding (the dars/NG5 setup-OOM)."""
    import dataclasses as _dc
    from fesom_jax.state import State
    from fesom_jax import phc_ic
    if ic_dir and (Path(ic_dir) / "T_ic.npy").exists():
        T = np.load(Path(ic_dir) / "T_ic.npy")
        S = np.load(Path(ic_dir) / "S_ic.npy")
    else:
        res = phc_ic.load_phc_ic(mesh)
        T, S = np.asarray(res.T), np.asarray(res.S)
    mask = np.asarray(mesh.node_layer_mask)
    st = State.rest(mesh, T0=10.0, S0=35.0, xp=np)
    return _dc.replace(st, T=T, S=S, T_old=np.where(mask, 10.0, 0.0),
                       S_old=np.where(mask, 35.0, 0.0))


def _gpu_peak_gb():
    """Max peak GPU bytes-in-use across local devices (GiB), or -1 on CPU / if unavailable.
    Phase 8b B.3 setup-OOM probe: confirms the host-build keeps GPU 0 below the OOM ceiling."""
    try:
        peaks = [d.memory_stats().get("peak_bytes_in_use", 0) for d in jax.local_devices()]
        return max(peaks) / (1024 ** 3) if peaks else -1.0
    except Exception:
        return -1.0


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
    ap.add_argument("--halo", choices=["allgather", "ragged", "padded", "coloured"],
                    default=None,
                    help="halo transport (overrides --ragged): allgather (oracle, any backend), "
                         "ragged (GPU-only, forward-only), padded (Phase 8c slot-padded dense "
                         "all_to_all — any backend, AD-correct), coloured (Phase 8d ppermute "
                         "rounds — any backend, AD-correct, no pad factor). Default: "
                         "ragged if --ragged 1, else allgather.")
    ap.add_argument("--dt", type=float, default=1800.0,
                    help="timestep (s). Kokkos: CORE2 1800, farc 900, dars/NG5 180 (cold-start CFL).")
    ap.add_argument("--full", type=int, default=0,
                    help="1 = COMPLETE model (KPP+GM/Redi+prognostic sea-ice + REAL JRA55 forcing + "
                         "PHC IC), like Kokkos; 0 = ocean-only (no ice/forcing).")
    ap.add_argument("--ic-dir", default=None,
                    help="dir with cached T_ic.npy/S_ic.npy (PHC IC); else interpolate live.")
    ap.add_argument("--year", type=int, default=1958, help="JRA55 forcing year (Kokkos used 1958).")
    ap.add_argument("--out-zarr", default=None,
                    help="if set, write the final State to this Zarr store — each GPU writes its "
                         "own shard in parallel (no gather), the scaling output path for NG5.")
    # component toggles (default on under --full) — to DECOMPOSE the per-step cost
    ap.add_argument("--ice", type=int, default=1)
    ap.add_argument("--kpp", type=int, default=1)
    ap.add_argument("--gm", type=int, default=1)
    # production-physics variants (paper §5): defaults 0 keep every existing invocation
    # byte-identical (model tag + configs unchanged).
    ap.add_argument("--mevp", type=int, default=0, help="ice rheology whichEVP=1 (mEVP)")
    ap.add_argument("--tke", type=int, default=0, help="cvmix-TKE mixing (forces kpp off)")
    ap.add_argument("--zstar", type=int, default=0, help="zstar ALE (default linfs)")
    ap.add_argument("--wsplit", type=int, default=0,
                    help="vertical-velocity splitting (implies zstar; dars production)")
    args = ap.parse_args()
    DT = args.dt
    import os
    jax.config.update("jax_enable_x64", True)
    if os.environ.get("JDIST_CPU"):           # multi-PROCESS CPU: 1 device/process, gloo
        # collectives (the topology that beats in-process fake devices ~1.7x AND is the
        # only way past the XLA:CPU all_gather rendezvous crash at >=32 in-process
        # devices — docs/PARALLELISM.md). Launch: srun -n <npes> with
        # XLA_FLAGS=--xla_force_host_platform_device_count=1 per process.
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
        jax.distributed.initialize()          # SLURM auto-detect (srun sets the env)
    elif os.environ.get("JDIST"):             # multi-node: 1 process/node, all local GPUs
        jax.distributed.initialize(
            local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))))

    from fesom_jax import partit, shard_mesh, ssh
    from fesom_jax import integrate_sharded as ish
    from fesom_jax.mesh import load_mesh

    proc0 = jax.process_index() == 0
    plat = jax.devices()[0].platform
    ndev = len(jax.devices())
    halo_mode = args.halo or ("ragged" if args.ragged else "allgather")
    use_ragged = halo_mode == "ragged"
    use_padded = halo_mode == "padded"
    use_coloured = halo_mode == "coloured"
    if ndev < args.npes:
        print(f"[bench] SKIP {args.name} npes={args.npes}: only {ndev} devices"); return
    if use_ragged and plat == "cpu":
        print(f"[bench] SKIP ragged on CPU (ragged_all_to_all unimplemented; "
              f"use --halo padded)"); return

    mesh = load_mesh(args.mesh_dir)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    if args.npes == 1:
        # single-device baseline (1 GPU / 1 CPU node): the identity partition, zero halo —
        # no dist_1 needed; the sharded path reduces to the dense model (S.2 no-op invariant).
        part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    else:
        part = partit.read_partition(Path(args.dist_dir), args.npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)

    full_kw = {}
    if args.full:
        from fesom_jax import ice, ice_evp, surface_forcing
        from fesom_jax.ice import IceConfig
        from fesom_jax.kpp import KppConfig
        from fesom_jax.gm import GMConfig
        from fesom_jax.shard_mesh import _shard_along_axis, local_sizes
        # REAL Kokkos setup: PHC winter IC + JRA55 `year` forcing + prognostic ice.
        state = phc_state(mesh, args.ic_dir)
        sst0 = np.asarray(state.T[:, 0])
        state = ice.seed_ice(state, mesh, sst0)                        # seed BEFORE partition
        cf = surface_forcing.build_surface_forcing(mesh, args.year, sst_ic=sst0)
        y, d, s, mo = surface_forcing.dates_for_steps(args.year, DT, 1)[0]
        sf, fs = cf.step_forcing(y, d, s, mo), cf.static               # real JRA, run start date
        bn = np.asarray(ice_evp.boundary_node_mask(mesh))
        _, Lmax = local_sizes(part)
        from fesom_jax.ale import AleConfig
        from fesom_jax.tke import TkeConfig
        full_kw = dict(ice_cfg=IceConfig(whichEVP=1 if args.mevp else 0) if args.ice else None,
                       kpp_cfg=KppConfig() if (args.kpp and not args.tke) else None,
                       tke_cfg=TkeConfig() if args.tke else None,
                       ale_cfg=(AleConfig(use_wsplit=bool(args.wsplit))
                                if (args.zstar or args.wsplit) else None),
                       gm_cfg=GMConfig() if args.gm else None,
                       step_forcing=shard_mesh.partition_step_forcing(sf, part),
                       forcing_static=shard_mesh.partition_forcing_static(fs, part),
                       boundary_node_p=_shard_along_axis(bn, part.myList_nod2D,
                                                         Lmax["nod"], 0, False))
    else:
        state = perturbed_state(mesh)

    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = np.zeros((args.npes, sm.Lmax["elem"], 2))      # host (Phase 8b B.3)
    if proc0:
        print(f"[bench] {args.name} npes={args.npes}: host setup done  "
              f"peak_gpu_after_setup={_gpu_peak_gb():.2f} GiB", flush=True)

    # Per-step time via the SUBTRACTION method (the JAX analog of Kokkos "omit the first W
    # steps"): time a warm N-step run and a warm W-step run, both with XLA compile EXCLUDED
    # (compile once, time the reused 2nd call), then per_step = (t_N - t_W)/(N - W) — this
    # cancels compile, the fixed per-call dispatch overhead, AND the first-W-step transient
    # (the AB2 is_first_step branch + early spin-up).
    def warm_time(n):
        # The stage prints are load-bearing at scale: a wall-clock kill between "host setup
        # done" and the final row used to be indistinguishable between a slow trace, a slow
        # compile and a hung collective (the padded halo at 64 GPUs, job 26228433). They are
        # flushed, proc0-only, and outside every timed region.
        tb = time.perf_counter()
        jfn, jargs, _ = ish.run_steps_sharded(sm, state_p, sop, stress_p, n, dt=DT,
                                              npes=args.npes, use_ragged=use_ragged,
                                              use_padded=use_padded,
                                              use_coloured=use_coloured,
                                              return_executable=True, **full_kw)
        if proc0:
            print(f"[bench] graph built (trace+lower) in {time.perf_counter() - tb:6.1f}s "
                  f"— entering compile+1st run", flush=True)
        tc = time.perf_counter()
        jax.block_until_ready(jfn(*jargs))        # 1st call: trace + compile + run (excluded)
        comp = time.perf_counter() - tc
        if proc0:
            print(f"[bench] compile+1st run done in {comp:6.1f}s — entering timed run",
                  flush=True)
        t0 = time.perf_counter()
        jax.block_until_ready(jfn(*jargs))        # 2nd call: reuse executable → pure run
        return time.perf_counter() - t0, comp, jfn, jargs

    # ONE compile per config (the full-model XLA compile is minutes — the subtraction method's
    # 2nd compile doesn't fit the QOS window). per_step = warm run / N (compile excluded by the
    # 2nd-call timing); the single is_first step is ~1/N ≈ 4% at N=25, within ±10% node noise.
    t_N, compile_s, jfn, jargs = warm_time(args.steps)
    per_step = t_N / args.steps
    per_step_ms = per_step * 1e3

    # Finiteness gate (docs/memo/DARS_INSTABILITY_FINDING.md lesson: the perf runs never
    # verified the state — dars at dt=180 was timing a blowing-up ocean). One extra
    # execution; a timing row from a non-finite run must not enter the paper.
    out_state = jfn(*jargs)
    nbad_T = int(jnp.sum(~jnp.isfinite(out_state.T)))
    nbad_uv = int(jnp.sum(~jnp.isfinite(out_state.uv)))
    max_uv = float(jnp.max(jnp.where(jnp.isfinite(out_state.uv),
                                     jnp.abs(out_state.uv), 0.0)))
    if proc0:
        print(f"[bench-finite] {args.name} npes={args.npes} steps={args.steps} "
              f"nonfinite_T={nbad_T} nonfinite_uv={nbad_uv} max_uv={max_uv:.3f} m/s",
              flush=True)

    # Optional gather-free output: each GPU writes its own shard of the FOLDED final State to
    # Zarr in parallel (the NG5 scaling output path — no global gather to rank 0). jfn(*jargs)
    # returns the folded [P*Lmax, …] sharded State; write_state_zarr writes addressable shards.
    if args.out_zarr:
        from fesom_jax import zarr_output
        out_state = jfn(*jargs)
        jax.block_until_ready(out_state)
        zarr_output.write_state_zarr(args.out_zarr, out_state, sm, part,
                                     attrs={"mesh": args.name, "dt": float(DT), "steps": int(args.steps)})
        if proc0:
            print(f"[bench] wrote sharded Zarr output → {args.out_zarr}  "
                  f"(peak_gpu={_gpu_peak_gb():.2f} GiB)", flush=True)
    tput = mesh.nod2D * mesh.nl / per_step / 1e6   # M node-levels / s
    tag = halo_mode
    if args.full:
        comp = "+".join(c for c, on in [
            ("mevp" if args.mevp else "ice", args.ice),
            ("kpp", args.kpp and not args.tke), ("tke", args.tke),
            ("gm", args.gm), ("zstar", args.zstar), ("wsplit", args.wsplit)] if on)
        model = f"FULL({comp or 'oce'}+JRA{args.year})"
    else:
        model = "ocean-only"
    if proc0:
        print(f"[bench] mesh={args.name:6s} nod2D={mesh.nod2D} nl={mesh.nl} npes={args.npes} "
              f"halo={tag:9s} model={model:20s} steps={args.steps}  per_step={per_step_ms:8.2f} ms  "
              f"throughput={tput:8.1f} Mnodlev/s  compile={compile_s:6.1f}s  plat={plat}  "
              f"peak_gpu={_gpu_peak_gb():.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
