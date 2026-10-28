#!/usr/bin/env python
"""Single-invocation, config-driven model run (Task A6 CLI).

    python scripts/run_from_config.py CONFIG.yaml [overrides]

Loads one :class:`~fesom_jax.run_config.RunConfig` YAML and does exactly ONE thing:
**load restart (or cold PHC IC) → run N steps (or a duration) → write a portable restart + exit.**
Multi-job campaigns chain these via SLURM ``--dependency=afterok`` (``scripts/chain_submit.sh``) —
there is NO in-process orchestration.

The forcing is interpolated AT RUNTIME (like FESOM — `JRA55Reader` builds the bilinear weights once,
then interpolates each step); nothing is pre-staged. The per-step forcing is fed to the sharded scan
in fine (≈ few-day) time-chunks (`--chunk-steps`) so a multi-year run never pre-stacks its forcing.

⚠️ The mesh/partition/IC wiring below is the CORE2 path (the available, smallest mesh — de-risk here
first). farc/dars/NG5 reuse the same `run_from_config`; point `--mesh-dir`/`--dist-dir`/`--ic-dir` at
the staged mesh (Task A5/B0). Example, not part of the unit-tested surface (the tested core is
`fesom_jax.run.run_from_config`; this just wires the production components to it).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax

# Multi-node (dars/NG5): 1 process per NODE, each owning all its local GPUs. Must run BEFORE any
# jax device op. JDIST=1 + srun (1 task/node) — see run_from_config.sbatch. Single-node leaves it off.
if os.environ.get("JDIST"):
    jax.distributed.initialize(
        local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))))
_IS_LEAD = (not os.environ.get("JDIST")) or jax.process_index() == 0

from fesom_jax import surface_forcing, partit, shard_mesh, ssh
from fesom_jax.mesh import load_mesh
from fesom_jax.run import run_from_config
from fesom_jax.run_config import load_yaml


def _build_partition(mesh, partition_spec, dist_dir):
    """`partition_spec` = ``"dist_<N>"`` (read the FESOM partition) or an int / ``"serial"``."""
    if partition_spec in (None, "serial", "1", 1):
        return partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    npes = int(str(partition_spec).replace("dist_", ""))
    if npes == 1:
        return partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    return partit.read_partition(Path(dist_dir), npes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="RunConfig YAML")
    ap.add_argument("--mesh-dir", help="override cfg.mesh (the mesh directory)")
    ap.add_argument("--dist-dir", help="directory holding the dist_<N> partition (for cfg.partition)")
    ap.add_argument("--ic-dir", default=None,
                    help="cold IC cache directory (T_ic.npy/S_ic.npy). Default: the packaged "
                         "pi IC when running the packaged pi mesh, else $FESOM_IC_DIR, else "
                         "data/ic_core2")
    ap.add_argument("--year", type=int, default=1958, help="forcing start year")
    ap.add_argument("--chunk-steps", type=int, default=None,
                    help="time-chunk length (steps) for the per-step forcing scan (~few days); default "
                         "gpu.chunk_steps, else 240")
    ap.add_argument("--restart-in", help="override cfg.restart_in (resume here; any device count)")
    ap.add_argument("--restart-out", help="override cfg.restart_out (write the restart here)")
    ap.add_argument("--steps", type=int, help="override cfg.n_steps")
    ap.add_argument("--params-file", default=None,
                   help="override cfg.params_file (.npz of trained gm_nn weights; absent = "
                        "default GM/Redi)")
    ap.add_argument("--partition", help="override cfg.partition (e.g. dist_8 -> dist_16 for a "
                                        "device-count-change restart)")
    ap.add_argument("--ragged", action="store_true",
                    help="ragged-halo path (no global all_gather) — needed at dars/NG5 scale; "
                         "GPU-only, forward-only (the sharded ragged AD bug doesn't apply here)")
    ap.add_argument("--padded", action="store_true",
                    help="slot-padded dense-all_to_all halo (Phase 8c) — the point-to-point "
                         "exchange that runs on EVERY backend (the only one usable on CPU past "
                         "16 in-process devices) and has a correct autodiff transpose; ships "
                         "~2-12x the ragged volume (P<=32) but 50-140x less than all_gather. "
                         "See docs/PARALLELISM.md for which halo to pick when.")
    ap.add_argument("--coloured", "--colored", action="store_true", dest="coloured",
                    help="coloured-ppermute halo (Phase 8d) — every backend and AD-correct like "
                         "--padded, but WITHOUT its P-growing pad factor: the neighbour graph is "
                         "bipartite-edge-coloured into K=Delta partial permutations, one "
                         "lax.ppermute each, so the wire volume stays 1.0-1.4x the true halo at "
                         "any P (padded ships 41x at 128 GPUs). Costs K collectives instead of 1.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="after the run, reduce the final State to gather-free scalar health "
                         "numbers (NaN/Inf scan + magnitude bounds) and print a finite/non-finite "
                         "verdict — the NG5 cold-spin-up stability gate (Task B2/R0)")
    ap.add_argument("--progress", action="store_true",
                    help="flushed per-setup-phase + per-chunk timing (host build vs device steps) "
                         "so a long multi-node run is diagnosable live (lead process only)")
    ap.add_argument("--local-forcing", action="store_true",
                    help="interpolate the per-step forcing for ONLY this process's local-partition "
                         "nodes (~npes× less host work; bit-identical) — the NG5 host-forcing fix")
    ap.add_argument("--checkpoint-every", type=int, default=None,
                    help="write a rolling intermediate restart every N steps (crash-safety on a "
                         "multi-hour run + segment-chaining); gather-free, coarse cadence ⇒ <<1%%")
    ap.add_argument("--restart-archive-out",
                    help="write ARCHIVAL restarts here: an immutable, uniquely-named directory "
                         "(fesom.<YYYY>.<DDD>.<SSSSS>, mirrors FESOM3) per firing, never "
                         "overwritten/deleted — resume any of them via --restart-in directly. "
                         "A SEPARATE stream from --restart-out/--checkpoint-every.")
    ap.add_argument("--restart-archive-period", choices=["year", "month", "day"], default="year",
                    help="archival-restart cadence unit (default: year)")
    ap.add_argument("--restart-archive-length", type=int, default=1,
                    help="fire every Nth --restart-archive-period boundary (default 1 = every one)")
    ap.add_argument("--daily-out", default=None,
                    help="write daily-mean ushow zarrs (sst/sss/temp100/u100/v100) to this dir, "
                         "one day_<YYYY>_<DOY> per calendar day; gather-free")
    ap.add_argument("--daily-start-step", type=int, default=0,
                    help="begin daily output only after this absolute step (e.g. the year-1 end "
                         "175200 ⇒ daily output from year 2 onward)")
    ap.add_argument("--monthly-out", default=None,
                    help="write monthly-mean ushow zarrs (full temp/salt/u/v + ssh/a_ice/m_ice) to "
                         "this dir, one <YYYY>_<MM> per calendar month; gather-free (the CORE2 "
                         "1958-2020 hindcast climatology output)")
    ap.add_argument("--monthly-start-step", type=int, default=0,
                    help="begin monthly output only after this absolute step")
    ap.add_argument("--chunk-diagnostics", action="store_true",
                    help="print gather-free max|uv|/max|eta| health after EACH chunk (a blow-up "
                         "trajectory probe — pair with a small --chunk-steps; lead process only)")
    ap.add_argument("--forcing-on-device", action="store_true",
                    help="override forcing.on_device=true: the per-step rdate·coef_a+coef_b "
                         "combine + wind rotation + unit conversions run IN-SCAN on device; the "
                         "host does getcoeffld only at bracket rolls (bit-identical values — "
                         "docs/plans/20260715-fesom-jax-ondevice-forcing.md). A/B knob.")
    ap.add_argument("--cheb-degree", type=int, default=None,
                    help="override ssh.cheb_degree=N: the degree-N Chebyshev CG preconditioner "
                         "(CGPOLY — 3x fewer solver iterations at equal tolerance; "
                         "solver-tolerance-equivalent, NOT bit-identical). A/B + cert knob.")
    ap.add_argument("--precision", choices=["double", "single"], default=None,
                    help="override cfg.precision: 'single' = FESOM2's single-precision build (#1207 policy; "
                         "CORE2 production physics only); default: the config's (double)")
    ap.add_argument("--ic-f32", action="store_true",
                    help="cold start only: round the initial T and S to float32 (the single-precision gate's "
                         "envelope: a double-precision run from a float32-sized perturbation)")
    ap.add_argument("--budget", action="store_true",
                    help="print float64 global volume/salt/heat content after every chunk ([budget] lines; "
                         "the single-precision conservation gate; host-gathers the State, not for speed runs)")
    ap.add_argument("--ssh-gathered", action="store_true",
                    help="override ssh.gathered=true: solve the SSH system on every device after "
                         "one all-gather instead of the sharded CG (small meshes only; off by "
                         "default — CORE2 GPU speed plan A4).")
    ap.add_argument("--lane-order", choices=["none", "rcm"], default=None,
                    help="override gpu.lane_order: the order of the lanes inside the shards (fesom_jax.lane_order; "
                         "'rcm' = reverse Cuthill-McKee nodes, elements and edges after them). A/B knob.")
    ap.add_argument("--ice-ave-steps", type=int, default=None,
                    help="override ice.ice_ave_steps=N: the sea-ice step every N ocean steps at "
                         "ice_dt = N*dt, FESOM2's scheme (plan Task 12; opt-in). A/B knob.")
    ap.add_argument("--output-layout", choices=["global", "folded"], default=None,
                    help="output + restart on-disk layout. 'global' = partition-independent canonical "
                         "node order — xarray/ushow read it directly (no unfold), byte-identical at "
                         "any device count; 'folded' = the gather-free sharded [P*Lmax] write — "
                         "multi-process OUTPUT uses all_gather (the restart auto-folds). DEFAULT: "
                         "'global' everywhere. Pass 'folded' for the gather-free fastest write, or if "
                         "all_gather output would OOM on a huge multi-node mesh.")
    args = ap.parse_args()
    # Default to the partition-independent 'global' canonical layout EVERYWHERE: single-process
    # host-gathers, multi-process output uses all_gather (and the restart auto-folds, since the
    # canonical restart writer is single-process — see run.py). Set --output-layout folded for the
    # gather-free fastest write or if all_gather output OOMs on a huge multi-node mesh.
    if args.output_layout is None:
        args.output_layout = "global"

    cfg = load_yaml(args.config)
    import dataclasses
    repl = {}
    if args.restart_in is not None:
        repl["restart_in"] = args.restart_in
    if args.restart_out is not None:
        repl["restart_out"] = args.restart_out
    if args.steps is not None:
        repl["n_steps"] = args.steps
    if args.partition is not None:
        repl["partition"] = args.partition
    if args.params_file is not None:
        repl["params_file"] = args.params_file
    if args.forcing_on_device:
        repl["forcing"] = dict(cfg.forcing or {}, on_device=True)
    if args.cheb_degree is not None:
        repl["ssh"] = dict(cfg.ssh or {}, cheb_degree=int(args.cheb_degree))
    _prec = args.precision if args.precision is not None else os.environ.get("FESOM_PRECISION") or None
    if _prec is not None:          # CLI --precision, else $FESOM_PRECISION (for the A/B timing wrappers)
        repl["precision"] = _prec
    if args.ssh_gathered:
        repl["ssh"] = dict(repl.get("ssh", cfg.ssh) or {}, gathered=True)
    if args.lane_order is not None:
        repl["gpu"] = dict(repl.get("gpu", cfg.gpu) or {}, lane_order=None if args.lane_order == "none"
                           else args.lane_order)
    if args.ice_ave_steps is not None:
        if cfg.ice is None:
            raise SystemExit("error: --ice-ave-steps needs `ice:` in the config")
        repl["ice"] = cfg.ice._replace(ice_ave_steps=int(args.ice_ave_steps))
    if repl:
        cfg = dataclasses.replace(cfg, **repl)

    # Flushed setup-phase timing (only the lead, only with --progress): the NG5 setup (16-node
    # 7.4 M mesh load, dist_64 partition, sharded-mesh build, host IC) is a candidate slow phase;
    # measure each so a re-run isn't a guess. Python block-buffers stdout to a file ⇒ flush=True.
    import time as _time
    _t0 = _time.perf_counter()

    def _lap(label):
        if args.progress and _IS_LEAD:
            print(f"[run.setup] {label}: +{_time.perf_counter() - _t0:.1f}s", flush=True)

    mesh_dir = args.mesh_dir or cfg.mesh          # None ⇒ the packaged pi mesh (mesh: null)
    mesh = load_mesh(mesh_dir) if mesh_dir is not None else load_mesh(); _lap("load_mesh")
    part = _build_partition(mesh, cfg.partition, args.dist_dir or mesh_dir); _lap("partition")
    if cfg.lane_order():            # the lanes inside the shards in a cache-friendly order (global ids unchanged)
        from fesom_jax.lane_order import edge_lane_distance, reorder_lanes
        _d0 = edge_lane_distance(part, mesh)
        part = reorder_lanes(part, mesh, cfg.lane_order()); _lap("lane_order")
        if _IS_LEAD:
            print(f"[run] lane_order={cfg.lane_order()}: median edge lane distance per shard "
                  f"{_d0.astype(int).tolist()} -> {edge_lane_distance(part, mesh).astype(int).tolist()}", flush=True)
    sm = shard_mesh.build_sharded_mesh(mesh, part); _lap("sharded_mesh")
    # single precision: upstream FESOM2 switches the preconditioner to the symmetric Jacobi variant (#995)
    op = ssh.build_ssh_operator(mesh, dt=cfg.dt, precond_variant=1 if cfg.precision == "single" else 0)
    cheb_deg, cheb_kap = cfg.ssh_cheb()
    if cheb_deg:                    # CGPOLY (ssh.cheb_degree): Chebyshev CG preconditioner
        op = ssh.enable_cheb_precond(op, cheb_deg, kappa_guess=cheb_kap)
        if _IS_LEAD:
            print(f"[run] CGPOLY: Chebyshev precond degree={cheb_deg} "
                  f"lam=[{op.cheb[1]:.4g}, {op.cheb[2]:.4g}]", flush=True)
    sop = ssh.partition_ssh_operator(op, part); _lap("ssh_op")

    # cold IC vs restart: run_from_config reads cfg.restart_in itself; only build the cold IC here.
    # HOST-build (xp=np) so the global State is never materialized on GPU 0 (the dars/NG5 setup-OOM
    # fix — a device-built 3.16 M State is ~50-80 GB, OOMs one GPU before partition_state can shard it).
    import numpy as np
    state0 = None
    if cfg.restart_in is None:
        from fesom_jax.phc_ic import cold_start_state, PACKAGED_PI_IC_DIR
        ic_dir = args.ic_dir
        if ic_dir is None and mesh_dir is None and PACKAGED_PI_IC_DIR.is_dir():
            # The packaged-pi run uses the packaged pi IC, NOT $FESOM_IC_DIR — that names the
            # fetch_data CORE2 cache, whose 126 858-node fields cannot fit the 3140-node pi mesh.
            ic_dir = PACKAGED_PI_IC_DIR
        # THE canonical cold start (one shared helper ⇒ no drift from the run scripts): PHC IC +
        # the seeded sea-ice IC (a_ice=0.9 where SST<0, NH/SH m_ice split — fesom_ice_initial_state).
        # xp=np keeps the global State on the host (the big-mesh setup-OOM fix).
        # ic_dir=None resolves $FESOM_IC_DIR → data/ic_core2 (fesom_jax.paths).
        state0 = cold_start_state(mesh, ic_dir, xp=np); _lap("cold_start_state")
        if args.ic_f32:     # the single-precision gate's envelope run: the cold T and S rounded to float32
            import dataclasses as _dc
            state0 = _dc.replace(state0, **{k: np.asarray(getattr(state0, k), np.float32).astype(np.float64)
                                            for k in ("T", "S", "T_old", "S_old")})
    sst0 = None if state0 is None else np.asarray(state0.T[:, 0])
    if cfg.forcing is None:
        # `forcing: null` = UNFORCED (the documented `null = OFF`). Never fall through to the
        # $FESOM_*/Levante default forcing paths — that silently runs the wrong physics
        # (usability review P2: a 1.65 °C SST divergence measured against the intended run).
        forcing = None
        if args.local_forcing or args.forcing_on_device:
            raise SystemExit("error: --local-forcing/--forcing-on-device need `forcing:` "
                             "configured (the run is unforced: `forcing: null`)")
        _lap("forcing_setup (unforced)")
    else:
        # Input-data paths: the run YAML's optional `forcing: {jra_dir,sss_path,runoff_path,chl_path}`
        # keys win; absent (all None) ⇒ each reader resolves $FESOM_* → the Levante default
        # (fesom_jax/paths.py, docs/DATA.md) — the historical behaviour.
        fpaths = cfg.forcing_paths()
        forcing = surface_forcing.build_surface_forcing(mesh, args.year, sst_ic=sst0,
                                                   dt=cfg.dt, **fpaths); _lap("forcing_setup")

    # The NG5 host-forcing fix: build a LOCAL-node forcing (interp only this process's owned
    # partitions, ~npes× less). forcing.static is reused for the (cheap, once) static path.
    local_forcing = None
    if args.local_forcing:
        from fesom_jax.forcing_local import build_local_forcing
        local_forcing = build_local_forcing(mesh, args.year, part, part.npes,
                                             static=forcing.static, sst_ic=sst0, dt=cfg.dt, **fpaths)
        _lap("local_forcing_setup")

    # the halo transport: a --padded / --coloured / --ragged flag wins over gpu.halo_transport
    _transport = ("padded" if args.padded else "coloured" if args.coloured else "ragged" if args.ragged
                  else cfg.halo_transport())
    if _IS_LEAD:
        print(f"[run] mesh={mesh_dir} npes={part.npes} devices={len(jax.devices())} "
              f"dt={cfg.dt} steps={cfg.n_steps or cfg.duration} restart_in={cfg.restart_in} "
              f"local_forcing={args.local_forcing} halo={_transport}", flush=True)
    res = run_from_config(cfg, mesh=mesh, part=part, sm=sm, sop=sop, forcing=forcing,
                          state0=state0, start_step=0, year=args.year,
                          chunk_steps=args.chunk_steps or cfg.chunk_steps() or 240, out_dir=cfg.restart_out,
                          use_ragged=_transport == "ragged", use_padded=_transport == "padded",
                          use_coloured=_transport == "coloured",
                          progress=(args.progress and _IS_LEAD),
                          local_forcing=local_forcing, checkpoint_every=args.checkpoint_every,
                          restart_archive_out=args.restart_archive_out,
                          restart_archive_period=args.restart_archive_period,
                          restart_archive_length=args.restart_archive_length,
                          daily_out=args.daily_out, daily_start_step=args.daily_start_step,
                          monthly_out=args.monthly_out, monthly_start_step=args.monthly_start_step,
                          chunk_diagnostics=args.chunk_diagnostics, output_layout=args.output_layout,
                          budget=args.budget)
    if _IS_LEAD:
        print(f"[run] DONE step={res.step} dt_stage={res.dt_stage} restart_out={cfg.restart_out}")
        print("RUN_DRIVER_OK")

    if cfg.ice is None and cfg.forcing is not None:
        # `ice: null` on a FORCED run is a diagnostic setting, not supported physics: without ice
        # thermodynamics the polar ocean cools far below freezing while staying finite and exiting 0
        # (inherited FESOM behaviour — usability review P8 measured −24.6 °C over 22 % of ocean
        # nodes with no warning at all). Collective (all ranks call), like --diagnostics below.
        from fesom_jax.diagnostics import state_diagnostics as _sd
        _tmin = float(_sd(res.state_p)["T_min"])
        if _IS_LEAD and _tmin < -2.5:
            print(f"WARNING: T_min={_tmin:.1f} degC is below seawater freezing and this run has "
                  f"`ice: null` under forcing. Sub-freezing water is EXPECTED without ice "
                  f"thermodynamics — treat the run as diagnostic, or enable ice (`ice: {{}}`).",
                  flush=True)

    if args.diagnostics:
        # Gather-free scalar reductions on the FOLDED sharded final State (no restart re-read, no
        # global gathered to one device) — runs on EVERY process (each global reduction is a
        # replicated all-reduce ⇒ same scalars everywhere); only the lead prints + emits the token.
        from fesom_jax.diagnostics import format_diagnostics, state_diagnostics, verdict
        diags = state_diagnostics(res.state_p)
        ok, _ = verdict(diags)
        if _IS_LEAD:
            print(format_diagnostics(diags, label=f"step{res.step}"))
            print("NG5_R0_FINITE" if ok else "NG5_R0_NONFINITE")
        if not ok:
            # non-finite ⇒ FAIL the process (every rank computes the same replicated verdict) so a
            # self-chaining runner STOPS the chain instead of propagating NaNs into the next restart.
            sys.stdout.flush()
            sys.exit(1)


if __name__ == "__main__":
    main()
