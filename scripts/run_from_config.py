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
from pathlib import Path

import jax

# Multi-node (dars/NG5): 1 process per NODE, each owning all its local GPUs. Must run BEFORE any
# jax device op. JDIST=1 + srun (1 task/node) — see run_from_config.sbatch. Single-node leaves it off.
if os.environ.get("JDIST"):
    jax.distributed.initialize(
        local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))))
_IS_LEAD = (not os.environ.get("JDIST")) or jax.process_index() == 0

from fesom_jax import core2_forcing, partit, shard_mesh, ssh
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
    ap.add_argument("--ic-dir", default="data/ic_core2", help="cold PHC IC cache directory")
    ap.add_argument("--year", type=int, default=1958, help="forcing start year")
    ap.add_argument("--chunk-steps", type=int, default=240,
                    help="time-chunk length (steps) for the per-step forcing scan (~few days)")
    ap.add_argument("--restart-in", help="override cfg.restart_in (resume here; any device count)")
    ap.add_argument("--restart-out", help="override cfg.restart_out (write the restart here)")
    ap.add_argument("--steps", type=int, help="override cfg.n_steps")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    import dataclasses
    repl = {}
    if args.restart_in is not None:
        repl["restart_in"] = args.restart_in
    if args.restart_out is not None:
        repl["restart_out"] = args.restart_out
    if args.steps is not None:
        repl["n_steps"] = args.steps
    if repl:
        cfg = dataclasses.replace(cfg, **repl)

    mesh_dir = args.mesh_dir or cfg.mesh
    mesh = load_mesh(mesh_dir)
    part = _build_partition(mesh, cfg.partition, args.dist_dir or mesh_dir)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=cfg.dt), part)

    # cold IC vs restart: run_from_config reads cfg.restart_in itself; only build the cold IC here.
    state0 = None
    if cfg.restart_in is None:
        from fesom_jax.phc_ic import core2_initial_state
        state0 = core2_initial_state(mesh, args.ic_dir)
    sst0 = None if state0 is None else __import__("numpy").asarray(state0.T[:, 0])
    forcing = core2_forcing.build_core_forcing(mesh, args.year, sst_ic=sst0)

    if _IS_LEAD:
        print(f"[run] mesh={mesh_dir} npes={part.npes} devices={len(jax.devices())} "
              f"dt={cfg.dt} steps={cfg.n_steps or cfg.duration} restart_in={cfg.restart_in}")
    res = run_from_config(cfg, mesh=mesh, part=part, sm=sm, sop=sop, forcing=forcing,
                          state0=state0, start_step=0, year=args.year,
                          chunk_steps=args.chunk_steps, out_dir=cfg.restart_out)
    if _IS_LEAD:
        print(f"[run] DONE step={res.step} dt_stage={res.dt_stage} restart_out={cfg.restart_out}")
        print("RUN_DRIVER_OK")


if __name__ == "__main__":
    main()
