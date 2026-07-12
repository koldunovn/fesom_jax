#!/bin/bash
# §0 / B1 — CORE2 all-on (zstar+TKE+mEVP+GM) MULTI-YEAR climate run for the paper foundation.
# FORWARD-ONLY (no AD) ⇒ single-GPU is fine, and this runs CONCURRENTLY with the single-GPU A1(b)
# adjoint job (separate SLURM jobs, separate GPUs ⇒ no conflict; the single-GPU rule is per-job).
# Produces, for the Fig 1 / FOUNDATION_BASELINE diagnostics:
#   * monthly-mean netCDF (SST/SSS/SSH/a_ice/m_ice/T/S)  -> $WORK/foundation_climate
#   * monthly State snapshots (full State w/ hnode ⇒ per-month MLD, the D2a Jensen-safe recipe)
#   * the final developed State pickle
# It also logs s/step ⇒ a single-GPU all3 throughput number for the §0 performance note.
# ⚠️ A few-year run is a FIRST-LOOK baseline: surface metrics (SST/MLD/ice) equilibrate in a few
#    years, deep T/S does NOT — label it so; a decadal spin-up is the proper foundation (external).
#SBATCH --job-name=core2_paper_found
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_foundation_run.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_foundation_run.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
YEARS=${YEARS:-3}
WORK=/work/ab0995/a270088/port_jax
echo "=== $(hostname) $(date) — §0 foundation all-on ${YEARS}-yr climate (zstar+TKE+mEVP+GM) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/archive/core2_kpp_climate_run.py --years "$YEARS" --start-year 1958 --dt 1800 \
    --ale on --tke on --mevp on --ic-dir data/ic_core2_dist864 \
    --out "$WORK/foundation_climate" --every 500 \
    --snapshot-dir "$WORK/foundation_snaps" --snapshot-every 1440 \
    --save-state "$WORK/foundation_final.pkl"
echo "exit: $?"
echo "=== done $(date) ==="
