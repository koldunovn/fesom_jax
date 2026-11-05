#!/bin/bash
# Albedo: the BGC gradient horizon (Phase 1, docs/plans/20261105-recom-parameter-estimation.md) on one CPU compute node.
#   sbatch scripts/albedo/run_bgc_horizon.sbatch --days 1 --cost chl          (from the repo root)
# on a GPU: sbatch -p gpu --gpus=a100:1 --export=ALL,JAX_PLATFORMS=cuda scripts/albedo/run_bgc_horizon.sbatch ...
# longer windows: sbatch --qos=12h --time=12:00:00 scripts/albedo/run_bgc_horizon.sbatch --days 5 30 90
#SBATCH --job-name=fj_bgchor
#SBATCH -p mpp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}"
source scripts/albedo/env.sh
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) ==="
$PY scripts/recom/bgc_gradient_horizon.py "$@"
echo "=== exit $? $(date) ==="
