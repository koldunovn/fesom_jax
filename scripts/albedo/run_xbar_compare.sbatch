#!/bin/bash
# Albedo: daily x-bar against the interpolated monthly x-bar (plan 20261105), on one CPU node, 8 tracers at a time.
#   sbatch scripts/albedo/run_xbar_compare.sbatch                    (from the repo root)
#SBATCH --job-name=fj_xbar
#SBATCH -p mpp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=12h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}"
source scripts/albedo/env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) ==="
$PY scripts/recom/xbar_daily_vs_monthly.py "$@"
echo "=== exit $? $(date) ==="
