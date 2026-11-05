#!/bin/bash
# Albedo: the 2-year pi spectral spin-up on one A100 (Phase 1 of docs/plans/20261105-recom-parameter-estimation.md).
#   sbatch scripts/albedo/run_pi_spinup.sbatch [run_from_config.py flags]          (from the repo root)
#SBATCH --job-name=fj_pi_spin
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -A clidyn.clidyn
#SBATCH --qos=12h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
source scripts/albedo/env.sh
OUT=/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/pi_recom_spectral_spin
mkdir -p $OUT
echo "=== $(hostname) $(date) job $SLURM_JOB_ID $(git rev-parse --short HEAD) ==="
$PY scripts/run_from_config.py configs/albedo/pi_recom_spectral_spin.yaml \
    --ic-dir /albedo/work/projects/p_specbic/sloza/fesom_jax_data/ic_pi_fork --progress \
    --restart-out $OUT/restart --monthly-out $OUT/monthly "$@"
rc=$?
echo "=== exit $rc $(date) ==="
exit $rc
