#!/bin/bash
# Albedo: pi + full REcoM, 30 days on one CPU node (the pi smoke run; docs/recom/RUNNING_ALBEDO.md).
#   sbatch scripts/albedo/run_pi_recom.sbatch [run_from_config.py flags]
#SBATCH --job-name=fj_pi_recom
#SBATCH -p mpp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} PYTHONUNBUFFERED=1
source scripts/albedo/env.sh
OUT=/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/pi_recom_bgc
mkdir -p $OUT
echo "=== $(hostname) $(date) job $SLURM_JOB_ID ==="
$PY scripts/run_from_config.py configs/albedo/pi_recom_bgc.yaml --progress --diagnostics \
    --restart-out $OUT/restart --monthly-out $OUT/monthly "$@"
echo "=== exit $? $(date) ==="
