#!/bin/bash
# Albedo: pi + the spectral REcoM fork (4 PFT, CDOM, 13-band OASIM light), 30 days on one CPU node (the pi smoke run; docs/recom/RUNNING_ALBEDO.md).
#   sbatch scripts/albedo/run_pi_recom_spectral.sbatch [run_from_config.py flags]
#SBATCH --job-name=fj_pi_recom_spectral
#SBATCH -p mpp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=12h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} PYTHONUNBUFFERED=1
source scripts/albedo/env.sh
OUT=/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/pi_recom_spectral
mkdir -p $OUT
echo "=== $(hostname) $(date) job $SLURM_JOB_ID ==="
# --ic-dir: the mesh is the fork's pi bundle (not the packaged pi), so the packaged pi PHC IC must be named
$PY scripts/run_from_config.py configs/albedo/pi_recom_spectral.yaml --ic-dir fesom_jax/data/ic_pi --progress --diagnostics \
    --restart-out $OUT/restart --monthly-out $OUT/monthly "$@"
echo "=== exit $? $(date) ==="
