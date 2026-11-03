#!/bin/bash
# Albedo: pytest on one CPU compute node (the login node kills XLA compiles).
#   sbatch scripts/albedo/run_pytest.sbatch fesom_jax/tests/test_recom_step_fork.py [-k expr ...]   (from the repo root)
# Environment variables of the gates (FESOM_RECOM_FORK_REF, FESOM_RECOM_FORK_RUN, ...) pass through --export=ALL.
#SBATCH --job-name=fj_pytest
#SBATCH -p mpp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}"
source scripts/albedo/env.sh                    # the Albedo input-data paths (the REcoM SAAR table among them)
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) ($(git branch --show-current)) ==="
echo "pytest $*"
$PY -m pytest -q -p no:cacheprovider -s --durations=10 "$@"
echo "=== exit $? $(date) ==="
