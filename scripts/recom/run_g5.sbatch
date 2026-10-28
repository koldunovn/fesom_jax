#!/bin/bash
# Gate G5 (lane W): the 30-day pi run with full REcoM vs the physics-matched Fortran twin.
# Submit from the worktree root: sbatch scripts/recom/run_g5.sbatch <outdir> [days]
#SBATCH --job-name=recom_g5
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH -o scripts/recom/run_g5.%j.out
set -u
cd "${SLURM_SUBMIT_DIR}"
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export PYTHONPATH=$PWD JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) $(git status --porcelain -- fesom_jax scripts configs | wc -l) modified"
$PY scripts/recom/g5_pi_30d.py --out "$1" --days "${2:-30}"
rc=$?
echo "=== exit $rc $(date)"
exit $rc
