#!/bin/bash
# A small CPU python job of the partial-cells branch on a compute node (never the login node).
#   cd <worktree> && sbatch --export=ALL,CMD="scripts/partial_cells/g1_geometry.py <args>" scripts/partial_cells/cpu_job.sbatch
#SBATCH --job-name=pc_cpu
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/partial_cells/cpu.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}" JAX_PLATFORMS=cpu OMP_NUM_THREADS=16
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) CMD=$CMD ==="
$PY $CMD
echo "=== rc=$? $(date) ==="
