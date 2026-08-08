#!/bin/bash
# NN-GM revised Stage-1 feasibility — D(t) + κ-branch signal evaluation (CPU, cheap).
# Needs: student_free_1958/ (student_year.sbatch) and whichever coarse_ng5_1958_MM.nc exist —
# it uses what is there, so it may run before all 12 coarsening jobs land (rerun any time).
#   sbatch scripts/nn_eddy/signal_eval.sbatch
#SBATCH --job-name=nn_signal_eval
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o scripts/nn_eddy/sigeval.%j.out
#SBATCH -e scripts/nn_eddy/sigeval.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date)  commit $(git rev-parse --short HEAD) ==="
"$PY" scripts/nn_eddy/signal_eval.py "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
