#!/bin/bash
# M3b — κ̂ from the eddy flux DIVERGENCE, over every coarse teacher month.
#
# CPU, one node, single process: ~35 s per month (one 1.3 GB product read, one `derive`, two
# divergences, one flux projection), so the full 36-month record is ~21 min. No GPU: nothing
# here is a model step.
#
# Usage:
#   sbatch scripts/nn_eddy/divergence_fit.sbatch                 # every month in COARSE_DIR
#   MONTHS="1958_06 1959_01" sbatch scripts/nn_eddy/divergence_fit.sbatch
#SBATCH --job-name=nn_divfit
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o scripts/nn_eddy/divfit.%j.out
#SBATCH -e scripts/nn_eddy/divfit.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu          # no GPU on this partition; skip the CUDA plugin probe
echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
if [ -n "${MONTHS:-}" ]; then
  "$PY" scripts/nn_eddy/divergence_fit.py --months $MONTHS "$@"
else
  "$PY" scripts/nn_eddy/divergence_fit.py --all "$@"
fi
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
