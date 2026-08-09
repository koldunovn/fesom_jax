#!/bin/bash
# M3 Stage 0 — fit gm_nn offline to the diffusivities the teacher's eddy fluxes imply.
#
# CPU, one node, single process. The cost is the ASSEMBLY, not the fit: one `derive` + one
# `project` per teacher mark, the same work divergence_fit.sbatch measures at ~35 s per month,
# so 219 marks over 36 months is ~25-40 min. The optax fit that follows is seconds.
#
# The holdout is named on the command line, never left implicit, and offline_fit.py writes it
# to holdout_declared.json before the first gradient step.
#
# Usage:
#   sbatch scripts/nn_eddy/offline_fit.sbatch
#   OUTTAG=grad3e7 EXTRA="--grad-floor 3.4e-7" sbatch scripts/nn_eddy/offline_fit.sbatch
#SBATCH --job-name=nn_stage0
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/stage0.%j.out
#SBATCH -e scripts/nn_eddy/stage0.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu          # no GPU on this partition; skip the CUDA plugin probe

# The gate holdout, per the child plan's own recommendation: the last two landed months.
HOLDOUT="${HOLDOUT:-1960_11,1960_12}"
# Output directory. A floor-sensitivity leg must NOT overwrite the primary fit's record, so
# every non-default configuration passes OUTTAG.
OUT="/work/ba1583/a270088/nn_eddy/stage0${OUTTAG:+_$OUTTAG}"

echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
echo "=== holdout=$HOLDOUT out=$OUT extra='${EXTRA:-}' ==="
"$PY" scripts/nn_eddy/offline_fit.py --holdout "$HOLDOUT" --out "$OUT" ${EXTRA:-} "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
