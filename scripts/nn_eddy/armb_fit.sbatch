#!/bin/bash
# M3 — fit the Arm B closure (a learned bolus streamfunction) to the teacher's eddy flux.
#
# CPU, one node. The cost is the ASSEMBLY: one `derive` + one `load_fluxes` + one divergence per
# teacher mark, ~20 s each, so 20 marks is ~7 min and the marks are cached per file afterwards.
# The fit itself is one gradient step per mark per epoch on the whole field.
#
# Memory: a cached mark is ~600 MB (whole-field features at interfaces plus six 3-D fields), and
# `fit` puts every mark on the device at once, so keep marks-per-month x months under ~60 at
# 128 GB. Raise --mem before raising the mark count.
#
# Usage:
#   sbatch scripts/nn_eddy/armb_fit.sbatch
#   OUTTAG=smoke EXTRA="--train 1959_03,1959_04 --marks-per-month 1 --epochs 3" \
#     sbatch scripts/nn_eddy/armb_fit.sbatch
#SBATCH --job-name=nn_armb
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH -o scripts/nn_eddy/armb.%j.out
#SBATCH -e scripts/nn_eddy/armb.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu          # no GPU on this partition; skip the CUDA plugin probe

# Same holdout as the Arm A legs, so the two arms are compared on identical unseen months.
HOLDOUT="${HOLDOUT:-1960_11,1960_12}"
OUT="/work/ba1583/a270088/nn_eddy/armb${OUTTAG:+_$OUTTAG}"

echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
echo "=== holdout=$HOLDOUT out=$OUT extra='${EXTRA:-}' ==="
"$PY" scripts/nn_eddy/armb_fit.py --holdout "$HOLDOUT" --out "$OUT" ${EXTRA:-} "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
