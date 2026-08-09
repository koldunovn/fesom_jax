#!/bin/bash
# M3 — the a-priori evaluation GATE M3 is read from.
#
# CPU. Only the holdout months are scored (~12 marks over two months), and each mark costs one
# `derive` + one `project` plus one forward flux per closure, so the whole thing is ~15 min.
#
# Usage — the rungs are passed as repeated TAG=PATH, and --const-from fixes BOTH the months and
# the train-fitted constant baseline from the record the fit wrote:
#
#   WEIGHTS="D0=/work/.../stage0/gm_nn_stage0_seed0.npz D2w=/work/.../stage0_w64/..." \
#   CONST_FROM=/work/ba1583/a270088/nn_eddy/stage0 \
#     sbatch scripts/nn_eddy/apriori_eval.sbatch
#SBATCH --job-name=nn_apriori
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o scripts/nn_eddy/apriori.%j.out
#SBATCH -e scripts/nn_eddy/apriori.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu

CONST_FROM="${CONST_FROM:-/work/ba1583/a270088/nn_eddy/stage0}"
ARGS=""
for w in ${WEIGHTS:?set WEIGHTS to one or more TAG=PATH}; do ARGS="$ARGS --weights $w"; done

echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
echo "=== rungs: $WEIGHTS ==="
"$PY" scripts/nn_eddy/apriori_eval.py $ARGS --const-from "$CONST_FROM" \
      --out "${OUT:-$CONST_FROM/apriori_eval.json}" ${EXTRA:-} "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
