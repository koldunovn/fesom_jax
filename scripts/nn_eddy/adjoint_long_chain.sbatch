#!/bin/bash
# A multi-year adjoint rung, chained across jobs by the reverse-sweep checkpoint.
#
#   NDAYS=731 STRIDE=27 MAXLEGS=6 sbatch scripts/nn_eddy/adjoint_long_chain.sbatch
#
# Each leg runs until STOPMIN, checkpoints, and exits 42; the leg then resubmits ITSELF with a
# dependency on its own job id, so the chain walks forward one 12-hour slot at a time. Exit 0
# means the sweep finished and no further leg is queued.
#
# ⚠️ THIS SCRIPT DELETES NOTHING, and must not learn to. Workspace house rule: no unattended
# deletion, ever — it has cost real data here once. The checkpoint reuses two fixed cotangent
# slots rather than accumulating files, which is bounded without removing anything, and the
# boundaries and the truth-mark cache are left in place for the user to prune by hand.
#
# ⚠️ MAXLEGS is a runaway guard, not a tuning knob. A bug that makes a leg do zero reverse
# chunks would otherwise resubmit forever.
#
# What each leg costs, from the measured 14.6 s/day forward and 71.2 s/day backward:
#   leg 1  = truth build (N x 14.6 s) + forward (N x 14.6 s) + as much reverse as fits
#   leg 2+ = truth mark from cache (instant) + boundaries from disk (no forward) + reverse
# so a year is 2 legs and two years is 3. FIVE years does not fit: leg 1 alone would need
# 7.4 h of truth + 7.4 h of forward = 14.8 h before the reverse begins, and neither of those
# is resumable yet — the forward checkpoint is the missing piece, not the wall clock.
#SBATCH --job-name=nn_adj_chain
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH -o scripts/nn_eddy/adjchain.%j.out
#SBATCH -e scripts/nn_eddy/adjchain.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

SNAP=${SNAP:-/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl}
SEASON=${SEASON:-winterNH}
NDAYS=${NDAYS:?set NDAYS}
TRUTH=${TRUTH:-$NDAYS}
STRIDE=${STRIDE:-1}
CHUNK=${CHUNK:-48}
FREEZE=${FREEZE:-mix}            # ⚠️ the experiment, not a tuning choice — see the ladder sbatch
FDH=${FDH:-}                     # empty = no finite differences (they are not checkpointed)
LEG=${LEG:-1}
MAXLEGS=${MAXLEGS:-6}
STOPMIN=${STOPMIN:-630}          # 10.5 h of a 12 h slot; the rest is compile + the final write
WEIGHTS=${WEIGHTS:-/work/ba1583/a270088/nn_eddy/horizon_truth/snap_step037440_m1.25_30d.npz}
RES=${RES:-scripts/nn_eddy/adjoint_horizon_results.jsonl}
CKPT=${CKPT:-/work/ba1583/a270088/nn_eddy/long_chain/N${NDAYS}_s${STRIDE}}
RESUME=""
[ "$LEG" -gt 1 ] && RESUME="--resume"

echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
echo "=== LEG $LEG of at most $MAXLEGS: N=$NDAYS d, truth=$TRUTH, stride=$STRIDE, "\
     "budget ${STOPMIN} min, ckpt $CKPT ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
free -g | head -2

"$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season "$SEASON" \
    --n-days "$NDAYS" --truth-days "$TRUTH" --chunk-steps "$CHUNK" \
    --boundary-stride "$STRIDE" --fd-h "$FDH" --freeze "$FREEZE" \
    ${WEIGHTS:+--weights-from "$WEIGHTS"} \
    --ckpt-dir "$CKPT" --ckpt-every 5 --stop-after-min "$STOPMIN" $RESUME \
    --run-start-year 1958 --results "$RES" --tag "chain-N${NDAYS}"
RC=$?
echo "=== leg $LEG exit $RC $(date) ==="

if [ "$RC" -eq 42 ]; then
    if [ "$LEG" -ge "$MAXLEGS" ]; then
        echo "REACHED MAXLEGS=$MAXLEGS without finishing — NOT resubmitting. The checkpoint in"
        echo "$CKPT is complete and consistent; raise MAXLEGS and resubmit by hand if this is"
        echo "just a long window, or look for a leg that did zero reverse chunks if it is not."
        exit 1
    fi
    NEXT=$((LEG + 1))
    echo "checkpointed; queueing leg $NEXT after this job"
    sbatch --dependency=afterany:$SLURM_JOB_ID \
        --export=ALL,NDAYS=$NDAYS,TRUTH=$TRUTH,STRIDE=$STRIDE,CHUNK=$CHUNK,FREEZE=$FREEZE,FDH=$FDH,LEG=$NEXT,MAXLEGS=$MAXLEGS,STOPMIN=$STOPMIN,SNAP=$SNAP,SEASON=$SEASON,WEIGHTS=$WEIGHTS,RES=$RES,CKPT=$CKPT \
        scripts/nn_eddy/adjoint_long_chain.sbatch
    exit 0
fi
exit $RC
