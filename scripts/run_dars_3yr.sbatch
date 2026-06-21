#!/bin/bash
# Self-chaining dars 3-YEAR run (JAX) — model-paper production: TKE + mEVP + zstar, GM OFF,
# dt 180→240 after year 1 (config dt_ramp). Each job resumes from the rolling restart, runs a
# bounded step budget (< 12 h walltime), writes a restart, and RESUBMITS itself until 3 model
# years (TOTAL steps, with the ramp) are done.
#
# Safety for the UNATTENDED weekend run:
#  * the FIRST (cold) job is a short CANARY — TKE+mEVP has never run at dars scale; if it goes
#    non-finite, run_from_config.py exits non-zero (NG5_R0_NONFINITE) and the chain STOPS.
#  * EVERY job ends with the finite check ⇒ a NaN never propagates into the next restart.
#  * a non-zero srun (non-finite OR crash OR walltime-kill) STOPS the chain (no resubmit) — a
#    halted chain is safe to inspect; better than burning budget on a broken state.
#  * a crash mid-job is covered by the rolling --checkpoint-every restart (resume from the last).
#  * runaway guard: a per-chain jobcount caps the chain length.
#SBATCH --job-name=dars3yr_jax
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=11:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_dars_3yr.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_dars_3yr.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="   # cmdbuf off (compile dead-end, PORTING_LESSONS)
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-$REPO/configs/dars_tke_mevp.yaml}   # override at launch for a fallback config
TAG=${TAG:-dars_3yr}                                  # names the output subdir (per-config)
MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
PART=dist_16
OUT=/work/ab0995/a270088/port_jax/runs/$TAG
RESTART=$OUT/restart
SELF=$REPO/scripts/run_dars_3yr.sbatch
TOTAL=438000          # 3 model years: 1958@180 (175200) + 1959-60@240 (262800). = _step_at_elapsed(3*365d)
CANARY=1440           # first (cold) job: 3 model days — de-risk TKE+mEVP finiteness cheaply
PERJOB=20000          # ~9-10 h at ≤1.7 s/step (margin under 11:45); a walltime-kill safely stops the chain
CHUNK=48

mkdir -p $OUT
# runaway guard (resume phase): cap chain length well above the ~22 real jobs expected.
CNT=$( cat $OUT/.jobcount 2>/dev/null || echo 0 ); CNT=$((CNT+1)); echo $CNT > $OUT/.jobcount
if [ "$CNT" -gt 35 ]; then echo "ABORT: jobcount $CNT > 35 (runaway guard)"; exit 1; fi

# current step = the rolling restart's step (0 ⇒ cold). Source of truth (survives crashes/kills).
CUR=$( $PY -c "import json;print(int(json.load(open('$RESTART/.zattrs')).get('step',0)))" 2>/dev/null || echo 0 )
[ -z "$CUR" ] && CUR=0
echo "=== $(date) dars3yr job#$CNT: CUR=$CUR / TOTAL=$TOTAL ==="
if [ "$CUR" -ge "$TOTAL" ]; then echo "DARS_3YR_DONE ($CUR steps = 3 model years)"; exit 0; fi

# step budget for THIS job: short canary on the cold start, else the per-job budget (capped at remaining).
if [ "$CUR" -eq 0 ]; then
  STEPS=$CANARY; RIN=""; rm -rf $OUT; mkdir -p $OUT; echo $CNT > $OUT/.jobcount
else
  REM=$((TOTAL-CUR)); STEPS=$PERJOB; [ "$STEPS" -gt "$REM" ] && STEPS=$REM; RIN="--restart-in $RESTART"
fi
echo "running $STEPS steps from $CUR (${RIN:-cold})"

srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition "$PART" \
    --steps "$STEPS" --chunk-steps "$CHUNK" --ragged --local-forcing \
    --checkpoint-every 2400 --diagnostics --progress \
    --restart-out "$RESTART" $RIN
R=$?
echo "dars3yr job#$CNT exit=$R"
if [ "$R" -ne 0 ]; then
  echo "DARS_3YR_STOP (exit=$R — non-finite/crash/kill; inspect $OUT, do NOT auto-continue)"; exit $R
fi

NEWCUR=$((CUR+STEPS))
if [ "$NEWCUR" -lt "$TOTAL" ]; then
  echo "resubmitting from $NEWCUR/$TOTAL"; cd $REPO && sbatch --export=ALL,CONFIG="$CONFIG",TAG="$TAG" $SELF
else
  echo "DARS_3YR_DONE ($NEWCUR steps = 3 model years)"
fi
