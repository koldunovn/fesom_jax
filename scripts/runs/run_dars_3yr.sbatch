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
MESH=${MESH:-/work/ab0995/a270088/fesom_jax_meshes/mesh_dars}
POOL=${POOL:-/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars}
PART=${PART:-dist_16}
IC=${IC:-$MESH}
NODES=${NODES:-4}     # GPU nodes (dist_16=4, dist_32=8); the resubmit re-requests --nodes=$NODES
OUT=/work/ab0995/a270088/port_jax/runs/$TAG
RESTART=$OUT/restart
SELF=$REPO/scripts/runs/run_dars_3yr.sbatch
TOTAL=${TOTAL:-438360}          # 3 model years: 1958@180 (175200) + 1959-60@240 (263160 = 731
                                # days INCL. the 1960 leap day; the old 438000 was a day short).
CANARY=1440           # first (cold) job: 3 model days — de-risk finiteness cheaply
PERJOB=${PERJOB:-20000}          # ~9-10 h margin under 11:45; a walltime-kill safely stops the chain
CHUNK=48

mkdir -p $OUT
# runaway guard (resume phase): cap chain length well above the ~22 real jobs expected.
CNT=$( cat $OUT/.jobcount 2>/dev/null || echo 0 ); CNT=$((CNT+1)); echo $CNT > $OUT/.jobcount
if [ "$CNT" -gt 35 ]; then echo "ABORT: jobcount $CNT > 35 (runaway guard)"; exit 1; fi

# current step = the rolling restart's step (0 ⇒ cold). Source of truth (survives crashes/kills).
# ⚠️ If the restart EXISTS but its step is unreadable (corrupted .zattrs / transient /work error),
# STOP the chain — the old silent `|| echo 0` fallback turned any read failure into a cold start
# (the 2026-07-03 data-loss incident class).
CUR=0
if [ -d "$RESTART" ]; then
  CUR=$( $PY -c "import json;print(int(json.load(open('$RESTART/.zattrs')).get('step',0)))" 2>/dev/null || echo FAIL )
  case "$CUR" in ''|*[!0-9]*)
    echo "ABORT: $RESTART exists but its step is unreadable — refusing the silent cold-start fallback; inspect + resubmit manually"; exit 1;;
  esac
fi
echo "=== $(date) dars3yr job#$CNT: CUR=$CUR / TOTAL=$TOTAL ==="
if [ "$CUR" -ge "$TOTAL" ]; then echo "DARS_3YR_DONE ($CUR steps = 3 model years)"; exit 0; fi

# step budget for THIS job: short canary on the cold start, else the per-job budget (capped at remaining).
if [ "$CUR" -eq 0 ]; then
  # COLD start — NO deletion (the old `rm -rf $OUT` here violated the no-silent-deletion rule
  # for unattended chains, 2026-07-03 review). A deliberate re-run over an old tag must clean
  # up MANUALLY first or use a fresh TAG.
  STEPS=$CANARY; RIN=""
else
  REM=$((TOTAL-CUR)); STEPS=$PERJOB; [ "$STEPS" -gt "$REM" ] && STEPS=$REM; RIN="--restart-in $RESTART"
fi
echo "running $STEPS steps from $CUR (${RIN:-cold})"

LOGFILE=$REPO/scripts/run_dars_3yr.${SLURM_JOB_ID}.out   # this job's own -o file (srun output lands here, live)
do_resubmit() {   # resubmit the chain from the rolling restart (shared by the hang-heal + normal-completion paths)
  cd $REPO && sbatch --nodes=$NODES \
    --export=ALL,CONFIG="$CONFIG",TAG="$TAG",MESH="$MESH",POOL="$POOL",PART="$PART",IC="$IC",NODES="$NODES",PERJOB="$PERJOB",TOTAL="$TOTAL" $SELF
}

# Run in the BACKGROUND so a compile-hang guard can supervise the first chunk. The observed failure
# (job#4, 2026-06-22) was an XLA compile-hang: the job reaches "setup done" then sits forever — 5h20m
# idle on 8 GPUs, no chunk 1, restart never advances. A fresh allocation clears it.
srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$IC" --partition "$PART" \
    --steps "$STEPS" --chunk-steps "$CHUNK" --ragged --local-forcing \
    --checkpoint-every 2400 --diagnostics --progress \
    --daily-out "$OUT/daily" --daily-start-step "${DAILY_START:-175200}" \
    --restart-out "$RESTART" $RIN &
SRUN_PID=$!

# COMPILE-HANG GUARD: a healthy job prints "chunk 1/" within minutes (setup ~1 min + XLA compile); a
# hang never does. Watch ONLY the first chunk — once it appears the job is healthy and we stop guarding
# (steady chunks run for hours). No "chunk 1/" within GUARD_SECS ⇒ kill + self-heal (resubmit from the
# intact restart) so the chain survives unattended. A consecutive-hang counter STOPS a hang-LOOP (a
# persistent fault, or a mis-firing guard) after 3 tries rather than burning the whole node budget.
GUARD_SECS=${GUARD_SECS:-1800}
t0=$SECONDS; HUNG=0
while kill -0 $SRUN_PID 2>/dev/null; do
  grep -q "chunk 1/" "$LOGFILE" 2>/dev/null && break
  [ $((SECONDS-t0)) -gt $GUARD_SECS ] && { HUNG=1; break; }
  sleep 30
done

if [ "$HUNG" -eq 1 ]; then
  HC=$(( $(cat $OUT/.hangcount 2>/dev/null || echo 0) + 1 )); echo $HC > $OUT/.hangcount
  echo "GUARD: COMPILE-HANG — no 'chunk 1/' in ${GUARD_SECS}s (consecutive hang #$HC); killing srun"
  kill $SRUN_PID 2>/dev/null; sleep 5; kill -9 $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null
  if [ "$HC" -ge 3 ]; then
    echo "DARS_3YR_STOP (3 consecutive compile-hangs — persistent fault; needs a human, chain halted)"; exit 1
  fi
  echo "GUARD: self-healing — resubmitting from $CUR/$TOTAL (restart intact, no data lost)"; do_resubmit; exit 0
fi

# healthy: chunk 1 reached ⇒ reset the consecutive-hang counter, then wait for the run to finish
echo 0 > $OUT/.hangcount
wait $SRUN_PID; R=$?
echo "dars3yr job#$CNT exit=$R"
if [ "$R" -ne 0 ]; then
  echo "DARS_3YR_STOP (exit=$R — non-finite/crash/kill; inspect $OUT, do NOT auto-continue)"; exit $R
fi

NEWCUR=$((CUR+STEPS))
if [ "$NEWCUR" -lt "$TOTAL" ]; then
  echo "resubmitting from $NEWCUR/$TOTAL"; do_resubmit
else
  echo "DARS_3YR_DONE ($NEWCUR steps = 3 model years)"
fi
