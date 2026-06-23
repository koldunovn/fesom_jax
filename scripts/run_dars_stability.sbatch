#!/bin/bash
# dars SUSTAINED-STABILITY confirmation (single bounded job): TKE + mEVP + zstar + w_split, dt=120,
# dist_64 (16 nodes — clears the TKE+mEVP XLA compile cliff that hung at dist_16). Runs ~5 model days
# (3600 steps @ dt=120) — 36x past the 100-step canary, well beyond the cold transient — with per-chunk
# health logging (max|uv|/cfl_z/per-level max|T|) so the velocity SETTLING is visible over days, plus
# rolling checkpoints for crash-safety. NOT self-chaining; extend STEPS for a longer confirmation.
#SBATCH --job-name=dars_stab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_dars_stability.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_dars_stability.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1

CONFIG=$REPO/configs/dars_tke_mevp_wsplit_dt120.yaml
MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
PART=dist_64
STEPS=${STEPS:-3600}            # 5 model days @ dt=120 (720 steps/day)
CHUNK=${CHUNK:-48}             # diag every 48 steps (~10/day) — watch the velocity settle
OUT=/work/ab0995/a270088/port_jax/runs/dars_tke_mevp_dt120
RESTART=$OUT/restart
mkdir -p $OUT
# resume from the rolling restart if present (crash-safety / extend); else cold start
CUR=$( $PY -c "import json;print(int(json.load(open('$RESTART/.zattrs')).get('step',0)))" 2>/dev/null || echo 0 )
[ -z "$CUR" ] && CUR=0
if [ "$CUR" -gt 0 ]; then RIN="--restart-in $RESTART"; REM=$((STEPS-CUR)); else RIN=""; REM=$STEPS; fi
echo "=== $(date) dars stability: CUR=$CUR target=$STEPS (running $REM, ${RIN:-cold}) dist_64/16 nodes ==="
[ "$REM" -le 0 ] && { echo "already at target ($CUR>=$STEPS)"; exit 0; }

LOGFILE=$REPO/scripts/run_dars_stability.${SLURM_JOB_ID}.out   # this job's -o file (srun output, live)

# Run in BACKGROUND so a compile-hang guard supervises chunk 1. The dars TKE+mEVP compile hung once at
# dist_64 (2026-06-23, ~2h idle at 'setup done') — the same intermittent XLA hang as the production chain.
# Healthy => 'chunk 1/' within minutes; hang => stuck at 'setup done'. No chunk 1 in GUARD_SECS => kill +
# self-heal (resubmit this job from the rolling restart). 3 consecutive hangs => STOP (persistent fault).
srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition "$PART" \
    --steps "$REM" --chunk-steps "$CHUNK" --ragged --local-forcing \
    --checkpoint-every 720 --chunk-diagnostics --diagnostics --progress \
    --restart-out "$RESTART" $RIN &
SRUN_PID=$!

GUARD_SECS=${GUARD_SECS:-1800}
t0=$SECONDS; HUNG=0
while kill -0 $SRUN_PID 2>/dev/null; do
  grep -q "chunk 1/" "$LOGFILE" 2>/dev/null && break
  [ $((SECONDS-t0)) -gt $GUARD_SECS ] && { HUNG=1; break; }
  sleep 30
done

if [ "$HUNG" -eq 1 ]; then
  HC=$(( $(cat $OUT/.hangcount 2>/dev/null || echo 0) + 1 )); echo $HC > $OUT/.hangcount
  echo "GUARD: COMPILE-HANG — no 'chunk 1/' in ${GUARD_SECS}s (consecutive #$HC); killing srun ($(date))"
  kill $SRUN_PID 2>/dev/null; sleep 5; kill -9 $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null
  if [ "$HC" -ge 3 ]; then echo "DARS_STAB_STOP (3 consecutive compile-hangs — needs a human)"; exit 1; fi
  echo "GUARD: self-healing — resubmitting ($(date))"; cd $REPO && sbatch scripts/run_dars_stability.sbatch; exit 0
fi

echo 0 > $OUT/.hangcount
wait $SRUN_PID; R=$?
echo "dars stability exit=$R ($(date))"
[ "$R" -eq 0 ] && echo "DARS_STAB_OK (TKE+mEVP+w_split dt=120 ran $STEPS steps = $((STEPS/720)) model days)" \
              || echo "DARS_STAB_FAIL (exit=$R — non-finite/crash/kill; inspect $OUT)"
