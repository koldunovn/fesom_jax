#!/bin/bash
# Self-chaining FORCA20 1958-1960 (3 model yr) HINDCAST (JAX) — the R1-rerun equivalent for FORCA20:
# lands the 2026-07-01 salinity + SST-true-mean fixes, canonical output, and the extended daily-output
# fields (ssh, sea ice, surface/100m/500m u,v) on a FRESH cold start. The OLD forca20_3yr chain (run via
# a repointed run_dars_3yr.sbatch, a filename trap) is STALLED at 62% (cancelled 2026-06-27, predates
# both fixes) — do NOT resume it; this is a clean restart under its own name and a new TAG.
#
# Physics: TKE + mEVP (whichEVP=1) + zstar, GM off, dt 180 (yr1) -> 240 (yr2+) — configs/forca20_tke_mevp_prod.yaml.
# Partition: dist_16 (4 nodes / 16 GPU) — chosen for queue-wait balance over dist_32/64 (2026-07-01
# scaling bench: dist_16 compiles cleanly, ~0.61 s/step steady-state, no compile-hang seen; dist_32/64
# comparison still pending but 4-node allocations queue much faster than 8/16-node ones).
#
# Safety (mirrors run_core2_hindcast.sbatch + the multi-node bits from run_dars_3yr.sbatch):
#  * first (cold) job is a CANARY (1440 steps = 30 model days) — de-risk finiteness at dist_16 cheaply.
#  * every job ends with --diagnostics ⇒ a NaN never propagates into the next restart.
#  * a non-zero srun (non-finite/crash/walltime-kill) STOPS the chain (no resubmit) — inspect.
#  * chain continuity + crash recovery is the ARCHIVAL restart (--restart-archive-out, MONTHLY +
#    unconditionally at the end of every job segment — 2026-07-02, replaces the old plain
#    --checkpoint-every rolling restart): each firing is an IMMUTABLE, uniquely-named directory,
#    never overwritten, resolved via restart.latest — so ANY past monthly restart stays resumable
#    for a branching experiment, not just the newest. A mid-segment CRASH (not a clean resubmit)
#    falls back to the last calendar-month boundary (bounded: ~2.4 h of FORCA20 wall-clock at the
#    measured ~0.6 s/step steady rate) — accepted explicitly (user call, 2026-07-02). Monthly
#    archival restarts cost ~27 GB each (this mesh) — over the 3-yr run that's ~36 restarts ≈ 972 GB
#    kept forever (no auto-pruning, matches FESOM3's own restart_keep=0 default) — a deliberate,
#    user-confirmed storage tradeoff, not an oversight.
#  * compile-hang guard: kill + self-heal-resubmit if "chunk 1/" never appears.
#  * runaway guard: a per-chain jobcount caps the chain length.
#SBATCH --job-name=forca20_hind
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=11:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_forca20_hindcast.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_forca20_hindcast.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1
export JAX_COMPILATION_CACHE_DIR=/work/ab0995/a270088/port_jax/jax_cache/forca20_hindcast
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

CONFIG=${CONFIG:-$REPO/configs/forca20_tke_mevp_prod.yaml}
TAG=${TAG:-forca20_hindcast_v2}
MESH=${MESH:-/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20}
POOL=${POOL:-/work/ab0995/a270088/meshes/FORCA20}
PART=${PART:-dist_16}
IC=${IC:-$MESH}
NODES=${NODES:-4}
OUT=/work/ab0995/a270088/port_jax/runs/$TAG
ARCHIVE=$OUT/restart_archive     # the ONLY restart stream now (2026-07-02) — immutable, monthly + end-of-job
SELF=$REPO/scripts/runs/run_forca20_hindcast.sbatch
TOTAL=${TOTAL:-438360}         # 3 model years: 1958 dt=180 (175200) + 1959-60 dt=240 (263160 =
                               # 731 days INCL. the 1960 leap day — the old 438000/262800 was
                               # 730 days and ended 1960-12-31, one day short; 2026-07-03 review).
CANARY=1440                    # first (cold) job: 30 model days — de-risk finiteness at dist_16 cheaply
PERJOB=${PERJOB:-55000}        # ~10.5h margin under 11:45 at the measured ~0.61 s/step + checkpoint I/O
CHUNK=48

mkdir -p $OUT
CNT=$( cat $OUT/.jobcount 2>/dev/null || echo 0 ); CNT=$((CNT+1)); echo $CNT > $OUT/.jobcount
if [ "$CNT" -gt 35 ]; then echo "ABORT: jobcount $CNT > 35 (runaway guard)"; exit 1; fi

# current restart = whatever restart.latest in the archive resolves to (empty ⇒ cold). Source of
# truth (survives crashes/kills) — a plain pointer-file read (mirrors zarr_output.resolve_latest_restart,
# reimplemented in pure bash so this doesn't drag in `import jax` just to read one line of text — that
# noisily fails CUDA init outside srun's GPU-scoped allocation and needlessly slows every resubmit).
RESTART_DIR=""
if [ -f "$ARCHIVE/restart.latest" ]; then
  LTAG=$(cat "$ARCHIVE/restart.latest"); [ -n "$LTAG" ] && RESTART_DIR="$ARCHIVE/$LTAG"
fi
CUR=0
if [ -n "$RESTART_DIR" ]; then
  # the pointer EXISTS ⇒ an unreadable step must STOP the chain, not silently cold-start —
  # the old `|| echo 0` fallback here was the mechanism class of the 2026-07-03 data loss
  # (a cold start overwrites period stores + same-tag archival dirs). Transient /work read
  # errors land here too: inspect + resubmit manually.
  CUR=$( $PY -c "import json;print(int(json.load(open('$RESTART_DIR/.zattrs')).get('step',0)))" 2>/dev/null || echo FAIL )
  case "$CUR" in ''|*[!0-9]*)
    echo "ABORT: restart pointer -> $RESTART_DIR but its step is unreadable (corrupted .zattrs or transient FS error) — refusing the silent cold-start fallback"; exit 1;;
  esac
fi
echo "=== $(date) forca20_hindcast job#$CNT: CUR=$CUR / TOTAL=$TOTAL (restart=${RESTART_DIR:-none}) ==="
if [ "$CUR" -ge "$TOTAL" ]; then echo "FORCA20_HINDCAST_DONE ($CUR steps = 3 model years)"; exit 0; fi

if [ "$CUR" -eq 0 ]; then
  # COLD start: refuse when this chain has EVER archived (a cold start would overwrite the
  # existing period stores under $OUT) — no silent deletion/overwrite in unattended chains.
  # A deliberate re-run over an old tag needs an explicit FORCE_COLD=1 (or a fresh TAG).
  if [ -n "$(ls -A "$ARCHIVE" 2>/dev/null)" ] && [ "${FORCE_COLD:-0}" != "1" ]; then
    echo "ABORT: cold start requested but $ARCHIVE is non-empty — refusing to overwrite an existing chain (FORCE_COLD=1 overrides)"; exit 1
  fi
  STEPS=$CANARY; RIN=""; mkdir -p $OUT; echo $CNT > $OUT/.jobcount
else
  REM=$((TOTAL-CUR)); STEPS=$PERJOB; [ "$STEPS" -gt "$REM" ] && STEPS=$REM; RIN="--restart-in $RESTART_DIR"
fi
echo "running $STEPS steps from $CUR (${RIN:-cold})"

LOGFILE=$REPO/scripts/run_forca20_hindcast.${SLURM_JOB_ID}.out
do_resubmit() {   # resubmit the chain from the latest archival restart (hang-heal + normal-completion paths)
  cd $REPO && sbatch --nodes=$NODES \
    --export=ALL,CONFIG="$CONFIG",TAG="$TAG",MESH="$MESH",POOL="$POOL",PART="$PART",IC="$IC",NODES="$NODES",PERJOB="$PERJOB",TOTAL="$TOTAL" $SELF
}

srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$IC" --partition "$PART" \
    --steps "$STEPS" --chunk-steps "$CHUNK" --ragged --local-forcing \
    --diagnostics --progress \
    --monthly-out "$OUT/monthly" --monthly-start-step 0 \
    --daily-out "$OUT/daily" --daily-start-step 0 \
    --restart-out "$OUT/restart" \
    --restart-archive-out "$ARCHIVE" --restart-archive-period month --restart-archive-length 1 \
    $RIN &
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
  echo "GUARD: COMPILE-HANG — no 'chunk 1/' in ${GUARD_SECS}s (consecutive hang #$HC); killing srun"
  kill $SRUN_PID 2>/dev/null; sleep 5; kill -9 $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null
  if [ "$HC" -ge 3 ]; then
    echo "FORCA20_HINDCAST_STOP (3 consecutive compile-hangs — persistent fault; needs a human)"; exit 1
  fi
  echo "GUARD: self-healing — resubmitting from $CUR/$TOTAL (restart intact, no data lost)"; do_resubmit; exit 0
fi

echo 0 > $OUT/.hangcount
wait $SRUN_PID; R=$?
echo "forca20_hindcast job#$CNT exit=$R"
if [ "$R" -ne 0 ]; then
  echo "FORCA20_HINDCAST_STOP (exit=$R — non-finite/crash/kill; inspect $OUT, do NOT auto-continue)"; exit $R
fi

NEWCUR=$((CUR+STEPS))
if [ "$NEWCUR" -lt "$TOTAL" ]; then
  echo "resubmitting from $NEWCUR/$TOTAL"; do_resubmit
else
  echo "FORCA20_HINDCAST_DONE ($NEWCUR steps = 3 model years)"
fi
