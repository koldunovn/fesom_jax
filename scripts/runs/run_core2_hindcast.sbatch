#!/bin/bash
# Self-chaining CORE2 1958-2020 HINDCAST (JAX) — model-paper R1, the fidelity flagship.
# All-on physics (zstar + TKE + mEVP + GM, tracer (0,1)) at dt=1800 on dist_4 / 1 node (4 GPUs) —
# CORE2 ANTI-scales past 4 GPU (halo-bound), so 1 node is OPTIMAL. Each job resumes from the latest
# archival restart, runs a bounded step budget (< 12 h walltime), writes a restart + monthly-mean zarrs, and
# RESUBMITS itself until 63 model years (TOTAL = 1,104,528 steps = 1958-01-01 -> 2021-01-01) are done.
# The forcing reader rolls calendar years automatically from the absolute step (run.py _year_boundaries
# + reopen_year); a job's start_step is read from the restart metadata, so resumes land in the right year.
#
# Safety for the UNATTENDED multi-day run (mirrors run_dars_3yr.sbatch):
#  * the FIRST (cold) job is a short CANARY (1 model month) — the cold all-on CORE2 finiteness gate;
#    non-finite ⇒ run_from_config.py exits non-zero (NG5_R0_NONFINITE) and the chain STOPS.
#  * EVERY job ends with --diagnostics ⇒ a NaN never propagates into the next restart.
#  * a non-zero srun (non-finite / crash / walltime-kill) STOPS the chain (no resubmit) — inspect.
#  * chain continuity + crash recovery is the ARCHIVAL restart (--restart-archive-out, yearly +
#    unconditionally at the end of every job segment — 2026-07-02, replaces the old plain
#    --checkpoint-every rolling restart): each firing is an IMMUTABLE, uniquely-named directory,
#    never overwritten, resolved via restart.latest — so ANY past yearly restart stays resumable
#    for a branching experiment, not just the newest. A mid-segment CRASH (not a clean resubmit)
#    falls back to the last calendar-year boundary (bounded: ~44 min of CORE2 wall-clock at the
#    measured ~0.15 s/step steady rate) — accepted explicitly (user call, 2026-07-02).
#  * compile-hang guard: kill + self-heal-resubmit if "chunk 1/" never appears (a stuck XLA compile).
#  * runaway guard: a per-chain jobcount caps the chain length.
#SBATCH --job-name=core2_hind
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=11:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_core2_hindcast.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_core2_hindcast.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="   # cmdbuf off (compile dead-end, PORTING_LESSONS)
export PYTHONUNBUFFERED=1                              # 1 node, single process: NO JDIST, NO --ragged
# COMPILE-REUSE (the perf fix). The driver re-jits a fresh closure per chunk ⇒ a full ~27 s XLA
# recompile EVERY chunk (the perf decomposition found a 96 ms/step model running the hindcast at
# 562 ms/step). Two layers:
#  * FESOM_REUSE_EXE: run.py caches the compiled executable in-process and reuses it for same-shape
#    chunks ⇒ ~6 s/chunk (125 ms/step, ~4.5x). VALIDATED: reuse-vs-fresh divergence == fresh-vs-fresh
#    divergence (the multi-GPU forced path is non-deterministic at roundoff — atomic scatter-adds in
#    the halo/element→node/EVP reductions — so reuse adds NO error; negligible for a climatology run).
#  * persistent XLA disk cache: warms each job's first (compile) chunks from the prior job's cache
#    (provably identical executable). /work = persistent (not purged).
export FESOM_REUSE_EXE=1
export JAX_COMPILATION_CACHE_DIR=/work/ab0995/a270088/port_jax/jax_cache/core2_hindcast
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

CONFIG=${CONFIG:-$REPO/configs/core2_full.yaml}
TAG=${TAG:-core2_hindcast}
MESH=${MESH:-data/mesh_core2}
POOL=${POOL:-/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2}
PART=${PART:-dist_4}
IC=${IC:-data/ic_core2}
OUT=/work/ab0995/a270088/port_jax/runs/$TAG
ARCHIVE=$OUT/restart_archive     # the ONLY restart stream now (2026-07-02) — immutable, yearly + end-of-job
SELF=$REPO/scripts/runs/run_core2_hindcast.sbatch
TOTAL=${TOTAL:-1104528}        # 63 model years: 1958-01-01 -> 2021-01-01 (23011 days x 48 steps/day).
CANARY=1536                    # first (cold) job: 32 model days — de-risk finiteness + 1 month of output
# Every job in the chain REQUESTS the same fixed 11:45 walltime (the #SBATCH --time above) regardless
# of PERJOB — Slurm schedules/backfills on the REQUEST, not on how long the job actually runs, so a
# smaller PERJOB does NOT queue faster; it just finishes its request early and pays a fresh queue-wait
# for the next leg. So PERJOB should use as much of the 11:45 request as safely possible, not as little.
# MEASURED (core2_hindcast_v2 R1 rerun, live log): steady-state chunk wall ~0.15 s/step (host+device+I/O,
# FESOM_REUSE_EXE hot) confirmed over 2 consecutive production jobs. 240000 steps ~= 42(setup) + 50(compile
# warmup) + 240000*0.15(compute) + ~14(yearly archival-restart writes, ~10s each) + 10(final archival write)
# ~= 10.5 h, leaving ~75 min margin under 11:45. ~5 production jobs for the 63 yr (down from ~8 at the old
# 150000, which was sized for the SLOWER pcache-only fallback ~0.24 s/step ⇒ ~16.6 h at 240000 steps — i.e.
# this sizing trades the worst-case-fallback safety margin for using the (now twice-validated) fast rate.
# If reuse ever regresses on some future leg, that job walltime-hard-kills (SIGKILL, no graceful exit) ⇒ the
# in-script auto-resubmit never fires (it runs only after a clean exit) ⇒ the chain silently STALLS until
# someone notices and manually resubmits — data loss is bounded to <=1 calendar year (the archival restart
# cadence, see the header) either way, since the archival stream is independent of PERJOB.
PERJOB=${PERJOB:-240000}
CHUNK=48                       # perf knob only: the monthly mean now accumulates EVERY step (run.py
                               # _MeanStream, split at month boundaries) ⇒ a TRUE time-mean that averages
                               # the diurnal cycle at any dt. CHUNK no longer sets the sampling cadence
                               # (the old 1-day value froze a fixed time-of-day ⇒ the wavenumber-1 SST
                               # artifact); larger CHUNK = fewer chunks = faster, bounded by forcing memory.

mkdir -p $OUT
# runaway guard (resume phase): cap chain length above the ~20 real jobs expected (19 prod + canary,
# + slack for self-heal resubmits).
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
echo "=== $(date) core2_hindcast job#$CNT: CUR=$CUR / TOTAL=$TOTAL (restart=${RESTART_DIR:-none}) ==="
if [ "$CUR" -ge "$TOTAL" ]; then echo "CORE2_HINDCAST_DONE ($CUR steps = 63 model years)"; exit 0; fi

# step budget for THIS job: short canary on the cold start, else the per-job budget (capped at remaining).
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

LOGFILE=$REPO/scripts/run_core2_hindcast.${SLURM_JOB_ID}.out   # this job's own -o file (srun output, live)
do_resubmit() {   # resubmit the chain from the latest archival restart (hang-heal + normal-completion paths)
  cd $REPO && sbatch \
    --export=ALL,CONFIG="$CONFIG",TAG="$TAG",MESH="$MESH",POOL="$POOL",PART="$PART",IC="$IC",PERJOB="$PERJOB",TOTAL="$TOTAL" $SELF
}

# Run in the BACKGROUND so the compile-hang guard can supervise the first chunk.
srun --ntasks=1 --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$IC" --partition "$PART" \
    --steps "$STEPS" --chunk-steps "$CHUNK" \
    --diagnostics --progress \
    --monthly-out "$OUT/monthly" --monthly-start-step 0 \
    --restart-out "$OUT/restart" \
    --restart-archive-out "$ARCHIVE" --restart-archive-period year --restart-archive-length 1 \
    $RIN &
SRUN_PID=$!

# COMPILE-HANG GUARD: a healthy job prints "chunk 1/" within minutes (setup + XLA compile). No
# "chunk 1/" within GUARD_SECS ⇒ kill + self-heal (resubmit from the intact restart). A consecutive-
# hang counter STOPS a hang-LOOP after 3 tries rather than burning the node budget.
GUARD_SECS=${GUARD_SECS:-900}
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
    echo "CORE2_HINDCAST_STOP (3 consecutive compile-hangs — persistent fault; needs a human)"; exit 1
  fi
  echo "GUARD: self-healing — resubmitting from $CUR/$TOTAL (restart intact, no data lost)"; do_resubmit; exit 0
fi

# healthy: chunk 1 reached ⇒ reset the consecutive-hang counter, then wait for the run to finish
echo 0 > $OUT/.hangcount
wait $SRUN_PID; R=$?
echo "core2_hindcast job#$CNT exit=$R"
if [ "$R" -ne 0 ]; then
  echo "CORE2_HINDCAST_STOP (exit=$R — non-finite/crash/kill; inspect $OUT, do NOT auto-continue)"; exit $R
fi

NEWCUR=$((CUR+STEPS))
if [ "$NEWCUR" -lt "$TOTAL" ]; then
  echo "resubmitting from $NEWCUR/$TOTAL"; do_resubmit
else
  echo "CORE2_HINDCAST_DONE ($NEWCUR steps = 63 model years)"
fi
