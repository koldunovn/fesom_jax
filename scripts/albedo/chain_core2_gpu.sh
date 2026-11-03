#!/bin/bash
# Albedo: a CORE2 (+REcoM) run as a chain of one-GPU segments sequenced by SLURM --dependency=afterok (the
# scripts/chain_submit.sh pattern; one segment = one scripts/albedo/run_core2_gpu.sbatch job within the 12 h qos).
#   scripts/albedo/chain_core2_gpu.sh CONFIG.yaml ROOT NSEG STEPS_PER_SEG YEAR [RESTART_IN]
# Segment k writes ROOT/seg<k>/{restart,monthly}; segment 0 cold-starts (or resumes RESTART_IN), segment k>0 resumes
# ROOT/seg<k-1>/restart. The monthly stores of all segments are linked into ROOT/monthly/<YYYY_MM> as they appear
# (scripts/albedo/link_monthly.sh ROOT), so the analysis scripts read one directory. --year is the config's start_year.
set -euo pipefail
CONFIG=${1:?CONFIG}; ROOT=${2:?ROOT}; NSEG=${3:?NSEG}; STEPS=${4:?STEPS_PER_SEG}; YEAR=${5:?YEAR}; PREV=${6:-}
mkdir -p "$ROOT"
DEP=""
for k in $(seq 0 $((NSEG - 1))); do
    OUT="$ROOT/seg$k"
    ARGS=(--steps "$STEPS" --year "$YEAR" --restart-out "$OUT/restart" --monthly-out "$OUT/monthly" --diagnostics)
    [ -n "$PREV" ] && ARGS+=(--restart-in "$PREV")
    if [ -n "$DEP" ]; then
        JID=$(sbatch --parsable --dependency=afterok:"$DEP" --export=ALL,OUT="$OUT",CONFIG="$CONFIG" \
              scripts/albedo/run_core2_gpu.sbatch "${ARGS[@]}")
    else
        JID=$(sbatch --parsable --export=ALL,OUT="$OUT",CONFIG="$CONFIG" scripts/albedo/run_core2_gpu.sbatch "${ARGS[@]}")
    fi
    echo "segment $k job $JID out $OUT ${PREV:+resumes $PREV}"
    DEP="$JID"; PREV="$OUT/restart"
done
echo "$DEP" > "$ROOT/last_job"
