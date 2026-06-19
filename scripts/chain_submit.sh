#!/bin/bash
# A6 example: chain N model segments with SLURM --dependency=afterok — the "no in-model
# orchestration" pattern. Each segment loads the previous restart, runs a fixed number of steps,
# writes the next restart, and exits; SLURM (not the model) sequences them. This is how a 2-3 yr
# NG5 ladder run is driven (one config, a series of dependent jobs), de-risked on farc/dars first.
#
# Usage:  scripts/chain_submit.sh CONFIG.yaml N_SEGMENTS STEPS_PER_SEGMENT [RESTART_ROOT]
# Example: scripts/chain_submit.sh configs/dars.yaml 6 4800 /work/ab0995/a270088/port_jax/runs/dars
#
# Segment 0 cold-starts from the PHC IC (no --restart-in); segment k>0 resumes
# RESTART_ROOT/seg<k-1> and writes RESTART_ROOT/seg<k>. A failing segment stops the chain
# (afterok) — the cheap localization the ladder gating wants.
set -euo pipefail

CONFIG="${1:?usage: chain_submit.sh CONFIG.yaml N_SEGMENTS STEPS_PER_SEGMENT [RESTART_ROOT]}"
NSEG="${2:?N_SEGMENTS}"
STEPS="${3:?STEPS_PER_SEGMENT}"
ROOT="${4:-runs/$(basename "${CONFIG%.yaml}")}"
mkdir -p "$ROOT"

DEP=""
PREV=""
for k in $(seq 0 $((NSEG - 1))); do
  OUT="$ROOT/seg$k"
  ARGS=(--restart-out "$OUT" --steps "$STEPS")
  if [ -n "$PREV" ]; then ARGS+=(--restart-in "$PREV"); fi   # k>0 resumes the prior restart

  if [ -n "$DEP" ]; then
    JID=$(sbatch --parsable --dependency=afterok:"$DEP" scripts/run_from_config.sbatch "$CONFIG" "${ARGS[@]}")
  else
    JID=$(sbatch --parsable scripts/run_from_config.sbatch "$CONFIG" "${ARGS[@]}")
  fi
  echo "segment $k -> job $JID  (restart_out=$OUT, depends_on=${DEP:-none})"
  DEP="$JID"
  PREV="$OUT"
done
echo "chain of $NSEG segments submitted; final restart = $ROOT/seg$((NSEG - 1))"
