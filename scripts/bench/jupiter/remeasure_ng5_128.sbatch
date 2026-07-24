#!/bin/bash
# REMEASURE the ng5 128-GPU anomaly (user request 2026-07-24).
#
# THE ANOMALY.  In the campaign (job 1030306) ng5/coloured measured:
#   P=32 316.8 | P=64 243.4 | **P=128 644.0 / 638.5** | P=256 257.2 / 260.4
# A point 2.6x slower than its neighbours ON BOTH SIDES cannot be a strong-scaling
# turnover — a turnover does not recover at 2x the device count.
#
# WHAT IS ALREADY RULED OUT (do not re-test these):
#  - Partition quality: ng5 dist_128 is as balanced as every other rung
#    (myDim max/mean 1.006, padfac 1.013).
#  - A single bad node: the two slow reps ran on DISJOINT 32-node subsets
#    (jpbo-038-[17-48] and jpbo-036-[33-48]+jpbo-037-[01-16]) and agreed to 0.9%,
#    while the UNION of those same 64 nodes ran P=256 at 257 ms.
#  - Colouring rounds K: K=14 at P=128 but K=15 at P=256, which is FAST.
#
# THE LIVE CLUE — compile time.  NOTE: compile is NOT part of the measurement — the
# bench times the 2nd warm call on an already-compiled executable, so `per_step` is
# pure run time and compile never enters it.  Compile is used here ONLY as a
# diagnostic about the CAUSE, and it is ~2x at exactly P=128:
#    P=32 75-87 s | P=64 76 s | **P=128 155-166 s** | P=256 69-75 s
# Compile is host-side XLA work, so a slow network cannot inflate it.  Runtime and
# compile being ~2x at the SAME single value of P suggests the COMPILED GRAPH DIFFERS
# there (a different/worse XLA lowering at that shape) — a mechanism that would slow
# the measured per_step for reasons unrelated to the fabric.  That is a hypothesis to
# test, not a conclusion.
#
# WHAT THIS JOB DISCRIMINATES:
#  1. Fresh allocation (different nodes entirely) -> is it reproducible at all?
#  2. P=128 under padded and ragged, which the campaign NEVER ran at 128
#     (AB_MAX_NPES=64 capped the A/B). If padded/ragged are ALSO ~2x slow with ~2x
#     compile, the anomaly is P-SPECIFIC (graph/lowering). If they are normal, it is
#     COLOURED-SPECIFIC at P=128.
#  3. In-allocation anchors at P=64 and P=256, run BEFORE and AFTER, so the 128
#     numbers are interpretable and any drift during the job is visible.
#
#   sbatch scripts/bench/jupiter/remeasure_ng5_128.sbatch
#
#SBATCH --job-name=ng5_128
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=64
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/ng5128_%j.out
#SBATCH -e scripts/logs/jupiter/ng5128_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1
RUN_TIMEOUT="${RUN_TIMEOUT:-3000}"

C="--name ng5 --mesh-dir $MESH/mesh_ng5 --dist-dir $RAW_ng5 --ic-dir $MESH/mesh_ng5 \
   --steps 150 --dt 180 --full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --year 1958"

# Node subsets so the P=128 legs can be placed on two DISJOINT halves of this
# allocation (the campaign's two slow reps were disjoint too — this repeats that
# control on fresh hardware).
ALL=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
HALF_A=$(echo "$ALL" | head -32 | paste -sd, -)
HALF_B=$(echo "$ALL" | tail -32 | paste -sd, -)

echo "=== $(date) ng5-128 REMEASURE on $SLURM_JOB_NODELIST ==="
echo "=== HALF_A=$HALF_A"
echo "=== HALF_B=$HALF_B"
echo "=== other jobs of ours (want: none) ==="; squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true

# run <label> <nodes> <npes> <halo> [nodelist]
run () {
  local label="$1" N="$2" NP="$3" TR="$4" NL="${5:-}"
  local place=""; [ -n "$NL" ] && place="--nodelist=$NL"
  echo "#### ng5 ${NP}GPU (${N}N) halo=$TR dt_prod=240 [$label] ####"
  timeout --kill-after=120 "$RUN_TIMEOUT" \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none $place \
      $PY scripts/bench/bench_forward_scaling.py $C --npes "$NP" --halo "$TR"
  local rc=$?
  [ $rc -eq 0 ] || echo "#### [$label] rc=$rc $([ $rc -ge 124 ] && echo "(TIMEOUT)") ####"
}

# --- anchors FIRST: prove this allocation reproduces the campaign's good points ---
run anchor-pre-64  16 64  coloured
run anchor-pre-256 64 256 coloured

# --- THE POINT: P=128, coloured, three reps across BOTH disjoint halves ---
run p128-coloured-A1 32 128 coloured "$HALF_A"
run p128-coloured-B1 32 128 coloured "$HALF_B"
run p128-coloured-A2 32 128 coloured "$HALF_A"

# --- P=128 under the transports the campaign never ran there ---
#     (padded/ragged at 128 were capped out by AB_MAX_NPES=64)
run p128-padded-A1 32 128 padded "$HALF_A"
run p128-padded-B1 32 128 padded "$HALF_B"
run p128-ragged-A1 32 128 ragged "$HALF_A"

# --- anchors LAST: same two points again -> any drift during the job is visible ---
run anchor-post-64  16 64  coloured
run anchor-post-256 64 256 coloured

echo "=== done $(date) ==="
echo "=== per_step is THE measurement (compile is excluded from it — 2nd warm call)."
echo "=== compile= is only a DIAGNOSTIC: it is host-side, so if it too is ~2x at P=128"
echo "=== the cause is the compiled graph, not the fabric. ==="
