#!/bin/bash
# THE JUPITER TRANSPORT MAP — measure, from scratch, which halo transport wins at each
# (mesh, P) on THIS machine.  Do not inherit Levante's "padded <=16, coloured beyond": the
# fabric is a different machine.  Levante is HDR-class with ~1 NIC/node; JUPITER is NDR with
# **4x200 Gb/s per node, one NIC per GH200 superchip**, i.e. roughly 4x the injection
# bandwidth per node.  A bandwidth-hungry transport can afford to be bandwidth-hungry here,
# so the crossovers must move -- and the campaign data already shows they do (core2/8 on a
# QUIET fabric: padded 46.2 ms BEATS coloured 51.1 ms; Levante's rule would have predicted
# padded winning too at P=8, but it also predicted padded winning to P=16, and the point of
# this job is to stop predicting).
#
# THE PROTOCOL THAT MAKES IT VALID: run with NOTHING ELSE OF OURS QUEUED.  During the main
# campaign four of our own jobs (116 nodes) ran concurrently and the padded points inflated
# ~4x (core2/8: 46 ms solo -> 204-224 ms contended; diag_2node.sbatch legs A-D isolate it).
# Coloured barely moved (51.1 -> 49.1). Contention sensitivity is itself a result, but it is
# NOT the like-for-like transport comparison, and the two must not be conflated.
#
#   sbatch scripts/bench/jupiter/transport_map_jupiter.sbatch      # only when the queue is ours alone
#
# Transports measured: padded, coloured, ragged.
#   - ragged (lax.ragged_all_to_all) ships only the true halo but its cost is a P-WAY
#     all-to-all, so it grows with the device count -- on a fat fabric it should hold out
#     further than it did on Levante (where it owned only a ~32-GPU middle band).
#     CAVEAT: ragged is FORWARD-ONLY in this project (JAX's transpose is wrong, O(1) error --
#     docs/JAX_RAGGED_A2A_BUG.md).  This benchmark is forward-only so it is fair game here,
#     but a point whose winner is `ragged` cannot be used for sharded GRADIENTS; those must
#     fall back to the best of coloured/padded.  The map below reports all three so that
#     fallback is a known number, not a guess.
#   - allgather is omitted: known slowest of the four, and O(global) per device.
#
#SBATCH --job-name=tmap
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/tmap_%j.out
#SBATCH -e scripts/logs/jupiter/tmap_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1

TRANSPORTS="${TRANSPORTS:-padded coloured ragged}"
REPS="${REPS:-2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"

# (mesh:npes) points spanning the regimes: small-mesh/latency-bound -> big-mesh/bandwidth-bound.
POINTS="${POINTS:-core2:8 core2:16 farc:8 farc:32 dars:32 dars:64 ng5:64 ng5:128}"

echo "=== $(date) JUPITER TRANSPORT MAP on $SLURM_JOB_NODELIST ==="
echo "=== QUIET-FABRIC CHECK — other jobs of ours (want: only this one) ==="
squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true
echo "=== points=[$POINTS] transports=[$TRANSPORTS] reps=$REPS ==="

for PT in $POINTS; do
  M="${PT%%:*}"; NP="${PT##*:}"
  N=$(( (NP + GPUS_PER_NODE - 1) / GPUS_PER_NODE )); [ "$N" -lt 1 ] && N=1
  RAW_VAR="RAW_${M}"; DDIR="${!RAW_VAR}"
  case "$M" in
    core2) DT=1800; DTP=1800; PHYS="--mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1" ;;
    farc)  DT=900;  DTP=1200; PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
    dars)  DT=120;  DTP=240;  PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
    ng5)   DT=180;  DTP=240;  PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
  esac
  C="--name $M --mesh-dir $MESH/mesh_$M --dist-dir $DDIR --ic-dir $MESH/mesh_$M \
     --steps 150 --dt $DT --full 1 $PHYS --year 1958 --npes $NP"
  # Interleave transports WITHIN a rep so any slow drift in the allocation hits all three
  # equally -- a transport measured entirely in the first half would otherwise win/lose on
  # timing rather than merit.
  for REP in $(seq 1 "$REPS"); do
    for TR in $TRANSPORTS; do
      echo "#### $M ${NP}GPU (${N}N) halo=$TR rep=$REP dt_prod=$DTP [TMAP] ####"
      timeout --kill-after=120 "$RUN_TIMEOUT" \
        srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
          $PY scripts/bench/bench_forward_scaling.py $C --halo "$TR"
      rc=$?
      [ $rc -eq 0 ] || echo "#### $M ${NP}GPU halo=$TR rep=$REP rc=$rc $([ $rc -ge 124 ] && \
        echo "(TIMEOUT ${RUN_TIMEOUT}s)") ####"
    done
  done
done
echo "=== done $(date) ==="
