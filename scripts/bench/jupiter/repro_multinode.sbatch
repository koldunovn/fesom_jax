#!/bin/bash
# REPRODUCIBILITY PROBE, MULTI-NODE POINTS (user request 2026-07-24: "machine is unstable,
# check yesterday's results still stand").
#
# MUST run with nothing else of ours queued — chain it:
#   sbatch --dependency=afterany:<ng5_128 jobid> scripts/bench/jupiter/repro_multinode.sbatch
# Yesterday's campaign ran FOUR of our jobs concurrently (116 nodes); that self-contention is
# one of the things this probe exists to control for, so re-creating it here would be
# self-defeating.
#
# These are the points the paper's two headline claims rest on, so they are the ones that must
# reproduce.  Campaign values (coloured, 2026-07-23):
#   dars  16  183.0 | 32  133.4 | 64  121.1 | 128 137.5   <- THE "dars turns over at 64" claim
#   farc   8  156.3 | 16  170.3                            <- THE "farc peaks at one node" claim
#   core2  8   48.9 | 16   60.8
# All are ALSO re-run in a single allocation, which is how the campaign ran them — so a
# disagreement here is a machine-state difference, not a protocol difference.
#
#SBATCH --job-name=repromn
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/repromn_%j.out
#SBATCH -e scripts/logs/jupiter/repromn_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"

DARS="--name dars --mesh-dir $MESH/mesh_dars --dist-dir $RAW_dars --ic-dir $MESH/mesh_dars \
   --steps 150 --dt 120 --full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --year 1958"
FARC="--name farc --mesh-dir $MESH/mesh_farc --dist-dir $RAW_farc --ic-dir $MESH/mesh_farc \
   --steps 150 --dt 900 --full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --year 1958"
CORE2="--name core2 --mesh-dir $MESH/mesh_core2 --dist-dir $RAW_core2 --ic-dir $MESH/mesh_core2 \
   --steps 150 --dt 1800 --full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1 --year 1958"

echo "=== $(date) multi-node reproducibility probe on $SLURM_JOB_NODELIST ==="
echo "=== other jobs of ours (want: none) ==="; squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true

run () {   # run <cfgvar> <npes> <halo> <rep>
  local CFG="$1" NP="$2" TR="$3" REP="$4"
  local N=$(( (NP + 3) / 4 ))
  local NAME=$(echo "$CFG" | tr 'A-Z' 'a-z')
  echo "#### $NAME ${NP}GPU (${N}N) halo=$TR rep=$REP [REPRO] ####"
  timeout --kill-after=120 "$RUN_TIMEOUT" \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/bench_forward_scaling.py ${!CFG} --npes "$NP" --halo "$TR"
  local rc=$?
  [ $rc -eq 0 ] || echo "#### $NAME ${NP}GPU halo=$TR rep=$REP rc=$rc ####"
}

# THE HEADLINE CURVE FIRST — dars, whose turnover at 64 is the paper's key scaling claim.
for REP in 1 2; do
  for NP in 16 32 64 128; do run DARS "$NP" coloured "$REP"; done
done

# farc: the "peaks at one node" claim rests on 8 and 16 being SLOWER than the 4-GPU point.
for REP in 1 2; do
  for NP in 8 16; do run FARC "$NP" coloured "$REP"; done
done

# core2 multi-node.
for REP in 1 2; do
  for NP in 8 16; do run CORE2 "$NP" coloured "$REP"; done
done

echo "=== done $(date) ==="
