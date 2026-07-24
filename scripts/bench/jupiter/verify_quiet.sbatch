#!/bin/bash
# QUIET-MACHINE VERIFICATION — run AFTER every other campaign job has drained (submit with
# --dependency=afterany:<all campaign jobids>), so this job owns its share of the fabric.
#
# Why this job exists.  During the campaign four of our own jobs (116 nodes) ran concurrently.
# Under that self-contention the PADDED transport collapsed ~4x at 8-16 GPUs (core2/8: 46 ms
# solo -> 204-224 ms contended, reproduced in diag_2node.sbatch legs A-D), while COLOURED
# barely moved (51.1 solo -> 49.1 contended).  Padded pads every halo to the global max and
# exchanges P-way, so its volume is large and bandwidth-bound; coloured ships ~1.0-1.4x the
# true halo in K rounds at ANY P.  Two things must therefore be checked on a quiet fabric:
#   1. the campaign's best-per-point (coloured) numbers are NOT depressed  -> re-measure them
#   2. the padded-vs-coloured comparison is FAIR                          -> re-run both
# Without (2) it would be dishonest to report "coloured beats padded 4x": that gap is a
# contention-robustness result, not a like-for-like transport result.
#
#   sbatch --dependency=afterany:JID1:JID2:JID3 scripts/bench/jupiter/verify_quiet.sbatch
#
#SBATCH --job-name=verify
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/verify_%j.out
#SBATCH -e scripts/logs/jupiter/verify_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1

echo "=== $(date) QUIET verification on $SLURM_JOB_NODELIST ==="
echo "=== concurrent jobs of mine (want: none but this one) ==="
squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true

CORE2="--name core2 --mesh-dir $MESH/mesh_core2 --dist-dir $RAW_core2 --ic-dir $MESH/mesh_core2 \
   --dt 1800 --full 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1 --year 1958 --steps 150"
DARS="--name dars --mesh-dir $MESH/mesh_dars --dist-dir $RAW_dars --ic-dir $MESH/mesh_dars \
   --dt 120 --full 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --year 1958 --steps 150"

# (2) the FAIR transport A/B, quiet fabric, both transports interleaved so any residual
#     drift hits both equally.
for REP in 1 2; do
  for TR in padded coloured; do
    echo "#### core2 8GPU (2N) halo=$TR rep=$REP dt_prod=1800 [QUIET] ####"
    srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/bench_forward_scaling.py $CORE2 --npes 8 --halo $TR || echo "rc=$?"
    echo "#### core2 16GPU (4N) halo=$TR rep=$REP dt_prod=1800 [QUIET] ####"
    srun -N4 -n4 --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/bench_forward_scaling.py $CORE2 --npes 16 --halo $TR || echo "rc=$?"
  done
done

# (1) re-measure the campaign's headline coloured points on a quiet fabric.
for REP in 1 2; do
  for TR in padded coloured; do
    echo "#### dars 32GPU (8N) halo=$TR rep=$REP dt_prod=240 [QUIET] ####"
    srun -N8 -n8 --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/bench_forward_scaling.py $DARS --npes 32 --halo $TR || echo "rc=$?"
  done
  echo "#### dars 64GPU (16N) halo=coloured rep=$REP dt_prod=240 [QUIET] ####"
  srun -N16 -n16 --ntasks-per-node=1 --cpu-bind=none \
    $PY scripts/bench/bench_forward_scaling.py $DARS --npes 64 --halo coloured || echo "rc=$?"
done

echo "=== done $(date) ==="
