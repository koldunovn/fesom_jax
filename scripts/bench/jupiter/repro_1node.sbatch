#!/bin/bash
# REPRODUCIBILITY PROBE, SINGLE-NODE POINTS (user request 2026-07-24: "machine is unstable,
# check yesterday's results still stand").
#
# WHY SINGLE-NODE FIRST: these points use ZERO inter-node fabric (all 4 GPUs are on one node,
# NVLink only).  So they (a) cannot be disturbed by, and cannot disturb, the ng5-128 remeasure
# running concurrently on 64 nodes, and (b) isolate the KERNEL from the network entirely.  If
# the single-node numbers reproduce, then yesterday's per-GPU performance and therefore the
# GH200-vs-A100 ratios are sound, and any instability is purely a fabric story.
#
# Campaign values to reproduce (job 1030278 / 1030279, 2026-07-23):
#   core2 1 GPU  coloured 93.2   padded 98.6/98.8
#   core2 2 GPU  coloured 64.1   padded 64.1/64.3
#   core2 4 GPU  coloured 50.0   padded 43.7/44.5      <- feeds the 2.01x A100 ratio
#   farc  4 GPU  coloured 156.5  padded 146.5          <- feeds the 1.93x A100 ratio
#
#   sbatch scripts/bench/jupiter/repro_1node.sbatch
#
#SBATCH --job-name=repro1n
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/repro1n_%j.out
#SBATCH -e scripts/logs/jupiter/repro1n_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
unset JDIST                                  # single node: no jax.distributed

CORE2="--name core2 --mesh-dir $MESH/mesh_core2 --dist-dir $RAW_core2 --ic-dir $MESH/mesh_core2 \
   --steps 150 --dt 1800 --full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1 --year 1958"
FARC="--name farc --mesh-dir $MESH/mesh_farc --dist-dir $RAW_farc --ic-dir $MESH/mesh_farc \
   --steps 150 --dt 900 --full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --year 1958"

echo "=== $(date) single-node reproducibility probe on $(hostname) ==="
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.max_limit --format=csv,noheader

# Interleave transports within each rep so any drift hits both equally.
for REP in 1 2; do
  for TR in padded coloured; do
    for NP in 1 2 4; do
      echo "#### core2 ${NP}GPU (1N) halo=$TR rep=$REP dt_prod=1800 [REPRO] ####"
      srun -N1 -n1 --ntasks-per-node=1 --cpu-bind=none \
        $PY scripts/bench/bench_forward_scaling.py $CORE2 --npes $NP --halo $TR || echo "rc=$?"
    done
    echo "#### farc 4GPU (1N) halo=$TR rep=$REP dt_prod=1200 [REPRO] ####"
    srun -N1 -n1 --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/bench_forward_scaling.py $FARC --npes 4 --halo $TR || echo "rc=$?"
  done
done
echo "=== done $(date) ==="
