#!/bin/bash
# Phase 8b B.3 — CORE2 (127k x nl48) FULL model on 4 GPU NODES (16 A100, dist_16),
# real JRA1958+PHC+ice, dt=1800. Multi-node CORE2 = comms-dominated (small mesh, ~7937k nod/GPU).
# ragged (point-to-point, the Kokkos-comparable halo) + allgather. Both fit at CORE2 scale (no OOM).
# vs Kokkos M524 CUDA CORE2: 1N 0.117 / 2N 0.095 / 4N 0.112 s/step (CORE2 stops scaling past 2N — over-decomp).
#SBATCH --job-name=bench_core2_4n
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_core2_4n.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_core2_4n.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
C="--name core2 --mesh-dir /home/a/a270088/port_jax/data/mesh_core2 --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --npes 16 --steps 25 --warmup 5 \
   --dt 1800 --full 1 --ic-dir /home/a/a270088/port_jax/data/ic_core2 --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST  CORE2 4N (16 GPU) ==="
echo "#### CORE2 FULL 16GPU ragged ####";    srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 1
echo "#### CORE2 FULL 16GPU allgather ####"; srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0
echo "=== done $(date) ==="
