#!/bin/bash
# Phase 8b B.3 — CORE2 (127k x nl48) FULL model on 2 GPU NODES (8 A100, dist_8),
# real JRA1958+PHC+ice, dt=1800. Multi-node CORE2 = comms-dominated (small mesh, ~15875k nod/GPU).
# ragged (point-to-point, the Kokkos-comparable halo) + allgather. Both fit at CORE2 scale (no OOM).
# vs Kokkos M524 CUDA CORE2: 1N 0.117 / 2N 0.095 / 4N 0.112 s/step (CORE2 stops scaling past 2N — over-decomp).
#SBATCH --job-name=bench_core2_2n
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_core2_2n.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_core2_2n.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
C="--name core2 --mesh-dir /home/a/a270088/port_jax/data/mesh_core2 --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --npes 8 --steps 25 --warmup 5 \
   --dt 1800 --full 1 --ic-dir /home/a/a270088/port_jax/data/ic_core2 --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST  CORE2 2N (8 GPU) ==="
echo "#### CORE2 FULL 8GPU ragged ####";    srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 1
echo "#### CORE2 FULL 8GPU allgather ####"; srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0
echo "=== done $(date) ==="
