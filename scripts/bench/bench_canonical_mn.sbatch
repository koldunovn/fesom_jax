#!/bin/bash
# MULTI-PROCESS (2-node, 8-GPU) validation + throughput of the canonical output writers. Each process
# sees only its LOCAL shards, so this exercises the distributed write path (rank-0 store create +
# barrier + per-process disjoint chunk writes) and the true multi-node collective cost. Validates
# all_gather + redistribute are canonical + byte-identical, and times folded vs all_gather vs
# redistribute. Adjust --nodes (+ the dist_<N> partition the .py reads = nodes*4) to test other scales.
#SBATCH --job-name=bench_canon_mn
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_canonical_mn.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_canonical_mn.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
srun --cpu-bind=none "$PY" scripts/bench/bench_canonical_mn.py
