#!/bin/bash
# MULTI-PROCESS (2-node, 8-GPU) validation of the canonical (all_gather) RESTART writer. Builds a
# signature full State (every leaf = field_offset + gid), shards it folded across 8 GPUs (2 processes),
# writes a canonical restart via the multi-process all_gather path, and on rank 0 reconstructs every
# leaf from disk and asserts == the original — covering BOTH nod- and elem-kind State leaves. Prints
# RESTART_MN_CANONICAL_OK. Adjust --nodes (+ the dist_<N> partition = nodes*4) for other scales.
#SBATCH --job-name=restart_mn
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/validate_restart_mn.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/validate_restart_mn.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
srun --cpu-bind=none "$PY" scripts/validate_restart_mn.py
