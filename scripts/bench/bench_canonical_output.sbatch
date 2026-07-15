#!/bin/bash
# Single-node (4-GPU, 1 process) correctness + throughput of the output writers (folded / host_gather /
# all_gather / redistribute). Validates the GPU-only redistribute (ragged_all_to_all) path is canonical
# + byte-identical, then times each writer on a realistic CORE2 monthly payload. See PORTING_LESSONS
# ("Multi-process canonical output: all_gather beats no-gather redistribute").
#SBATCH --job-name=bench_canon
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_canonical_output.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_canonical_output.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1
srun --ntasks=1 --cpu-bind=none "$PY" scripts/bench/bench_canonical_output.py
