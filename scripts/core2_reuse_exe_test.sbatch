#!/bin/bash
# Bit-identity + speedup test for the executable-reuse fix (reuse_executable=True). Short (1 node,
# 4 GPU, 20 min ⇒ backfill-friendly). Must print REUSE_BITIDENTICAL_OK before enabling FESOM_REUSE_EXE
# on the production hindcast chain.
#SBATCH --job-name=c2_reuse
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_reuse_exe_test.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_reuse_exe_test.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" PYTHONUNBUFFERED=1
echo "=== $(date) executable-reuse bit-identity test ==="
srun --ntasks=1 --cpu-bind=none "$PY" scripts/core2_reuse_exe_test.py
echo "=== done $(date) ==="
