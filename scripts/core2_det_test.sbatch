#!/bin/bash
#SBATCH --job-name=c2_det
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_det_test.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_det_test.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" PYTHONUNBUFFERED=1
echo "=== $(date) determinism test (fresh vs fresh) ==="
srun --ntasks=1 --cpu-bind=none "$PY" scripts/core2_reuse_diag.py det
echo "=== done $(date) ==="
