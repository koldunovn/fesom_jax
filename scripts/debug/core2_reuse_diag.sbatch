#!/bin/bash
# Diagnose the reuse-exe bit-identity failure (fresh per-chunk forcing, per-leaf diff) + measure the
# persistent-compile-cache fallback. Runs BOTH modes in one allocation. Short (1 node, 4 GPU, 25 min).
#SBATCH --job-name=c2_rdiag
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_reuse_diag.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_reuse_diag.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" PYTHONUNBUFFERED=1
echo "=== $(date) (A) per-leaf divergence diagnosis (fresh per-chunk forcing) ==="
srun --ntasks=1 --cpu-bind=none "$PY" scripts/debug/core2_reuse_diag.py diag
echo "=== $(date) (B) persistent compilation cache speedup ==="
srun --ntasks=1 --cpu-bind=none "$PY" scripts/debug/core2_reuse_diag.py pcache
echo "=== done $(date) ==="
