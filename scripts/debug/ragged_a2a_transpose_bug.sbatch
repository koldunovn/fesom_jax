#!/bin/bash
# GPU-verify the self-contained ragged_all_to_all transpose-bug repro (for upstream filing).
# 2 GPUs, short. See docs/JAX_RAGGED_A2A_BUG.md.
#SBATCH --job-name=ragged_bug
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/ragged_bug.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/ragged_bug.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== node $(hostname) $(date) ==="; nvidia-smi -L 2>/dev/null | head -2
"$PY" scripts/debug/ragged_a2a_transpose_bug.py
echo ">>>> exit: $? ($(date))"
