#!/bin/bash
# Phase 6 Task 6.7 (GATE 6) — CORE2 prognostic-ice gradient gate on an A100.
# The backward pass (the EVP 120-subcycle scan nested in the N-step time-loop scan + the
# thermo Newton + the FCT limiter) is memory-heavy → GPU. Small N (the EVP scan inflates
# memory vs the Phase-5 37.8 GB @ N=20).
#SBATCH --job-name=core2_ice_grad
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_ice_grad_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_ice_grad_gate.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date) ==="
nvidia-smi -L 2>/dev/null || true
"$PY" scripts/core2_ice_grad_gate.py --n 4
echo "exit: $?"
