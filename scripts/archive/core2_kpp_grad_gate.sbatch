#!/bin/bash
# Phase 6C Task K.10 (GATE 6C AD half) — CORE2 KPP gradient gate on an A100.
# The backward pass (the bldepth OBL search + blmix cubic + smooth_blmc + node→elem + the
# mo_convect, nested in the N-step time-loop scan + FCT limiter) is memory-heavy → GPU.
# Small N (KPP ON, GM + ice OFF — matching the KPP dump config, isolating the KPP gradient).
#SBATCH --job-name=core2_kpp_grad
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_kpp_grad_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_kpp_grad_gate.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date) ==="
nvidia-smi -L 2>/dev/null || true
"$PY" scripts/archive/core2_kpp_grad_gate.py --n 4
echo "exit: $?"
