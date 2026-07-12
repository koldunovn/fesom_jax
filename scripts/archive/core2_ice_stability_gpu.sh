#!/bin/bash
# Phase 6 Task 6.7 (GATE 6) — CORE2 prognostic-ice multi-day stability on an A100.
# The assembled ice step (120-subcycle EVP + FCT + thermo + the ocean step) is heavy →
# always GPU. Checks the supercooling cap + ice growth + numerical stability past day 8.
#SBATCH --job-name=core2_ice_stab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_ice_stability_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_ice_stability_gpu.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
# ~10 model days (1728 steps, dt=500) WITH prognostic sea ice.
"$PY" scripts/archive/core2_ice_stability_run.py --steps 1728 --every 50
echo "exit: $?"
