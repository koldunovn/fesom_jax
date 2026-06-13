#!/bin/bash
# Phase 9b Task JT.4 (GATE 9b AD core) — CORE2 TKE gradient gate on an A100.
# The backward pass (the mxl min-scans + the reciprocal-multiply Thomas + the node→elem Av
# + mo_convect, nested in the N-step checkpointed time-loop scan with the new tke carry +
# the FCT limiter) is memory-heavy → GPU. Small N (TKE ON, KPP/GM/ice OFF — the TKE-only path).
#SBATCH --job-name=core2_tke_grad
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_tke_grad_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_tke_grad_gate.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date) ==="
nvidia-smi -L 2>/dev/null || true
"$PY" scripts/core2_tke_grad_gate.py --n 4
echo "exit: $?"
