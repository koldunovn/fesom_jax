#!/bin/bash
# Phase 9b Task JT.5 — CORE2 TKE stability + TKE↔KPP discrimination on an A100.
# 10-day linfs+TKE forward run (KPP/GM/ice off) — no NaN, bounded SSH/vel — then KPP for the
# surface-RMS scheme contrast (must be clearly resolved, the C measured 11–18× SST).
#SBATCH --job-name=core2_tke_stab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_tke_stability.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_tke_stability.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date) ==="
nvidia-smi -L 2>/dev/null || true
PYTHONPATH=/home/a/a270088/port_jax/scripts "$PY" scripts/core2_tke_stability.py --steps 480
echo "exit: $?"
