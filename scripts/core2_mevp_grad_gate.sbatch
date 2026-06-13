#!/bin/bash
# Phase 9c JM.4 — CORE2 mEVP gradient gate on an A100. Confirms the trainable seams survive
# mEVP-ON: the d(SST)/d(k_ver) plateau (unchanged vs EVP), the assembled masked-NaN probe
# d(SST)/d(T0), and the ice→ocean d(SST)/d(m_ice0). Assembled backward ⇒ GPU.
#SBATCH --job-name=core2_mevp_grad
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_mevp_grad_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_mevp_grad_gate.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
N=${1:-4}
echo "=== $(hostname) $(date)  N=$N (mEVP grad gate) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/core2_mevp_grad_gate.py --n "$N"
echo "exit: $?"
