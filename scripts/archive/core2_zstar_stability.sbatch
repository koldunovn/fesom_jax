#!/bin/bash
# Phase 9a JZ.8 (GATE 9a) — CORE2 zstar multi-day stability on an A100. The FULL z2_cdump
# 4-config (KPP + GM/Redi + 120-subcycle EVP ice + FCT + thermo + zstar live geometry / D2
# stiffness / forcing flip) is heavy → always GPU. Checks 10-day numerical stability with the
# moving vertical coordinate live: no NaN, bounded SSH/vel, POSITIVE layer thickness (no column
# collapse), and global-mean-SSH (steric/volume) conservation.
#
# Args (optional):  $1 = steps (default 480 ≈ 10 days at dt=1800)
# Usage:  sbatch scripts/archive/core2_zstar_stability.sbatch          # ~10 days
#         sbatch scripts/archive/core2_zstar_stability.sbatch 1440     # ~30 days
#SBATCH --job-name=core2_zstar_stab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_zstar_stability.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_zstar_stability.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
STEPS=${1:-480}
echo "=== $(hostname) $(date)  steps=$STEPS (zstar 4-config) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/archive/core2_zstar_stability.py --steps "$STEPS" --every 20
echo "exit: $?"
