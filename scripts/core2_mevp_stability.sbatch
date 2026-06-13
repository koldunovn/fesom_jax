#!/bin/bash
# Phase 9c JM.3 — CORE2 mEVP multi-day stability + liveness fields on an A100. Runs the FULL
# model (KPP + GM/Redi + prognostic ice + FCT + thermo, linfs) for BOTH ice rheologies (mEVP
# whichEVP=1, then EVP control) → stability gate + saved surface fields for the diff-of-diffs
# liveness compare. Two configs ⇒ two XLA compiles (~15 min each) → 3h walltime.
#
# Args (optional):  $1 = steps (default 480 ≈ 10 days at dt=1800)
# Usage:  sbatch scripts/core2_mevp_stability.sbatch          # ~10 days
#         sbatch scripts/core2_mevp_stability.sbatch 1440     # ~30 days (stronger liveness signal)
#SBATCH --job-name=core2_mevp_stab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_mevp_stability.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_mevp_stability.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
STEPS=${1:-480}
echo "=== $(hostname) $(date)  steps=$STEPS (mEVP + EVP, KPP+GM+ice linfs) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/core2_mevp_stability.py --steps "$STEPS" --every 48
echo "exit: $?"
