#!/bin/bash
# Phase 6C Task K.9 (GATE 6C) — CORE2 KPP + GM/Redi + sea-ice multi-day stability + the
# KPP-vs-PP climate-difference sign, on an A100. The full production step (KPP OBL profile +
# GM coefficient TDMA + Redi scatters + 120-subcycle EVP + FCT + thermo + the ocean step) is
# heavy → always GPU. ``--mixing both`` runs KPP then a matched PP baseline and reports the
# genuine scheme difference (the discriminating check; "JAX-KPP ≈ C-KPP" is the K.8 step gate).
#
# Args (optional):  $1 = steps (default 1728 ≈ 10 days)   $2 = mixing kpp|pp|both (default both)
# Usage:  sbatch scripts/archive/core2_kpp_stability_gpu.sh             # both, ~10 days
#         sbatch scripts/archive/core2_kpp_stability_gpu.sh 5184 kpp    # KPP only, ~30 days
#SBATCH --job-name=core2_kpp_stab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_kpp_stability_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_kpp_stability_gpu.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
STEPS=${1:-1728}
MIX=${2:-both}
echo "=== $(hostname) $(date)  steps=$STEPS mixing=$MIX ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/archive/core2_kpp_stability_run.py --steps "$STEPS" --every 50 --mixing "$MIX"
echo "exit: $?"
