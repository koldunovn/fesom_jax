#!/bin/bash
# Phase 6B Task G.7 (GATE 6B) — CORE2 GM/Redi + sea-ice multi-day stability on an A100.
# The full production step (GM coefficient TDMA + Redi scatters + 120-subcycle EVP + FCT +
# thermo + the ocean step) is heavy → always GPU. Checks multi-day numerical stability with
# GM/Redi live + the front-smoothing sanity sign (vs the --gm off baseline).
#
# Args (optional):  $1 = steps (default 1728 ≈ 10 days)   $2 = gm mode on|off (default on)
# Usage:  sbatch scripts/core2_gm_stability_gpu.sh             # GM ON, ~10 days
#         sbatch scripts/core2_gm_stability_gpu.sh 1728 off    # ice-only baseline
#SBATCH --job-name=core2_gm_stab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_gm_stability_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_gm_stability_gpu.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
STEPS=${1:-1728}
GM=${2:-on}
echo "=== $(hostname) $(date)  steps=$STEPS gm=$GM ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/core2_gm_stability_run.py --steps "$STEPS" --every 50 --gm "$GM"
echo "exit: $?"
