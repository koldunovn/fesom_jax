#!/bin/bash
# Phase 9c follow-up — CORE2 all-3-options year-1 climate STABILITY (zstar+TKE+mEVP all on). ⚠️
# JAX-FIRST: the C never ran all 3 together (no combined oracle), so this is a stability /
# physical-plausibility check ONLY (no NaN, bounded SSH/vel/m_ice, hnode>0), NOT a fidelity gate.
# Extends the 3-step ALL_ON_SMOKE to a full year.
#SBATCH --job-name=core2_all3_clim
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_all3_climate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_all3_climate.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date) — all-3 (zstar+TKE+mEVP) year-1 STABILITY (no C oracle) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/archive/core2_kpp_climate_run.py --years 1 --start-year 1958 --dt 1800 \
    --ale on --tke on --mevp on --ic-dir data/ic_core2_dist864 \
    --out /home/a/a270088/port_jax/data/all3_climate_1yr --every 500
echo "exit: $?"
