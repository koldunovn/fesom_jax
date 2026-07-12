#!/bin/bash
# Phase 9c follow-up — CORE2 mEVP year-1 climate (the year-scale C comparison mEVP lacked, to
# match how zstar/TKE were validated). Full model KPP+GM+ice(mEVP)+linfs, 864r IC (matches the
# c_mevp_2yr oracle's dist_864 partition — the IC-provenance rule). Writes 12 monthly means →
# data/mevp_climate_1yr; the 4-way annual-mean RMS vs c_mevp_2yr / fortran_mevp_2yr is the
# arbiter (core2_mevp_climate_compare.py --annual).
#SBATCH --job-name=core2_mevp_clim
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_mevp_climate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_mevp_climate.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date) — mEVP year-1 climate (KPP+GM+mEVP+linfs, dist864 IC) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/archive/core2_kpp_climate_run.py --years 1 --start-year 1958 --dt 1800 \
    --mevp on --ic-dir data/ic_core2_dist864 \
    --out /home/a/a270088/port_jax/data/mevp_climate_1yr --every 500
echo "exit: $?"
