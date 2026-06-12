#!/bin/bash
# Phase 9a JZ.8 — re-run the JAX-zstar 1-yr climate from the dist_864-faithful IC (to match the
# c_zstar_2yr 864-rank oracle's partition). Closes the IC-provenance gap. Usage: sbatch after the
# dist_864 IC build (scripts/rebuild_ic_dist864.sbatch) completes.
#SBATCH --job-name=zstar_clim_864
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_zstar_climate_dist864.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_zstar_climate_dist864.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date)  zstar climate dist_864 IC ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/core2_kpp_climate_run.py --years 1 --start-year 1958 --dt 1800 --ale on \
    --ic-dir data/ic_core2_dist864 \
    --out /home/a/a270088/port_jax/data/zstar_climate_1yr_dist864 --every 500
echo "exit: $?"
