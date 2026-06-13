#!/bin/bash
# Phase 9b JT.5 — JAX-TKE 1-yr climate (linfs+TKE, dt=1800) from the dist_864-faithful IC, to
# compare against the c_tke_2yr 864-rank oracle (the IC-provenance rule). THE arbiter of whether
# the step-1 forcing-gap (7e-4) matters at climate scale or washes out as a transient.
#SBATCH --job-name=tke_climate
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_tke_climate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_tke_climate.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== $(hostname) $(date)  TKE climate 1yr dist_864 IC ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/core2_kpp_climate_run.py --years 1 --start-year 1958 --dt 1800 --tke on \
    --ic-dir data/ic_core2_dist864 \
    --out /home/a/a270088/port_jax/data/tke_climate_1yr_dist864 --every 500
echo "exit: $?"
