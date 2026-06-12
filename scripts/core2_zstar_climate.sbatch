#!/bin/bash
# Phase 9a JZ.8 (GATE 9a) — CORE2 zstar 1-yr CLIMATE run on an A100 (monthly-mean SST/SSS in
# the C-port format), for the discriminating year-1 comparison: JAX-zstar ↔ C-zstar RMS must be
# ≪ the zstar↔linfs contrast (the C measured 3–9×). Reuses the validated K.9 monthly-mean
# machinery with --ale on (ale_cfg=AleConfig() + the dist16 IC).
#
# Args (optional):  $1 = years (default 1 = 1958)
# Usage:  sbatch scripts/core2_zstar_climate.sbatch
#SBATCH --job-name=core2_zstar_clim
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_zstar_climate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_zstar_climate.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
YEARS=${1:-1}
echo "=== $(hostname) $(date)  zstar climate, years=$YEARS ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/core2_kpp_climate_run.py --years "$YEARS" --start-year 1958 --dt 1800 \
    --ale on --out /home/a/a270088/port_jax/data/zstar_climate_1yr --every 500
echo "exit: $?"
