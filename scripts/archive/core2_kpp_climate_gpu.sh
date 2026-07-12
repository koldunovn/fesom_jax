#!/bin/bash
# Phase 6C follow-up — CORE2 KPP+GM/Redi+ice multi-year climate run on an A100, writing
# MONTHLY-MEAN fields per variable in the C-port format (<var>.fesom.<yr>.monthly.nc) —
# ushow-readable AND directly comparable via port_kokkos/scripts/m32_climate_compare.py
# against the C-port-KPP + Fortran-KPP references. dt=1800 (the Fortran KPP timestep).
#
# Args (optional):  $1 = years (default 2)   $2 = dt (default 1800)   $3 = out dir
# Usage:  sbatch scripts/archive/core2_kpp_climate_gpu.sh        # 2 years, dt=1800, monthly means
#SBATCH --job-name=core2_kpp_clim
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_kpp_climate_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_kpp_climate_gpu.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
YEARS=${1:-2}; DT=${2:-1800}; OUT=${3:-/home/a/a270088/port_jax/data/kpp_climate_2yr}
echo "=== $(hostname) $(date)  years=$YEARS dt=$DT out=$OUT (monthly means) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/archive/core2_kpp_climate_run.py --years "$YEARS" --dt "$DT" --out "$OUT"
echo "exit: $?"
