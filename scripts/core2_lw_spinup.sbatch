#!/bin/bash
# Long-window Task A2 — 5-YEAR all-on (zstar+TKE+mEVP+GM) SPIN-UP → the frozen base state.
# FORWARD-ONLY (no AD), single-GPU (reuses the validated foundation path verbatim; the
# foundation 3-yr ran 0.13 s/step ⇒ 5 yr ≈ 3.2 h compute + ~15 min compile). Produces the
# frozen base that ALL long experiments start from:
#   * $WORK/longwindow/spinup5_state.pkl       — the final developed 5-yr State (--save-state)
#   * $WORK/longwindow/spinup5_climate/        — monthly means (so core2_lw_foundation_check
#                                                gates the year-4→5 drift = SPINUP_5YR_OK)
#   * $WORK/longwindow/spinup5_snaps/          — a few late-spin-up snapshots (IC-sensitivity)
# JRA calendar 1958→1962 (5 consecutive years; well within the 1958– record).
#SBATCH --job-name=lw_spinup5
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/lw_spinup5.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/lw_spinup5.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
WORK=/work/ab0995/a270088/port_jax
mkdir -p "$WORK/longwindow/spinup5_climate" "$WORK/longwindow/spinup5_snaps"
echo "=== $(hostname) $(date) — A2 5-yr all-on spin-up (zstar+TKE+mEVP+GM) → frozen base ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/core2_kpp_climate_run.py --years 5 --start-year 1958 --dt 1800 \
    --ale on --tke on --mevp on --ic-dir data/ic_core2_dist864 \
    --out "$WORK/longwindow/spinup5_climate" --every 500 \
    --snapshot-dir "$WORK/longwindow/spinup5_snaps" --snapshot-every 7200 \
    --save-state "$WORK/longwindow/spinup5_state.pkl"
echo "exit: $? ($(date))"
