#!/bin/bash
# Long-window Task B1 — the 10-YEAR REFERENCE TRAJECTORY from the spun-up base.
# RESTART from spinup5_state.pkl (via the A3 --load-state seam: is_first_step=False throughout)
# and run 10 yr all-on FORWARD (no AD), continuing the JRA calendar at 1963 (spin-up = 1958-1962).
# Saves dense State checkpoints (every 5 d ≈ 730 over 10 yr ⇒ the D1 burst-seed pool) + monthly
# means (the 10-yr-mean MLD target). 5-d cadence (not 10-d) so a WEEKLY-probing D1 re-run (~7-d
# seeds, K≈520) is possible LATER from the SAME reference — no 10-yr re-run (user req 2026-06-18;
# the suggested ~18-d K=200 run sub-samples the same pool). Single-GPU (~6.5 h + ~36 min snap I/O).
# ⚠️ SUBMIT ONLY AFTER A2's SPINUP_5YR_OK (verified by core2_lw_foundation_check on spinup5_climate).
#   sbatch scripts/paper/core2_lw_reference.sbatch
#SBATCH --job-name=lw_ref10
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/lw_ref10.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/lw_ref10.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
WORK=/work/ab0995/a270088/port_jax
SPIN="$WORK/longwindow/spinup5_state.pkl"
if [ ! -f "$SPIN" ]; then echo "ABORT: $SPIN missing (run+verify A2 first)"; exit 1; fi
mkdir -p "$WORK/longwindow/ref10_climate" "$WORK/longwindow/ref10_snaps"
echo "=== $(hostname) $(date) — B1 10-yr reference from $SPIN (JRA 1963-1972) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/archive/core2_kpp_climate_run.py --years 10 --start-year 1963 --dt 1800 \
    --ale on --tke on --mevp on --ic-dir data/ic_core2_dist864 \
    --load-state "$SPIN" \
    --out "$WORK/longwindow/ref10_climate" --every 1000 \
    --snapshot-dir "$WORK/longwindow/ref10_snaps" --snapshot-every 240 \
    --save-state "$WORK/longwindow/ref10_final.pkl"
echo "exit: $? ($(date))"
