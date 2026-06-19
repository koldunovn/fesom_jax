#!/bin/bash
# A2 RESTART/PROBE — the 5-yr spin-up tripped the (too-tight) 3.0 m/s cap at a localized summer
# Somali Current jet (year 5, July 1962). Cap now 5.0. Restart from the last pre-trip snapshot
# (step 79200 ≈ 1962-07-08, |vel|=2.59) and finish the 5 yr (8400 steps), reporting |vel| often:
#   * if the Somali jet PEAKS (~3–3.5) and SUBSIDES into autumn → physical seasonal current ⇒ base OK
#   * if it keeps CLIMBING toward 5.0 → a real localized instability ⇒ investigate (viscosity/limiter)
# Saves the completed 5-yr base → spinup5_state.pkl (if it survives).
#SBATCH --job-name=lw_spin_fin
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/lw_spinup_finish.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/lw_spinup_finish.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
WORK=/work/ab0995/a270088/port_jax
SNAP="$WORK/longwindow/spinup5_snaps/snap_step079200.pkl"
echo "=== $(hostname) $(date) — A2 finish/probe from $SNAP (1962-07-08, cap=5.0) ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/core2_kpp_climate_run.py --steps 8400 --start-year 1962 --start-month 7 --start-day 8 \
    --dt 1800 --ale on --tke on --mevp on --ic-dir data/ic_core2_dist864 \
    --load-state "$SNAP" \
    --out "$WORK/longwindow/spinup5_finish_climate" --every 200 \
    --save-state "$WORK/longwindow/spinup5_state.pkl"
echo "exit: $? ($(date))"
