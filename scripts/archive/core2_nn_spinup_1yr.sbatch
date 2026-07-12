#!/bin/bash
# §3 E1 prep — produce a DEVELOPED 1-year all-on (zstar+TKE+mEVP+GM) state for the NN twin (and E2).
# The 10-day S0 has developed surface mixing but not the thermocline/deep state; a ~1-year forward is
# the "more or less developed" ocean. Forward-only ⇒ single GPU is fine (sharding-safe too, but the
# 1-yr forward is ~35 min so it's not needed). Reuses the proven all3_climate_1yr config + --save-state
# to pickle the final restartable State. Streamed forcing (no pre-stack/OOM). Single A100.
#SBATCH --job-name=core2_nn_spinup1yr
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_nn_spinup_1yr.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_nn_spinup_1yr.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
YEARS=${1:-1}
SAVE=${2:-/work/ab0995/a270088/port_jax/nn_twin_S0_1yr.pkl}
OUT=/home/a/a270088/port_jax/data/nn_spinup_${YEARS}yr

echo "=== $(hostname) $(date)  all3 ${YEARS}-yr spin-up → $SAVE ==="
nvidia-smi -L 2>/dev/null || true

"$PY" scripts/archive/core2_kpp_climate_run.py --years "$YEARS" --start-year 1958 --dt 1800 \
    --ale on --tke on --mevp on --ic-dir data/ic_core2_dist864 \
    --out "$OUT" --every 1000 --save-state "$SAVE"
rc=$?
echo "=== done $(date)  (exit $rc) ==="
[ "$rc" -eq 0 ] && ls -lah "$SAVE"
exit $rc
