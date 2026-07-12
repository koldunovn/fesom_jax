#!/bin/bash
# Paper-experiments Task D2b — CORE2 GM→T/S stratification calibration via EKI, single A100-80GB.
# Forward-only (NO adjoint) ⇒ immune to the mEVP sea-ice rheology adjoint instability (A8) ⇒ the
# FULL all-on model runs with live ice. Args pass straight through to the driver, e.g.:
#   sbatch scripts/paper/core2_paper_calib_gm_eki.sbatch --mode budget --n 240
#   sbatch scripts/paper/core2_paper_calib_gm_eki.sbatch --mode twin   --n 240 --members 12 --iters 5 --gamma-auto
#   sbatch scripts/paper/core2_paper_calib_gm_eki.sbatch --mode obs    --n 480 --n-spin 480 --members 12 --iters 6
#SBATCH --job-name=core2_gm_eki
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_calib_gm_eki.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_calib_gm_eki.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

echo "=== $(hostname) $(date)  args: $* ==="
nvidia-smi -L 2>/dev/null || true
"$PY" scripts/paper/core2_paper_calib_gm_eki.py "$@"
rc=$?
echo "=== done $(date)  exit=$rc ==="
exit $rc
