#!/bin/bash
# Why does the sea-ice adjoint blow up after three steps? Three attacks in one job:
#   C. forward thresholds per step  — what crosses a limit between step 3 and step 4
#   A. backward one step at a time  — which state field's cotangent jumps, and when
#   B. freeze one ice component     — which of the six pieces of an ice step is responsible
#SBATCH --job-name=nn_icedig
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/icedig.%j.out
#SBATCH -e scripts/nn_eddy/icedig.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
echo "=== $(hostname) $(date) — digging into the sea-ice adjoint blow-up ==="
"$PY" scripts/nn_eddy/ice_adjoint_dig.py --snap "$SNAP" --steps 8
rc=$?
echo "=== done $(date) rc=$rc ==="
