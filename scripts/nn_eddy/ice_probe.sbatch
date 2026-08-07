#!/bin/bash
#SBATCH --job-name=nn_iceprobe
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/iceprobe.%j.out
#SBATCH -e scripts/nn_eddy/iceprobe.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
echo "=== $(hostname) $(date) — where does the sea-ice adjoint go non-finite? ==="
"$PY" scripts/nn_eddy/ice_adjoint_probe.py --snap "$SNAP" --steps 1,2,3,4,6,8,12,16,24,32,48
rc=$?
echo "=== done $(date) rc=$rc ==="
