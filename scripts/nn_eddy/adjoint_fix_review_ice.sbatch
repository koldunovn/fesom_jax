#!/bin/bash
# Measure why the CORE2 adjoint-knob tests fail. CPU: one step, ~15 compiles, no GPU needed —
# and CPU is the right platform, because the point is to separate structural zeros from the
# GPU nondeterminism that is a SEPARATE failure cluster in the same test files.
#SBATCH --job-name=nn_revice
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH -o scripts/nn_eddy/revice.%j.out
#SBATCH -e scripts/nn_eddy/revice.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date) ==="
echo "=== checkout: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null) ==="
"$PY" scripts/nn_eddy/adjoint_fix_review_ice.py
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
