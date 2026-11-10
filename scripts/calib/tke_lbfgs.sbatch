#!/bin/bash
# The TKE-constant calibration (scripts/calib/tke_lbfgs.py) on one A100.
#
#   sbatch --parsable --export=ALL,WT=<checkout> [--time=..] scripts/calib/tke_lbfgs.sbatch \
#       calib --mode twin --optimizer lbfgs --prior off --out <run dir> [options]
#   sbatch ... scripts/calib/tke_lbfgs.sbatch fd --months 1 --window-steps 12 --out <run dir>
#
# Every argument goes to tke_lbfgs.py (see its docstring). A calibration is resumable: resubmit
# with the same arguments and --out, and the optimiser replays its log and continues.
# For a CPU run use scripts/adjoint/job_cpu.sbatch with `scripts/calib/tke_lbfgs.py ...`
# (--mem=200G).
#
# Guards:
#   * fesom_jax is an editable install that points at ANOTHER checkout, so PYTHONPATH=$WT is
#     exported and the job refuses to run unless `import fesom_jax` resolves inside $WT.
#   * TMPDIR / CUDA_CACHE_PATH on /scratch (home quota), both echoed for provenance.
#   * No persistent XLA compilation cache: JAX_COMPILATION_CACHE_DIR must be unset.
#SBATCH --job-name=tke_lbfgs
#SBATCH -p gpu
#SBATCH -A ab0995_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=a100_80
#SBATCH --time=04:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/calib/logs/%x.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
WT=${WT:-$SLURM_SUBMIT_DIR}
cd "$WT" || exit 2
export PYTHONPATH=$WT
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=/scratch/a/a270088/fesom_jax_dev_tmp/nv
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then
    echo "FATAL: JAX_COMPILATION_CACHE_DIR is set ($JAX_COMPILATION_CACHE_DIR); no persistent XLA cache"
    exit 2
fi

echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-none} ==="
echo "=== checkout $WT  HEAD $(git rev-parse HEAD 2>/dev/null)  branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="
echo "=== TMPDIR=$TMPDIR  CUDA_CACHE_PATH=$CUDA_CACHE_PATH  XLA_FLAGS=${XLA_FLAGS:-} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

"$PY" -c "
import os, fesom_jax, jax
wt = os.path.realpath('$WT') + os.sep
got = os.path.realpath(fesom_jax.__file__)
print('[import] fesom_jax <-', got, '| devices', jax.devices())
assert got.startswith(wt), f'fesom_jax imported from {got}, not from {wt}'
" || exit 2

echo "=== cmd: $PY scripts/calib/tke_lbfgs.py $* ==="
"$PY" scripts/calib/tke_lbfgs.py "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
