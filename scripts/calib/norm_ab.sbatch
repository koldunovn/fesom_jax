#!/bin/bash
# The control-transform A/B driver (scripts/calib/norm_ab.py) on one A100.
#
#   sbatch --parsable --export=ALL,WT=<checkout> scripts/calib/norm_ab.sbatch \
#       gate|profile|twin <tke_lbfgs options> [norm_ab options] --out <run dir>
#
# Every argument goes to norm_ab.py (see its docstring); the tke_lbfgs options build the same
# Calibration as tke_lbfgs.py. A twin run is resumable: resubmit with the same arguments and --out.
#
# Guards:
#   * fesom_jax is an editable install that points at ANOTHER checkout, so PYTHONPATH=$WT is
#     exported and the job refuses to run unless `import fesom_jax` resolves inside $WT.
#   * TMPDIR / CUDA_CACHE_PATH on /scratch (home quota), both echoed for provenance.
#   * No persistent XLA compilation cache: JAX_COMPILATION_CACHE_DIR must be unset.
#SBATCH --job-name=norm_ab
#SBATCH -p gpu
#SBATCH -A ab0995_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=a100_80
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
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

echo "=== cmd: $PY scripts/calib/norm_ab.py $* ==="
"$PY" scripts/calib/norm_ab.py "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
