#!/bin/bash
# Generic CPU job for the adjoint work: runs `$PY <args>` inside a checkout with the import,
# temp-dir and cache guards every job needs.
#
#   sbatch --export=ALL,WT=<checkout> [-o <log>] [--time=..] [--mem=..] \
#       scripts/adjoint/job_cpu.sbatch -m pytest -rs -q fesom_jax/tests/test_ssh.py
#
# stderr goes to the -o log (no -e directive, so a -o on the command line captures both).
# WT defaults to the submit directory. Walltime and memory go on the sbatch command line:
# compute nodes are exclusive, so leave --mem out for unit work (a --mem only caps the cgroup and
# OOMs CORE2 tests) and pass --mem=200G for CORE2 files. Sharded tests: add
# XLA_FLAGS=--xla_force_host_platform_device_count=4 to --export.
#
# Guards:
#   * fesom_jax is an editable install that points at ANOTHER checkout, so PYTHONPATH=$WT is
#     exported and the job refuses to run unless `import fesom_jax` resolves inside $WT.
#   * TMPDIR / CUDA_CACHE_PATH on /scratch (home quota), both echoed for provenance.
#   * No persistent XLA compilation cache: JAX_COMPILATION_CACHE_DIR must be unset.
#SBATCH --job-name=fj_cpu
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
WT=${WT:-$SLURM_SUBMIT_DIR}
cd "$WT" || exit 2
export PYTHONPATH=$WT
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
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
[ -e "$WT/data" ] || echo "WARNING: $WT/data is missing: every CORE2 test will SKIP"

"$PY" -c "
import os, fesom_jax
wt = os.path.realpath('$WT') + os.sep
got = os.path.realpath(fesom_jax.__file__)
print('[import] fesom_jax <-', got)
assert got.startswith(wt), f'fesom_jax imported from {got}, not from {wt}'
" || exit 2

echo "=== cmd: $PY $* ==="
"$PY" "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
