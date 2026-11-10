#!/bin/bash
# One round of a multi-job TKE-constant calibration as ONE multi-node job: srun starts one
# `scripts/calib/tke_lbfgs.py winjob` per node, task i evaluating month i of MJ_MONTHS (a colon
# list, "1:2:3", because --export splits on commas) for evaluation MJ_EVAL. Submitted by
# scripts/calib/mj_round.sh with MJ_MULTI=1:
#
#   sbatch --nodes=<n> --export=ALL,WT=<checkout>,MJ_MONTHS=<m1:m2:..>,MJ_EVAL=<k> \
#       scripts/calib/mj_round.sbatch --out <run_dir> [tke_lbfgs options]
#
# One job counts once against the per-user running-job limit, however many windows it runs.
# Guards as scripts/adjoint/job_cpu.sbatch (import root, TMPDIR, no persistent XLA cache).
#SBATCH --job-name=mj_round
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/calib/logs/%x.%j.out

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
: "${MJ_MONTHS:?MJ_MONTHS (colon list) is required}"
: "${MJ_EVAL:?MJ_EVAL is required}"
n=$(echo "$MJ_MONTHS" | tr ':' '\n' | grep -c .)
if [ "$n" -ne "${SLURM_NNODES:-0}" ]; then
    echo "FATAL: $n months but ${SLURM_NNODES:-?} nodes"
    exit 2
fi

echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-none} nodes $SLURM_JOB_NODELIST ==="
echo "=== checkout $WT  HEAD $(git rev-parse HEAD 2>/dev/null)  months $MJ_MONTHS  eval $MJ_EVAL ==="
"$PY" -c "
import os, fesom_jax
wt = os.path.realpath('$WT') + os.sep
got = os.path.realpath(fesom_jax.__file__)
print('[import] fesom_jax <-', got)
assert got.startswith(wt), f'fesom_jax imported from {got}, not from {wt}'
" || exit 2

# Slurm raises the CPU count to cover --mem (SLURM_CPUS_PER_TASK=52 for 200G) but keeps
# SLURM_TRES_PER_TASK=cpu=16, and srun refuses the pair; give the tasks the CPUs the job has.
export SLURM_TRES_PER_TASK="cpu=${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== cmd: winjob --eval $MJ_EVAL $* (one task per month, ${SLURM_CPUS_PER_TASK:-?} CPUs) ==="
srun --kill-on-bad-exit=0 --label bash -c \
    'IFS=: read -ra M <<< "$MJ_MONTHS"; exec "$0" "$WT/scripts/calib/tke_lbfgs.py" winjob --months "${M[$SLURM_PROCID]}" --eval "$MJ_EVAL" "$@"' \
    "$PY" "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
