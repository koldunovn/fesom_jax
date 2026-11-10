#!/bin/bash
# Window evaluations of SEVERAL multi-job calibrations in ONE multi-node job (submitted by
# scripts/calib/mj_drive.py): task i runs line i+1 of the task file, which holds the complete
# argument list of one `scripts/calib/tke_lbfgs.py winjob` call (its --out, --months, --eval and
# the calibration's options), one node per task.
#
#   sbatch --nodes=<number of lines> --export=ALL,WT=<checkout> scripts/calib/mj_batch.sbatch <task file>
#
# One job counts once against the per-user running-job limit. Guards as
# scripts/adjoint/job_cpu.sbatch (import root, TMPDIR, no persistent XLA cache).
#SBATCH --job-name=mj_batch
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
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=/scratch/a/a270088/fesom_jax_dev_tmp/nv
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then
    echo "FATAL: JAX_COMPILATION_CACHE_DIR is set ($JAX_COMPILATION_CACHE_DIR); no persistent XLA cache"
    exit 2
fi
TASKS=${1:?task file required}
n=$(grep -c . "$TASKS")
if [ "$n" -ne "${SLURM_NNODES:-0}" ]; then
    echo "FATAL: $n tasks but ${SLURM_NNODES:-?} nodes"
    exit 2
fi
# Slurm raises the CPU count to cover --mem (SLURM_CPUS_PER_TASK=52 for 200G) but keeps
# SLURM_TRES_PER_TASK=cpu=16, and srun refuses the pair; give the tasks the CPUs the job has.
export SLURM_TRES_PER_TASK="cpu=${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-none} nodes $SLURM_JOB_NODELIST tasks $TASKS ==="
echo "=== checkout $WT  HEAD $(git rev-parse HEAD 2>/dev/null) ==="
"$PY" -c "
import os, fesom_jax
wt = os.path.realpath('$WT') + os.sep
got = os.path.realpath(fesom_jax.__file__)
print('[import] fesom_jax <-', got)
assert got.startswith(wt), f'fesom_jax imported from {got}, not from {wt}'
" || exit 2
srun --kill-on-bad-exit=0 --label bash -c \
    'read -ra A <<< "$(sed -n "$((SLURM_PROCID + 1))p" "$1")"; exec "$0" "$WT/scripts/calib/tke_lbfgs.py" winjob "${A[@]}"' \
    "$PY" "$TASKS"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
