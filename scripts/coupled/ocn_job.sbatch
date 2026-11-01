#!/bin/bash
# The ocean lane of the coupled AWI-CM JAX system: one Python command on ONE A100, from a frozen snapshot of a
# commit (git archive), outputs under $OUT.
#
#   SNAP=$(scripts/coupled/snapshot.sh <commit>)
#   sbatch --export=ALL,SNAP=$SNAP,OUT=<new dir> scripts/coupled/ocn_job.sbatch <script.py> [args]
#
# The script path is relative to the snapshot. CMDBUF=on keeps XLA's command buffers (off by default, as every
# production run). No persistent XLA cache, no XLA dump.
#SBATCH --job-name=ocn_cpl
#SBATCH -p gpu
#SBATCH -A ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH -o /work/ba1583/a270088/awicm/ocn/slurm/%x.%j.out
set -u
SNAP=${SNAP:?SNAP (the snapshot directory) must be set}
OUT=${OUT:?OUT (the output directory) must be set}
source /sw/etc/profile.levante 2>/dev/null
PY=${PY:-/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python}
export PYTHONPATH=$SNAP
case "$OUT" in
    /work/ab0995/a270088/port_jax/runs/*) echo "FATAL: OUT=$OUT is under the production runs"; exit 2;;
esac
cd "$SNAP" || exit 2
export TMPDIR=/scratch/${USER:0:1}/$USER/awicm_ocn_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1
if [ "${CMDBUF:-off}" = on ]; then CB=""; else CB="--xla_gpu_enable_command_buffer="; fi
export XLA_FLAGS="$(echo $CB ${GPU_XLA_FLAGS:-})"
echo "=== $(hostname) $(date -u) job ${SLURM_JOB_ID:-local} snapshot $SNAP ($(cat $SNAP/SNAPSHOT_COMMIT 2>/dev/null)) ==="
echo "=== OUT=$OUT XLA_FLAGS=$XLA_FLAGS  cmd: $* ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" -c "
import sys, fesom_jax
p = fesom_jax.__file__
print('[import] fesom_jax <-', p)
sys.exit(0 if p.startswith('$SNAP/') else 4)
" || { echo "FATAL: fesom_jax does not import from $SNAP"; exit 4; }
"$PY" "$@"
rc=$?
echo "=== exit $rc $(date -u) ==="
exit $rc
