#!/bin/bash
# CORE2 GPU speed plan: one model run through scripts/run_from_config.py on the 4 GPUs of one node, from the
# checkout $WT (a frozen snapshot worktree; never the main checkout: the editable install points there, so
# PYTHONPATH=$WT and an import guard), with outputs only under $OUT (never a production run directory).
#
#   sbatch --export=ALL,WT=<checkout>,OUT=<new dir>,CONFIG=<yaml> scripts/perf/run_core2_gpu.sbatch [run flags]
#
# Run flags go to run_from_config.py unchanged (--restart-in, --steps, --chunk-steps, --monthly-out, ...).
# Refuses an OUT that already holds a restart or output store (no overwrite). ARM_ENV_SH (set by
# run_core2_gpu_gh200.sbatch) selects the aarch64 Python on dolpung. CMDBUF=on keeps XLA's command buffers.
#SBATCH --job-name=fj_run
#SBATCH -p gpu
#SBATCH -A ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
OUT=${OUT:?OUT (a new output directory) must be set}
CONFIG=${CONFIG:?CONFIG must be set}
if [ -n "${ARM_ENV_SH:-}" ]; then
    source "$ARM_ENV_SH"                    # sets PY; unsets PYTHONPATH
    export PYTHONPATH=$WT:/work/ab0995/a270088/fesom_jax_dev/envs/arm_extra/site
else
    source /sw/etc/profile.levante 2>/dev/null
    PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
    export PYTHONPATH=$WT
fi
case "$OUT" in
    /work/ab0995/a270088/port_jax/runs/*) echo "FATAL: OUT=$OUT is under the production runs"; exit 2;;
esac
if [ -e "$OUT/restart" ] || [ -e "$OUT/monthly" ] || [ -e "$OUT/restart_archive" ]; then
    echo "FATAL: $OUT already holds a run (restart/monthly/restart_archive); pick a new OUT"; exit 2
fi
cd "$WT" || exit 2
export TMPDIR=${FJ_TMPDIR:-/scratch/${USER:0:1}/$USER/fesom_jax_dev_tmp}
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$OUT"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then
    echo "FATAL: JAX_COMPILATION_CACHE_DIR is set; no persistent XLA cache"; exit 2
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1
# CMDBUF=on keeps XLA's command buffers (CUDA graphs; -3.6 % in single precision on 4 GH200, job 28295889);
# the default turns them off, as every production run so far
if [ "${CMDBUF:-off}" = on ]; then CB=""; else CB="--xla_gpu_enable_command_buffer="; fi
export XLA_FLAGS="$(echo $CB ${GPU_XLA_FLAGS:-})"
gd=$(sed -n 's/^gitdir: //p' "$WT/.git" 2>/dev/null); gd=${gd:-$WT/.git}
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local}  checkout $WT  HEAD $(cat "$gd/HEAD" 2>/dev/null) ==="
echo "=== CONFIG=$CONFIG OUT=$OUT XLA_FLAGS=$XLA_FLAGS  flags: $* ==="
cp "$CONFIG" "$OUT/config.${SLURM_JOB_ID:-local}.yaml"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" -c "
import sys, fesom_jax
p = fesom_jax.__file__
print('[import] fesom_jax <-', p)
sys.exit(0 if p.startswith('$WT/') else 4)
" || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }
"$PY" scripts/run_from_config.py "$CONFIG" --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 "$@"
rc=$?
echo "=== run exit $rc $(date) ==="
exit $rc
