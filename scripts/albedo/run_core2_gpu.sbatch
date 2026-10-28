#!/bin/bash
# Albedo: one CORE2 (+REcoM) run through scripts/run_from_config.py on ONE GPU (docs/recom/RUNNING_ALBEDO.md).
#   sbatch --export=ALL,OUT=<new dir>,CONFIG=<yaml> scripts/albedo/run_core2_gpu.sbatch [run flags]
# Run flags go to run_from_config.py unchanged (--restart-in, --steps, --monthly-out, --precision single, ...).
# Refuses an OUT that already holds a restart or output store. GPU=a100 (default) or a40 picks the node type.
#SBATCH --job-name=fj_core2
#SBATCH -p gpu
#SBATCH -A clidyn.clidyn
#SBATCH --qos=12h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
set -u
OUT=${OUT:?OUT (a new output directory) must be set}
CONFIG=${CONFIG:?CONFIG must be set}
WT=${WT:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
if [ -e "$OUT/restart" ] || [ -e "$OUT/monthly" ] || [ -e "$OUT/restart_archive" ]; then
    echo "FATAL: $OUT already holds a run (restart/monthly/restart_archive); pick a new OUT"; exit 2
fi
cd "$WT" || exit 2
export TMPDIR=/albedo/scratch/user/$USER/fesom_jax_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
source "$WT/scripts/albedo/env.sh"
export XLA_FLAGS="--xla_gpu_enable_command_buffer=${GPU_XLA_FLAGS:+ $GPU_XLA_FLAGS}"
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local}  checkout $WT  HEAD $(git -C "$WT" rev-parse --short HEAD 2>/dev/null) ==="
echo "=== CONFIG=$CONFIG OUT=$OUT  flags: $* ==="
cp "$CONFIG" "$OUT/config.${SLURM_JOB_ID:-local}.yaml"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" scripts/run_from_config.py "$CONFIG" --progress "$@"
rc=$?
echo "=== run exit $rc $(date) ==="
exit $rc
