#!/bin/bash
# Single-precision sub-plan, Task 8: steady ms/step of double vs single precision on the 4 GPUs of one node, same
# snapshot, alternating (dp sp dp sp), with gpu_profile.py time (2 warm + 4 timed chunks of K). The GPU option set
# comes from the environment (FESOM_* switches, exported by the caller) plus the run flags passed to this script.
#
#   sbatch --export=ALL,WT=<checkout>,FESOM_...=1 scripts/perf/sp_speed.sbatch --cheb-degree 3 ...
#SBATCH --job-name=fj_spspeed
#SBATCH -p gpu
#SBATCH -A ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT must be set}
K=${K:-96}
CONFIG=${CONFIG:-configs/core2_full_bench.yaml}
REPS=${REPS:-2}
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export PYTHONPATH=$WT
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: JAX_COMPILATION_CACHE_DIR is set"; exit 2; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${GPU_XLA_FLAGS:-}"
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-spspeed
mkdir -p "$OUT"
gd=$(sed -n 's/^gitdir: //p' "$WT/.git" 2>/dev/null); gd=${gd:-$WT/.git}
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} checkout $WT HEAD $(cat "$gd/HEAD" 2>/dev/null) ==="
env | grep -E "^FESOM_" | sort
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" -c "
import sys, fesom_jax
p = fesom_jax.__file__
print('[import] fesom_jax <-', p)
sys.exit(0 if p.startswith('$WT/') else 4)
" || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }
for rep in $(seq 1 "$REPS"); do
    for P in double single; do
        D="$OUT/$P.$rep"; mkdir -p "$D"
        echo "=== $P rep $rep ($(date +%T))"
        "$PY" scripts/perf/gpu_profile.py time "$D" --chunk-steps "$K" --steps $((4 * K)) -- \
            "$CONFIG" --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --partition dist_4 \
            --precision "$P" "$@" > "$D/time.log" 2>&1
        echo "exit $?"; grep -E "steady|finite|Error" "$D/time.log" | tail -n 4
    done
done
