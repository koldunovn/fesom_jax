#!/bin/bash
# Single-precision sub-plan, Task 9: the conservation/fidelity triplet on the 4 GPUs of one node, from a frozen
# snapshot worktree $WT — three runs of $STEPS steps from the cold start 1958-01-01 with CONFIG (default
# configs/core2_gpu.yaml), one after the other:
#   dp      double precision
#   dpic32  double precision from the float32-rounded initial T and S (the envelope of an f32-sized perturbation)
#   sp      precision single
# each with --budget (float64 volume / salt / heat content after every chunk) and a final restart under
# $OUT/<label>/restart. Compare with scripts/perf/compare_precision.py.
#
#   sbatch --export=ALL,WT=<checkout>,OUT=<new dir>,STEPS=1440 [--time=..] scripts/perf/sp_triplet.sbatch
#SBATCH --job-name=fj_sp3
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
WT=${WT:?WT (the checkout) must be set}
OUT=${OUT:?OUT (a new output directory) must be set}
STEPS=${STEPS:?STEPS must be set}
CONFIG=${CONFIG:-configs/core2_gpu.yaml}
K=${K:-96}
RUNS=${RUNS:-"dp dpic32 sp"}
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export PYTHONPATH=$WT
case "$OUT" in /work/ab0995/a270088/port_jax/runs/*) echo "FATAL: OUT under production runs"; exit 2;; esac
if [ -e "$OUT" ]; then echo "FATAL: $OUT exists; pick a new OUT"; exit 2; fi
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$OUT"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: JAX_COMPILATION_CACHE_DIR is set"; exit 2; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${GPU_XLA_FLAGS:-}"
gd=$(sed -n 's/^gitdir: //p' "$WT/.git" 2>/dev/null); gd=${gd:-$WT/.git}
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} checkout $WT HEAD $(cat "$gd/HEAD" 2>/dev/null) STEPS=$STEPS K=$K ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" -c "
import sys, fesom_jax
p = fesom_jax.__file__
print('[import] fesom_jax <-', p)
sys.exit(0 if p.startswith('$WT/') else 4)
" || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }
rc_all=0
for R in $RUNS; do
    case $R in
        dp) X=();; dpic32) X=(--ic-f32);; sp) X=(--precision single);;
        *) echo "unknown run $R"; exit 2;;
    esac
    mkdir -p "$OUT/$R"
    echo "=== run $R ($(date +%T)) flags ${X[*]}"
    t0=$(date +%s)
    "$PY" scripts/run_from_config.py "$CONFIG" --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 \
        --steps "$STEPS" --chunk-steps "$K" --budget --diagnostics --progress \
        --restart-out "$OUT/$R/restart" "${X[@]}" > "$OUT/$R/run.log" 2>&1
    rc=$?
    echo "=== run $R exit $rc wall $(( $(date +%s) - t0 )) s"
    grep -E "NG5_R0|\[budget\]" "$OUT/$R/run.log" | tail -n 2
    [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
