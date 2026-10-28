#!/bin/bash
# CORE2 GPU speed plan (docs/plans/20261020-core2-gpu-speed.md), Task 1/2: time and profile the production
# run path on P GPUs of one node, from the checkout $WT (never the main checkout: the editable install
# points there, so PYTHONPATH=$WT and an import guard).
#
#   sbatch --export=ALL,WT=<checkout>,P=4,LABEL=base [--gpus=4] scripts/perf/gpu_profile.sbatch [run flags]
#
# Run flags go to scripts/run_from_config.py unchanged (e.g. --padded, --cheb-degree 3, --forcing-on-device).
# Environment:
#   P (GPUs, default 4; P=1 runs --partition 1), LABEL, CONFIG (default configs/core2_full_bench.yaml: no
#   output), K (chunk steps, default 96), NSYS=0 (skip the profile), NOISE=R (also R repeat runs and their
#   noise floor), NOISE_STEPS (default 96), NOISE_CHUNK (their chunk length, default NOISE_STEPS), BASE_NOISE=<a base job's noise/ dir> (compare this run's
#   state_0 against the base's state_0 and floor), NOISE_FACTOR (default 2: a same-code run from another job
#   reaches 1.85x the in-job floor, docs/PERFORMANCE_CORE2_GPU.md Protocol), CMDBUF=on (keep XLA command buffers; production turns them
#   off, scripts/run_from_config.sbatch), GPU_XLA_FLAGS (appended), ARM_ENV_SH (set by
#   gpu_profile_gh200.sbatch: the aarch64 Python on dolpung).
# Results: /work/ab0995/a270088/fesom_jax_dev/runs/perf/<job>-<label>/ (time.json, prof.json, profile.json,
# areas.txt, lines.txt, nccl.txt, noise/).
#SBATCH --job-name=fj_perf
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
P=${P:-4}
LABEL=${LABEL:-P$P}
K=${K:-96}
CONFIG=${CONFIG:-configs/core2_full_bench.yaml}
if [ -n "${ARM_ENV_SH:-}" ]; then
    source "$ARM_ENV_SH"                    # sets PY; unsets PYTHONPATH
    export PYTHONPATH=$WT:/work/ab0995/a270088/fesom_jax_dev/envs/arm_extra/site
    NSYS_BIN=${NSYS_BIN:-/sw/spack-levante-0.23.1/linux-rhel9-neoverse_v2/nvhpc-25.7-dsupkck/Linux_aarch64/25.7/profilers/Nsight_Systems/bin/nsys}
else
    source /sw/etc/profile.levante 2>/dev/null
    PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
    export PYTHONPATH=$WT
    NSYS_BIN=${NSYS_BIN:-/sw/spack-levante/nvhpc-25.7-3wn2ha/Linux_x86_64/25.7/profilers/Nsight_Systems/bin/nsys}
fi
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then
    echo "FATAL: JAX_COMPILATION_CACHE_DIR is set; no persistent XLA cache"; exit 2
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
if [ "${CMDBUF:-off}" = on ]; then BASE_FLAGS=""; else BASE_FLAGS="--xla_gpu_enable_command_buffer="; fi
XLA_FLAGS="$BASE_FLAGS ${GPU_XLA_FLAGS:-}"
export XLA_FLAGS="$(echo $XLA_FLAGS)"      # no stray spaces: XLA reads a non-"--" value as a flag FILE name
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-$LABEL
mkdir -p "$OUT/nsys"
if [ "$P" = 1 ]; then PART=1; else PART=dist_$P; fi
RFC=("$CONFIG" --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --partition "$PART" "$@")
GP=("$PY" scripts/perf/gpu_profile.py)

gd=$(sed -n 's/^gitdir: //p' "$WT/.git" 2>/dev/null); gd=${gd:-$WT/.git}
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local}  checkout $WT  HEAD $(cat "$gd/HEAD" 2>/dev/null) ==="
echo "=== P=$P K=$K out=$OUT  XLA_FLAGS=$XLA_FLAGS  run flags: ${RFC[*]} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"$PY" -c "
import sys, fesom_jax
p = fesom_jax.__file__
print('[import] fesom_jax <-', p)
sys.exit(0 if p.startswith('$WT/') else 4)
" || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }

echo "=== time: 2 warm chunks + 4 timed chunks of $K ($(date +%T))"
"${GP[@]}" time "$OUT" --chunk-steps "$K" --steps $((4 * K)) -- "${RFC[@]}" 2>&1 | tee "$OUT/time.log" | tail -n 5
echo "time exit ${PIPESTATUS[0]}"

if [ "${NSYS:-1}" != 0 ]; then
    echo "=== nsys over one timed chunk, kernels in CUDA graphs traced one by one ($(date +%T))"
    XLA_FLAGS="$(echo $XLA_FLAGS --xla_dump_to=$OUT/hlo --xla_dump_hlo_as_text)" \
    "$NSYS_BIN" profile -t cuda,nvtx -s none --cpuctxsw=none --capture-range=cudaProfilerApi \
        --capture-range-end=stop --cuda-graph-trace=node -o "$OUT/nsys/prof" -f true \
        "${GP[@]}" prof "$OUT" --chunk-steps "$K" --steps "$K" -- "${RFC[@]}" > "$OUT/nsys/prof.log" 2>&1
    echo "nsys exit $?"
    "$NSYS_BIN" stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv \
        --output "$OUT/nsys/prof" "$OUT/nsys/prof.nsys-rep" > "$OUT/nsys/stats.log" 2>&1
    echo "stats exit $?"
    "${GP[@]}" stats "$OUT" "$OUT/nsys/prof" --steps "$K" --devices "$P"
    HLO=$(ls -S "$OUT"/hlo/*after_optimizations.txt 2>/dev/null | head -n 1)
    KCSV="$OUT/nsys/prof_cuda_gpu_kern_sum.csv"
    if [ -n "$HLO" ] && [ -f "$KCSV" ]; then
        echo "=== kernels by area ($(basename "$HLO"))"
        "$PY" scripts/perf/hlo_kernel_map.py "$HLO" "$KCSV" --steps $((K * P)) --by area | tee "$OUT/areas.txt"
        "$PY" scripts/perf/hlo_kernel_map.py "$HLO" "$KCSV" --steps $((K * P)) --by line --top 40 > "$OUT/lines.txt"
    fi
    if [ "$P" -gt 1 ] && [ -f "$OUT/nsys/prof.sqlite" ]; then
        "$PY" scripts/perf/nccl_waits.py "$OUT/nsys/prof.sqlite" --steps "$K" > "$OUT/nccl.txt"
        head -n 4 "$OUT/nccl.txt"
    fi
fi

if [ "${NOISE:-0}" != 0 ]; then
    echo "=== noise: $NOISE runs of ${NOISE_STEPS:-96} steps from the cold start ($(date +%T))"
    "${GP[@]}" noise "$OUT/noise" --reps "$NOISE" --steps "${NOISE_STEPS:-96}" --chunk-steps "${NOISE_CHUNK:-${NOISE_STEPS:-96}}" \
        -- "${RFC[@]}" 2>&1 | tee "$OUT/noise.log" | grep -E "floor|saved|Error|error" | tail -n 40
    echo "noise exit ${PIPESTATUS[0]}"
    if [ -n "${BASE_NOISE:-}" ] && [ -f "$BASE_NOISE/noise.json" ]; then
        echo "=== this run against the base's noise floor ($BASE_NOISE)"
        "${GP[@]}" compare "$OUT/noise/state_0.npz" "$BASE_NOISE/state_0.npz" --floor "$BASE_NOISE/noise.json" \
            --factor "${NOISE_FACTOR:-2}" \
            | tee "$OUT/compare.txt" | tail -n 12
    fi
fi
echo "=== done $(date +%T)"
