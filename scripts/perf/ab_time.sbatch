#!/bin/bash
# CORE2 GPU speed plan: an A/B timing on ONE node — variants A and B alternate (A B A B ...) inside one allocation,
# so the ~5 % spread between Levante A100 nodes does not decide a small difference. Each run is
# `gpu_profile.py time` (2 warm + 4 timed chunks of K steps), no profile.
#
#   sbatch --export=ALL,WT=<checkout>,A="KEY=VAL ...",B="KEY=VAL ...",REPS=2,LABEL=<name> \
#       scripts/perf/ab_time.sbatch [run flags]
#
# A and B are space-separated environment settings applied to that variant only (e.g. A="FESOM_HALO_COMPACT=1"
# B="FESOM_HALO_COMPACT=2"); the common environment and the run flags are shared. A_FLAGS / B_FLAGS are extra run
# flags for that variant only (e.g. B_FLAGS="--precision single"; $A_ARGS / $B_ARGS work too). VARIANTS="A B C D" runs more variants, each
# with its own $C / $C_FLAGS, in that order every rep; $C_K overrides the chunk length K for that variant, $C_CONFIG the
# run configuration (default $CONFIG). ARM_ENV_SH as in gpu_profile.sbatch (set by
# ab_time_gh200.sbatch on dolpung).
#SBATCH --job-name=fj_ab
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
REPS=${REPS:-2}
K=${K:-96}
LABEL=${LABEL:-ab}
CONFIG=${CONFIG:-configs/core2_full_bench.yaml}
if [ -n "${ARM_ENV_SH:-}" ]; then
    source "$ARM_ENV_SH"
    export PYTHONPATH=$WT:/work/ab0995/a270088/fesom_jax_dev/envs/arm_extra/site
else
    source /sw/etc/profile.levante 2>/dev/null
    PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
    export PYTHONPATH=$WT
fi
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: no persistent XLA cache"; exit 2; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
export XLA_FLAGS="$(echo --xla_gpu_enable_command_buffer= ${GPU_XLA_FLAGS:-})"
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" \
    || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-$LABEL
mkdir -p "$OUT"
P=${P:-4}
if [ "$P" = 1 ]; then PART=1; else PART=dist_$P; fi
VARIANTS=${VARIANTS:-A B}
echo "=== $(hostname) job ${SLURM_JOB_ID:-local} reps=$REPS flags: $* ==="
for v in $VARIANTS; do f=${v}_FLAGS; kk=${v}_K; vc=${v}_CONFIG; echo "    $v: env [${!v:-}] flags [${!f:-}] K=${!kk:-$K} config ${!vc:-$CONFIG}"; done
for r in $(seq 1 "$REPS"); do
    for v in $VARIANTS; do
        vars=${!v:-}
        f=${v}_FLAGS; va=${v}_ARGS; vflags="${!f:-} ${!va:-}"
        kk=${v}_K; vk=${!kk:-$K}
        vc=${v}_CONFIG; vconf=${!vc:-$CONFIG}
        d="$OUT/$v.$r"; mkdir -p "$d"
        env $vars "$PY" scripts/perf/gpu_profile.py time "$d" --chunk-steps "$vk" --steps $((4 * vk)) -- \
            "$vconf" --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --partition "$PART" "$@" $vflags \
            > "$d/time.log" 2>&1
        echo "$v rep $r: $(grep -h 'steady' "$d/time.log" | tail -1)"
    done
done
echo "=== done $(date +%T)"
