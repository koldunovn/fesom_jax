#!/bin/bash
# Production-loop A/B on one GH200 node of dolpung: variants A B ... alternate (A B A B ...) inside one allocation,
# each a `run_from_config.py --progress` run of $STEPS steps in chunks of $K with monthly output (the production
# loop: forcing on the host worker, output on the writer thread, month-end chunks), timed by scripts/perf/loop_ms.py
# (the loop from the end of the leading compile chunks, later compile chunks excluded; the host's wait for the
# forcing included, unlike gpu_profile.py time). At most two dolpung nodes for the whole user (the user's rule).
#
#   sbatch --export=ALL,WT=<checkout>,CONFIG=<yaml>,A="KEY=VAL ...",B="KEY=VAL ...",REPS=2,LABEL=<name> \
#       scripts/perf/loop_ab.sbatch [run flags]
# $V_CONFIG / $V_FLAGS / $V_K: a variant's own configuration / extra run flags / chunk length; VARIANTS="A B C" runs
# more variants.
#SBATCH --job-name=fj_loop_ab
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
CONFIG=${CONFIG:?CONFIG must be set}
REPS=${REPS:-2}
K=${K:-96}
STEPS=${STEPS:-2920}
LABEL=${LABEL:-loop_ab}
source "${ARM_ENV_SH:-/home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh}"     # sets PY
export PYTHONPATH=$WT:/work/ab0995/a270088/fesom_jax_dev/envs/arm_extra/site
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: no persistent XLA cache"; exit 2; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1 FESOM_REUSE_EXE=1
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" \
    || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-$LABEL
mkdir -p "$OUT"
VARIANTS=${VARIANTS:-A B}
echo "=== $(hostname) job ${SLURM_JOB_ID:-local} reps=$REPS steps=$STEPS K=$K flags: $* ==="
for v in $VARIANTS; do f=${v}_FLAGS; vc=${v}_CONFIG; kk=${v}_K; echo "    $v: env [${!v:-}] flags [${!f:-}] K=${!kk:-$K} config ${!vc:-$CONFIG}"; done
for r in $(seq 1 "$REPS"); do
    for v in $VARIANTS; do
        vars=${!v:-}; f=${v}_FLAGS; vc=${v}_CONFIG; kk=${v}_K
        d="$OUT/$v.$r"; mkdir -p "$d"
        env $vars "$PY" scripts/run_from_config.py "${!vc:-$CONFIG}" --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 \
            --steps "$STEPS" --chunk-steps "${!kk:-$K}" --progress --monthly-out "$d/monthly" "$@" ${!f:-} > "$d/run.log" 2>&1
        rc=$?
        echo "$v rep $r rc=$rc: $("$PY" scripts/perf/loop_ms.py "$d/run.log" | sed 's|^[^:]*: ||') host wait $(grep -o 'host=[0-9.]*' "$d/run.log" | awk -F= '{s+=$2} END {printf "%.1f s", s}')"
    done
done
echo "=== done $(date +%T)"
