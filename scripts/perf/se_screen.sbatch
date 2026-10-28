#!/bin/bash
# Split-explicit stability screens (plan docs/plans/20261021-core2-split-explicit.md, Task 7) on one GH200 node of
# dolpung: every configuration named in $LIST (one name per line, $DIR/<name>.yaml) runs $STEPS steps in chunks of
# $K with --chunk-diagnostics (max|uv|, max|eta|, the vertical CFL after every chunk), one after another in the
# allocation. Names containing "r1990" start from $RESTART_IN, the others cold; names starting with "se_" get $SE_ENV
# on top of $COMMON_ENV. At most two dolpung nodes for the whole user at a time (the user's rule): check
# `squeue -p dolpung -u a270088` and chain with --dependency.
#
#   sbatch --export=ALL,WT=<checkout>,DIR=<dir>,LIST=<file>,COMMON_ENV="K=V ...",SE_ENV="K=V ..." \
#       scripts/perf/se_screen.sbatch [run flags]
#SBATCH --job-name=fj_se_screen
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
DIR=${DIR:?DIR (the configuration directory) must be set}
LIST=${LIST:?LIST (the file of configuration names) must be set}
STEPS=${STEPS:-3000}
K=${K:-32}
RESTART_IN=${RESTART_IN:-/work/ab0995/a270088/port_jax/runs/core2_hindcast_v3/restart_archive/fesom.1990.001.00000}
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
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/se_screen/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
echo "=== $(hostname) job ${SLURM_JOB_ID:-local} steps=$STEPS K=$K common [${COMMON_ENV:-}] se [${SE_ENV:-}] flags: $* ==="
while read -r name; do
    [ -z "$name" ] && continue
    vars="${COMMON_ENV:-}"
    case "$name" in se_*) vars="$vars ${SE_ENV:-}";; esac
    rin=(); case "$name" in *r1990*) rin=(--restart-in "$RESTART_IN");; esac
    t0=$(date +%s)
    env $vars "$PY" scripts/run_from_config.py "$DIR/$name.yaml" --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 \
        --steps "$STEPS" --chunk-steps "$K" --chunk-diagnostics "${rin[@]}" "$@" > "$OUT/$name.log" 2>&1
    rc=$?
    last=$(grep '\[chunk-diag\]' "$OUT/$name.log" | tail -1 | cut -c1-160)
    echo "$name rc=$rc $(( $(date +%s) - t0 ))s chunks=$(grep -c '\[chunk-diag\]' "$OUT/$name.log") last: $last"
done < "$LIST"
echo "=== done $(date +%T)"
