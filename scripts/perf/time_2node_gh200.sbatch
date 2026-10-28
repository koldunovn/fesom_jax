#!/bin/bash
# CORE2 on one or two dolpung GH200 nodes: `gpu_profile.py time` with JDIST=1 under srun, for each variant in turn
# inside one two-node allocation. A variant is "label|config|run flags|nodes|processes per node|env" (the last three
# optional: 2, 1, none); nodes 1 or 2 run dist_4 / dist_8, 4 processes a node run one per GPU pinned to its Grace
# (scripts/perf/proc_per_gpu.sh), env is space-separated KEY=VAL (e.g. FESOM_HALO_NONE=1, timing only). The config
# is copied with the async writers off (single-process only, fesom_jax.run) and the variant's partition. The two-node limit of dolpung covers
# every session of the user: check `squeue -p dolpung -u a270088` and chain with --dependency.
#
#   sbatch --export=ALL,WT=<checkout> [--dependency=afterany:<job>] scripts/perf/time_2node_gh200.sbatch \
#       [common run flags, e.g. --restart-in DIR]
#
# VARIANTS (newline-separated) overrides the default list below. Results: runs/perf/<job>-2node/<label>/time.p<i>.json.
#SBATCH --job-name=fj_2node_gh200
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
export ARM_ENV_SH=${ARM_ENV_SH:-/home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh}
source "$ARM_ENV_SH"                    # sets PY; unsets PYTHONPATH
export PYTHONPATH=$WT:/work/ab0995/a270088/fesom_jax_dev/envs/arm_extra/site
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: no persistent XLA cache"; exit 2; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
if [ "${CMDBUF:-on}" = on ]; then export XLA_FLAGS=""; else export XLA_FLAGS="--xla_gpu_enable_command_buffer="; fi
export JDIST=1
nvidia-smi topo -m 2>/dev/null | head -n 8
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" \
    || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-2node
mkdir -p "$OUT"
DEFAULT_VARIANTS="fast_padded|configs/core2_gpu_fast.yaml|
fast_coloured|configs/core2_gpu_fast.yaml|--coloured
recom_dp_padded|configs/core2_gpu_recom.yaml|
recom_dp_coloured|configs/core2_gpu_recom.yaml|--coloured
recom_sp_padded|configs/core2_gpu_recom.yaml|--precision single
recom_sp_coloured|configs/core2_gpu_recom.yaml|--precision single --coloured"
VARIANTS=${VARIANTS:-$DEFAULT_VARIANTS}
echo "=== $SLURM_JOB_NODELIST job ${SLURM_JOB_ID:-local} checkout $WT  XLA_FLAGS=[$XLA_FLAGS]  common flags: $* ==="
while IFS='|' read -r label cfg flags nodes ppn venv; do
    [ -n "$label" ] || continue
    nodes=${nodes:-2}; ppn=${ppn:-1}; part=dist_$((4 * nodes))
    if [ "$ppn" = 4 ]; then
        launch=(srun -N "$nodes" --ntasks-per-node=4 --cpus-per-task=72 --cpu-bind=none bash scripts/perf/proc_per_gpu.sh)
    else
        launch=(srun -N "$nodes" --ntasks-per-node=1 --cpus-per-task=288 env GPUS_PER_NODE=4)
    fi
    mkdir -p "$OUT/$label"
    sed -e 's/^\(  output_async:\) *true/\1 false/' -e 's/^\(  restart_async:\) *true/\1 false/' \
        -e "s/^partition: .*/partition: $part/" "$cfg" > "$OUT/$label/config.yaml"
    grep -E "^  (output|restart)_async:|^partition:" "$OUT/$label/config.yaml" | tr '\n' ' '; echo
    echo "--- $label: $cfg $flags, $nodes node(s) x $ppn process(es), env [${venv:-}] ($(date +%T))"
    # shellcheck disable=SC2086
    env ${venv:-} "${launch[@]}" "$PY" scripts/perf/gpu_profile.py time "$OUT/$label" \
        --chunk-steps 96 --steps 384 -- "$OUT/$label/config.yaml" \
        --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --partition "$part" $flags "$@" \
        < /dev/null > "$OUT/$label/time.log" 2>&1      # srun reads stdin: without /dev/null it eats the variant list
    echo "exit $?"; grep -E "steady|Error|error" "$OUT/$label/time.log" | tail -n 4
done <<< "$VARIANTS"
echo "=== done $(date +%T)"
