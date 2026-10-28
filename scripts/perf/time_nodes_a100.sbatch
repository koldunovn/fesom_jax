#!/bin/bash
# CORE2 on 1-N Levante A100 nodes (4 A100 each): `gpu_profile.py time` with JDIST=1 under srun, one process per node,
# for each variant in turn inside one allocation; the partition is dist_<4 x nodes>. A variant is
# "label|config|run flags|env" (env: space-separated KEY=VAL, e.g. FESOM_HALO_NONE=1, timing only). The config is
# copied with the async writers off (single-process only, fesom_jax.run) and the job's partition. The scaling twin of
# scripts/perf/time_2node_gh200.sbatch, so the GH200 nodes of dolpung are not needed for it.
#
#   sbatch --nodes=N --export=ALL,WT=<checkout> scripts/perf/time_nodes_a100.sbatch [common run flags, e.g. --restart-in DIR]
#
# VARIANTS (newline-separated) overrides the default list below. Results: runs/perf/<job>-a100n<N>/<label>/time.p<i>.json.
#SBATCH --job-name=fj_nodes_a100
#SBATCH --partition=gpu
#SBATCH --account=ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export PYTHONPATH=$WT
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: no persistent XLA cache"; exit 2; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
if [ "${CMDBUF:-on}" = on ]; then export XLA_FLAGS=""; else export XLA_FLAGS="--xla_gpu_enable_command_buffer="; fi
export JDIST=1 GPUS_PER_NODE=4
N=${SLURM_JOB_NUM_NODES:-1}
part=dist_$((4 * N))
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" \
    || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-a100n$N
mkdir -p "$OUT"
DEFAULT_VARIANTS="fast|configs/core2_gpu_fast.yaml|
fast_none|configs/core2_gpu_fast.yaml||FESOM_HALO_NONE=1
recom_dp|configs/core2_gpu_recom.yaml|
recom_dp_none|configs/core2_gpu_recom.yaml||FESOM_HALO_NONE=1
recom_sp|configs/core2_gpu_recom.yaml|--precision single"
VARIANTS=${VARIANTS:-$DEFAULT_VARIANTS}
echo "=== $SLURM_JOB_NODELIST ($N nodes, $part) job ${SLURM_JOB_ID:-local} checkout $WT  XLA_FLAGS=[$XLA_FLAGS]  common flags: $* ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
while IFS='|' read -r label cfg flags venv; do
    [ -n "$label" ] || continue
    mkdir -p "$OUT/$label"
    sed -e 's/^\(  output_async:\) *true/\1 false/' -e 's/^\(  restart_async:\) *true/\1 false/' \
        -e "s/^partition: .*/partition: $part/" "$cfg" > "$OUT/$label/config.yaml"
    echo "--- $label: $cfg $flags, env [${venv:-}] ($(date +%T))"
    # shellcheck disable=SC2086
    env ${venv:-} srun -N "$N" --ntasks-per-node=1 --cpus-per-task=128 "$PY" scripts/perf/gpu_profile.py time "$OUT/$label" \
        --chunk-steps 96 --steps 384 -- "$OUT/$label/config.yaml" \
        --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --partition "$part" $flags "$@" \
        < /dev/null > "$OUT/$label/time.log" 2>&1      # srun reads stdin: without /dev/null it eats the variant list
    echo "exit $?"; grep -E "steady|Error|error" "$OUT/$label/time.log" | grep -v -i grpc | tail -n 4
done <<< "$VARIANTS"
echo "=== done $(date +%T)"
