#!/bin/bash
# CORE2 GPU-scaling bench WITH REAL I/O (monthly canonical zarr + periodic checkpoint) — unlike
# bench_core2_*node.sbatch / bench_core2_full.sbatch (bench_forward_scaling.py, pure step timing,
# no I/O). Cold PHC start, ~2 model-months (STEPS steps @ dt=1800, 1440 steps/model-month) so the
# measured rate includes real monthly write + checkpoint overhead, not just the compute step.
# Separate TAG/output dir/JAX cache from the live core2_hindcast_v2 production chain — safe to run
# concurrently with it.
#
# Submit once per GPU count (NODES controls whether JDIST + --ragged multi-node comms are used):
#   sbatch --nodes=1 --gres=gpu:1 --export=ALL,PART=dist_1,NODES=1,TAG=core2_scaling_1gpu $0
#   sbatch --nodes=1 --gres=gpu:4 --export=ALL,PART=dist_4,NODES=1,TAG=core2_scaling_4gpu $0
#   sbatch --nodes=2 --gres=gpu:4 --export=ALL,PART=dist_8,NODES=2,TAG=core2_scaling_8gpu $0
#SBATCH --job-name=core2_scale
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_core2_scaling_io.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_core2_scaling_io.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1
export JAX_COMPILATION_CACHE_DIR=/work/ab0995/a270088/port_jax/jax_cache/core2_scaling_bench
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

CONFIG=$REPO/configs/core2_full.yaml
MESH=data/mesh_core2
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
IC=data/ic_core2
PART=${PART:?set PART=dist_1|dist_4|dist_8}
NODES=${NODES:-1}
TAG=${TAG:?set TAG}
STEPS=${STEPS:-2880}     # 2 model-months (1440 steps/month @ dt=1800) -> real monthly-zarr + checkpoint I/O
CHUNK=${CHUNK:-48}        # matches the production hindcast chunk size
PROGRESS=${PROGRESS:-1}   # 0 = drop --progress (no per-chunk jax.block_until_ready) to test host/GPU overlap
OUT=/work/ab0995/a270088/port_jax/runs/perf/$TAG
rm -rf "$OUT"; mkdir -p "$OUT"

EXTRA=""
if [ "$NODES" -gt 1 ]; then
  export JDIST=1 GPUS_PER_NODE=4
  EXTRA="--ragged"
fi
[ "$PROGRESS" = "1" ] && EXTRA="$EXTRA --progress"

echo "=== $(date) $TAG: PART=$PART NODES=$NODES STEPS=$STEPS PROGRESS=$PROGRESS nodelist=$SLURM_JOB_NODELIST ==="
T0=$(date +%s.%N)
srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$IC" --partition "$PART" \
    --steps "$STEPS" --chunk-steps "$CHUNK" $EXTRA \
    --checkpoint-every 1440 --diagnostics \
    --monthly-out "$OUT/monthly" --monthly-start-step 0 \
    --restart-out "$OUT/restart"
RC=$?
T1=$(date +%s.%N)
echo "$TAG exit=$RC  wall_elapsed=$(echo "$T1 - $T0" | bc)s"
