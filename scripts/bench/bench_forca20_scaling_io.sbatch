#!/bin/bash
# FORCA20 (2.13M nodes, 70 levels) GPU-scaling bench WITH REAL I/O (daily canonical zarr + periodic
# checkpoint) — mirrors bench_core2_scaling_io.sbatch. Cold PHC start, 5 model-days (STEPS @ dt=180,
# forca20_tke_mevp_prod.yaml's year-1 dt) so the measured rate includes real daily-zarr + checkpoint
# I/O, not just the compute step. Separate TAG/output dir/JAX cache from any production run.
#
# Only dist_16/32/64 exist for this mesh (4/8/16 nodes) — all multi-node, so JDIST + --ragged +
# --local-forcing are always on here (mirrors the proven run_dars_3yr.sbatch multi-node flag set for
# this mesh scale; --local-forcing is the NG5/FORCA20-scale host-forcing fix, not optional at this size).
#
# Submit once per GPU count:
#   sbatch --nodes=4  --gres=gpu:4 --export=ALL,PART=dist_16,TAG=forca20_scaling_16gpu $0
#   sbatch --nodes=8  --gres=gpu:4 --export=ALL,PART=dist_32,TAG=forca20_scaling_32gpu $0
#   sbatch --nodes=16 --gres=gpu:4 --export=ALL,PART=dist_64,TAG=forca20_scaling_64gpu $0
#SBATCH --job-name=forca20_scale
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_forca20_scaling_io.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_forca20_scaling_io.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1
export JAX_COMPILATION_CACHE_DIR=/work/ab0995/a270088/port_jax/jax_cache/forca20_scaling_bench
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

CONFIG=$REPO/configs/forca20_tke_mevp_prod.yaml
MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
POOL=/work/ab0995/a270088/meshes/FORCA20
IC=$MESH
PART=${PART:?set PART=dist_16|dist_32|dist_64}
TAG=${TAG:?set TAG}
STEPS=${STEPS:-2400}     # 5 model-days @ dt=180 (480 steps/day) -> real daily-zarr + checkpoint I/O
CHUNK=${CHUNK:-48}
OUT=/work/ab0995/a270088/port_jax/runs/perf/$TAG
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== $(date) $TAG: PART=$PART STEPS=$STEPS nodelist=$SLURM_JOB_NODELIST ==="
srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$IC" --partition "$PART" \
    --steps "$STEPS" --chunk-steps "$CHUNK" --ragged --local-forcing \
    --checkpoint-every 960 --diagnostics --progress \
    --daily-out "$OUT/daily" --daily-start-step 0 \
    --restart-out "$OUT/restart"
echo "$TAG exit=$?"
