#!/bin/bash
# Parameterized PERF MEASUREMENT run (dars by default; 4 nodes / dist_16). Device-timed via
# --progress (block_until_ready per chunk). A/B by CONFIG (ice on/off, evp sub-cycles) + CMDBUF.
#SBATCH --job-name=perf
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_perf.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_perf.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
CMDBUF=${CMDBUF:-0}
if [ "$CMDBUF" = "1" ]; then export XLA_FLAGS=""; else export XLA_FLAGS="--xla_gpu_enable_command_buffer="; fi
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
MESH=${MESH:-/work/ab0995/a270088/fesom_jax_meshes/mesh_dars}
POOL=${POOL:-/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars}
CONFIG=${CONFIG:-configs/dars.yaml}; PART=${PART:-dist_16}
STEPS=${STEPS:-96}; CHUNK=${CHUNK:-48}; TAG=${TAG:-perf}
OUT=/work/ab0995/a270088/port_jax/runs/perf/$TAG; rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== $(date) PERF $TAG: CONFIG=$CONFIG PART=$PART CMDBUF=$CMDBUF STEPS=$STEPS ==="
srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition "$PART" \
    --steps "$STEPS" --chunk-steps "$CHUNK" --ragged --local-forcing --progress --restart-out "$OUT"
echo "perf $TAG exit=$?"
