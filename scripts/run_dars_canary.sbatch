#!/bin/bash
# dars COMPILE-CANARY: de-risk a config at dars/dist_16 (4 nodes) with a HARD compile cap. A healthy
# dars compile is <25 min (KPP+EVP perf); the TKE+mEVP hang was >66 min. `timeout 1800` (30 min)
# auto-kills a pathological compile (exit 124) so no node-hours are burned waiting. 60 steps is just
# enough to confirm compile + early finiteness. Launch: sbatch --export=ALL,CONFIG=...,TAG=... <this>
#SBATCH --job-name=dars_canary
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_dars_canary.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_dars_canary.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-$PWD/configs/dars_kpp_mevp.yaml}
TAG=${TAG:-canary}
MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
OUT=/work/ab0995/a270088/port_jax/runs/dars_canary/$TAG; rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== $(date) dars canary CONFIG=$CONFIG (30-min compile cap, 60 steps, dist_16) ==="
T0=$SECONDS
timeout 1800 srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition dist_16 \
    --steps 60 --chunk-steps 30 --ragged --local-forcing --progress --diagnostics \
    --restart-out "$OUT"
R=$?
echo ">>> dars canary $TAG exit=$R elapsed=$((SECONDS-T0))s $([ $R -eq 124 ] && echo 'COMPILE-HANG' || echo 'completed')"
[ $R -eq 0 ] && echo "DARS_CANARY_OK ($TAG compiles + runs at dars/dist_16)" || echo "DARS_CANARY_FAIL ($TAG exit=$R)"
