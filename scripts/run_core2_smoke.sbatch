#!/bin/bash
# CORE2 HINDCAST SMOKE (R1 de-risk): a short cold CORE2 run on dist_4 / 1 node (4 GPUs, the
# anti-scaling-optimal layout) that exercises the EXACT production path of the 1958-2020 hindcast —
# cold PHC IC -> chunked forced scan -> --monthly-out (the NEW monthly-mean code) -> --diagnostics
# (finite gate). 1536 steps = 32 days at dt=1800 ⇒ January 1958 fully flushes (month flips to Feb)
# and a partial February flushes at the end. Verifies: (a) the monthly-out zarr is written, finite,
# correctly shaped; (b) the cold CORE2 all-on run is finite; (c) real s/step to tune PERJOB; (d) the
# config + CLI wiring. ~10 min. MUST be green before launching the self-chaining production hindcast.
#SBATCH --job-name=core2_smoke
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_core2_smoke.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_core2_smoke.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="   # cmdbuf off (compile dead-end, PORTING_LESSONS)
export PYTHONUNBUFFERED=1                              # 1 node, single process: NO JDIST

MESH=data/mesh_core2
IC=data/ic_core2
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
OUT=/work/ab0995/a270088/port_jax/runs/core2_smoke
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== $(date) CORE2 smoke: cold, 1536 steps (32 days), dist_4/1node, --monthly-out ==="
srun --ntasks=1 --cpu-bind=none "$PY" scripts/run_from_config.py configs/core2_full.yaml \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$IC" --partition dist_4 \
    --steps 1536 --chunk-steps 48 --checkpoint-every 480 \
    --monthly-out "$OUT/monthly" --monthly-start-step 0 \
    --diagnostics --progress --restart-out "$OUT/restart"
R=$?
echo "=== smoke exit=$R ==="
echo "--- monthly zarrs written ---"; ls -d "$OUT"/monthly/* 2>/dev/null
[ $R -eq 0 ] && echo "CORE2_SMOKE_OK" || echo "CORE2_SMOKE_FAILED (exit=$R)"
