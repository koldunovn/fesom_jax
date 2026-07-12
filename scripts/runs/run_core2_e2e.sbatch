#!/bin/bash
# CORE2 end-to-end de-risk (the user's "test everything cheaply before NG5"): the FULL long-run
# pipeline on the cheap CORE2 mesh, MULTI-PROCESS (dist_12 = 3 nodes, the NG5 jax.distributed path):
# cold-start (seeded ice) -> run -> rolling CHECKPOINT -> write restart -> RESTART-RESUME in a fresh
# process -> continue -> CHECKPOINT -> restart -> ushow OUTPUT. Catches integration surprises (CLI,
# local-forcing, checkpoint, restart chain, output) that step-level tests can't.
#SBATCH --job-name=core2_e2e
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=3
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_core2_e2e.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_core2_e2e.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
MESH=data/mesh_core2; IC=data/ic_core2; POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
OUT=/work/ab0995/a270088/port_jax/runs/core2_e2e
rm -rf "$OUT"; mkdir -p "$OUT"
COMMON="--mesh-dir $MESH --dist-dir $POOL --ic-dir $IC --partition dist_12 --chunk-steps 40 \
        --checkpoint-every 40 --local-forcing --progress"
echo "=== $(date) seg0: cold, 120 steps, checkpoint every 40 ==="
srun --cpu-bind=none "$PY" scripts/run_from_config.py configs/core2_full.yaml $COMMON \
    --steps 120 --restart-out "$OUT/seg0"; R0=$?
echo "=== seg1: RESTART-RESUME from seg0, +120 steps ==="
srun --cpu-bind=none "$PY" scripts/run_from_config.py configs/core2_full.yaml $COMMON \
    --steps 120 --restart-in "$OUT/seg0" --restart-out "$OUT/seg1"; R1=$?
echo "=== ushow output from the final restart (single process) ==="
"$PY" scripts/tools/restart_to_ushow.py "$OUT/seg1" "$MESH" "$OUT/seg1_ushow.zarr"; R2=$?
echo "seg0=$R0 seg1=$R1 ushow=$R2"
[ $R0 -eq 0 ] && [ $R1 -eq 0 ] && [ $R2 -eq 0 ] && echo "CORE2_E2E_OK" || echo "CORE2_E2E_FAILED"
