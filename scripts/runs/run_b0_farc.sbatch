#!/bin/bash
# B0 de-risk (farc, 638 k nodes, 1 node x 4 A100): the FIRST real-scale integration of the Part-A
# machinery — a forced sharded run -> portable restart -> resume -> continue. Two segments in one
# job (cold 16 steps -> restart -> +16 steps); both must print RUN_DRIVER_OK. Proves run.py +
# run_steps_sharded_forced + the chunked forcing + write/read_restart work at scale on real GPUs.
#
#   sbatch scripts/runs/run_b0_farc.sbatch
#
#SBATCH --job-name=b0_farc
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_b0_farc.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_b0_farc.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_farc   # C-export == prepare_mesh.py output; has the IC
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
OUT=/work/ab0995/a270088/port_jax/runs/farc_b0
mkdir -p "$OUT"

echo "=== nvidia-smi (4 GPUs?) ==="; nvidia-smi -L 2>/dev/null | head

echo "=== B0 segment 0: cold farc, 16 forced steps, dist_4, write restart ==="
$PY scripts/run_from_config.py configs/farc.yaml \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" \
    --steps 16 --chunk-steps 16 --restart-out "$OUT/seg0"
S0=$?

echo "=== B0 segment 1: resume $OUT/seg0, +16 forced steps, dist_4, write restart ==="
$PY scripts/run_from_config.py configs/farc.yaml \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" \
    --steps 16 --chunk-steps 16 --restart-in "$OUT/seg0" --restart-out "$OUT/seg1"
S1=$?

echo ""
if [ $S0 -eq 0 ] && [ $S1 -eq 0 ]; then
  echo "MACHINERY_SCALE_OK (farc forced run + portable restart + resume at 638k on 4 GPU)"
else
  echo "B0_FAILED (seg0=$S0 seg1=$S1)"
fi
