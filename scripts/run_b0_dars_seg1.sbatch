#!/bin/bash
# B0 de-risk (dars, 3.16 M nodes) — SEGMENT 1: RESUME seg0's restart on dist_16 (4 nodes x4 A100 =
# 16 GPU) — a DIFFERENT device count than seg0's dist_8 (8 GPU). This is the headline A1 feature at
# multi-node scale: save on 8 devices, resume on 16. Submit with --dependency=afterok on seg0.
#
#SBATCH --job-name=b0_dars_s1
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_b0_dars_s1.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_b0_dars_s1.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4

MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
OUT=/work/ab0995/a270088/port_jax/runs/dars_b0

echo "=== $(date) dars B0 seg1: nodes=$SLURM_JOB_NODELIST  dist_16 (16 GPU) RESUME seg0 (dist_16) +8 steps ==="
srun --cpu-bind=none "$PY" scripts/run_from_config.py configs/dars.yaml \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition dist_16 \
    --steps 8 --chunk-steps 8 --ragged --restart-in "$OUT/seg0" --restart-out "$OUT/seg1"
R=$?
echo "seg1 exit=$R"
if [ $R -eq 0 ]; then
  echo "MACHINERY_SCALE_DARS_OK (dars forced run + multi-node forced run + restart at dist_16)"
else
  echo "B0_DARS_FAILED (seg1 exit=$R)"
fi
