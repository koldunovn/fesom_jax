#!/bin/bash
# B0 de-risk (dars, 3.16 M nodes) — SEGMENT 0: cold forced run on dist_8 (2 nodes x4 A100 = 8 GPU,
# MULTI-NODE jax.distributed), write a portable restart. Segment 1 (run_b0_dars_seg1.sbatch) then
# RESUMES it on dist_16 (4 nodes = 16 GPU) — the device-count-change restart at multi-node scale,
# the first multi-node test of run.py / run_from_config.py.
#
#SBATCH --job-name=b0_dars_s0
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_b0_dars_s0.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_b0_dars_s0.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4

MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
OUT=/work/ab0995/a270088/port_jax/runs/dars_b0
mkdir -p "$OUT"

echo "=== $(date) dars B0 seg0: nodes=$SLURM_JOB_NODELIST  dist_8 (8 GPU) cold 8 steps ==="
srun --cpu-bind=none "$PY" scripts/run_from_config.py configs/dars.yaml \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition dist_8 \
    --steps 8 --chunk-steps 8 --restart-out "$OUT/seg0"
echo "seg0 exit=$?"
