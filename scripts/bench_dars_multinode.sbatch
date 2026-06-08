#!/bin/bash
# Phase 8b B.3 STEP 3 — dars (3.16M) FULL model across 2 GPU NODES (8 A100, dist_8), real
# JRA1958+PHC+ice, dt=180, subtraction-timed. 1 process/node × 4 local GPUs (jax.distributed).
# allgather then ragged. vs Kokkos dars GPU (SCALING_M524). Needs the cached dars PHC IC.
#SBATCH --job-name=bench_dars_mn
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_dars_mn.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_dars_mn.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --npes 8 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### dars FULL 8GPU allgather ####"; srun --cpu-bind=none "$PY" scripts/bench_forward_scaling.py $C --ragged 0
echo "#### dars FULL 8GPU ragged ####";    srun --cpu-bind=none "$PY" scripts/bench_forward_scaling.py $C --ragged 1
echo "=== done $(date) ==="
