#!/bin/bash
# dars (3.16M x nl57) FULL model at dist_32 = 8 GPU NODES (paper §5: third dars scaling
# point after 8 and 16 GPUs). Same harness/protocol as bench_dars_dist16.sbatch.
# READ-ONLY inputs; writes only this log.
#SBATCH --job-name=bench_dars_d128
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_dars_d128.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_dars_d128.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --npes 128 --steps 25 --warmup 5 \
   --dt 60 --full 1 --ic-dir $M --year 1958 --ragged 1"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### dars FULL 128 GPU (32 nodes) ragged ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
echo "=== done $(date) ==="
