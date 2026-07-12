#!/bin/bash
# Phase 8b B.3 STEP 4 — NG5 (7.4M x nl70) FULL model at dist_64 = 16 GPU NODES (64 A100), the next
# scaling point after dist_32 (8N, 0.840 s/step). 116k nod/GPU (8.1M node-lev/GPU) — half of dist_32,
# so peak_gpu ~20 GiB (comfortable). real JRA1958+PHC+ice, dt=180, ragged + sharded Zarr output.
# vs Kokkos M524 CUDA NG5-16N = 0.492 s/step. 30-min walltime to backfill the 16-node allocation.
#SBATCH --job-name=bench_ng5_d64
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_ng5_d64.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_ng5_d64.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
ZD=/work/ab0995/a270088/fesom_jax_meshes/ng5_out_d64.zarr
C="--name ng5 --mesh-dir $M --dist-dir $POOL/ng5 --npes 64 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958 --ragged 1 --out-zarr $ZD"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### NG5 FULL 64GPU (16 nodes) ragged + zarr output ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
echo ">>>> ng5 d64 ragged exit: $? ($(date))"
echo "=== done $(date) ==="
