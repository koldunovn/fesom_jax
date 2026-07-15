#!/bin/bash
# Phase 8b B.3 STEP 3 — dars (3.16M x nl57) FULL model at dist_16 = 4 GPU NODES (16 A100), the
# next scaling-curve point after dars-8 (2N, 0.934 s/step). 197k nod/GPU (11.3M node-lev/GPU) —
# half of dars-8's per-GPU load, fits comfortably. real JRA1958+PHC+ice, dt=180, ragged only
# (allgather OOMs+hangs multi-node, dars-8 proved it). vs Kokkos M524 CUDA dars-4N = 0.475 s/step.
#SBATCH --job-name=bench_dars_d16
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_dars_d16.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_dars_d16.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --npes 16 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958 --ragged 1"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### dars FULL 16GPU (4 nodes) ragged ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
echo ">>>> dars d16 ragged exit: $? ($(date))"
echo "=== done $(date) ==="
