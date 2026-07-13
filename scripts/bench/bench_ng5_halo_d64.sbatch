#!/bin/bash
# Phase 8c — NG5 (7.4M x nl70) at dist_64 (16 nodes, 64 A100): padded vs ragged.
# The last decision point for "padded everywhere": it wins/ties ragged at <=16 GPUs
# (CORE2 4/8, dars 16) and [dars-32 pending]; NG5-64 is where the paper's scaling curves
# approach their turnover, and where the pad factor is largest. No zarr output (pure step
# timing, matching the CORE2/dars halo A/Bs). Tabulated ragged reference: 0.584 s/step.
#SBATCH --job-name=bench_ng5_halo64
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_ng5_halo64.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_ng5_halo64.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
C="--name ng5 --mesh-dir $M --dist-dir $POOL/ng5 --npes 64 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### NG5-64 padded ####"; srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo padded
echo "#### NG5-64 ragged ####"; srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo ragged
echo "=== done $(date) ==="
