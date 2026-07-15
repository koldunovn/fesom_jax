#!/bin/bash
# Phase 8d diagnostic — WHY the coloured transport wins or loses, per exchange, on GPU.
# dars at dist_16 (4 nodes) — cheap, and already in the A/B table, so it is directly comparable.
#
# The full-model A/B gives one number per step, which cannot separate the two exchange
# populations that pull in opposite directions: the CG's 2-D ssh exchange (tiny, ~254 per step
# => per-collective LATENCY dominates, and coloured pays K collectives where padded pays 1) and
# the 3-D tracer exchanges ([Lmax,nl] => BYTES dominate, and coloured ships 5-40x less). CORE2-4
# already showed the latency side winning (coloured 92.5 ms vs padded 80.4). This prices both
# ranks directly, so the hybrid question -- padded for the CG, coloured for the 3-D fields --
# can be answered with numbers instead of arithmetic.
#SBATCH --job-name=bench_halo_micro
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
# .log, NOT .out — keep transport A/Bs out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_halo_micro.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_halo_micro.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
srun --cpu-bind=none "$PY" scripts/bench/bench_halo_micro.py \
    --mesh-dir $M --dist-dir $POOL/dars --npes 16 --kind nod \
    --halo padded,coloured,ragged
echo "=== done $(date) ==="
