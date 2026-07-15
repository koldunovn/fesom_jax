#!/bin/bash
# Phase 8c — dars (3.16M x nl57) at dist_32 (8 nodes, 32 A100): padded vs ragged.
# Padded wins/ties ragged at <=16 GPUs (bench_core2_halo3, bench_dars_halo16); the pad factor
# grows with P (11.6x @P32 at CORE2), so >=32 is where ragged could pull ahead — this is the
# first decision point for whether the paper's ragged scaling curves would change with padded.
#SBATCH --job-name=bench_dars_halo32
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
# .log, NOT .out: the paper reducer globs scripts/logs/bench_*.out and would pull these
# A/B re-runs into scaling.csv (figure takes best-per-ngpu, macros take first-in-glob-order
# => they silently disagree). Keep transport A/Bs out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_dars_halo32.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_dars_halo32.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --npes 32 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### dars-32 padded ####"; srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo padded
echo "#### dars-32 ragged ####"; srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo ragged
echo "=== done $(date) ==="
