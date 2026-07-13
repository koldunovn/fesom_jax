#!/bin/bash
# Phase 8c — dars (3.16M x nl57) FULL model at dist_16 (4 nodes, 16 A100): padded vs ragged.
# At CORE2 the padded (AD-correct) transport beats ragged at 4 AND 8 GPUs (jobs 26227814/5);
# this asks whether that holds on a multi-million-node mesh at its production GPU count, where
# the pad factor is larger (docs/PARALLELISM.md). Same flags as bench_dars_dist16.sbatch
# (dt=180 bench; per-step time is dt-independent, SYPD is reported at dt_prod=120).
#SBATCH --job-name=bench_dars_halo16
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_dars_halo16.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_dars_halo16.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --npes 16 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### dars-16 padded ####"; srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo padded
echo "#### dars-16 ragged ####"; srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo ragged
echo "=== done $(date) ==="
