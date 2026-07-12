#!/bin/bash
# FORCA20 (2.13M x nl70) FULL model at dist_32 = 8 GPU NODES (paper §5: second FORCA20
# scaling point). ⚠️ Known risk: the production run_from_config path COMPILE-HUNG on
# forca20 dist_32 (2026-07 hindcast v2 notes); this is a bounded 30-min attempt via the
# bench path — if it hangs, the walltime kills it harmlessly and the hang is confirmed
# as partition-dependent. READ-ONLY inputs; writes only this log.
#SBATCH --job-name=bench_forca20_d32
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_forca20_d32.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_forca20_d32.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
D=/work/ab0995/a270088/meshes/FORCA20
C="--name forca20 --mesh-dir $M --dist-dir $D --npes 32 --steps 25 --warmup 5 \
   --dt 240 --full 1 --ic-dir $M --year 1958 --ragged 1"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### forca20 FULL 32 GPU (8 nodes) ragged ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
echo "=== done $(date) ==="
