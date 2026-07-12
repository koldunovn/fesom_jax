#!/bin/bash
# FORCA20 (2.13M x nl70) FULL-model scaling point at dist_16 = 4 GPU NODES (paper §5:
# the production high-res mesh was missing from the scaling table). Same harness and
# protocol as bench_dars_dist16.sbatch: real JRA1958 + cached PHC IC + prognostic ice,
# ragged halo (allgather OOMs/hangs multi-node). dt_prod=240 is applied by the paper's
# reduce_scaling.py; the bench's internal dt only sets the operator.
# READ-ONLY inputs (mesh export /work/.../mesh_forca20, dist files /work/.../meshes/FORCA20,
# JRA pool); writes ONLY this log — the running forca20_hindcast_v3 chain is untouched.
#SBATCH --job-name=bench_forca20
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_forca20.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_forca20.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
D=/work/ab0995/a270088/meshes/FORCA20
C="--name forca20 --mesh-dir $M --dist-dir $D --npes 16 --steps 25 --warmup 5 \
   --dt 240 --full 1 --ic-dir $M --year 1958 --ragged 1"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### forca20 FULL 16GPU (4 nodes) ragged ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
echo ">>>> forca20 d16 ragged exit: $? ($(date))"
echo "=== done $(date) ==="
