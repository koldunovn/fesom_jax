#!/bin/bash
# RE-MEASURE forca20 64 GPU (remeasure/README.md; user re-added the far points 2026-07-16). ragged, dt=240.
# ⚠️ Read the bench-finite line before banking.
#SBATCH --job-name=bench_rm_forca20_b
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_rm_forca20_b.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_rm_forca20_b.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
C="--name forca20 --mesh-dir $M --dist-dir /work/ab0995/a270088/meshes/FORCA20 --steps 150 --dt 240 \
   --full 1 --ic-dir $M --year 1958 --halo ragged"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST re-measure forca20 64 (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  echo "#### forca20 64 GPU ragged rep=$REP ####"
  srun -N16 -n16 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 64
done
echo "=== done $(date) ==="
