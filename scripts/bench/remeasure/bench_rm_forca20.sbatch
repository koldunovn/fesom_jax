#!/bin/bash
# RE-MEASURE forca20 16/32 GPU (remeasure/README.md). ragged (banked best/tie). dt=240.
# NOTE the production-chain memory: forca20 dist_32 COMPILE-HUNG in run_from_config chains
# (B0); the bench harness banked a forca20-32 row before — if the 32-point hangs here,
# kill it and keep 16 (do not let it eat the allocation).
#SBATCH --job-name=bench_rm_forca20
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_rm_forca20.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_rm_forca20.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
export JDIST=1 GPUS_PER_NODE=4
D=/work/ab0995/a270088/meshes/FORCA20
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
C="--name forca20 --mesh-dir $M --dist-dir $D --steps 150 --dt 240 \
   --full 1 --ic-dir $M --year 1958 --halo ragged"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST re-measure forca20 (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  echo "#### forca20 16 GPU ragged rep=$REP ####"
  srun -N4 -n4 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 16
  echo "#### forca20 32 GPU ragged rep=$REP ####"
  timeout 45m srun -N8 -n8 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 32
done
echo "=== done $(date) ==="
