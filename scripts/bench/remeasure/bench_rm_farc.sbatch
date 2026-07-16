#!/bin/bash
# RE-MEASURE farc 4/8/16 GPU (remeasure/README.md). padded (the ≤16-GPU winner —
# the 0714 §5 table flagged farc as a MATERIAL padded-vs-ragged change).
#SBATCH --job-name=bench_rm_farc
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_rm_farc.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_rm_farc.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_farc
C="--name farc --mesh-dir $M --dist-dir $POOL/farc --steps 150 --dt 900 \
   --full 1 --ic-dir $M --year 1958 --halo padded"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST re-measure farc (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  unset JDIST
  echo "#### farc 4 GPU padded rep=$REP ####"
  srun -N1 -n1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 4
  export JDIST=1 GPUS_PER_NODE=4
  echo "#### farc 8 GPU padded rep=$REP ####"
  srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 8
  echo "#### farc 16 GPU padded rep=$REP ####"
  srun -N4 -n4 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 16
done
echo "=== done $(date) ==="
