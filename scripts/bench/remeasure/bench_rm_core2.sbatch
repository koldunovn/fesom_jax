#!/bin/bash
# RE-MEASURE core2 1/2/4/8 GPU (see remeasure/README.md for the frozen protocol).
# padded transport (the ≤16-GPU winner); ngpu=1 is the transport-exempt anchor.
#SBATCH --job-name=bench_rm_core2
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_rm_core2.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_rm_core2.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB    # CGPOLY OFF; forcing.on_device OFF by default — the paper protocol
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir data/mesh_core2 --dist-dir $POOL/core2 --steps 150 \
   --dt 1800 --full 1 --ic-dir data/ic_core2 --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST re-measure core2 (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  unset JDIST
  echo "#### core2 1 GPU rep=$REP ####"
  srun -N1 -n1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 1
  echo "#### core2 2 GPU padded rep=$REP ####"
  srun -N1 -n1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 2 --halo padded
  echo "#### core2 4 GPU padded rep=$REP ####"
  srun -N1 -n1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 4 --halo padded
  export JDIST=1 GPUS_PER_NODE=4
  echo "#### core2 8 GPU padded rep=$REP ####"
  srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 8 --halo padded
done
echo "=== done $(date) ==="
