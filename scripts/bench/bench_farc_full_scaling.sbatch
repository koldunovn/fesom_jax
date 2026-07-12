#!/bin/bash
# farc (638k x nl48) FULL-model strong-scaling curve at 4/8/16 GPUs (paper §5: farc had
# only ocean-only "base" rows; this gives the 4th mesh curve between CORE2 and dars).
# Same harness/protocol as the dars/NG5 points: real JRA1958 + cached PHC IC + prognostic
# ice (ice+kpp+gm), ragged halo, warm-executable timing. One 4-node job hosts all three
# points via sub-allocation sruns. READ-ONLY inputs; writes only this log.
#SBATCH --job-name=bench_farc_full
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_farc_full.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_farc_full.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_farc
C="--name farc --mesh-dir $M --dist-dir $POOL/farc --steps 25 --warmup 5 --dt 900 \
   --full 1 --ic-dir $M --year 1958 --ragged 1"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### farc FULL 4 GPU (1 node, single process) ####"
unset JDIST
srun -N1 -n1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 4
export JDIST=1 GPUS_PER_NODE=4
echo "#### farc FULL 8 GPU (2 nodes) ####"
srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 8
echo "#### farc FULL 16 GPU (4 nodes) ####"
srun -N4 -n4 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 16
echo "=== done $(date) ==="
