#!/bin/bash
# RE-MEASURE dars 8/16/32 GPU (remeasure/README.md). dt=120 for ALL dars points (the
# dist_32 dt-180 blowup is partition-specific but 150-step horizons at 180 are unverified;
# one uniform caption note — SYPD uses DT_PROD=120 regardless). Transports: dars-8 padded
# (≤16-GPU winner), dars-16/32 ragged (banked best/tie). ⚠️ Bank NOTHING without reading
# the bench-finite line of that row.
#SBATCH --job-name=bench_rm_dars_a
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_rm_dars_a.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_rm_dars_a.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --steps 150 --dt 120 \
   --full 1 --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST re-measure dars 8/16/32 (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  echo "#### dars 8 GPU padded rep=$REP ####"
  srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 8 --halo padded
  echo "#### dars 16 GPU ragged rep=$REP ####"
  srun -N4 -n4 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 16 --halo ragged
  echo "#### dars 32 GPU ragged rep=$REP ####"
  srun -N8 -n8 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 32 --halo ragged
done
echo "=== done $(date) ==="
