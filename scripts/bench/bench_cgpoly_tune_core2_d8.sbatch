#!/bin/bash
# CGPOLY TUNING A/B at CORE2-8 (Gate 0 of the levers-ON figure plan,
# docs/plans/20260716-levers-on-scaling-figure.md): pick the (degree, kappa) that the
# combined 1-yr cert will certify. Six legs, same allocation, standard cold 25-step
# protocol (comparable with the shipped k=3/kappa=30 A/B = −20.7 %, job 26301616):
#   off · k2 · k3 (the shipped config) · k4 · k3/kappa=100 · k4/kappa=100
# Read bench-finite per leg before trusting any row.
#SBATCH --job-name=jax_cgpoly_tune
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:15:00
# .log NOT .out — A/Bs stay out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cgpoly_tune_core2_d8.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cgpoly_tune_core2_d8.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir data/mesh_core2 --dist-dir $POOL/core2 --npes 8 \
   --steps 25 --dt 1800 --full 1 --mevp 1 --halo padded \
   --ic-dir data/ic_core2 --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST  CGPOLY tuning core2-8 (sha $(git rev-parse --short HEAD)) ==="
run_leg () {  # $1=label $2=degree $3=kappa (0 = off)
  unset FESOM_CG_CHEB FESOM_CG_CHEB_KAPPA
  [ "$2" != 0 ] && export FESOM_CG_CHEB=$2 FESOM_CG_CHEB_KAPPA=$3
  echo "#### core2-8 tune LEG=$1 ####"
  srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
}
run_leg off      0 0
run_leg k2_k30   2 30
run_leg k3_k30   3 30
run_leg k4_k30   4 30
run_leg k3_k100  3 100
run_leg k4_k100  4 100
echo "=== done $(date) ==="
