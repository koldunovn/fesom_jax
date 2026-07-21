#!/bin/bash
# 128-GPU TURNOVER VERIFICATION (user question 2026-07-20: fluke or real?).
# Two confounds in the banked dars/forca20-128 turnover: (a) both reps of each row shared
# ONE allocation (a slow allocation hits both identically); (b) TRANSPORT — the two meshes
# that degrade run RAGGED (a P-way collective, cost grows with P; P doubled 64->128) while
# NG5, which keeps scaling, runs COLOURED (K=Delta rounds, P-INDEPENDENT). Four legs, one
# fresh allocation: dars/forca20 x ragged/coloured, 150 steps, prod physics (dt as banked).
#   ragged  legs = the FLUKE test (second allocation of the banked rows)
#   coloured legs = the TRANSPORT test (if coloured removes the turnover, the wall at 128
#   is largely the P-way transport, and the per-point best-transport policy flips there)
#SBATCH --job-name=bench_pr_128race
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_pr_128race.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_pr_128race.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
MD=/work/ab0995/a270088/fesom_jax_meshes
CD="--name dars --mesh-dir $MD/mesh_dars --dist-dir $POOL/dars --steps 150 --full 1 --dt 120 \
    --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --ic-dir $MD/mesh_dars --year 1958 --npes 128"
CF="--name forca20 --mesh-dir $MD/mesh_forca20 --dist-dir /work/ab0995/a270088/meshes/FORCA20 --steps 150 --full 1 --dt 240 \
    --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --ic-dir $MD/mesh_forca20 --year 1958 --npes 128"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST 128-GPU turnover verification (sha $(git rev-parse --short HEAD)) ==="
for TR in ragged coloured; do
  echo "#### dars 128 GPU $TR prod race ####"
  srun -N32 -n32 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $CD --halo $TR
  echo "#### forca20 128 GPU $TR prod race ####"
  srun -N32 -n32 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $CF --halo $TR
done
echo "=== done $(date) ==="
