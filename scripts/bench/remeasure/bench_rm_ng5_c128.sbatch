#!/bin/bash
# RE-MEASURE ng5 128 GPU (remeasure/README.md) — the optional far point, re-added
# 2026-07-16 (user); see bench_rm_dars_c128 for the rationale. coloured (the NG5
# transport winner), dt=180.
#SBATCH --job-name=bench_rm_ng5_c
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_rm_ng5_c.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_rm_ng5_c.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
C="--name ng5 --mesh-dir $M --dist-dir $POOL/ng5 --steps 150 --dt 180 \
   --full 1 --ic-dir $M --year 1958 --halo coloured"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST re-measure ng5 128 (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  echo "#### ng5 128 GPU coloured rep=$REP ####"
  srun -N32 -n32 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes 128
done
echo "=== done $(date) ==="
