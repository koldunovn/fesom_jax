#!/bin/bash
# fig10 v2 production-physics kernel scaling — ng5_b (protocol: fig10prod/README.md).
# Levers OFF (bit-identical code only); physics = the 63-yr class per mesh.
#SBATCH --job-name=bench_pr_ng5_b
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_pr_ng5_b.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_pr_ng5_b.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
C="--name ng5 --mesh-dir /work/ab0995/a270088/fesom_jax_meshes/mesh_ng5 --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5 --steps 150 --full 1 --dt 180 \
   --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --ic-dir /work/ab0995/a270088/fesom_jax_meshes/mesh_ng5 --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST fig10prod ng5_b (sha $(git rev-parse --short HEAD)) ==="
run_pt() { local NP=$1 SN=$2 TR=$3 REP=$4
  echo "#### ng5 $NP GPU $TR prod rep=$REP ####"
  if [ "$SN" -gt 1 ]; then export JDIST=1 GPUS_PER_NODE=4; else unset JDIST; fi
  srun -N$SN -n$SN --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes $NP --halo $TR
}
run_pt 64 16 coloured 1
run_pt 64 16 coloured 2
echo "=== done $(date) ==="
