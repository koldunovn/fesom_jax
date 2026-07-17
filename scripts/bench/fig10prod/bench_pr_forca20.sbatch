#!/bin/bash
# fig10 v2 production-physics kernel scaling — forca20 (protocol: fig10prod/README.md).
# Levers OFF (bit-identical code only); physics = the 63-yr class per mesh.
#SBATCH --job-name=bench_pr_forca20
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_pr_forca20.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_pr_forca20.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
C="--name forca20 --mesh-dir /work/ab0995/a270088/fesom_jax_meshes/mesh_forca20 --dist-dir /work/ab0995/a270088/meshes/FORCA20 --steps 150 --full 1 --dt 240 \
   --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 --ic-dir /work/ab0995/a270088/fesom_jax_meshes/mesh_forca20 --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST fig10prod forca20 (sha $(git rev-parse --short HEAD)) ==="
run_pt() { local NP=$1 SN=$2 TR=$3 REP=$4
  echo "#### forca20 $NP GPU $TR prod rep=$REP ####"
  if [ "$SN" -gt 1 ]; then export JDIST=1 GPUS_PER_NODE=4; else unset JDIST; fi
  srun -N$SN -n$SN --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes $NP --halo $TR
}
run_pt 16 4 ragged 1
run_pt 32 8 ragged 1
run_pt 16 4 ragged 2
run_pt 32 8 ragged 2
echo "=== done $(date) ==="
