#!/bin/bash
# fig10 v2 production-physics kernel scaling — core2 (protocol: fig10prod/README.md).
# Levers OFF (bit-identical code only); physics = the 63-yr class per mesh.
#SBATCH --job-name=bench_pr_core2
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_pr_core2.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_pr_core2.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
unset FESOM_CG_CHEB
C="--name core2 --mesh-dir data/mesh_core2 --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --steps 150 --full 1 --dt 1800 \
   --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1 --ic-dir data/ic_core2 --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST fig10prod core2 (sha $(git rev-parse --short HEAD)) ==="
run_pt() { local NP=$1 SN=$2 TR=$3 REP=$4
  echo "#### core2 $NP GPU $TR prod rep=$REP ####"
  if [ "$SN" -gt 1 ]; then export JDIST=1 GPUS_PER_NODE=4; else unset JDIST; fi
  # npes=1 = single device, zero halo — run WITHOUT a transport flag (the banked
  # baseline's convention; the padded machinery is untested at P=1)
  local HF="--halo $TR"; [ "$NP" -eq 1 ] && HF=""
  srun -N$SN -n$SN --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --npes $NP $HF
}
run_pt 1 1 padded 1
run_pt 2 1 padded 1
run_pt 4 1 padded 1
run_pt 8 2 padded 1
run_pt 1 1 padded 2
run_pt 2 1 padded 2
run_pt 4 1 padded 2
run_pt 8 2 padded 2
echo "=== done $(date) ==="
