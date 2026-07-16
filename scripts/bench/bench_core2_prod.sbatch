#!/bin/bash
# CORE2 bench with the EXACT production hindcast physics (paper §5): zstar + cvmix-TKE +
# mEVP + GM (the all-on core2_full.yaml combination), vs the historical bench config
# (linfs + KPP + std-EVP). Removes the ~10% caveat in the paper's protocol paragraph and
# supplies the missing 4-GPU peak_gpu. Same compile-once protocol. READ-ONLY inputs;
# writes only this log — the running hindcast chains are untouched.
#SBATCH --job-name=bench_core2_prod
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_core2_prod.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_core2_prod.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --npes 4 --steps 150 \
   --warmup 5 --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958 \
   --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1"
echo "=== node $(hostname)  $(date) ==="
echo "#### production physics (zstar+TKE+mEVP+GM) padded ####";    "$PY" scripts/bench/bench_forward_scaling.py $C --halo padded
echo "#### production physics (zstar+TKE+mEVP+GM) ragged ####";    "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 1
echo "#### production physics (zstar+TKE+mEVP+GM) allgather ####"; "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0
echo "=== done $(date) ==="
