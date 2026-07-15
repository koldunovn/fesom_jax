#!/bin/bash
# Phase 8b — CORE2 COMPLETE-model per-step bench, CORRECTED timing (compile-once, time the
# reused 2nd call; compile reported separately). Real JRA1958 + PHC IC, npes=4, dt=1800.
# Decomposes the cost to locate the gap vs Kokkos (full=0.117 s/step):
#   full     = ice+KPP+GM (allgather, then ragged)
#   oce      = ocean+forcing only        (ice0 kpp0 gm0)  -> baseline
#   oce+ice  = + prognostic ice (120-subcycle EVP)        -> ice cost = this - oce
#   oce+kg   = + KPP + GM                                 -> kpp+gm cost = this - oce
#SBATCH --job-name=bench_core2_full
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_core2_full.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_core2_full.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --npes 4 --steps 25 \
   --warmup 5 --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958"
echo "=== node $(hostname)  $(date) ==="
# essentials first (full, baseline, +ice) so the key numbers land even if a later compile runs long
echo "#### full allgather ####"; "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 1 --kpp 1 --gm 1
echo "#### oce (base) ####";     "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 0 --kpp 0 --gm 0
echo "#### oce+ice ####";        "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 1 --kpp 0 --gm 0
echo "#### oce+kg ####";         "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 0 --kpp 1 --gm 1
echo "#### full ragged ####";    "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 1 --ice 1 --kpp 1 --gm 1
echo "=== done $(date) ==="
