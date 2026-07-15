#!/bin/bash
# Phase 8b — DECOMPOSE the CORE2 full-model per-step cost (real JRA1958+PHC) to locate the ~10x
# gap vs Kokkos (JAX full=1.19s, Kokkos full=0.117s). All npes=4, allgather, dt=1800, real forcing.
# Configs isolate ice(EVP) vs KPP vs GM on top of ocean+forcing:
#   oce      = ocean + forcing only (ice/kpp/gm OFF)   -> baseline
#   oce+ice  = + prognostic ice (the 120-subcycle EVP) -> ice cost = this - oce
#   oce+kg   = + KPP + GM (no ice)                      -> kpp+gm cost = this - oce
# (full with all three on already measured = 1.19 s.)
#SBATCH --job-name=bench_core2_decomp
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_core2_decomp.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_core2_decomp.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
COMMON="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --npes 4 --steps 10 \
        --ragged 0 --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958"
echo "=== node $(hostname)  $(date) ==="
echo "######## oce (ice=0 kpp=0 gm=0) ########";     "$PY" scripts/bench/bench_forward_scaling.py $COMMON --ice 0 --kpp 0 --gm 0
echo "######## oce+ice (ice=1 kpp=0 gm=0) ########"; "$PY" scripts/bench/bench_forward_scaling.py $COMMON --ice 1 --kpp 0 --gm 0
echo "######## oce+kg (ice=0 kpp=1 gm=1) ########";  "$PY" scripts/bench/bench_forward_scaling.py $COMMON --ice 0 --kpp 1 --gm 1
echo "=== done $(date) ==="
