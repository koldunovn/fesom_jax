#!/bin/bash
# Phase 8b B.1 — farc (638k) COMPLETE-model per-step bench, real JRA1958+PHC+ice, npes=4, dt=900,
# CORRECTED timing (subtraction method, compile excluded). vs Kokkos farc full GPU 1N = 0.309 s/step.
# full allgather + ragged + the oce/+ice decomposition. Needs the cached farc PHC IC (cache_phc_ic_farc).
#SBATCH --job-name=bench_farc_full
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_farc_full.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_farc_full.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_farc
C="--name farc --mesh-dir $M --dist-dir $POOL/farc --npes 4 --steps 25 --warmup 5 \
   --dt 900 --full 1 --ic-dir $M --year 1958"
echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null | head -1 || true
echo "#### farc full allgather ####"; "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 1 --kpp 1 --gm 1
echo "#### farc oce (base) ####";     "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 0 --kpp 0 --gm 0
echo "#### farc oce+ice ####";        "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 1 --kpp 0 --gm 0
echo "#### farc full ragged ####";    "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 1 --ice 1 --kpp 1 --gm 1
echo "=== done $(date) ==="
