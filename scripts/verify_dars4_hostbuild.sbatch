#!/bin/bash
# Phase 8b B.3 host-build REWRITE verification — dars (3.16M x nl57) FULL model on a SINGLE
# GPU node (dist_4, 4xA100), real JRA1958 + PHC IC + prognostic ice, dt=180. This is the
# headline test of the host-build + device_put-sharded fix: BEFORE the fix the data pipeline
# built the full global State (~40 GB) on GPU 0 in SETUP and OOM'd ("1.34 GiB jit__where");
# AFTER, the global IC stays HOST (numpy) and device_put places only each GPU's 1/4 shard.
# WATCH: "peak_gpu_after_setup" (must be « 40 GiB now) + a real per_step (no setup OOM).
# ragged FIRST (forward-safe, the scaling target); then all_gather (the grad-correct default,
# should also fit at 4 GPU single-node — O(P*N_local) is only 4x here). ONE compile/config.
#SBATCH --job-name=verify_dars4
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/verify_dars4.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/verify_dars4.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --npes 4 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958"
echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null | head -4 || true
echo "#### dars-4 FULL single-node ragged ####"
"$PY" scripts/bench_forward_scaling.py $C --ragged 1
echo ">>>> ragged exit: $? ($(date))"
echo "#### dars-4 FULL single-node allgather ####"
"$PY" scripts/bench_forward_scaling.py $C --ragged 0
echo ">>>> allgather exit: $? ($(date))"
echo "=== done $(date) ==="
