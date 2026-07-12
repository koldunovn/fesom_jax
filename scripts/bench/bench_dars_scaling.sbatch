#!/bin/bash
# Phase 8b B.2 — dars (3.16M x nl57) forward bench at npes=4 (1 GPU node), dt=180, ragged vs
# all_gather. ⚠️ Kokkos (SCALING_M524) reports dars does NOT fit 4xA100 for the FULL step (starts
# at 2N); our OCEAN-ONLY bench is leaner (no ice/forcing state) so it MAY fit — and if all_gather
# OOMs but ragged fits, that's a concrete single-node memory win for ragged. Run allgather first
# so an OOM there still leaves the (leaner) ragged run to land.
#SBATCH --job-name=bench_dars
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_dars.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_dars.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
MD=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null | head -1 || true
for r in 0 1; do
  echo "######## dars npes=4 ragged=$r dt=180 ########"
  "$PY" scripts/bench/bench_forward_scaling.py --name dars --mesh-dir "$MD" --dist-dir "$POOL/dars" \
      --npes 4 --steps 50 --ragged $r --dt 180
done
echo "=== done $(date) ==="
