#!/bin/bash
# Phase 8b B.1 — forward-scaling benchmark at npes=4 on real A100s: CORE2 (small, below the
# crossover) vs farc (638k, above it — per-device halo ~1019 nod vs ~160k owned, so all_gather
# moves ~600x more for the node exchange). Full-ragged now (substep + the dominant CG/SSH halo).
# ragged vs all_gather. ≤30 min for the fast QOS.
#SBATCH --job-name=bench_farc
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_farc.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_farc.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null | head -1 || true
# spec: name  mesh-dir  dist-dir  dt  (Kokkos cold-start CFL: CORE2 1800, farc 900, dars/NG5 180)
for spec in "core2 $PWD/data/mesh_core2 $POOL/core2 1800" \
            "farc /work/ab0995/a270088/fesom_jax_meshes/mesh_farc $POOL/farc 900" ; do
  set -- $spec; name=$1; md=$2; dd=$3; dtv=$4
  for r in 0 1; do
    echo "######## $name npes=4 ragged=$r dt=$dtv ########"
    "$PY" scripts/bench_forward_scaling.py --name "$name" --mesh-dir "$md" --dist-dir "$dd" \
        --npes 4 --steps 100 --ragged $r --dt $dtv
  done
done
echo "=== done $(date) ==="
