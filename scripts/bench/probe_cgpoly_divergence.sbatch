#!/bin/bash
# HOW un-bit-identical is CGPOLY? "Not bit-identical" is a category, not a magnitude. The
# Chebyshev preconditioner changes the CG iteration path, so the solve lands on a different
# point WITHIN the same convergence tolerance. This measures the actual per-step divergence
# in T, S, uv and the SSH increment against the degree-0 reference.
#
# NOTE this is a SHORT-RUN bound (3 and 20 steps), not a verdict: whether the difference stays
# bounded over a year is what the certification run is for.
#SBATCH --job-name=cgpoly_diverge
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cgpoly_diverge.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cgpoly_diverge.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=32
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
OUT=/scratch/a/a270088/cgpoly_diverge; mkdir -p $OUT
for steps in 3 20; do
  for k in 0 6; do
    echo "#### steps=$steps degree=$k ####"
    FESOM_CG_CHEB=$k XLA_FLAGS="--xla_force_host_platform_device_count=4" \
      "$PY" scripts/bench/bench_forward_scaling.py --name core2 --mesh-dir $PWD/data/mesh_core2 \
      --dist-dir $POOL/core2 --ic-dir $PWD/data/ic_core2 --npes 4 --steps $steps --warmup 1 \
      --dt 1800 --full 1 --year 1958 --halo coloured --state-out $OUT/s${steps}_k${k}.npz 2>&1 \
      | grep -E "^\[bench\] mesh|^\[bench-finite\]" | tail -2
  done
done
"$PY" - <<'PYEOF'
import numpy as np
for steps in (3, 20):
    a = np.load(f"/scratch/a/a270088/cgpoly_diverge/s{steps}_k0.npz")
    b = np.load(f"/scratch/a/a270088/cgpoly_diverge/s{steps}_k6.npz")
    print(f"--- {steps} steps: degree 6 vs degree 0 ---", flush=True)
    for k in a.files:
        x, y = a[k], b[k]
        d = np.abs(x - y)
        scale = np.max(np.abs(x)) or 1.0
        print(f"  {k:<6} max|diff|={d.max():.3e}  rel={d.max()/scale:.3e}  "
              f"rms={np.sqrt((d**2).mean()):.3e}", flush=True)
PYEOF
echo "=== done $(date) ==="
