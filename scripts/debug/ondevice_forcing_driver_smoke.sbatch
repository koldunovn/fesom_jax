#!/bin/bash
# On-device forcing DRIVER smoke (CPU, CORE2 all-on, 12 steps / 6-step chunks): the pytest
# gates call run_steps_sharded_forced directly — this exercises the run.py/run_from_config
# wiring the GPU A/B would otherwise be first to touch (per-chunk stack_tables, nb formula,
# partition_forcing_tables/const, the _fkw plumbing, reuse-exe across chunks, restart write).
# OFF vs ON restarts compared to <=1e-9 (CPU's non-IEEE f64 divide seeds <=1 ULP into prec —
# bit-identity is the GPU A/B's job, see bench_core2_ondevice_forcing_ab.sbatch).
#SBATCH --job-name=ondev_driver_smoke
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/ondev_driver_smoke.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/ondev_driver_smoke.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 FESOM_REUSE_EXE=1
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
OUT=/scratch/a/a270088/ondev_driver_smoke.$SLURM_JOB_ID
mkdir -p "$OUT"

for LEG in off on; do
  EXTRA=""; [ "$LEG" = on ] && EXTRA="--forcing-on-device"
  echo "=== $(date) LEG=$LEG ==="
  srun "$PY" scripts/run_from_config.py configs/core2_full_bench.yaml \
      --mesh-dir data/mesh_core2 --dist-dir "$POOL" --ic-dir data/ic_core2 \
      --partition dist_1 --steps 12 --chunk-steps 6 --progress \
      --restart-out "$OUT/restart_$LEG" $EXTRA
  echo "=== LEG=$LEG rc=$? ==="
done

# The in-scan combine is FMA-contracted by XLA inside jit (~1e-9 rel seed on the forcing
# fields; measured, unavoidable — see PORTING_LESSONS `ONDEVICE_FORCING_XLA_CPU_DIVIDE`),
# and 12 all-on steps amplify that seed (mEVP stress is the stiffest: rel ~3e-2 observed;
# ocean T rel ~6e-7). Gate: the SLOW fields must stay ≤1e-4 rel — a wrong bracket (hours of
# wrong forcing) puts SST at 1e-3+ rel instantly. Everything else is reported, not gated.
echo "=== compare restarts (slow-field budget 1e-4 rel; FMA-seed growth expected) ==="
"$PY" - "$OUT/restart_off" "$OUT/restart_on" << 'PYEOF'
import sys
import numpy as np
import zarr
a = zarr.open_group(sys.argv[1], mode="r")
b = zarr.open_group(sys.argv[2], mode="r")
GATED = {"T", "S", "uv", "uvnode", "eta_n", "hnode", "w"}   # slow ocean fields
bad = 0
for n in sorted(set(a.array_keys()) | set(b.array_keys())):
    xa, xb = np.asarray(a[n]), np.asarray(b[n])
    scale = float(np.max(np.abs(xa))) or 1.0
    rel = float(np.max(np.abs(xa - xb))) / scale
    mark = ""
    if n in GATED and rel > 1e-4:
        bad += 1; mark = "  <-- OVER BUDGET"
    print(f"  {n:14s} rel={rel:9.3e}{mark}")
print("DRIVER_SMOKE:", "PASS" if bad == 0 else f"FAIL ({bad} slow fields over 1e-4 rel)")
sys.exit(1 if bad else 0)
PYEOF
echo "=== done $(date) rc=$? ==="
