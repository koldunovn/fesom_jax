#!/bin/bash
# LOCAL-forcing on-device DRIVER smoke (CPU, CORE2 all-on, dist_4 + 4 fake devices — the
# multi-partition scatter is actually exercised): the NG5 increment routes
# --local-forcing + forcing.on_device through LocalForcing.stack_tables_partitioned /
# forcing_const_partitioned instead of raising. Three legs:
#   A  --local-forcing                      (host per-step stack — the production NG5 path)
#   B  --local-forcing --forcing-on-device  (the NEW combined path)
#   C  --forcing-on-device                  (global tables — the session-2 certified path)
# Gates: B==C restarts BIT-IDENTICAL (both consume byte-identical table inputs through the
# SAME executable — the pytest gate pins the inputs, this pins the driver wiring), and
# B vs A within the slow-field 1e-4 rel budget (the in-jit FMA seed, PORTING_LESSONS
# `ONDEVICE_FORCING_XLA_CPU_DIVIDE` — bit-identity vs the HOST path is impossible in-scan).
#SBATCH --job-name=ondev_local_smoke
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/ondev_local_smoke.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/ondev_local_smoke.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 FESOM_REUSE_EXE=1
export XLA_FLAGS=--xla_force_host_platform_device_count=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
OUT=/scratch/a/a270088/ondev_local_smoke.$SLURM_JOB_ID
mkdir -p "$OUT"

run_leg () {  # $1 = leg name, $2… = extra flags
  local LEG=$1; shift
  echo "=== $(date) LEG=$LEG ==="
  srun "$PY" scripts/run_from_config.py configs/core2_full_bench.yaml \
      --mesh-dir data/mesh_core2 --dist-dir "$POOL" --ic-dir data/ic_core2 \
      --partition dist_4 --steps 12 --chunk-steps 6 --progress \
      --restart-out "$OUT/restart_$LEG" "$@"
  echo "=== LEG=$LEG rc=$? ==="
}

run_leg A --local-forcing
run_leg B --local-forcing --forcing-on-device
run_leg C --forcing-on-device

echo "=== compare: B==C bit-identical; B vs A slow-field 1e-4 rel budget ==="
"$PY" - "$OUT" << 'PYEOF'
import sys
import numpy as np
import zarr
out = sys.argv[1]
A = zarr.open_group(f"{out}/restart_A", mode="r")
B = zarr.open_group(f"{out}/restart_B", mode="r")
C = zarr.open_group(f"{out}/restart_C", mode="r")
fail = 0

# B == C: the local and global TABLE paths must be byte-identical end to end.
nbit = 0
for n in sorted(set(B.array_keys()) | set(C.array_keys())):
    xb, xc = np.asarray(B[n]), np.asarray(C[n])
    if not (xb.shape == xc.shape and np.array_equal(xb, xc, equal_nan=True)):
        nbit += 1
        print(f"  B!=C {n:14s} max|d|={float(np.max(np.abs(xb - xc))):.3e}")
print("LOCAL_VS_GLOBAL_TABLES:", "BIT-IDENTICAL" if nbit == 0 else f"FAIL ({nbit} fields differ)")
fail += nbit

# B vs A: the FMA-seed budget on the slow ocean fields (same gate as the global smoke).
GATED = {"T", "S", "uv", "uvnode", "eta_n", "hnode", "w"}
bad = 0
for n in sorted(set(A.array_keys()) | set(B.array_keys())):
    xa, xb = np.asarray(A[n]), np.asarray(B[n])
    scale = float(np.max(np.abs(xa))) or 1.0
    rel = float(np.max(np.abs(xa - xb))) / scale
    mark = ""
    if n in GATED and rel > 1e-4:
        bad += 1; mark = "  <-- OVER BUDGET"
    print(f"  {n:14s} rel={rel:9.3e}{mark}")
fail += bad
print("LOCAL_SMOKE:", "PASS" if fail == 0 else f"FAIL (bit={nbit}, slow-over-budget={bad})")
sys.exit(1 if fail else 0)
PYEOF
echo "=== done $(date) rc=$? ==="
