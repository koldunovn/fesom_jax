#!/bin/bash
# COMBINED ALL-LEVERS 1-YEAR CLIMATE CERTIFICATION (levers-ON figure Gate 1) — the port_kokkos M7 discipline for
# value-equivalent levers (their per-tier "1-yr climate" gate): forcing.on_device is
# VALUE-equivalent, not bit-identical (in-jit FMA contraction; PORTING_LESSONS
# `ONDEVICE_FORCING_XLA_CPU_DIVIDE`), so short-window gates + a timing A/B are not enough.
#
# Design: THREE legs, same allocation — off_1, off_2 (the run-to-run noise CONTROL: the
# multi-GPU forced path is non-deterministic at roundoff, so two OFF reruns of one year
# decorrelate in weather), on. Verdict per monthly-mean field: |on − off_1| must be within
# ~the |off_2 − off_1| floor (climate-equivalent), NOT zero. CORE2 all-on, dist_4, 1958,
# 17520 steps @ dt=1800, monthly means out.
#SBATCH --job-name=jax_allon_1yr_cert
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80; one slow node paces the job
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/allon_1yr_cert.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/allon_1yr_cert.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1

CONFIG=$REPO/configs/core2_full_bench.yaml
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
OUT=/work/ab0995/a270088/port_jax/runs/perf/allon_1yr_cert.$SLURM_JOB_ID
mkdir -p "$OUT"
STEPS=17520   # 365 d @ dt=1800 (year 1958)

run_leg() {  # run_leg <label> <extra-args...>
    local label=$1; shift
    echo "#### 1yr LEG=$label $(date +%T) ####"
    srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
        --mesh-dir data/mesh_core2 --dist-dir "$POOL" --ic-dir data/ic_core2 \
        --steps $STEPS --chunk-steps 48 \
        --monthly-out "$OUT/monthly_$label" \
        --restart-out "$OUT/restart_$label" "$@"
    echo "#### LEG=$label rc=$? $(date +%T) ####"
}

echo "=== $(date) node=$SLURM_JOB_NODELIST  on-device forcing 1-yr climate cert ==="
run_leg off_1
run_leg off_2
run_leg on --forcing-on-device --cheb-degree ${CHEB_DEGREE:?set CHEB_DEGREE from the tuning verdict}

echo "=== climate-equivalence verdict (floor-controlled, monthly means) ==="
"$PY" - "$OUT" << 'PYEOF'
import sys
from pathlib import Path
import numpy as np
import zarr

out = Path(sys.argv[1])
months = sorted(p.name for p in (out / "monthly_off_1").iterdir() if p.is_dir())
print(f"months found: {len(months)}")
fields = ("temp", "salt", "u", "v", "ssh", "a_ice", "m_ice")

def monthly(leg, mo, f):
    g = zarr.open_group(str(out / f"monthly_{leg}" / mo), mode="r")
    return np.asarray(g[f])

bad = 0
# Per-field ANNUAL mean comparison (weather averages down; climate signal remains) +
# the worst single month, both against the off_2-off_1 rerun floor.
for f in fields:
    ann = {leg: np.mean([monthly(leg, mo, f) for mo in months], axis=0)
           for leg in ("off_1", "off_2", "on")}
    scale = float(np.max(np.abs(ann["off_1"]))) or 1.0
    floor = float(np.max(np.abs(ann["off_2"] - ann["off_1"]))) / scale
    delta = float(np.max(np.abs(ann["on"] - ann["off_1"]))) / scale
    rms_floor = float(np.sqrt(np.mean((ann["off_2"] - ann["off_1"]) ** 2))) / scale
    rms_delta = float(np.sqrt(np.mean((ann["on"] - ann["off_1"]) ** 2))) / scale
    mark = ""
    if delta > max(3 * floor, 1e-12) or rms_delta > max(3 * rms_floor, 1e-12):
        bad += 1; mark = "  <-- ABOVE 3x RERUN FLOOR"
    print(f"  {f:6s} annual-mean: floor max={floor:9.3e} rms={rms_floor:9.3e} | "
          f"on-off max={delta:9.3e} rms={rms_delta:9.3e}{mark}")
print("ONDEV_1YR_CERT:", "PASS (climate-equivalent within the rerun floor)"
      if bad == 0 else f"FAIL ({bad} fields above 3x the rerun floor)")
sys.exit(1 if bad else 0)
PYEOF
echo "=== done $(date) rc=$? ==="
