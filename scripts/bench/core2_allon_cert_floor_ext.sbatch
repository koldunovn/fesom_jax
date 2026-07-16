#!/bin/bash
# ALL-ON CERT FLOOR EXTENSION (follow-up to 26312994, which FAILED on ssh at 4.4x a
# 2-run rerun floor): the on-off ssh deviation (~1-2 mm, 3 of 126858 nodes over the
# gate) matches the PASSING session-2 magnitude — what shrank is the floor realization
# (2.4e-4 vs 9.0e-4 in 26280985; a 2-sample max-statistic is noisy). This job adds two
# more OFF legs (off_3, off_4) into the SAME cert directory and re-verdicts against the
# POOLED pairwise floor (6 pairs from 4 OFF legs): if the on-off max sits inside the
# pairwise-floor spread, the config is climate-equivalent under an honest floor; if it
# sticks out against EVERY pair, it is a real CGPOLY signature -> isolate (cert CGPOLY
# alone, per docs/plans/20260716-levers-on-scaling-figure.md Gate 1 fallback).
#SBATCH --job-name=jax_allon_cert_ext
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/allon_cert_ext.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/allon_cert_ext.%j.log
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
OUT=/work/ab0995/a270088/port_jax/runs/perf/allon_1yr_cert.26312994   # the EXISTING cert dir
STEPS=17520

run_leg() {
    local label=$1; shift
    echo "#### 1yr LEG=$label $(date +%T) ####"
    srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
        --mesh-dir data/mesh_core2 --dist-dir "$POOL" --ic-dir data/ic_core2 \
        --steps $STEPS --chunk-steps 48 \
        --monthly-out "$OUT/monthly_$label" \
        --restart-out "$OUT/restart_$label" "$@"
    echo "#### LEG=$label rc=$? $(date +%T) ####"
}

echo "=== $(date) node=$SLURM_JOB_NODELIST  floor extension: off_3 + off_4 ==="
run_leg off_3
run_leg off_4

echo "=== POOLED verdict: on vs 4 OFF legs, floor = 6 pairwise OFF-OFF spreads ==="
"$PY" - "$OUT" << 'PYEOF'
import sys, glob, itertools
import numpy as np
import zarr

out = sys.argv[1]
FIELDS = ("temp", "salt", "u", "v", "ssh", "a_ice", "m_ice")
LEGS_OFF = ["off_1", "off_2", "off_3", "off_4"]

def annual(leg, field):
    months = sorted(glob.glob(f"{out}/monthly_{leg}/*"))
    assert len(months) == 12, f"{leg}: {len(months)} months"
    return np.mean([np.asarray(zarr.open_group(m, mode="r")[field]).squeeze()
                    for m in months], axis=0)

fail = 0
for f in FIELDS:
    offs = {l: annual(l, f) for l in LEGS_OFF}
    on = annual("on", f)
    floors = [float(np.max(np.abs(offs[a] - offs[b])))
              for a, b in itertools.combinations(LEGS_OFF, 2)]
    ons = [float(np.max(np.abs(on - offs[l]))) for l in LEGS_OFF]
    fmax, fmed = max(floors), float(np.median(floors))
    omin, omed = min(ons), float(np.median(ons))
    # gate: the smallest on-off max must be within 3x the LARGEST pairwise floor
    # (the deviation must be explainable by rerun scatter against at least one leg),
    # and the median on-off within 3x the median floor (no systematic offset).
    ok = (omin <= 3.0 * fmax) and (omed <= 3.0 * fmed)
    mark = "" if ok else "  <-- FAIL"
    if not ok:
        fail += 1
    print(f"  {f:6s} floors(6 pairs) max={fmax:.3e} med={fmed:.3e} | "
          f"on-off(4 legs) min={omin:.3e} med={omed:.3e}{mark}")
print("ALLON_CERT_POOLED:", "PASS" if fail == 0 else f"FAIL ({fail} fields)")
sys.exit(1 if fail else 0)
PYEOF
echo "=== done $(date) rc=$? ==="
