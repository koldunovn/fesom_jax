#!/bin/bash
# ON-DEVICE FORCING A/B — CORE2 all-on production loop (run_from_config), 1 node / 4 GPU
# (dist_4, the hindcast layout), SAME allocation, interleaved x2:
#   leg off = per-step host stack (production today)
#   leg on  = --forcing-on-device (bracket tables + in-scan combine,
#             docs/plans/20260715-fesom-jax-ondevice-forcing.md)
# Measures what the lever actually targets: the per-chunk `[run.progress] host= device=`
# split (host forcing build + H2D). 240 steps / 48-step chunks; chunk 1 carries the
# compile — reduce over chunks 2-5. The end-of-run restarts double as the GPU
# INTEGRATION-LEVEL IDENTITY GATE: on GPU (IEEE div.rn.f64) ON must equal OFF
# bit-for-bit — compared array-by-array at the end of the job.
#SBATCH --job-name=jax_ondev_ab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80; one slow node paces the job
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
# .log, NOT .out — keep A/Bs out of the paper's bench_*.out glob (20260714 gotcha 1).
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/ondev_forcing_gpu_ab.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/ondev_forcing_gpu_ab.%j.log
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
OUT=/work/ab0995/a270088/port_jax/runs/perf/ondev_ab.$SLURM_JOB_ID
mkdir -p "$OUT"

run_leg() {  # run_leg <label> <extra-args...>
    local label=$1; shift
    echo "#### CORE2 all-on dist_4 LEG=$label $(date +%T) ####"
    srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
        --mesh-dir data/mesh_core2 --dist-dir "$POOL" --ic-dir data/ic_core2 \
        --steps 240 --chunk-steps 48 --progress \
        --restart-out "$OUT/restart_$label" "$@"
    echo "#### LEG=$label rc=$? ####"
}

echo "=== $(date) node=$SLURM_JOB_NODELIST  on-device forcing A/B (240 steps, 48/chunk) ==="
for REP in 1 2; do
  run_leg "off_$REP"
  run_leg "on_$REP" --forcing-on-device
done

# Identity gate WITH the nondeterminism control (learned from job 26278428): the multi-GPU
# forced path is non-deterministic at roundoff run-to-run, so at 240 steps a plain OFF
# rerun decorrelates as much as switching the flag (off1↔off2: T rel 5.5e-3, uv 8.6e-3 —
# same order as off1↔on1). The honest gate: the flag's effect must not exceed ~10× the
# measured off1↔off2 floor per field. Bit-identity is unobservable here; short-window
# correctness is pinned by the CPU gates (test_jra55 + test_forcing_sharded).
echo "=== GPU identity gate (floor-controlled): off_1 vs on_1, floor = off_1 vs off_2 ==="
"$PY" - "$OUT/restart_off_1" "$OUT/restart_off_2" "$OUT/restart_on_1" << 'PYEOF'
import sys
import numpy as np
import zarr
o1 = zarr.open_group(sys.argv[1], mode="r")
o2 = zarr.open_group(sys.argv[2], mode="r")
n1 = zarr.open_group(sys.argv[3], mode="r")
GATED = {"T", "S", "uv", "uvnode", "eta_n", "hnode", "w"}
bad, allbit = 0, True
for n in sorted(set(o1.array_keys()) & set(n1.array_keys())):
    a, b, c = (np.asarray(g[n]) for g in (o1, o2, n1))
    scale = float(np.max(np.abs(a))) or 1.0
    floor = float(np.max(np.abs(a - b))) / scale
    delta = float(np.max(np.abs(a - c))) / scale
    if delta > 0:
        allbit = False
    mark = ""
    if n in GATED and delta > max(10 * floor, 1e-12):
        bad += 1; mark = "  <-- ABOVE 10x NOISE FLOOR"
    print(f"  {n:14s} floor={floor:9.3e}  on-off={delta:9.3e}{mark}")
if allbit:
    print("IDENTITY_GATE: PASS-BIT (bit-identical)")
    sys.exit(0)
print("IDENTITY_GATE:", "PASS (flag effect within the GPU nondeterminism floor)"
      if bad == 0 else f"FAIL ({bad} slow fields above 10x the rerun floor)")
sys.exit(1 if bad else 0)
PYEOF
echo "=== done $(date) rc=$? ==="
