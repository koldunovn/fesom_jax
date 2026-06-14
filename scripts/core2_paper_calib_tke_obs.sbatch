#!/bin/bash
# Paper-experiments Task D2a (obs application) — CORE2 TKE->WOA(MLD+SST) calibration, single A100-80GB.
# Spin up forward (no AD) to a model-consistent state, then a short adjoint window calibrates
# {tke_c_k, tke_c_eps} to the WOA monthly-derived MLD + SST through the all-on + frozen-ice adjoint
# (A8) and the obs operator. Reports recovered VALUES + plausibility + baseline->calibrated misfit.
# Descend the WINDOW N on OOM (the spin-up is forward-only/light). Same XLA flags as the twins.
#SBATCH --job-name=core2_paper_tkeobs
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_calib_tke_obs.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_calib_tke_obs.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/calib_tke_obs_results.jsonl
NSPIN=${1:-480}
PARAMS=${2:-ck_ceps}
NLIST=${3:-"12 8 6 4"}
CONFIG=${4:-tkegm}      # tkegm (ocean-only) fits the obs-calib backward; all3 OOMs (mEVP recompute)

echo "=== $(hostname) $(date)  n-spin=$NSPIN params=$PARAMS NLIST='$NLIST' ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

# 1) spin-up ONCE in its own process → S0.pkl (its retained allocator pool dies with the process,
#    so the calibration backward gets a fresh pool — the in-process version OOMed at every N).
echo ""
echo "===== spin-up n-spin=$NSPIN config=$CONFIG  $(date) ====="
"$PY" scripts/core2_paper_calib_tke_obs.py --mode spinup --n-spin "$NSPIN" --config "$CONFIG"
if [ "$?" -ne 0 ]; then echo "spin-up FAILED; aborting"; exit 1; fi

# 2) calibrate per N (loads S0), descending on OOM
for N in $NLIST; do
    echo ""
    echo "===== calibrate N=$N params=$PARAMS config=$CONFIG  $(date) ====="
    "$PY" scripts/core2_paper_calib_tke_obs.py --mode calibrate --n-spin "$NSPIN" --n "$N" \
        --params "$PARAMS" --config "$CONFIG" --results "$RESULTS"
    rc=$?
    echo "  (N=$N exit: $rc)"
    if [ "$rc" -eq 0 ]; then break; fi
    if [ "$rc" -ne 2 ]; then echo "  (hard-failed rc=$rc; not retrying)"; break; fi
done
echo ""
echo "=== done $(date) ==="
