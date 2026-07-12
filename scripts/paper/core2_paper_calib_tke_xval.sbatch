#!/bin/bash
# Paper-experiments Task D2c — held-out (independent-cell) CROSS-VALIDATION of the TKE->WOA(MLD+SST)
# calibration, single A100-80GB. Reuses the D2a spin-up state S0 (config tkegm; spins only if absent),
# then runs THREE short-adjoint calibrations from the same S0:
#   (1) full-domain (--holdout none)  — the D2a headline reference for Fig 3
#   (2) lon fold 0  — train on odd 60deg sectors, SCORE the held-out even sectors
#   (3) lon fold 1  — train on even sectors, SCORE the held-out odd sectors
# 2-fold CV: the recovered c_k must AGREE across folds (cross-fold scatter = the statistical floor,
# since the staged EN4 is a seasonal climatology, not the multi-year series an interannual spread
# needs) and each TRAIN-only c_k must reduce its HELD-OUT misfit (generalization, not overfitting).
# ck-only (well-constrained; ck_ceps overfits c_eps->0). Same XLA flags / descend-N-on-OOM as D2a.
#SBATCH --job-name=core2_paper_tkexval
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_calib_tke_xval.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_calib_tke_xval.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/calib_tke_xval_results.jsonl
NSPIN=${1:-480}
DEG=${2:-60}
NLIST=${3:-"12 8 6 4"}
CONFIG=tkegm           # ocean-only; the obs-calib backward fits (all3 OOMs on the mEVP recompute)
PARAMS=ck              # the well-constrained headline parameter
S0=/work/ab0995/a270088/port_jax/calib_tke_obs_S0.pkl

echo "=== $(hostname) $(date)  n-spin=$NSPIN deg=$DEG NLIST='$NLIST' config=$CONFIG ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

# 0) spin-up ONCE → S0.pkl (reuse the D2a state if present; its retained pool dies with the process
#    so each calibration backward gets a fresh allocator pool).
if [ ! -f "$S0" ]; then
    echo ""
    echo "===== spin-up n-spin=$NSPIN config=$CONFIG  $(date) ====="
    "$PY" scripts/paper/core2_paper_calib_tke_obs.py --mode spinup --n-spin "$NSPIN" --config "$CONFIG" --s0 "$S0"
    if [ "$?" -ne 0 ]; then echo "spin-up FAILED; aborting"; exit 1; fi
else
    echo "reusing existing S0 -> $S0"
fi

# helper: calibrate (loads S0), descending N on OOM
run_calib () {   # $1=holdout $2=fold
    local HO=$1 FOLD=$2
    for N in $NLIST; do
        echo ""
        echo "===== calibrate holdout=$HO fold=$FOLD N=$N  $(date) ====="
        "$PY" scripts/paper/core2_paper_calib_tke_obs.py --mode calibrate --n-spin "$NSPIN" --n "$N" \
            --params "$PARAMS" --config "$CONFIG" --holdout "$HO" --holdout-deg "$DEG" --fold "$FOLD" \
            --s0 "$S0" --results "$RESULTS"
        rc=$?
        echo "  (holdout=$HO fold=$FOLD N=$N exit: $rc)"
        if [ "$rc" -eq 0 ]; then return 0; fi
        if [ "$rc" -ne 2 ]; then echo "  (hard-failed rc=$rc; not retrying)"; return "$rc"; fi
    done
    return 1
}

# 1) full-domain reference  2) + 3) the two held-out folds
run_calib none 0
run_calib lon  0
run_calib lon  1

echo ""
echo "=== done $(date) ==="
