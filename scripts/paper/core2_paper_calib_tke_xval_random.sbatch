#!/bin/bash
# Paper-experiments Task D2c — the CLEAN overfitting test: a seeded 50/50 RANDOM-cell 2-fold CV of
# the TKE->WOA MLD calibration (complements core2_paper_calib_tke_xval.sbatch's blocked lon/lat folds).
# A random split makes train+held-out share the SAME bias structure, so a non-overfit c_k reduces the
# held-out misfit ~as much as the train misfit (vs the blocked split, where the c_k-sensitive deep-
# convection signal is spatially concentrated and the held-out moves less). APPENDS to the xval
# results so Fig 3 sees both fold families. Reuses the D2a S0; single A100-80; ck-only; tkegm.
#SBATCH --job-name=core2_paper_tkexvalR
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_calib_tke_xval_random.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_calib_tke_xval_random.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/calib_tke_xval_results.jsonl       # APPEND (do not truncate — keep the lon/none folds)
NSPIN=${1:-480}
NLIST=${2:-"12 8 6 4"}
CONFIG=tkegm
PARAMS=ck
S0=/work/ab0995/a270088/port_jax/calib_tke_obs_S0.pkl

echo "=== $(hostname) $(date)  random-fold xval  n-spin=$NSPIN NLIST='$NLIST' ==="
nvidia-smi -L 2>/dev/null || true

if [ ! -f "$S0" ]; then
    echo "===== spin-up n-spin=$NSPIN config=$CONFIG  $(date) ====="
    "$PY" scripts/paper/core2_paper_calib_tke_obs.py --mode spinup --n-spin "$NSPIN" --config "$CONFIG" --s0 "$S0"
    if [ "$?" -ne 0 ]; then echo "spin-up FAILED; aborting"; exit 1; fi
else
    echo "reusing existing S0 -> $S0"
fi

run_calib () {   # $1=fold
    local FOLD=$1
    for N in $NLIST; do
        echo ""
        echo "===== calibrate holdout=random fold=$FOLD N=$N  $(date) ====="
        "$PY" scripts/paper/core2_paper_calib_tke_obs.py --mode calibrate --n-spin "$NSPIN" --n "$N" \
            --params "$PARAMS" --config "$CONFIG" --holdout random --fold "$FOLD" \
            --s0 "$S0" --results "$RESULTS"
        rc=$?
        echo "  (random fold=$FOLD N=$N exit: $rc)"
        if [ "$rc" -eq 0 ]; then return 0; fi
        if [ "$rc" -ne 2 ]; then echo "  (hard-failed rc=$rc; not retrying)"; return "$rc"; fi
    done
    return 1
}

run_calib 0
run_calib 1

echo ""
echo "=== done $(date) ==="
