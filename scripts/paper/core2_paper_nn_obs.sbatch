#!/bin/bash
# §3 / E2 — NN-of-TKE OBS TRAINING + long-forward deployment validation on a single A100-80GB.
# (1) TRAIN: train the embedded tke_nn over K seasonal short windows (the 12 monthly snapshots) to
#     reduce MLD/SST vs WOA through the all3 frozen-ice adjoint; held-out cells ⇒ NN_OBS_OK.
# (2) VALIDATE: a LONG forward-only run (default NN→0 vs the trained NN) — drift ≤ default + stable +
#     persisted MLD benefit ⇒ NN_DRIFT_OK. Run as a SEPARATE process (fresh allocator pool for the
#     forward-only validation — the D2a spin-up/calibrate split). Reuses the E1 remat_blocks fix.
# Single A100; descend N on OOM. PREALLOCATE=false + command-buffers-off (the C1/E1 lessons) mandatory.
#SBATCH --job-name=core2_nn_obs
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_nn_obs.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_nn_obs.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/nn_obs_results.jsonl
CONFIG=${CONFIG:-all3}
K=${K:-8}                       # seasonal chunks (subsample of the 12 monthly snapshots)
NLIST=${NLIST:-"12 10 8"}       # per-chunk window (descend on OOM)
ITERS=${ITERS:-120}
LR=${LR:-0.1}
HOLDOUT=${HOLDOUT:-random}
FOLD=${FOLD:-0}
WSST=${WSST:-0.3}
REG=${REG:-0.03}                # trust-region reg (keep the closure mild ⇒ deploys; offline/online fix)
MMAX=${MMAX:-2.0}               # bounded multiplier cap (structural stability)
LONG_STEPS=${LONG_STEPS:-4320}  # forward-only deployment horizon (90 d) — cheap
LONG_SEG=${LONG_SEG:-240}       # re-stacked forcing segment (5 d)
NNPKL=${NNPKL:-/work/ab0995/a270088/port_jax/nn_obs_trained.pkl}
EXTRA=${EXTRA:-}

echo "=== $(hostname) $(date)  E2 NN-obs config=$CONFIG K=$K NLIST='$NLIST' iters=$ITERS holdout=$HOLDOUT/$FOLD ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

# ---- (1) TRAIN: descend N on OOM ----
TRAIN_RC=1
for N in $NLIST; do
    echo ""
    echo "===== TRAIN config=$CONFIG N=$N K=$K  $(date) ====="
    "$PY" scripts/paper/core2_paper_nn_obs.py --mode train --config "$CONFIG" --n "$N" --windows "$K" \
        --iters "$ITERS" --lr "$LR" --holdout "$HOLDOUT" --fold "$FOLD" --w-sst "$WSST" \
        --reg "$REG" --m-max "$MMAX" --nn-pkl "$NNPKL" --results "$RESULTS" $EXTRA
    TRAIN_RC=$?
    echo "  (TRAIN config=$CONFIG N=$N K=$K exit: $TRAIN_RC)"
    [ "$TRAIN_RC" -eq 0 ] && break
    [ "$TRAIN_RC" -ne 2 ] && { echo "  (hard-failed rc=$TRAIN_RC; not retrying)"; break; }
done

# ---- (2) VALIDATE: long forward-only (separate process, fresh pool) ----
if [ -f "$NNPKL" ]; then
    echo ""
    echo "===== VALIDATE long-forward $LONG_STEPS steps seg $LONG_SEG  $(date) ====="
    "$PY" scripts/paper/core2_paper_nn_obs.py --mode validate --config "$CONFIG" \
        --long-steps "$LONG_STEPS" --long-seg "$LONG_SEG" \
        --nn-pkl "$NNPKL" --results "$RESULTS" $EXTRA
    echo "  (VALIDATE exit: $?)"
else
    echo "  (no trained NN pkl at $NNPKL — skipping validate)"
fi
echo ""
echo "=== done $(date) ==="
