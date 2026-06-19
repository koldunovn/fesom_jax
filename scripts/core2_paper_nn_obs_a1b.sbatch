#!/bin/bash
# §3 / A1(b) PROBE — physically-informed-prior NN-of-TKE: warm-start the closure at the
# const-mult-validated 2× ("more mixing", the side that IMPROVES the 90-d deployed MLD) and anchor
# the trust-region reg THERE, so the short-window adjoint refines SPATIAL structure around the right
# anchor instead of collapsing to the fast-response m<1 (the E2 offline/online failure).
# Honest framing: the short-window objective itself prefers less mixing, so this is a well-posed
# PROBE — does spatial structure beat the uniform 2× (the known +0.5% deployed-MLD bar)? The 90-d
# VALIDATE forward is the gate. If it collapses (to E2 or to uniform-2×), the principled fix is
# EKI-on-deployed-MLD (forward-only ⇒ immune to the fast/slow misalignment).
# Separate RESULTS + NNPKL ⇒ the committed E2 NN_OBS_OK artifacts are untouched. Single A100-80.
#SBATCH --job-name=core2_nn_a1b
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_nn_obs_a1b.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_nn_obs_a1b.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/nn_obs_a1b_results.jsonl          # A1(b)-specific (E2's nn_obs_results.jsonl untouched)
NNPKL=/work/ab0995/a270088/port_jax/nn_obs_a1b_trained.pkl
CONFIG=${CONFIG:-all3}
K=${K:-8}
NLIST=${NLIST:-"12 10 8"}
ITERS=${ITERS:-120}
LR=${LR:-0.1}
HOLDOUT=${HOLDOUT:-random}
FOLD=${FOLD:-0}
WSST=${WSST:-0.3}
REG=${REG:-0.5}                  # FIRM anchor (vs E2's 0.03) ⇒ hold near 2× while spatial structure emerges
MMAX=${MMAX:-3.0}                # room around the 2× anchor (vs E2's 2.0 cap)
REGTGT=${REGTGT:-2.0}            # A1(b): anchor toward the 2× "more-mixing" side
LONG_STEPS=${LONG_STEPS:-4320}   # 90-d deployment horizon
LONG_SEG=${LONG_SEG:-240}
EXTRA=${EXTRA:-}

echo "=== $(hostname) $(date)  A1(b) NN-obs reg-target=$REGTGT reg=$REG m_max=$MMAX config=$CONFIG K=$K ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

# ---- (1) TRAIN: warm-start at 2×, anchored reg; descend N on OOM ----
TRAIN_RC=1
for N in $NLIST; do
    echo ""
    echo "===== TRAIN A1(b) config=$CONFIG N=$N K=$K reg-target=$REGTGT  $(date) ====="
    "$PY" scripts/core2_paper_nn_obs.py --mode train --config "$CONFIG" --n "$N" --windows "$K" \
        --iters "$ITERS" --lr "$LR" --holdout "$HOLDOUT" --fold "$FOLD" --w-sst "$WSST" \
        --reg "$REG" --m-max "$MMAX" --reg-target "$REGTGT" \
        --nn-pkl "$NNPKL" --results "$RESULTS" $EXTRA
    TRAIN_RC=$?
    echo "  (TRAIN A1(b) N=$N exit: $TRAIN_RC)"
    [ "$TRAIN_RC" -eq 0 ] && break
    [ "$TRAIN_RC" -ne 2 ] && { echo "  (hard-failed rc=$TRAIN_RC; not retrying)"; break; }
done

# ---- (2) VALIDATE: 90-d forward-only (trained vs default NN→0); compare to the uniform-2× +0.5% bar ----
if [ -f "$NNPKL" ]; then
    echo ""
    echo "===== VALIDATE A1(b) long-forward $LONG_STEPS steps seg $LONG_SEG  $(date) ====="
    "$PY" scripts/core2_paper_nn_obs.py --mode validate --config "$CONFIG" \
        --long-steps "$LONG_STEPS" --long-seg "$LONG_SEG" \
        --nn-pkl "$NNPKL" --results "$RESULTS" $EXTRA
    echo "  (VALIDATE A1(b) exit: $?)"
else
    echo "  (no trained NN pkl at $NNPKL — skipping validate)"
fi
echo ""
echo "=== done $(date) ==="
