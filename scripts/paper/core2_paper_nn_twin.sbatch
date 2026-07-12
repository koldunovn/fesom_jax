#!/bin/bash
# Paper-experiments Task E1 (§3 hybrid-ML) — CORE2 NN-of-TKE perfect-model twin on a single A100-80GB.
# Injects a seeded tke_nn (non-trivial bounded multiplier) → synthetic T/S evolution, then trains a
# trainee tke_nn FROM NN→0 back to it through the FULL paper model (zstar+TKE+mEVP+GM) with the
# FROZEN-ICE adjoint (D1/A8). Proves end-to-end NN training through the global ocean model:
#   (1) T/S evolution recovered (misfit ≪ NN→0 start), (2) induced-mixing field recovered, plus a
#   FD↔AD spot check on the NN-weight gradient. Single-GPU only; descend N on OOM (a twin is
# well-posed at any N). Same A100-80 / XLA flags as the D1/D2a twins (the C1 OOM + cmd-buffer lessons).
#SBATCH --job-name=core2_paper_nntwin
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_nn_twin.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_nn_twin.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
# PREALLOC=true gives BFC one big pool up front — more robust for the single large backward
# arena (the NN twin's ~38 GiB temp) than incremental growth, which fragments after the truth
# forward and fails the contiguous allocation even with headroom. Default false (the D2a precedent).
export XLA_PYTHON_CLIENT_PREALLOCATE=${PREALLOC:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/nn_twin_results.jsonl
CONFIG=${1:-all3}      # all3 (the proper model + frozen-ice adjoint); model_full removed ⇒ N=12 fits
NLIST=${2:-"12 8 6 4"}
ITERS=${3:-200}
# strong-signal start: the DEVELOPED 1-year all3 state (the cold IC / 10-day state under-constrain
# the induced field). Pass S0="" to fall back to the cold IC.
S0=${S0-/work/ab0995/a270088/port_jax/nn_twin_S0_1yr.pkl}
FROM_S0=""
[ -n "$S0" ] && [ -f "$S0" ] && FROM_S0="--from-s0 $S0"

# EXTRA forwards driver flags (e.g. --no-grad-check for long-window runs once the gradient is proven,
# --misfit-tol, --lr) without editing this file.
EXTRA=${EXTRA:-}
echo "=== $(hostname) $(date)  config=$CONFIG  NLIST='$NLIST'  iters=$ITERS  from_s0='$FROM_S0'  extra='$EXTRA' ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

for N in $NLIST; do
    echo ""
    echo "===== NN-of-TKE twin config=$CONFIG N=$N  $(date) ====="
    "$PY" scripts/paper/core2_paper_nn_twin.py --config "$CONFIG" --n "$N" --iters "$ITERS" $FROM_S0 --results "$RESULTS" $EXTRA
    rc=$?
    echo "  (config=$CONFIG N=$N exit: $rc)"
    if [ "$rc" -eq 0 ]; then break; fi
    if [ "$rc" -ne 2 ]; then echo "  (hard-failed rc=$rc; not retrying)"; break; fi
done
echo ""
echo "=== done $(date) ==="
