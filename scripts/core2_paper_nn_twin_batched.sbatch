#!/bin/bash
# §3 / E2 — batched-window NN-of-TKE twin on a single A100-80GB. The single-window E1 adjoint is
# memory-capped at N≈10 (~5 h), too short to recover a rich per-column NN field. This trains over K
# chained N-step chunks starting from K different truth states, ACCUMULATING the gradient across chunks
# (one chunk's backward live at a time ⇒ per-chunk memory = the fitting N, effective window = K×N).
# Reuses the E1 remat_blocks fix + PREALLOCATE=false (mandatory). Single A100; descend K on OOM.
#SBATCH --job-name=core2_nntwin_batched
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_nn_twin_batched.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_nn_twin_batched.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/nn_twin_batched_results.jsonl
CONFIG=${1:-all3}
N=${2:-10}                 # per-chunk steps (the memory-feasible window)
KLIST=${3:-"8 6 4"}        # K chunks (descend on OOM)
ITERS=${4:-120}
S0=${S0-/work/ab0995/a270088/port_jax/nn_twin_S0_1yr.pkl}
FROM_S0=""
[ -n "$S0" ] && [ -f "$S0" ] && FROM_S0="--from-s0 $S0"
EXTRA=${EXTRA:-}
# SEASONAL mode: SNAPS=<snapshot dir> ⇒ chunk starts from different months (the diversity the field
# needs). Overrides the chained-from-S0 path. Produced by core2_nn_snapshots_1yr.sbatch.
SNAPS=${SNAPS:-}
[ -n "$SNAPS" ] && { EXTRA="$EXTRA --snapshots-dir $SNAPS"; FROM_S0=""; }

echo "=== $(hostname) $(date)  config=$CONFIG N=$N KLIST='$KLIST' iters=$ITERS from_s0='$FROM_S0' extra='$EXTRA' ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

for K in $KLIST; do
    echo ""
    echo "===== batched NN-twin config=$CONFIG N=$N K=$K  $(date) ====="
    "$PY" scripts/core2_paper_nn_twin_batched.py --config "$CONFIG" --n "$N" --windows "$K" \
        --iters "$ITERS" $FROM_S0 --results "$RESULTS" $EXTRA
    rc=$?
    echo "  (config=$CONFIG N=$N K=$K exit: $rc)"
    if [ "$rc" -eq 0 ]; then break; fi
    if [ "$rc" -ne 2 ]; then echo "  (hard-failed rc=$rc; not retrying)"; break; fi
done
echo ""
echo "=== done $(date) ==="
