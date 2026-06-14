#!/bin/bash
# Paper-experiments Task A7 — CORE2 adjoint-window de-risking sweep on a single A100-80GB.
# Loops N = 4, 20, 50, 100, 200 as SEPARATE processes (peak GPU memory is a process-global
# high-water mark — a fresh process per N is the only clean per-N measurement). Each run appends
# a JSON line to the results file; scripts/fig_window_snr.py then finds N_max + the adjoint↔EKI
# boundary and emits WINDOW_DERISK_OK. Single-GPU (the sharded ragged-halo AD bug is forward-safe
# only). Config: zstar+TKE, GM/KPP/ice off (the TKE→MLD fast-target signal; lower-bounds memory).
# ⚠️ Levante's gpu partition has BOTH a100_80 (80GB) and a100_40 (40GB) nodes; --gres=gpu:1 can land
# on a 40GB card (it did — l40369), and the assembled CORE2 backward OOMs on CUBIN-load there. Request
# the 80GB variant explicitly (the guardrail), and disable XLA preallocation (the 75% prealloc leaves
# too little for module loading on the big executable → "Failed to load in-memory CUBIN ... OOM").
#SBATCH --job-name=core2_adj_window
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_adjoint_window_sweep.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_adjoint_window_sweep.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # allocate on demand (avoids OOM-during-module-load)
RESULTS=scripts/adjoint_window_sweep_results.jsonl

echo "=== $(hostname) $(date) ==="
nvidia-smi -L 2>/dev/null || true

# fresh results file for this sweep
: > "$RESULTS"

# ascending N so the cheap windows complete first (durable partial results if a big N OOMs/times out)
for N in 4 20 50 100 200; do
    echo ""
    echo "===== N=$N  $(date) ====="
    "$PY" scripts/core2_adjoint_window_sweep.py --n "$N" --results "$RESULTS"
    echo "  (N=$N exit: $?)"
done

echo ""
echo "===== building figure + decision ====="
"$PY" scripts/fig_window_snr.py --results "$RESULTS" --out scripts/fig_window_snr.png
echo "exit: $?"
