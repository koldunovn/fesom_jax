#!/bin/bash
# D1 DIAGNOSTIC — isolate the all-on (zstar+TKE+mEVP+GM) adjoint explosion seen in job 25591579
# (forward/bowl clean, but the backward gradient blew up to |g|~9e51 at N=12). Grad-check (one
# backward, no recovery) across (config, N) to answer: is it the WINDOW (shrinking N rescues the
# full all-on model) or the mEVP ice-rheology adjoint (the 120-iter pseudo-time solver)?
#   all3 N=4/8 : does a shorter window give a sane all-on gradient? (keeps the proper model)
#   tkegm N=12 : ocean-only (no ice) at the same N where all3 exploded -> is mEVP the culprit?
#   gm    N=12 : C1 baseline sanity (should be ~6.5e-7).
# Single A100-80GB; same XLA flags as the twin (C1 OOM lesson).
#SBATCH --job-name=core2_twin_diag
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_calib_twin_diag.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_calib_twin_diag.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/calib_twin_diag.jsonl

echo "=== $(hostname) $(date) — D1 adjoint-explosion diagnostic ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

for SPEC in "all3 4" "all3 8" "tkegm 12" "gm 12"; do
    set -- $SPEC; CFG=$1; N=$2
    echo ""
    echo "===== grad-check config=$CFG N=$N  $(date) ====="
    "$PY" scripts/core2_paper_calib_twin.py --config "$CFG" --n "$N" --grad-check --results "$RESULTS"
    echo "  ($CFG N=$N exit: $?)"
done
echo ""
echo "=== diagnostic done $(date) ==="
