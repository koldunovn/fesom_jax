#!/bin/bash
# Paper-experiments Task C1 — CORE2 instantaneous adjoint sensitivity maps on a single A100-80GB.
# Runs the two targets as SEPARATE processes (clean per-target peak memory, like the A7 sweep):
#   mld_ck : zstar+TKE  → d(mean MLD)/d(c_k field)
#   ts_kgm : zstar+GM   → d(upper-ocean T)/d(k_gm field) + the adjoint↔EKI cross-check
# Each appends a JSON line to the results file + writes sensitivity_<target>.npz; fig_sensitivity.py
# then draws Fig 2 and emits SENSITIVITY_MAP_OK. Single-GPU only (the sharded ragged-halo AD bug is
# forward-safe only). Window N=20 = A7's measured N_max (~10 h; the fast/instantaneous sensitivity).
# ⚠️ Levante's gpu partition mixes a100_40 + a100_80 — request the 80GB card explicitly (A7 lesson);
# PREALLOCATE=false avoids the CUBIN-load OOM and MEM_FRACTION=0.95 gives the full card.
#SBATCH --job-name=core2_paper_sens
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_sensitivity.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_sensitivity.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
RESULTS=scripts/sensitivity_results.jsonl
N=${1:-20}

echo "=== $(hostname) $(date)  N=$N ==="
nvidia-smi -L 2>/dev/null || true

# fresh results file for this run
: > "$RESULTS"

for TGT in mld_ck ts_kgm; do
    echo ""
    echo "===== target=$TGT  $(date) ====="
    "$PY" scripts/core2_paper_sensitivity.py --target "$TGT" --n "$N" --results "$RESULTS"
    echo "  ($TGT exit: $?)"
done

echo ""
echo "===== building Fig 2 + decision ====="
"$PY" scripts/fig_sensitivity.py --results "$RESULTS" --out scripts/fig_sensitivity.png
echo "exit: $?"
