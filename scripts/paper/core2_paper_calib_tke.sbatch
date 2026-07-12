#!/bin/bash
# Paper-experiments Task D2a — CORE2 TKE->MLD perfect-model twin on a single A100-80GB.
# Recovers an injected tke_c_k from the model's own MLD field, on the FULL paper model
# (zstar+TKE+mEVP+GM) with the FROZEN-ICE adjoint (D1/A8: full mEVP forward, backward skips the
# unstable ice rheology). The §2 mixing-calibration proof; MLD is the fast/adjoint-reachable target
# (C1: d(mean MLD)/d(c_k)=+2.91). Single-GPU only; descend N on OOM (a twin is well-posed at any N).
# Same A100-80 / XLA flags as the D1 twin (the C1 OOM + command-buffer lessons).
#SBATCH --job-name=core2_paper_tke
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_calib_tke.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_calib_tke.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/calib_tke_results.jsonl
CONFIG=${1:-all3}
NLIST=${2:-"12 8 6 4"}

echo "=== $(hostname) $(date)  config=$CONFIG  NLIST='$NLIST' ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

for N in $NLIST; do
    echo ""
    echo "===== TKE->MLD twin config=$CONFIG N=$N  $(date) ====="
    "$PY" scripts/paper/core2_paper_calib_tke.py --config "$CONFIG" --n "$N" --results "$RESULTS"
    rc=$?
    echo "  (config=$CONFIG N=$N exit: $rc)"
    if [ "$rc" -eq 0 ]; then break; fi
    if [ "$rc" -ne 2 ]; then echo "  (hard-failed rc=$rc; not retrying)"; break; fi
done
echo ""
echo "=== done $(date) ==="
