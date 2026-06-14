#!/bin/bash
# Paper-experiments Task D1 — CORE2 perfect-model k_gm twin on a single A100-80GB.
# Default config = the FULL paper model (zstar + TKE + mEVP + GM/Redi, all live; user steer
# 2026-06-14 "proper ocean model, not a toy"). Injects truth k_gm=1500, grid-scans the misfit
# bowl, then recovers k_gm 800->1500 with cosine-decay Adam through the global adjoint
# (calibrate.optimize) -> TWIN_RECOVER_OK. Single-GPU only (the sharded ragged-halo AD bug is
# forward-safe only). The all-on backward is heavy, so descend N on OOM (the script exits 2 on OOM);
# a perfect-model twin stays well-posed at any N (J=0 at the truth).
# ⚠️ Levante's gpu partition mixes a100_40 + a100_80 — request the 80GB card explicitly (A7 lesson);
# PREALLOCATE=false avoids the CUBIN-load OOM and MEM_FRACTION=0.95 gives the full card; the
# command-buffer disable is the C1 "38 alive graphs" OOM fix (loss + grad executables).
#SBATCH --job-name=core2_paper_twin
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_calib_twin.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_calib_twin.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/calib_results.jsonl
CONFIG=${1:-all3}
NLIST=${2:-"12 8 6 4"}        # all-on backward is heavy; descend until it fits 80 GB

echo "=== $(hostname) $(date)  config=$CONFIG  NLIST='$NLIST' ==="
nvidia-smi -L 2>/dev/null || true
: > "$RESULTS"

for N in $NLIST; do
    echo ""
    echo "===== twin config=$CONFIG N=$N  $(date) ====="
    "$PY" scripts/core2_paper_calib_twin.py --config "$CONFIG" --n "$N" --results "$RESULTS"
    rc=$?
    echo "  (config=$CONFIG N=$N exit: $rc)"
    # 0 = success (twin recovered) → done; 2 = OOM → try a smaller N; else hard fail → stop.
    if [ "$rc" -eq 0 ]; then break; fi
    if [ "$rc" -ne 2 ]; then echo "  (hard-failed rc=$rc; not retrying)"; break; fi
done
echo ""
echo "=== done $(date) ==="
