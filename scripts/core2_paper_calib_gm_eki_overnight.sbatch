#!/bin/bash
# Paper-experiments Task D2b — the full overnight D2b sequence on one A100-80GB:
#   (1) budget  — time the all-on forward, project the production GPU-hours (the A4/D2b budget);
#   (2) twin    — perfect-model EKI proof: recover a planted k_gm (GM_EKI_TWIN_OK);
#   (3) obs      — scoped EKI demonstration to WOA basin-mean T/S (GM_CALIB demonstration).
# Each stage runs in its own process (fresh allocator pool) and the script continues on failure,
# so a partial result (budget + twin) survives even if the obs stage runs long. Forward-only ⇒
# the full all-on model with live mEVP ice (no adjoint ⇒ immune to the A8 ice-rheology instability).
#SBATCH --job-name=core2_gm_eki_night
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_calib_gm_eki_night.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_calib_gm_eki_night.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
RESULTS=scripts/calib_gm_eki_results.jsonl

# window/ensemble sizing — set from the smoke timing (placeholders; safe conservative defaults)
TWIN_N=${TWIN_N:-240}
OBS_N=${OBS_N:-480}
NSPIN=${NSPIN:-480}
MEMBERS=${MEMBERS:-10}     # 1-D k_gm ⇒ a modest ensemble suffices (smoke: argmin exact at n=96)
ITERS=${ITERS:-5}          # EKI on a 1-D param converges in ~4-5 iters

echo "=== $(hostname) $(date)  TWIN_N=$TWIN_N OBS_N=$OBS_N NSPIN=$NSPIN MEMBERS=$MEMBERS ITERS=$ITERS ==="
nvidia-smi -L 2>/dev/null || true

echo ""; echo "===== (1) budget  $(date) ====="
"$PY" scripts/core2_paper_calib_gm_eki.py --mode budget --n "$TWIN_N" --n-spin 96 \
    --budget-members "$MEMBERS" --budget-iters "$ITERS" --results "$RESULTS"
echo "  (budget exit: $?)"

echo ""; echo "===== (2) twin  $(date) ====="
"$PY" scripts/core2_paper_calib_gm_eki.py --mode twin --n "$TWIN_N" --n-spin "$NSPIN" \
    --members "$MEMBERS" --iters 5 --gamma-auto --results "$RESULTS"
echo "  (twin exit: $?)"

echo ""; echo "===== (3) obs (scoped demonstration)  $(date) ====="
"$PY" scripts/core2_paper_calib_gm_eki.py --mode obs --n "$OBS_N" --n-spin "$NSPIN" \
    --members "$MEMBERS" --iters "$ITERS" --sigma-t 0.5 --sigma-s 0.1 --perturb-obs --results "$RESULTS"
echo "  (obs exit: $?)"

echo ""; echo "=== done $(date) ==="
