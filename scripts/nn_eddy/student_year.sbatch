#!/bin/bash
# NN-GM revised Stage-1 feasibility — student free year 1958 + κ-branch runs (one A100).
# Phase 1 (--year, ~2.5 h): 365 d forward from the PHC cold start, 5-day (T,S,eta) marks +
# 4 branch-State pickles. Skipped if marks_year.npz already exists (never overwritten).
# Phase 2 (--branches, ~1 h): 4 branch days x m in {1/3,1,3} x 10 d; per-file skip makes a
# re-run resume. Submit from the port_jax worktree root:
#   sbatch scripts/nn_eddy/student_year.sbatch
#SBATCH --job-name=nn_student_yr
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o scripts/nn_eddy/studentyr.%j.out
#SBATCH -e scripts/nn_eddy/studentyr.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
OUT=${OUT:-/work/ab0995/a270088/nn_eddy/student_free_1958}
echo "=== $(hostname) $(date)  commit $(git rev-parse --short HEAD)  OUT=$OUT ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
if [ ! -f "$OUT/marks_year.npz" ]; then
  "$PY" scripts/nn_eddy/student_year.py --year --out "$OUT" || exit $?
else
  echo "=== marks_year.npz exists — year phase skipped ==="
fi
"$PY" scripts/nn_eddy/student_year.py --branches --out "$OUT"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
