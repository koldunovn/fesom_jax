#!/bin/bash
# CPU pytest job for the TKE-calibration work (controls, host-side L-BFGS, calibration cost).
#
# Usage, from the checkout under test:
#   sbatch --parsable scripts/calib/job_cpu.sbatch <pytest args...>
#   sbatch --parsable --mem=200G --time=04:00:00 scripts/calib/job_cpu.sbatch <core2 test files>
#
# Guards (each one has cost a real job before):
#   - PYTHONPATH is the checkout and the imported fesom_jax is asserted to come from it: the
#     editable install points at a different working tree, and a job that silently tests that
#     tree reports a pass for code it never ran.
#   - TMPDIR and CUDA_CACHE_PATH live on /scratch (the home quota is nearly full).
#   - No persistent XLA compilation cache: JAX_COMPILATION_CACHE_DIR must be unset.
#   - Output streams to the log (no pipe through tail: a job killed at its time limit would
#     otherwise print nothing).
#   - pytest runs with -rs so every skip is listed with its reason; a skip is not a pass.
#SBATCH --job-name=calib_cpu
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/calib/logs/%x.%j.out
#SBATCH -e /work/ab0995/a270088/fesom_jax_dev/runs/calib/logs/%x.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
WT="${SLURM_SUBMIT_DIR:?submit from the checkout under test}"
cd "$WT" || exit 1
export PYTHONPATH="$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=/scratch/a/a270088/fesom_jax_dev_tmp/nv
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then
  echo "FATAL: JAX_COMPILATION_CACHE_DIR is set ($JAX_COMPILATION_CACHE_DIR); no persistent XLA cache" >&2
  exit 1
fi
echo "=== $(hostname) $(date) job=${SLURM_JOB_ID} ==="
echo "=== checkout: $WT  commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD) ==="
echo "=== dirty files: $(git status --porcelain --untracked-files=no | wc -l) ==="
"$PY" - <<'EOF' || exit 1
import os, pathlib
import fesom_jax
wt = pathlib.Path(os.environ["PYTHONPATH"]).resolve()
got = pathlib.Path(fesom_jax.__file__).resolve()
print(f"[import] fesom_jax <- {got}")
if wt not in got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {got}, not from the checkout {wt}")
if not (wt / "data").exists():
    raise SystemExit(f"FATAL: {wt}/data missing; every CORE2 test would SKIP")
EOF
"$PY" -m pytest -v -rs -p no:randomly "$@"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
