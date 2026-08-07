#!/bin/bash
# The heavy group: the CORE2 step gates the adjoint knobs must not move. GPU, because these
# build and compile a full CORE2 step many times over and CPU compiles are minutes each.
#SBATCH --job-name=nn_adjtests_gpu
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -o scripts/nn_eddy/adjtests_gpu.%j.out
#SBATCH -e scripts/nn_eddy/adjtests_gpu.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
echo "=== $(hostname) $(date) — CORE2 step gates + the TKE-path knob tests (GPU) ==="
echo "=== checkout: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="
# The editable install resolves fesom_jax to the MAIN checkout ~/port_jax; the worktree only
# wins because pytest prepends the rootdir. Assert it rather than trust it — an unnoticed
# main-checkout import silently invalidated an experiment on 2026-08-07.
"$PY" - <<'EOF' || exit 1
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import fesom_jax
here, got = pathlib.Path.cwd().resolve(), pathlib.Path(fesom_jax.__file__).resolve()
print(f"[import] fesom_jax <- {got}")
if here not in got.parents:
    sys.exit(f"FATAL: fesom_jax imported from {got}, not from the checkout under test {here}")
EOF
"$PY" -m pytest -v -p no:randomly \
  fesom_jax/tests/test_adjoint_config_tke.py \
  fesom_jax/tests/test_ice_adjoint_mode.py \
  fesom_jax/tests/test_core2_step.py \
  fesom_jax/tests/test_gm_step.py \
  fesom_jax/tests/test_tke_step.py \
  fesom_jax/tests/test_ice_step.py
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
