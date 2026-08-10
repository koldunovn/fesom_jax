#!/bin/bash
# The adjoint-knob tests, on CPU. Fast group only — pi mesh and the closure called directly.
#
# ⚠️ Output is NOT piped. Piping pytest through `tail` buffers everything until it exits, so a
# job that hits its time limit prints NOTHING and you learn only that it was too slow. That
# cost one 30-minute job here. Stream it, and let `tail` be the reader's problem afterwards.
#SBATCH --job-name=nn_adjtests
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00
#SBATCH -o scripts/runs/adjtests.%j.out
#SBATCH -e scripts/runs/adjtests.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export JAX_PLATFORMS=cpu
echo "=== $(hostname) $(date) — adjoint-knob tests (CPU group) ==="
echo "=== checkout: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="
# See adjoint_tests_gpu.sbatch: the editable install points at the MAIN checkout, so assert
# which fesom_jax we actually got before spending an hour testing the wrong one.
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
  fesom_jax/tests/test_adjoint_stab.py \
  fesom_jax/tests/test_adjoint_config.py \
  fesom_jax/tests/test_adjoint_config_tke.py \
  fesom_jax/tests/test_gradient.py
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
