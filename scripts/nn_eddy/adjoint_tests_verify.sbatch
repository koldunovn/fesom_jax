#!/bin/bash
# Verify the 2026-08-07 adjoint-knob test repair. Runs the four touched files.
# Submit twice: as-is for CPU, and with PLATFORM=gpu (see adjoint_tests_verify_gpu.sbatch) —
# several assertions are platform-conditional (exact forward equality is a CPU property), so a
# green CPU run does NOT imply a green GPU run and vice versa.
#SBATCH --job-name=nn_verify
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/verify.%j.out
#SBATCH -e scripts/nn_eddy/verify.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date) — CPU ==="
echo "=== checkout: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null) ==="
"$PY" - <<'EOF' || exit 1
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import fesom_jax
here, got = pathlib.Path.cwd().resolve(), pathlib.Path(fesom_jax.__file__).resolve()
print(f"[import] fesom_jax <- {got}")
if here not in got.parents:
    sys.exit(f"FATAL: fesom_jax imported from {got}, not {here}")
EOF
"$PY" -m pytest -v -p no:randomly -p no:cacheprovider --durations=15 \
  fesom_jax/tests/test_adjoint_config.py \
  fesom_jax/tests/test_adjoint_config_tke.py \
  fesom_jax/tests/test_ice_adjoint_mode.py \
  fesom_jax/tests/test_mevp.py
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
