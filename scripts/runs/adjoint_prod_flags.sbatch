#!/bin/bash
# The three AdjointConfig flags that only exist under zstar+GM (test_adjoint_config_prod.py).
# Its own job because the fixture is the most expensive in the adjoint set: one assembled CORE2
# step with zstar + TKE + mEVP + GM, and a separate jax.jit closure per configuration.
#SBATCH --job-name=nn_prodflags
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH -o scripts/runs/prodflags.%j.out
#SBATCH -e scripts/runs/prodflags.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
echo "=== $(hostname) $(date) — CPU ==="
echo "=== checkout: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null) ==="
"$PY" - <<'PYEOF' || exit 1
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import fesom_jax
here, got = pathlib.Path.cwd().resolve(), pathlib.Path(fesom_jax.__file__).resolve()
print(f"[import] fesom_jax <- {got}")
if here not in got.parents:
    sys.exit(f"FATAL: fesom_jax imported from {got}, not {here}")
PYEOF
"$PY" -m pytest -v -s -p no:randomly -p no:cacheprovider --durations=15 \
  fesom_jax/tests/test_adjoint_config_prod.py
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
