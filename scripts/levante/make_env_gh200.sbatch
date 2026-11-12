#!/bin/bash
# Step 2 of the GH200 environment: build the aarch64 venv $VENV on a dolpung node from the wheels of fetch_wheels.sh
# (offline), install this checkout editable, and check that JAX sees the four GH200s.
#   sbatch scripts/levante/make_env_gh200.sbatch            (from the checkout $WT, after fetch_wheels.sh)
# The base interpreter is the ARM spack tree's python 3.12 (as the dolpung env of scripts/perf/env_dolpung.sh);
# PYARM=/path/to/python3.12 overrides the search.
#SBATCH --job-name=fj_mkenv_gh200
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH -o /work/bk1415/a270044/fesom_jax_pool/runs/slurm/%x.%j.out
set -eu
WT=${WT:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$WT/scripts/levante/env_gh200.sh"
echo "=== $(hostname) $(uname -m) $(date) WT=$WT VENV=$VENV ==="
[ "$(uname -m)" = aarch64 ] || { echo "FATAL: not an aarch64 node"; exit 2; }
PYARM=${PYARM:-$(ls -d /sw/spack-levante*/linux-rhel9-neoverse_v2*/python-3.12*/bin/python3.12 2>/dev/null | sort -V | tail -1)}
[ -x "$PYARM" ] || { echo "FATAL: no ARM python 3.12 found; set PYARM"; ls -d /sw/spack-levante*/linux-rhel9-neoverse_v2*/python-3* 2>/dev/null; exit 2; }
echo "base interpreter: $PYARM ($($PYARM --version))"
# numcodecs 0.15.2 is built from source here (zarr 2 needs it; no aarch64 wheel): a C compiler is needed
export CC=${CC:-$(command -v gcc || command -v cc || ls -d /sw/spack-levante*/linux-rhel9-neoverse_v2*/gcc-*/bin/gcc 2>/dev/null | sort -V | tail -1)}
[ -x "$CC" ] || { echo "FATAL: no C compiler for numcodecs; set CC"; exit 2; }
echo "C compiler: $CC ($($CC --version | head -1))"
rm -rf "$VENV"
"$PYARM" -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-index --find-links "$WHEELS" --upgrade pip setuptools wheel || true
"$VENV/bin/python" -m pip install --no-index --find-links "$WHEELS" -r "$WT/scripts/levante/requirements_gh200.txt"
"$VENV/bin/python" -m pip install --no-index --no-deps --no-build-isolation -e "$WT"
"$VENV/bin/python" -m pip freeze > "$VENV/freeze.txt"
"$VENV/bin/python" - <<'EOF'
import fesom_jax, jax, jax.numpy as jnp
print("fesom_jax", fesom_jax.__file__)
print("jax", jax.__version__, "devices", jax.devices())
x = jnp.arange(1e6)
print("check", float(jax.jit(lambda a: (a * a).sum())(x)), "float64:", x.dtype)
assert len(jax.devices("gpu")) == 4, "expected four GH200s"
EOF
echo "=== exit $? $(date) ==="
