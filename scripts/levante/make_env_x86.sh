#!/bin/bash
# The x86 environment of fesom-jax for Levante's A100 nodes (partition gpu, 4x A100 80GB per node): on a LOGIN node
# (x86, internet), from the checkout:   bash scripts/levante/make_env_x86.sh
# A private Miniforge in $CONDA_X86 gives python 3.12; the pinned set of requirements_levante.txt is installed from
# PyPI (on x86 every package, numcodecs included, has a wheel) and the checkout editable. The GPU check runs in the
# first job (the login nodes have no GPU); here only the import and the CPU backend are checked.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/gh200.conf"
WT=$(cd "$HERE/../.." && pwd)
mkdir -p "$POOL/envs" "$RUNS/slurm"                # the slurm log directory must exist before the first sbatch
if [ ! -x "$CONDA_X86/bin/conda" ]; then
    curl -fsSL -o "$POOL/envs/Miniforge3-x86_64.sh" \
        https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh
    bash "$POOL/envs/Miniforge3-x86_64.sh" -b -p "$CONDA_X86"
fi
export PIP_CACHE_DIR=$POOL/envs/pip-cache CONDA_PKGS_DIRS=$POOL/envs/conda-pkgs
[ -x "$VENV_X86/bin/python" ] || "$CONDA_X86/bin/conda" create -y -p "$VENV_X86" python=3.12 pip
"$VENV_X86/bin/python" -m pip install --upgrade pip
"$VENV_X86/bin/python" -m pip install -r "$HERE/requirements_levante.txt"
"$VENV_X86/bin/python" -m pip install --no-deps --no-build-isolation -e "$WT"
"$VENV_X86/bin/python" -m pip freeze > "$VENV_X86/freeze.txt"
JAX_PLATFORMS=cpu taskset -c 0-7 "$VENV_X86/bin/python" -c "import fesom_jax, jax; print('fesom_jax', fesom_jax.__file__); print('jax', jax.__version__, jax.devices())"
echo "env ready: $VENV_X86 (python $("$VENV_X86/bin/python" -c 'import sys; print(sys.version.split()[0])'))"
