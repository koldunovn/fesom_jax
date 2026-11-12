#!/bin/bash
# Step 1 of the GH200 environment, on a Levante LOGIN node (x86, has internet; the dolpung nodes are aarch64 and
# may not): download what requirements_levante.txt needs for aarch64 / CPython 3.12 into $WHEELS.
#   bash scripts/levante/fetch_wheels.sh          (from the checkout; any python3 with pip >= 21 and curl)
# Then: sbatch scripts/levante/make_env_gh200.sbatch
# zarr 2 needs numcodecs < 0.16, which publishes no aarch64 wheels: numcodecs (and the pure-python asciitree) come as
# source from PyPI and are built on the node, so their build tools are fetched as aarch64 wheels too.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/gh200.conf"
mkdir -p "$WHEELS" "$RUNS/slurm"                 # the slurm log directory must exist before the first sbatch
PYDL=${PYDL:-python3}
"$PYDL" -m pip --version
PLAT=(--only-binary=:all: --implementation cp --python-version 3.12 --abi cp312 --abi abi3 --abi none
      --platform manylinux_2_28_aarch64 --platform manylinux_2_27_aarch64 --platform manylinux_2_17_aarch64
      --platform manylinux2014_aarch64 --platform any)
SRC="numcodecs asciitree"                        # source packages (sdists below)
grep -v -E "^(zarr|numcodecs|asciitree)==" "$HERE/requirements_levante.txt" > "$WHEELS/binary_requirements.txt"
# 1. the binary set with all its dependencies (jax[cuda12] brings the nvidia-* CUDA 12 runtime wheels)
"$PYDL" -m pip download -d "$WHEELS" -r "$WHEELS/binary_requirements.txt" "${PLAT[@]}"
# 2. zarr itself (pure python) without its numcodecs < 0.16 dependency, which no aarch64 wheel satisfies
"$PYDL" -m pip download -d "$WHEELS" --no-deps "${PLAT[@]}" "$(grep -E '^zarr==' "$HERE/requirements_levante.txt" | cut -d' ' -f1)"
# 3. the build tools of numcodecs (its pyproject [build-system]) and of asciitree
"$PYDL" -m pip download -d "$WHEELS" "${PLAT[@]}" "setuptools>=64" "setuptools-scm[toml]>=6.2" Cython py-cpuinfo \
    "numpy==$(grep -E '^numpy==' "$HERE/requirements_levante.txt" | cut -d= -f3 | cut -d' ' -f1)" wheel
# 4. the sdists, straight from PyPI (pip download would try to build their metadata here)
for pkg in $SRC; do
    ver=$(grep -E "^$pkg==" "$HERE/requirements_levante.txt" | cut -d= -f3 | cut -d' ' -f1)
    url=$(curl -fsS "https://pypi.org/pypi/$pkg/$ver/json" | "$PYDL" -c \
          "import json,sys; print([f['url'] for f in json.load(sys.stdin)['urls'] if f['packagetype']=='sdist'][0])")
    (cd "$WHEELS" && curl -fsSLO "$url")
    echo "sdist $pkg $ver"
done
echo "wheels in $WHEELS: $(ls "$WHEELS" | wc -l) files, $(du -sh "$WHEELS" | cut -f1)"
