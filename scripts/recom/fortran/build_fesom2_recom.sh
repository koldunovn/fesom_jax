#!/usr/bin/env bash
# Build the double-precision FESOM2 + REcoM reference used by the fesom-jax REcoM gates.
#
# Clone (once):
#   git clone https://github.com/FESOM/fesom2 /work/ab0995/a270088/fesom2_recom
#   cd /work/ab0995/a270088/fesom2_recom
#   git checkout -b recom-ref-dumps e96a2b13
#   git submodule update --init src/recom        # -> 692f8045e0d5bd296d9c0e8bd9ff4a2b5872f79d
#
# Build: FESOM2's own configure.sh (Intel oneAPI 2022.0.1 + OpenMPI 4.1.2, env/levante.dkrz.de/shell,
# CMAKE_BUILD_TYPE=Release, the default flags of src/CMakeLists.txt), DP (USE_SINGLE_PRECISION off,
# the default), RECOM_COUPLED=ON. The executable is installed to $FESOM2/bin/fesom.x.
set -euo pipefail
FESOM2=${FESOM2:-/work/ab0995/a270088/fesom2_recom}
JOBS=${JOBS:-16}
cd "$FESOM2"
echo "fesom2 $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "recom  $(git -C src/recom rev-parse HEAD)"
test "$(git -C src/recom rev-parse HEAD)" = 692f8045e0d5bd296d9c0e8bd9ff4a2b5872f79d
# configure.sh calls `make install -j$(nproc --all)`; cap the parallelism on shared login nodes.
nproc() { echo "$JOBS"; }
export -f nproc
BUILD_DIR=build_recom_dp ./configure.sh levante -DRECOM_COUPLED=ON -DUSE_SINGLE_PRECISION=OFF
ls -l "$FESOM2/bin/fesom.x"
