#!/bin/bash
# Build the standalone mocsy driver (gate G1, docs/plans/20261025-recom-lane-mocsy.md M2)
# from the REcoM 692f804 mocsy sources, with REcoM's own Intel flags
# (REcoM_src/CMakeLists.txt:31-37, the classic-ifort branch incl. -no-prec-sqrt -ip;
# USE_PRECISION stays undefined, as on the recom target).
#
# Usage: build_mocsy_driver.sh [build_dir]   (default /work/ab0995/a270088/recom_ref/mocsy_g1/build)
# Runs on the login node in ~1 min. Never deletes anything: a rebuild overwrites the objects.
set -euo pipefail

REC=${RECOM_SRC:-/work/ab0995/a270088/REcoM_src}
ENV_SHELL=/work/ab0995/a270088/fesom2_sp_zstar/env/levante.dkrz.de/shell
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD=${1:-/work/ab0995/a270088/recom_ref/mocsy_g1/build}

set +u
# shellcheck disable=SC1090
source "$ENV_SHELL" >/dev/null 2>&1
set -u

FLAGS="-O3 -r8 -i4 -fp-model precise -no-prec-div -fimf-use-svml -init=zero -no-wrap-margin -fpe0 -fpp"
FLAGS="$FLAGS -march=core-avx2 -fPIC -qopt-malloc-options=2 -qopt-prefetch=5 -unroll-aggressive"
FLAGS="$FLAGS -no-prec-sqrt -ip"

mkdir -p "$BUILD"
cd "$BUILD"
echo "REcoM_src: $(git -C "$REC" rev-parse HEAD)" > build_info.txt
echo "compiler:  $(mpif90 --version 2>&1 | head -1)" >> build_info.txt
echo "flags:     $FLAGS" >> build_info.txt

M=$REC/mocsy
# dependency order (modules first); the gsw_* bodies are needed by meos/gsw_mod_toolbox at link
SRCS="singledouble DNAD gsw_mod_kinds gsw_mod_teos10_constants gsw_mod_toolbox"
SRCS="$SRCS gsw_mod_baltic_data gsw_mod_error_functions gsw_mod_saar_data gsw_mod_specvol_coefficients"
for f in "$M"/gsw_*.F90; do
    b=$(basename "$f" .F90)
    case " $SRCS " in *" $b "*) ;; *) SRCS="$SRCS $b" ;; esac
done
SRCS="$SRCS sw_adtg sw_ptmp sw_temp p80 rho eos phsolvers varsolver constants vars p2fCO2 gasx"

OBJS=""
for b in $SRCS; do
    mpif90 $FLAGS -I"$M" -c "$M/$b.F90" -o "$b.o"
    OBJS="$OBJS $b.o"
done
mpif90 $FLAGS -c "$HERE/mocsy_driver.F90" -o mocsy_driver.o
mpif90 $FLAGS -o mocsy_driver.x mocsy_driver.o $OBJS
echo "built $BUILD/mocsy_driver.x"
