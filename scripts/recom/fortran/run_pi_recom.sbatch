#!/bin/bash
#SBATCH --job-name=recom_fref
#SBATCH --account=ab0995
#SBATCH --partition=compute
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --output=/work/ab0995/a270088/recom_ref/logs/%x_%j.out
#
# FESOM2 + REcoM (DP) on the pi mesh, one MPI rank: the Fortran reference of the fesom-jax REcoM port.
#
# Environment (all optional except RUN):
#   RUN            run directory name under /work/ab0995/a270088/recom_ref/ (required)
#   DAYS           run length in days (default 1)
#   START_DOY      day of year of a cold start in 1958 (default 1 = 1958-01-01; 182 = 1958-07-01)
#   CONTINUE       1 = continue an existing run directory from its fesom.clock and restart files
#                  (REcoM_restart = .true.); otherwise the directory must not exist (cold start,
#                  REcoM_restart = .false., BGC from the IC files)
#   EXE            fesom.x to copy into the run directory (default $FESOM2/bin/fesom.x); its
#                  ../lib64/libfesom.so is copied along (the executable finds it via $ORIGIN/../lib64)
#   FESOM_RECOM_DUMP_DIR, FESOM_RECOM_DUMP_STEPS   passed through to the instrumented build
#                  (dump directory; comma-separated mstep list, mstep = step within this segment)
#
# Example: RUN=smoke_1d DAYS=1 sbatch run_pi_recom.sbatch
set -euo pipefail
FESOM2=${FESOM2:-/work/ab0995/a270088/fesom2_recom}
BASE=/work/ab0995/a270088/recom_ref
NML=${NML:-/work/ab0995/a270088/fesom_jax_dev/wt_recom_fref/scripts/recom/fortran/namelists}
MESH=$BASE/mesh_pi_part/          # copies of test/meshes/pi + the 1-rank partition dist_1
CLIM=$FESOM2/test/input/recom/
: "${RUN:?set RUN}"
DAYS=${DAYS:-1}
START_DOY=${START_DOY:-1}
CONTINUE=${CONTINUE:-0}
EXE=${EXE:-$FESOM2/bin/fesom.x}
RD=$BASE/$RUN

if [ "$CONTINUE" = 1 ]; then
    test -f "$RD/fesom.clock" || { echo "CONTINUE=1 but no $RD/fesom.clock"; exit 1; }
    RECOM_RESTART=.true.
    SEG=$(ls "$RD"/seg_*.info 2>/dev/null | wc -l)
else
    if [ -e "$RD" ]; then echo "refusing to reuse existing run directory $RD"; exit 1; fi
    mkdir -p "$RD"
    printf '%s %d %d\n%s %d %d\n' 0.0 "$START_DOY" 1958 0.0 "$START_DOY" 1958 > "$RD/fesom.clock"
    RECOM_RESTART=.false.
    SEG=0
fi
SEG=$((SEG + 1))
cd "$RD"
for f in namelist.config namelist.oce namelist.dyn namelist.ice namelist.tra namelist.forcing \
         namelist.io namelist.recom namelist.cvmix; do
    sed -e "s|@MESH@|$MESH|g" -e "s|@CLIM@|$CLIM|g" -e "s|@RESULT@|$RD/|g" \
        -e "s|@RUN_LENGTH@|$DAYS|g" -e "s|@RECOM_RESTART@|$RECOM_RESTART|g" \
        "$NML/$f" > "$RD/$f"
    cp -p "$RD/$f" "$RD/$f.seg$SEG"
done
mkdir -p "$RD/exe_seg$SEG/bin" "$RD/exe_seg$SEG/lib64"
cp -p "$EXE" "$RD/exe_seg$SEG/bin/fesom.x"
cp -p "$(dirname "$EXE")/../lib64/libfesom.so" "$RD/exe_seg$SEG/lib64/libfesom.so"

set +u; source "$FESOM2/env/levante.dkrz.de/shell"; set -u
{
    echo "segment       $SEG"
    echo "job           ${SLURM_JOB_ID:-none}"
    echo "date          $(date -Is)"
    echo "fesom2        $(git -C "$FESOM2" rev-parse HEAD) $(git -C "$FESOM2" status --porcelain --ignore-submodules=all -- src | wc -l) modified src files"
    echo "recom         $(git -C "$FESOM2/src/recom" rev-parse HEAD) $(git -C "$FESOM2/src/recom" status --porcelain | wc -l) modified files"
    echo "exe           $EXE md5 $(md5sum "$EXE" | cut -d' ' -f1) libfesom.so md5 $(md5sum "$RD/exe_seg$SEG/lib64/libfesom.so" | cut -d' ' -f1)"
    echo "days          $DAYS"
    echo "start_doy     $START_DOY"
    echo "clock_in      $(tr '\n' ' ' < fesom.clock)"
    echo "recom_restart $RECOM_RESTART"
    echo "dump_dir      ${FESOM_RECOM_DUMP_DIR:-unset}"
    echo "dump_steps    ${FESOM_RECOM_DUMP_STEPS:-unset}"
} | tee "seg_$SEG.info"
if [ -n "${FESOM_RECOM_DUMP_DIR:-}" ]; then mkdir -p "$FESOM_RECOM_DUMP_DIR"; fi

t0=$(date +%s)
rc=0
srun -l --cpu_bind=cores --ntasks=1 "./exe_seg$SEG/bin/fesom.x" > "fesom.seg$SEG.out" 2>&1 || rc=$?
echo "exit $rc  wall $(( $(date +%s) - t0 )) s" | tee -a "seg_$SEG.info"
exit $rc
