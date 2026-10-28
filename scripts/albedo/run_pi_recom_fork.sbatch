#!/bin/bash
#SBATCH --job-name=recom_fref
#SBATCH -A clidyn.clidyn
#SBATCH -p mpp
#SBATCH --qos=30min
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=60G
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x_%j.out
#
# Albedo: the FESOM2-REcoM FORK (spectral light, 4 PFT + CDOM) on the pi mesh, one MPI rank, double precision:
# the Fortran reference of the fesom-jax spectral REcoM port (docs/plans/20261028-recom-spectral-port.md, lane 0).
# The Albedo counterpart of scripts/recom/fortran/run_pi_recom.sbatch (Levante).
#
#   RUN=<name> DAYS=<n> [START_DOY=1] [CONTINUE=0] [EXE=...] [NML=...] \
#   [FESOM_RECOM_DUMP_DIR=<dir> FESOM_RECOM_DUMP_STEPS=1,2,17] sbatch scripts/albedo/run_pi_recom_fork.sbatch
set -euo pipefail
FESOM2=${FESOM2:-/albedo/work/user/sloza/SPECTRAL/recom_spectral_r692}
DATA=/albedo/work/projects/p_specbic/sloza/fesom_jax_data
BASE=/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/recom_ref
NML=${NML:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom/scripts/recom/fortran/namelists_albedo_spectral}
MESH=$DATA/mesh_pi_part/            # the fork's test/meshes/pi + the 1-rank partition dist_1 (fesom_ini, npes=1 allowed)
CLIM=$DATA/recom_pi/                # FESOM2 test/input/recom (5-degree IC files, Albani dust on pi) + optics/
: "${RUN:?set RUN}"
DAYS=${DAYS:-1}
START_DOY=${START_DOY:-1}
CONTINUE=${CONTINUE:-0}
EXE=${EXE:-$FESOM2/bin/fesom.x}
RD=$BASE/$RUN
mkdir -p $BASE
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
for f in namelist.config namelist.oce namelist.dyn namelist.ice namelist.icepack namelist.tra namelist.forcing \
         namelist.io namelist.recom; do
    sed -e "s|@MESH@|$MESH|g" -e "s|@CLIM@|$CLIM|g" -e "s|@RESULT@|$RD/|g" \
        -e "s|@RUN_LENGTH@|$DAYS|g" -e "s|@RECOM_RESTART@|$RECOM_RESTART|g" \
        "$NML/$f" > "$RD/$f"
    cp -p "$RD/$f" "$RD/$f.seg$SEG"
done
cp -p $CLIM/optics/*.dat "$RD/"                      # the spectral tables are read as relative paths
mkdir -p "$RD/exe_seg$SEG/bin" "$RD/exe_seg$SEG/lib64"
cp -p "$EXE" "$RD/exe_seg$SEG/bin/fesom.x"
cp -p "$(dirname "$EXE")/../lib64/libfesom.so" "$RD/exe_seg$SEG/lib64/libfesom.so"
set +u; source "$FESOM2/env.sh"; set -u
export LD_LIBRARY_PATH=$RD/exe_seg$SEG/lib64:${LD_LIBRARY_PATH:-}
{
    echo "segment       $SEG"
    echo "job           ${SLURM_JOB_ID:-none}"
    echo "date          $(date -Is)"
    echo "fesom2        $(git -C "$FESOM2" rev-parse HEAD) $(git -C "$FESOM2" status --porcelain -- src | wc -l) modified src files ($(git -C "$FESOM2" branch --show-current))"
    echo "exe           $EXE md5 $(md5sum "$EXE" | cut -d' ' -f1) libfesom.so md5 $(md5sum "$RD/exe_seg$SEG/lib64/libfesom.so" | cut -d' ' -f1)"
    echo "days          $DAYS"
    echo "start_doy     $START_DOY"
    echo "clock_in      $(tr '\n' ' ' < fesom.clock)"
    echo "recom_restart $RECOM_RESTART"
    echo "dump_dir      ${FESOM_RECOM_DUMP_DIR:-unset}"
    echo "dump_steps    ${FESOM_RECOM_DUMP_STEPS:-unset}"
} | tee "seg_$SEG.info"
if [ -n "${FESOM_RECOM_DUMP_DIR:-}" ]; then mkdir -p "$FESOM_RECOM_DUMP_DIR"; fi
export OMP_NUM_THREADS=1
ulimit -s unlimited
date
srun --mpi=pmi2 "$RD/exe_seg$SEG/bin/fesom.x" > "fesom2.0.out.seg$SEG" 2>&1
rc=$?
date
tail -5 "fesom2.0.out.seg$SEG"
echo "=== fesom exit $rc ==="
exit $rc
