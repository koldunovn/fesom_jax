#!/bin/bash
# One FESOM2 CORE2 run (cold start) in RUNDIR with the clean upstream build (fesom2_icefw_base, 4413769a) — the
# Fortran reference of the partial-cells validation (docs/plans/20261125-fesom-jax-partial-cells.md, Tasks 1b/8).
# No resubmission, no cleanup.   sbatch --export=ALL,RUNDIR=<dir> [--time=..] fortran_run.sbatch
#SBATCH --job-name=pc_fortran
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=128
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/partial_cells/fortran/logs/run.%j.out
BIN=${BIN:-/work/ab0995/a270088/fesom2_icefw_runs/build_base/bin/fesom.x}
source /sw/etc/profile.levante 2>/dev/null
source /work/ab0995/a270088/fesom2_icefw/env/levante.dkrz.de/shell
ulimit -s unlimited
cd "$RUNDIR" || exit 2
echo "=== $(hostname) $(date) RUNDIR=$RUNDIR BIN=$BIN clock: $(tail -1 fesom.clock) ==="
T0=$(date +%s)
srun -l "$BIN" > fesom.out 2>&1
RC=$?
echo "fesom rc=$RC wall=$(( $(date +%s) - T0 )) s; clock now: $(tail -1 fesom.clock)"
grep -iE "nan|error|abort" fesom.out | head -5
