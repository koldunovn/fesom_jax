#!/bin/bash
# Coarse-grain a month of NG5 SSH snapshots onto CORE2 — the eta SIDE-product (M2).
#
# Cheap: one 2-D field, no level loop, no vertical remap — seconds of compute against a ~180 MB
# read, versus ~35 min for a 3-D month. Accepts several months in one job for that reason:
#   sbatch scripts/nn_eddy/coarsen_eta.sbatch 1958_01 1958_02 ...
#   sbatch scripts/nn_eddy/coarsen_eta.sbatch --all-1958
# The script REFUSES to overwrite an existing product and never touches the 3-D files.
#SBATCH --job-name=nn_coarsen_eta
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o scripts/nn_eddy/coarseneta.%j.out
#SBATCH -e scripts/nn_eddy/coarseneta.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date)  commit $(git rev-parse --short HEAD) ==="

MONTHS="$*"
if [ "$1" = "--all-1958" ]; then
    MONTHS="1958_01 1958_02 1958_03 1958_04 1958_05 1958_06 1958_07 1958_08 1958_09 1958_10 1958_11 1958_12"
fi
if [ -z "$MONTHS" ]; then echo "usage: sbatch $0 <month> [month ...] | --all-1958"; exit 2; fi

rc=0
for m in $MONTHS; do
    echo ""
    echo "=== month $m ==="
    "$PY" scripts/nn_eddy/coarsen_eta.py --month "$m" || rc=$?
done
echo ""
echo "=== done $(date) rc=$rc ==="
[ $rc -eq 0 ] && echo "ETA PRODUCTS OK" || echo "ETA PRODUCT FAILURE — see above"
exit $rc
