#!/bin/bash
# Coarse-grain a month of NG5 teacher snapshots onto CORE2 (Task 6 pipeline).
#
# CPU, one node. This is I/O and scatter bound, not flop bound: per fine level it reads ~180 MB
# of deflated netCDF and does ~130 M scatter-adds. A GPU would spend its time waiting for the
# filesystem, and the 16 GB mesh export plus the level buffers want host memory, not device.
#
# Args are passed straight through:
#   sbatch scripts/nn_eddy/coarsen_stream.sbatch --month 1958_01 --records 0 --max-levels 5
#   sbatch scripts/nn_eddy/coarsen_stream.sbatch --month 1958_01
#SBATCH --job-name=nn_coarsen
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH -o scripts/nn_eddy/coarsen.%j.out
#SBATCH -e scripts/nn_eddy/coarsen.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
echo "=== $(hostname) $(date) ==="
echo "=== checkout: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "=== args: $* ==="
"$PY" scripts/nn_eddy/coarsen_stream.py "$@"
rc=$?          # NB: read rc BEFORE anything else runs — $(date) would overwrite $?
echo "=== done $(date) rc=$rc ==="
exit $rc
