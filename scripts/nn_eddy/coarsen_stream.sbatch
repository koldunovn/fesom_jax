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

# Check every product this job wrote, in the same job. On 2026-08-08 a month completed with
# FESOM's fill value in 12.5% of its column and reported sensible surface statistics; it was
# caught by reading the file afterwards, which is a thing that happens once. Running the checker
# here makes a contaminated product a FAILED JOB instead of a file somebody trusts. Skipped for
# --verify-conservation and --max-levels runs, which write nothing usable.
case " $* " in
  *" --verify-conservation "*|*" --max-levels "*) ;;
  *)
    if [ "$rc" -eq 0 ]; then
        # ⚠️ ASK THE CODE where it writes; never hardcode it here. This line used to read
        # /work/ab0995/a270088/nn_eddy/coarse. After the data root moved to ba1583 (behind
        # nn_data.DATA_ROOT) the products went to the new path while the checker kept looking
        # at the old one, found nothing, printed "WARNING: ... nothing checked" and exited 0 —
        # so the whole point of running the checker in-job, that a contaminated product is a
        # FAILED JOB rather than a file somebody trusts, was silently off. Jobs
        # 26807590-26807598 ran with that hole and were checked by hand afterwards.
        OUTDIR=$("$PY" -c "import sys; sys.path.insert(0, 'scripts/nn_eddy'); \
import nn_data; print(nn_data.COARSE_DIR)")
        [ -z "$OUTDIR" ] && { echo "FATAL: could not resolve nn_data.COARSE_DIR"; exit 3; }
        for i in $(seq 1 $#); do
            eval "a=\${$i}"; eval "b=\${$((i+1))}"
            [ "$a" = "--out" ] && OUTDIR="$b"
            [ "$a" = "--month" ] && MONTH="$b"
        done
        PROD="$OUTDIR/coarse_ng5_${MONTH}.nc"
        if [ -f "$PROD" ]; then
            echo "=== checking $PROD ==="
            "$PY" scripts/nn_eddy/check_coarse_product.py "$PROD"
            crc=$?
            [ "$crc" -ne 0 ] && { echo "PRODUCT CHECK FAILED — do not train on this file"; rc=$crc; }
        else
            echo "WARNING: expected product $PROD not found; nothing checked"
        fi
    fi
    ;;
esac

echo "=== done $(date) rc=$rc ==="
exit $rc
