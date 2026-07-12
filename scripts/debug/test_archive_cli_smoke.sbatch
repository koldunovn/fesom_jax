#!/bin/bash
# End-to-end smoke test of the archival-restart CLI plumbing: run_from_config.py (not just the
# library function) through a real 2-job resume cycle, mimicking exactly what
# run_core2_hindcast.sbatch's RESTART_DIR/restart.latest bash logic does. Cheap (CORE2 test mesh,
# tiny step counts) -- a fast, isolated dry run of the exact production wiring before it goes live.
#SBATCH --job-name=test_archive_cli
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_archive_cli_smoke.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_archive_cli_smoke.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
REPO=/home/a/a270088/port_jax
cd $REPO
export JAX_PLATFORMS=cpu

OUT=/scratch/a/a270088/tmp/claude-24253/-home-a-a270088-port-jax/4aa79da0-783b-409a-9145-10c843e84004/scratchpad/archive_cli_smoke
rm -rf "$OUT"; mkdir -p "$OUT"
ARCHIVE=$OUT/restart_archive

echo "=== $(date) job#1: cold start, restart-archive-period=day (fires within a small step count) ==="
RESTART_DIR=""
if [ -f "$ARCHIVE/restart.latest" ]; then
  TAG=$(cat "$ARCHIVE/restart.latest"); [ -n "$TAG" ] && RESTART_DIR="$ARCHIVE/$TAG"
fi
echo "RESTART_DIR (should be empty, cold)='$RESTART_DIR'"
RIN=""; [ -n "$RESTART_DIR" ] && RIN="--restart-in $RESTART_DIR"

"$PY" scripts/run_from_config.py configs/core2_full.yaml \
    --mesh-dir data/mesh_core2 --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 \
    --ic-dir data/ic_core2 --partition dist_1 --year 1958 \
    --steps 60 --chunk-steps 12 --diagnostics --progress \
    --restart-archive-out "$ARCHIVE" --restart-archive-period day --restart-archive-length 1 \
    $RIN 2>&1 | tail -30
echo "job#1 exit=${PIPESTATUS[0]}"

echo "--- job#1 archive contents ---"
ls "$ARCHIVE" 2>&1
echo "restart.latest -> $(cat $ARCHIVE/restart.latest 2>&1)"

echo "=== $(date) job#2: resume from restart.latest, run 12 more steps ==="
RESTART_DIR=""
if [ -f "$ARCHIVE/restart.latest" ]; then
  TAG=$(cat "$ARCHIVE/restart.latest"); [ -n "$TAG" ] && RESTART_DIR="$ARCHIVE/$TAG"
fi
echo "RESTART_DIR (should be the job#1 final tag)='$RESTART_DIR'"
if [ -z "$RESTART_DIR" ]; then
  echo "TEST_FAILED: no restart.latest after job#1"; exit 1
fi

"$PY" scripts/run_from_config.py configs/core2_full.yaml \
    --mesh-dir data/mesh_core2 --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 \
    --ic-dir data/ic_core2 --partition dist_1 --year 1958 \
    --steps 12 --chunk-steps 12 --diagnostics --progress \
    --restart-archive-out "$ARCHIVE" --restart-archive-period day --restart-archive-length 1 \
    --restart-in "$RESTART_DIR" 2>&1 | tail -30
echo "job#2 exit=${PIPESTATUS[0]}"

echo "--- final archive contents (must be >=3: day boundary(48) + job#1 final(60) + job#2 final(72), none overwritten) ---"
ls "$ARCHIVE" 2>&1
N=$(ls -d "$ARCHIVE"/fesom.* 2>/dev/null | wc -l)
echo "N_ARCHIVAL_DIRS=$N"
if [ "$N" -ge 3 ]; then echo "ARCHIVE_CLI_SMOKE_OK"; else echo "ARCHIVE_CLI_SMOKE_FAILED (expected >=3 dirs)"; fi
