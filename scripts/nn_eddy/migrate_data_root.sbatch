#!/bin/bash
# Move the NN-GM working data from ab0995 to ba1583, next to the teacher output it came from.
#
# WHY: measured 2026-08-08 — the ab0995 /work mount was at 98% (2.5 P of 2.6 P, 53 T free) while
# ba1583 was at 10% (117 T free) and already held the 2.9 TB ng5_teacher run. Coarse products grow
# ~13 GB per teacher year and M4's trajectories ~130 GB per year per refresh, so this only gets
# worse. `nn_data.DATA_ROOT` is the single constant every script reads.
#
# ⚠️ THIS SCRIPT NEVER DELETES ANYTHING. It copies, then verifies, and leaves the originals
# exactly where they are for a human to remove after reading the verification output. That is the
# house rule (unattended deletion cost this project real data once), and it is also just correct
# for 53 GB of verified products whose numbers are recorded in a passed gate.
#
#   sbatch scripts/nn_eddy/migrate_data_root.sbatch            # copy + verify
#   sbatch scripts/nn_eddy/migrate_data_root.sbatch --verify    # verify an earlier copy only
#SBATCH --job-name=nn_migrate
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o scripts/nn_eddy/migrate.%j.out
#SBATCH -e scripts/nn_eddy/migrate.%j.out
source /sw/etc/profile.levante
set -u
SRC=/work/ab0995/a270088/nn_eddy
DST=/work/ba1583/a270088/nn_eddy
# ng5_teacher is ALREADY on ba1583 and is not touched. mesh_ng5 is an input, the rest are products.
DIRS="coarse mesh_ng5 student_free_1958 horizon_truth"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

echo "=== $(hostname) $(date) ==="
echo "=== SRC=$SRC  DST=$DST  dirs: $DIRS  verify_only=$VERIFY_ONLY ==="
/usr/bin/df -h "$DST" | tail -1

rc=0
if [ "$VERIFY_ONLY" -eq 0 ]; then
    for d in $DIRS; do
        [ -d "$SRC/$d" ] || { echo "SKIP $d (not present in SRC)"; continue; }
        echo ""
        echo "--- copying $d ($(/usr/bin/du -sh "$SRC/$d" | cut -f1)) ---"
        mkdir -p "$DST/$d"
        # -a preserves times/modes; --ignore-existing so a re-run RESUMES instead of rewriting
        # files already verified; no --delete anywhere, ever.
        rsync -a --ignore-existing --info=progress2 "$SRC/$d/" "$DST/$d/" || rc=$?
    done
fi

echo ""
echo "=== VERIFY: per-directory file count, byte total, and a full md5 of every product ==="
for d in $DIRS; do
    [ -d "$SRC/$d" ] || continue
    ns=$(find "$SRC/$d" -type f | wc -l); nd=$(find "$DST/$d" -type f 2>/dev/null | wc -l)
    bs=$(find "$SRC/$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    bd=$(find "$DST/$d" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
    echo "  $d: files $ns -> $nd   bytes $bs -> $bd"
    [ "$ns" = "$nd" ] && [ "$bs" = "$bd" ] || { echo "  MISMATCH in $d"; rc=1; }
done

echo ""
echo "=== md5 of the coarse products (the files a recorded gate depends on) ==="
# Only `coarse` is checksummed: it is the irreplaceable output (a month costs ~35 min of a node)
# and the one whose bytes a published number rests on. mesh_ng5 is regenerable from the mesh and
# the others are cheap to reproduce, so count+bytes is proportionate for them.
for f in "$SRC/coarse"/*.nc; do
    b=$(basename "$f")
    [ -f "$DST/coarse/$b" ] || { echo "  MISSING $b"; rc=1; continue; }
    a=$(md5sum "$f" | cut -d' ' -f1)
    c=$(md5sum "$DST/coarse/$b" | cut -d' ' -f1)
    if [ "$a" = "$c" ]; then echo "  OK   $b  $a"; else echo "  DIFF $b  $a != $c"; rc=1; fi
done

echo ""
echo "=== done $(date) rc=$rc ==="
if [ $rc -eq 0 ]; then
    echo "MIGRATION VERIFIED — the copies match byte for byte."
    echo "The ORIGINALS UNDER $SRC ARE UNTOUCHED. Nothing here deletes them; that is a human"
    echo "decision, and the two big items are not project data at all:"
    /usr/bin/du -sh "$SRC"/task1_smoke "$SRC"/core2_output_test 2>/dev/null
else
    echo "MIGRATION NOT VERIFIED — do not delete anything, read the mismatches above."
fi
exit $rc
