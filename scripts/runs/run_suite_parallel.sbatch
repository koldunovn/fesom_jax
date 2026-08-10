#!/bin/bash
# Full JAX test suite, every COST group as a job array so they run concurrently.
#
# History, because the shape of this script is a response to two specific failures.
#
# `run_suite.sbatch` runs the groups sequentially in one job, so the wall time must cover their
# SUM and the last group is always the one time-cut. That happened twice (2026-06-18,
# 2026-08-06) and both times read as a sharding failure rather than a scheduling one. An array
# makes the wall time the LONGEST group and gives each its own budget.
#
# The groups then had a second problem: they were "ice", "sharding", and "everything else".
# "Everything else" is a default, not a category. It reached 920 tests and 3:28 of wall time
# without anyone deciding that — a new file joined it by doing nothing, and nothing reported
# per-test cost, so the growth was invisible. Three changes fix that:
#
#   1. Groups come from `scripts/runs/suite_groups.txt`, one line per file. There is no
#      residual group left for a file to fall into.
#   2. `fesom_jax/tests/test_suite_groups.py` FAILS if a test file is in no group, so the
#      omission surfaces on whoever adds the file rather than months later.
#   3. `--durations=25` on every group. Per-test cost is now always in the log, which is the
#      only reason anyone would notice the next time a group starts growing.
#
#   sbatch scripts/runs/run_suite_parallel.sbatch
#   grep -hE "GROUP|passed|failed|slowest" scripts/run_suite_par.*.out
#
#SBATCH --job-name=jax_suite_par
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH -o scripts/run_suite_par.%A_%a.out
#SBATCH -e scripts/run_suite_par.%A_%a.out
# NB --mem=300G, not 200G: the sharding group OOM'd at 209 GB against a 200 GB request on
# 2026-08-07. That is a real failure and it looked exactly like a sharding bug. The array shares
# one --mem for all tasks, so the largest group sets it.
#
# ⚠️ 8:00:00, not 6:00:00, and 8 h is the `compute` partition maximum. Measured on 2026-08-08
# (job 26794462): sharding 5:02:17, core2 3:23:32, ice 0:31:07, unit 0:33:31, scripts 0:00:31.
# The SUM is ~9.5 h, which is why the sequential run_suite.sbatch cannot produce a green baseline
# at all inside one job — it was time-cut at 2:30 on 2026-08-10 (job 26832970) partway through
# its second group, and no wall time available on this partition would have saved it. The array's
# budget only has to cover the LONGEST group, and sharding at 5 h is that group.

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/port_jax}" || exit 1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

MANIFEST=scripts/runs/suite_groups.txt
echo "=== node $(hostname)  $(date)  array task ${SLURM_ARRAY_TASK_ID} ==="
echo "=== checkout: $(pwd)  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST"; exit 2; }

# ⚠️ NOT `GROUPS`. That is a bash BUILT-IN array holding the current user's group IDs, and
# assigning to it is silently ignored — the array tasks read out gid 1221/1260/1291 as group
# names and the run died with "group 1221 is empty" (2026-08-08). The empty-group guard below is
# the only reason that took one minute to find instead of looking like a pytest problem.
#
# ⚠️ The group NAMES come from the manifest too, sorted, and are NOT written out here. They used
# to be: `SUITE_GROUPS=(unit core2 ice sharding scripts)`. When the `ci` group was added on
# 2026-08-10 the manifest, VALID_GROUPS in test_suite_groups.py and .github/workflows/ci.yml were
# all updated and this line was not, so the array would have run five of six groups and printed a
# pass — 35 files silently untested. That is the same registry-drift failure the manifest exists
# to end, one level up. A list in one file describing the contents of another always drifts.
mapfile -t SUITE_GROUPS < <(awk '!/^#/ && NF {print $1}' "$MANIFEST" | sort -u)
NGROUPS=${#SUITE_GROUPS[@]}
GROUP=${SUITE_GROUPS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== manifest groups (sorted): ${SUITE_GROUPS[*]} ==="

# The --array range is a static SBATCH directive and cannot read the manifest, so it is the one
# number that CAN still disagree with it. Refuse to run rather than test a subset: with 6 groups
# and --array=0-4, tasks 0-4 would each pass and nobody would be told about group 5.
if [ -n "$SLURM_ARRAY_TASK_COUNT" ] && [ "$SLURM_ARRAY_TASK_COUNT" -ne "$NGROUPS" ]; then
    echo "ABORT: --array covers $SLURM_ARRAY_TASK_COUNT tasks but the manifest has $NGROUPS"
    echo "groups (${SUITE_GROUPS[*]}). Set --array=0-$((NGROUPS - 1)) in this file."
    exit 2
fi
[ -n "$GROUP" ] || { echo "no group for array task ${SLURM_ARRAY_TASK_ID}"; exit 2; }

# Every test file must be in the manifest. Checked HERE too, not only in the pytest gate: if a
# new file is unclassified, this run must not quietly test less than the whole suite and then
# print a pass.
#
# ⚠️ BOTH directories. Until 2026-08-08 every runner globbed fesom_jax/tests/ only, so the seven
# files under scripts/tests/ had never run in the suite at all — including the 12 mesh-diagnostic
# tests and the 7 horizon-scan harness tests the plan records as passing. They passed; they just
# passed by hand.
ON_DISK=$( { ls fesom_jax/tests/test_*.py; ls scripts/tests/test_*.py; } 2>/dev/null | sort)
LISTED=$(awk '!/^#/ && NF {print $2}' "$MANIFEST" | sort)
UNCLASSIFIED=$(comm -23 <(echo "$ON_DISK") <(echo "$LISTED"))
if [ -n "$UNCLASSIFIED" ]; then
    echo "ABORT: test files in no suite group — this run would silently skip them:"
    echo "$UNCLASSIFIED" | sed 's/^/  /'
    echo "Add each to $MANIFEST ('core2' if it loads the CORE2 mesh/IC/forcing, else 'unit')."
    exit 2
fi

FILES=$(awk -v g="$GROUP" '!/^#/ && $1 == g {printf "%s ", $2}' "$MANIFEST")
NFILES=$(wc -w <<<"$FILES")
echo "=== GROUP: $GROUP  ($NFILES files, from $MANIFEST) ==="
[ "$NFILES" -gt 0 ] || { echo "group $GROUP is empty"; exit 2; }

# `--durations=25` everywhere: the only reason the ocean group's growth to 3:28 was invisible is
# that nothing ever printed which tests were expensive.
PYTEST_ARGS="-q -p no:cacheprovider --durations=25"

if [ "$GROUP" = "scripts" ]; then
    # These import their target by path and need the repo root on sys.path, which running pytest
    # from the root already provides. Kept as their own group for that reason, not for cost.
    $PY -m pytest $PYTEST_ARGS $FILES
elif [ "$GROUP" = "sharding" ]; then
    # 4 CPU fake devices; these tests skip cleanly when only 1 device is visible, which is why
    # they must NOT be folded into another group — they would silently pass by skipping.
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        $PY -m pytest $PYTEST_ARGS $FILES
else
    $PY -m pytest $PYTEST_ARGS $FILES
fi
rc=$?          # read rc immediately: a $(date) in between would overwrite $?

echo ""
echo "=== GROUP $GROUP rc=$rc  $(date) ==="
exit $rc
