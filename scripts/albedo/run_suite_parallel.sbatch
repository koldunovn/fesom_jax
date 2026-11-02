#!/bin/bash
# Albedo: the full suite as one array task per cost group (scripts/runs/run_suite_parallel.sbatch, its Levante
# original, holds the history and the guards; this copy changes only the SLURM header, the interpreter and the
# environment: one 128-core 256 GB mpp node per group, at most 240 GB, the 12 h qos, scripts/albedo/env.sh).
#   sbatch scripts/albedo/run_suite_parallel.sbatch            (from the repo root)
#SBATCH --job-name=fj_suite
#SBATCH -p mpp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=12h
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%A_%a.out

source "${SLURM_SUBMIT_DIR:-.}/scripts/albedo/env.sh"
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:?run from the repo root}" || exit 1
export PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH}   # this checkout, not the env's editable install
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
    rc=$?          # read rc immediately: a $(date) in between would overwrite $?
elif [ "$GROUP" = "sharding" ] || [ "$GROUP" = "sharding_slow" ]; then
    # 4 CPU fake devices; these tests skip cleanly when only 1 device is visible, which is why
    # they must NOT be folded into another group — they would silently pass by skipping.
    # One pytest process per file: in one process the group's jit caches outgrew the 300 GB
    # request a quarter of the way through (OOM, 2026-10-28, 24 files).
    rc=0
    for f in $FILES; do
        XLA_FLAGS=--xla_force_host_platform_device_count=4 \
            $PY -m pytest $PYTEST_ARGS "$f" || rc=1
    done
else
    $PY -m pytest $PYTEST_ARGS $FILES
    rc=$?
fi

echo ""
echo "=== GROUP $GROUP rc=$rc  $(date) ==="
exit $rc
