#!/bin/bash
# Full JAX test suite, three groups AS A JOB ARRAY so they run concurrently.
#
# Why this exists. run_suite.sbatch runs ocean, ice and sharding sequentially in one job,
# and the wall time has to cover their SUM: ocean ~1:14 + ice ~2:28 + sharding ~0:30 = ~4:12
# against a 2:30 cap. The sharding group is last, so it is the one that gets time-cut — which
# has now happened twice (2026-06-18, 2026-08-06) and both times looked like a sharding
# failure rather than a scheduling one. Running the groups in parallel makes the wall time the
# LONGEST group (~2:30) instead of the sum, and gives each its own budget.
#
# Same three groups, same flags, same environment as run_suite.sbatch — only the scheduling
# differs. Submit from the repo root of whichever checkout should be tested.
#
#   sbatch scripts/runs/run_suite_parallel.sbatch
#   # then: grep -h "passed\|failed\|GROUP" scripts/run_suite_par.*.out
#
#SBATCH --job-name=jax_suite_par
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH -o scripts/run_suite_par.%A_%a.out
#SBATCH -e scripts/run_suite_par.%A_%a.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/port_jax}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

ICE_FILES="test_ice_adv test_ice_coupling test_ice_evp test_ice_ic test_ice_step test_ice_thermo"
SHARD_FILES="fesom_jax/tests/test_halo.py"
for f in fesom_jax/tests/test_*_sharded.py; do [ -e "$f" ] && SHARD_FILES="$SHARD_FILES $f"; done

echo "=== node $(hostname)  $(date)  array task ${SLURM_ARRAY_TASK_ID} ==="
echo "=== checkout: $(pwd)  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    echo "=== GROUP: ocean (everything except ice and sharding) ==="
    IGNORE=""
    for f in $ICE_FILES;  do IGNORE="$IGNORE --ignore=fesom_jax/tests/$f.py"; done
    for f in $SHARD_FILES; do IGNORE="$IGNORE --ignore=$f"; done
    $PY -m pytest fesom_jax/tests/ -q -p no:cacheprovider $IGNORE
    rc=$?
    ;;
  1)
    echo "=== GROUP: ice (separate process — the jit cache of the whole suite plus the"
    echo "===        assembled ice step exceeds a generous RAM budget in one) ==="
    $PY -m pytest fesom_jax/tests/test_ice_*.py -q -p no:cacheprovider
    rc=$?
    ;;
  2)
    echo "=== GROUP: sharding (4 CPU fake-devices; skips cleanly in the 1-device group) ==="
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        $PY -m pytest $SHARD_FILES -q -p no:cacheprovider
    rc=$?
    ;;
  *) echo "unknown array task"; rc=2 ;;
esac

echo ""
echo "=== GROUP ${SLURM_ARRAY_TASK_ID} rc=$rc  $(date) ==="
exit $rc
