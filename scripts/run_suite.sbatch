#!/bin/bash
# Full JAX test suite on a COMPUTE node (257 GB RAM, 256 cores) — the login node
# has too little RAM for the heavy CORE2 backprop tests (test_gradient_core2 /
# test_step_core2) and only allows one CPU-JAX process. Runs the ocean group and
# the ice group as SEPARATE pytest invocations (the 423-in-one-process jit-cache +
# the heavy assembled test_ice_step exceeds even a generous RAM budget cleanly).
# JAX_PLATFORMS=cpu — the tight bit-exact gates are calibrated for CPU FP.
#
# Usage: sbatch scripts/run_suite.sbatch
#SBATCH --job-name=jax_suite
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
# Wall budget — the suite GREW (ocean now 665 tests ~44 min + ice ~2 min + sharding ~30 min):
# the old 65-min cap time-cut the sharding group (2026-06-18). 1:45 clears the full set with margin.
#SBATCH --time=01:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_suite.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_suite.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
# 128 physical cores/node, SMT off (--hint=nomultithread). The --cpus-per-task cgroup
# binds all threads to the 16 allocated physical cores (no oversubscription / the
# login-node pthread_create OOM); cap the OpenMP pool to match.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

ICE_FILES="test_ice_adv test_ice_coupling test_ice_evp test_ice_ic test_ice_step test_ice_thermo"
IGNORE=""
for f in $ICE_FILES; do IGNORE="$IGNORE --ignore=fesom_jax/tests/$f.py"; done

# Phase-8 sharding tests use shard_map collectives ⇒ need CPU fake-devices; they
# run as a SEPARATE group with --xla_force_host_platform_device_count=4 (and SKIP
# cleanly in the 1-device ocean group). Ignore them in ocean so they only run there.
SHARD_FILES="fesom_jax/tests/test_halo.py"
for f in fesom_jax/tests/test_*_sharded.py; do [ -e "$f" ] && SHARD_FILES="$SHARD_FILES $f"; done
for f in $SHARD_FILES; do IGNORE="$IGNORE --ignore=$f"; done

echo "=== node $(hostname)  $(date) ==="
echo "=== OCEAN group (incl. heavy CORE2 tests) ==="
$PY -m pytest fesom_jax/tests/ -q -p no:cacheprovider $IGNORE
OCEAN=$?

echo ""
echo "=== ICE group (separate process) ==="
$PY -m pytest fesom_jax/tests/test_ice_*.py -q -p no:cacheprovider
ICE=$?

echo ""
echo "=== SHARDING group (4 CPU fake-devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    $PY -m pytest $SHARD_FILES -q -p no:cacheprovider
SHARD=$?

echo ""
echo "=== SUMMARY: ocean rc=$OCEAN  ice rc=$ICE  shard rc=$SHARD ==="
[ $OCEAN -eq 0 ] && [ $ICE -eq 0 ] && [ $SHARD -eq 0 ] && echo "ALL GREEN" || echo "FAILURES (see above)"
