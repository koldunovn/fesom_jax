#!/bin/bash
# Sharding-group-only run (test_halo + test_*_sharded) on CPU fake-devices — the group the
# full run_suite.sbatch time-cut after it grew (ocean 44 min + ice 2 min + sharding 30 min >
# the 65-min wall). Standalone confirm that the A3/B0 changes don't regress sharding.
#SBATCH --job-name=jax_shard
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_suite_shard.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_suite_shard.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
SHARD_FILES="fesom_jax/tests/test_halo.py"
for f in fesom_jax/tests/test_*_sharded.py; do [ -e "$f" ] && SHARD_FILES="$SHARD_FILES $f"; done
echo "=== node $(hostname)  $(date) — SHARDING group (PER-FILE, fresh process each) ==="
RC=0
for f in $SHARD_FILES; do
  echo "----- $f -----"
  XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      "$PY" -m pytest "$f" -q -p no:cacheprovider; rc=$?
  [ $rc -ne 0 ] && RC=$rc
done
echo "exit: $RC ($(date))"
