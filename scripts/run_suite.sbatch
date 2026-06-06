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
#SBATCH --time=00:30:00
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

echo "=== node $(hostname)  $(date) ==="
echo "=== OCEAN group (incl. heavy CORE2 tests) ==="
$PY -m pytest fesom_jax/tests/ -q -p no:cacheprovider $IGNORE
OCEAN=$?

echo ""
echo "=== ICE group (separate process) ==="
$PY -m pytest fesom_jax/tests/test_ice_*.py -q -p no:cacheprovider
ICE=$?

echo ""
echo "=== SUMMARY: ocean rc=$OCEAN  ice rc=$ICE ==="
[ $OCEAN -eq 0 ] && [ $ICE -eq 0 ] && echo "ALL GREEN" || echo "FAILURES (see above)"
