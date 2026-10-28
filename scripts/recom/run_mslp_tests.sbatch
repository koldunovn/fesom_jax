#!/bin/bash
# Lane P (REcoM plan Task 5): the l_mslp tests, the identity gates with mslp off, and the pi physics floor.
# Submit from the worktree root: sbatch scripts/recom/run_mslp_tests.sbatch [floor|tests|all]
#SBATCH --job-name=recom_mslp
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH -o scripts/recom/run_mslp_tests.%j.out
set -u
WHAT=${1:-all}
cd "${SLURM_SUBMIT_DIR}"
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export PYTHONPATH=$PWD JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) $(git status --porcelain -- fesom_jax scripts configs | wc -l) modified"
rc=0
if [ "$WHAT" = floor ] || [ "$WHAT" = all ]; then
    echo "=== physics floor (JAX pi_recom vs the Fortran twin, steps 1-2)"
    $PY scripts/recom/physics_floor.py --steps 2 || rc=1
fi
if [ "$WHAT" = tests ] || [ "$WHAT" = all ]; then
    echo "=== one device: test_mslp, the DP lock, the JRA55 reader, the REcoM transport"
    $PY -m pytest -q -rs -p no:cacheprovider -s fesom_jax/tests/test_mslp.py fesom_jax/tests/test_dp_lock.py \
        fesom_jax/tests/test_jra55.py fesom_jax/tests/test_recom_transport.py fesom_jax/tests/test_run_config.py \
        fesom_jax/tests/test_partition_state.py fesom_jax/tests/test_forcing_adj.py || rc=1
    echo "=== four host devices: test_mslp (sharded), test_recom_transport (sharded), test_forcing_sharded"
    XLA_FLAGS=--xla_force_host_platform_device_count=4 $PY -m pytest -q -rs -p no:cacheprovider -s \
        fesom_jax/tests/test_mslp.py fesom_jax/tests/test_recom_transport.py fesom_jax/tests/test_forcing_sharded.py || rc=1
fi
echo "=== exit $rc $(date)"
exit $rc
