#!/bin/bash
# Lane P part 2 (REcoM plan Task 7): the REcoM per-step input tests and the identity gates with REcoM off.
# Submit from the worktree root: sbatch scripts/recom/run_inputs_tests.sbatch [tests]
#SBATCH --job-name=recom_inputs
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH -o scripts/recom/run_inputs_tests.%j.out
set -u
WHAT=${1:-tests}
cd "${SLURM_SUBMIT_DIR}"
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export PYTHONPATH=$PWD JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) $(git status --porcelain -- fesom_jax scripts configs | wc -l) modified"
rc=0
if [ "$WHAT" = tests ] || [ "$WHAT" = all ]; then
    echo "=== one device: test_recom_inputs, the DP lock, mslp, the JRA55 reader, the REcoM transport, configs"
    $PY -m pytest -q -rs -p no:cacheprovider -s fesom_jax/tests/test_recom_inputs.py fesom_jax/tests/test_dp_lock.py \
        fesom_jax/tests/test_mslp.py fesom_jax/tests/test_jra55.py fesom_jax/tests/test_recom_transport.py \
        fesom_jax/tests/test_run_config.py fesom_jax/tests/test_partition_state.py fesom_jax/tests/test_forcing_adj.py \
        fesom_jax/tests/test_recom_params.py fesom_jax/tests/test_precision.py || rc=1
    echo "=== four host devices: test_recom_inputs, test_mslp, test_recom_transport, test_forcing_sharded (sharded)"
    XLA_FLAGS=--xla_force_host_platform_device_count=4 $PY -m pytest -q -rs -p no:cacheprovider -s \
        fesom_jax/tests/test_recom_inputs.py fesom_jax/tests/test_mslp.py fesom_jax/tests/test_recom_transport.py \
        fesom_jax/tests/test_forcing_sharded.py || rc=1
fi
echo "=== exit $rc $(date)"
exit $rc
