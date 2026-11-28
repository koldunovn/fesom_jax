#!/bin/bash
# Selected tests of the partial-cells branch on a compute node (CPU JAX; never on the login node).
#   cd <worktree> && sbatch --export=ALL,TESTS="fesom_jax/tests/test_partial_cells.py" scripts/partial_cells/pytest.sbatch
# DEVICES=N adds --xla_force_host_platform_device_count=N for the sharded tests.
#SBATCH --job-name=pc_pytest
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/partial_cells/pytest.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
if [ -n "$DEVICES" ]; then export XLA_FLAGS="--xla_force_host_platform_device_count=$DEVICES"; fi
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) TESTS=$TESTS ==="
$PY -c "import fesom_jax, sys; print('fesom_jax from', fesom_jax.__file__)"
$PY -m pytest $TESTS -q -p no:cacheprovider ${PYTEST_ARGS}
echo "=== pytest rc=$? $(date) ==="
