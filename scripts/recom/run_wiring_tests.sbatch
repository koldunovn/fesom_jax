#!/bin/bash
# Lane W (REcoM wiring): pytest on a compute node. Submit from the worktree root:
#   sbatch scripts/recom/run_wiring_tests.sbatch [DEV=1|4] <pytest args...>
# DEV=4 runs with 4 host devices (the sharded tests).
#SBATCH --job-name=recom_wire
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH -o scripts/recom/run_wiring_tests.%j.out
set -u
cd "${SLURM_SUBMIT_DIR}"
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export PYTHONPATH=$PWD JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
DEV=1
case "${1:-}" in DEV=*) DEV=${1#DEV=}; shift;; esac
[ "$DEV" != 1 ] && export XLA_FLAGS=--xla_force_host_platform_device_count=$DEV
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) $(git status --porcelain -- fesom_jax scripts configs | wc -l) modified, DEV=$DEV"
$PY -m pytest -q -rs -p no:cacheprovider -s "$@"
rc=$?
echo "=== exit $rc $(date)"
exit $rc
