#!/bin/bash
# Confirm test_step_sharded.py (incl. the new npes=2 owned-match gate) on a COMPUTE node.
#SBATCH --job-name=test_step_shard
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=80G
#SBATCH --time=00:25:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_step_sharded.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_step_sharded.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -q -p no:cacheprovider -rs
echo "exit: $?"
