#!/bin/bash
# Phase 8d gates — the coloured-ppermute halo, on CPU (fake devices), on a compute node.
# CORE2-scale pytest does not belong on the login node; the step gates are the slow ones
# (a full sharded ocean step compiles per transport).
#   test_halo.py            standalone exchange: forward bit-exact + grad vs the all_gather
#                           oracle + the partial-permutation invariant (8 devices)
#   test_step_sharded.py    END-TO-END: drives the exchange through the CG's SSHHalo
#                           (~2 exchanges x ~127 iters = the dominant per-step comm)
#SBATCH --job-name=test_coloured
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/test_coloured.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/test_coloured.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu

echo "=== $(date) halo gates (8 fake devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=8 "$PY" -m pytest fesom_jax/tests/test_halo.py -q

echo "=== $(date) end-to-end sharded-step gates (4 fake devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 "$PY" -m pytest \
    fesom_jax/tests/test_step_sharded.py -q -k "coloured or padded"

echo "=== done $(date) ==="
