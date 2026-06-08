#!/bin/bash
# Phase 8b B.0b/c — gate the ragged_all_to_all halo primitive on CPU fake-devices.
# Runs the whole S.3 halo suite (existing all_gather identity/grad gates MUST stay green)
# + the new test_ragged_primitive_matches_allgather (fwd byte-id + transpose == all_gather).
# Also tells us whether lax.ragged_all_to_all lowers on the CPU backend at all.
#SBATCH --job-name=test_halo_ragged
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --time=20:00
#SBATCH --mem=60G
#SBATCH -o /home/a/a270088/port_jax/scripts/test_halo_ragged.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_halo_ragged.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
export JAX_PLATFORMS=cpu
echo "=== node $(hostname)  $(date) ==="
"$PY" -m pytest fesom_jax/tests/test_halo.py -q -p no:cacheprovider -rs 2>&1
echo ">>>> exit: $?"
echo "=== done $(date) ==="
