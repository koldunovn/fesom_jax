#!/bin/bash
# S.9 finalize — verify the GPU-aware gate edits on CPU fake-devices (Rule 1: heavy shard_map
# compiles go through sbatch on -p compute, NEVER the login node). Confirms (a) the platform-aware
# _BYTE_ID_ATOL keeps the CPU byte-id gate tight (1e-9), (b) the _owned_match 4-tuple unpack runs
# (OCEAN owned-match = the lightest caller), (c) the EVP stress σ now lands in the DIAG print/skip
# path, not the assert loop (ICE owned-match exercises _DIAG_FIELDS).
#SBATCH --job-name=verify_s9_gate
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --time=30:00
#SBATCH --mem=120G
#SBATCH -o /home/a/a270088/port_jax/scripts/verify_s9_gate_cpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/verify_s9_gate_cpu.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_FLAGS="--xla_force_host_platform_device_count=2"
export JAX_PLATFORMS=cpu
echo "=== node $(hostname)  $(date) ==="
for t in \
  "test_serial_sharded_step_matches_dense" \
  "test_sharded_step_owned_matches" \
  "test_ice_sharded_step_owned_matches" ; do
  echo "######## $t ########"
  "$PY" -m pytest "fesom_jax/tests/test_step_sharded.py::$t" -q -p no:cacheprovider -rs -s
  echo ">>>> $t exit: $?"
done
echo "=== done $(date) ==="
