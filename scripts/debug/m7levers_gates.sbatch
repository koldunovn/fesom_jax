#!/bin/bash
# perf/kokkos-m7-levers gates — the four levers transferred from the port_kokkos M7
# campaign, on CPU (fake devices), on a compute node (pytest does not belong on login).
#   test_jra55.py               clamp-release fix + trig cache: the d1_s0 C-dump compare
#                               runs INSIDE the old clamp window ⇒ it is the bit-identity
#                               proof; the new clamp/trig tests pin the no-rebuild contract
#   test_halo.py                exchange_pair == two singles (bit) + the standing transports
#   test_reductions_sharded.py  global_dot_pair == two global_dots (bit)
#   test_ssh.py + test_ssh_sharded.py   the CG with the fused (r·z, r·r) reduction
#   test_ice_evp.py + test_mevp.py + test_ice_step.py   dense EVP/mEVP paths (must be
#                               byte-identical — exch=None returns the pair untouched)
#   test_step_sharded.py        END-TO-END: sharded step through the fused CG psum and
#                               the fused EVP u,v exchange
#SBATCH --job-name=m7levers_gates
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/m7levers_gates.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/m7levers_gates.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
set -o pipefail
rc=0

echo "=== $(date) jra55 reader (clamp + trig cache; C-dump bit compare) ==="
"$PY" -m pytest fesom_jax/tests/test_jra55.py fesom_jax/tests/test_forcing.py -q || rc=1

echo "=== $(date) halo exchange_pair + transports (8 fake devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=8 "$PY" -m pytest \
    fesom_jax/tests/test_halo.py fesom_jax/tests/test_reductions_sharded.py -q || rc=1

echo "=== $(date) ssh CG (dense + sharded, 4 fake devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 "$PY" -m pytest \
    fesom_jax/tests/test_ssh.py fesom_jax/tests/test_ssh_sharded.py -q || rc=1

echo "=== $(date) EVP/mEVP dense paths ==="
"$PY" -m pytest fesom_jax/tests/test_ice_evp.py fesom_jax/tests/test_mevp.py \
    fesom_jax/tests/test_ice_step.py -q || rc=1

echo "=== $(date) end-to-end sharded step (4 fake devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 "$PY" -m pytest \
    fesom_jax/tests/test_step_sharded.py -q || rc=1

echo "=== done $(date) rc=$rc ==="
exit $rc
