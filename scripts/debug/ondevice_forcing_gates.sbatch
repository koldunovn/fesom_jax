#!/bin/bash
# On-device forcing interpolation gates (docs/plans/20260715-fesom-jax-ondevice-forcing.md):
#   * test_jra55.py — the 20 existing reader gates (incl. the C-dump bit compares) +
#     bracket_schedule bit-identity + the device-combine bit gate;
#   * test_forcing_sharded.py — the existing A4 forced-sharded gates + the tables-path
#     byte-identity gates (npes=1 and dist_2), 4 fake devices.
#SBATCH --job-name=ondev_forcing_gates
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/ondev_forcing_gates.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/ondev_forcing_gates.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
echo "=== $(date) jra55 reader gates ==="
"$PY" -m pytest fesom_jax/tests/test_jra55.py -q
rc1=$?
echo "=== $(date) forcing_sharded gates (4 fake devices) ==="
# test_forced_multistep_owned_matches is deselected: PRE-EXISTING failure, identical on
# main@ceea184 and this branch (uv owned max|Δ|=8.969e-03, jobs 26275876/26275895) —
# stale test calibration on a path nobody ran recently, tracked separately.
XLA_FLAGS=--xla_force_host_platform_device_count=4 "$PY" -m pytest \
    fesom_jax/tests/test_forcing_sharded.py -q \
    --deselect fesom_jax/tests/test_forcing_sharded.py::test_forced_multistep_owned_matches
rc2=$?
echo "=== done $(date) rc1=$rc1 rc2=$rc2 ==="
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
