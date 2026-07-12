#!/bin/bash
#SBATCH --job-name=test_time_dim
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_time_dim.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_time_dim.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
echo "=== $(date) full regression after giving data fields a leading time dim ==="
echo "--- ushow_output / canonical_redist (4 fake devices) ---"
XLA_FLAGS="--xla_force_host_platform_device_count=4" "$PY" -m pytest \
  fesom_jax/tests/test_ushow_output.py fesom_jax/tests/test_canonical_redist.py -x -s -q 2>&1 | tail -20
echo "pytest1 exit=${PIPESTATUS[0]}"
echo "--- daily/monthly output (real run, single device) ---"
"$PY" -m pytest fesom_jax/tests/test_run_entry.py -k "daily_output or monthly_output" -x -s -q 2>&1 | tail -30
echo "pytest2 exit=${PIPESTATUS[0]}"
