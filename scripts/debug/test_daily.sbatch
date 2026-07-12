#!/bin/bash
#SBATCH --job-name=test_daily
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_daily.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_daily.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
echo "=== $(date) test_daily_output (CORE2 single-device, CPU) ==="
"$PY" -m pytest fesom_jax/tests/test_run_entry.py::test_daily_output -x -s -q 2>&1 | tail -28
echo "pytest exit=${PIPESTATUS[0]}"
