#!/bin/bash
#SBATCH --job-name=test_restart_archive
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_restart_archive.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_restart_archive.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
echo "=== $(date) restart-archive feature tests ==="
echo "--- run_config.py: fast, no fixtures ---"
"$PY" -m pytest fesom_jax/tests/test_run_config.py -x -s -q 2>&1 | tail -20
echo "pytest1 exit=${PIPESTATUS[0]}"
echo "--- run_entry.py: pure-logic archive tests (fast) ---"
"$PY" -m pytest fesom_jax/tests/test_run_entry.py -k "archive_tag or archive_boundaries or year_boundaries or plan_chunks" -x -s -q 2>&1 | tail -20
echo "pytest2 exit=${PIPESTATUS[0]}"
echo "--- run_entry.py: full integration (real forcing/mesh, slower) ---"
"$PY" -m pytest fesom_jax/tests/test_run_entry.py -k "checkpoint_cadence or restart_archive_cadence or restart_seam or daily_output or monthly_output" -x -s -q 2>&1 | tail -40
echo "pytest3 exit=${PIPESTATUS[0]}"
