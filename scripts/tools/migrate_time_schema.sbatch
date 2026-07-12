#!/bin/bash
# Migrate the already-written core2_hindcast_v2 / forca20_hindcast_v2 monthly+daily output to the
# new ushow time schema: every data field gains a leading time[1] axis, and the time coordinate
# itself becomes int64 (not float64) -- both required for ushow's per-point time-series feature,
# see fesom_jax/ushow_output.py's _put_time docstring + docs/PORTING_LESSONS.md. Tested 100%
# byte-identical on scratch copies before this was submitted. Skips each root's most-recently-
# modified dir as a safety margin against the still-running self-chaining production jobs.
#SBATCH --job-name=migrate_time
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/migrate_time_schema.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/migrate_time_schema.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
echo "=== $(date) migrate_time_schema ==="
"$PY" /scratch/a/a270088/tmp/claude-24253/-home-a-a270088-port-jax/4aa79da0-783b-409a-9145-10c843e84004/scratchpad/migrate_time_schema.py --roots \
  /work/ab0995/a270088/port_jax/runs/core2_hindcast_v2/monthly \
  /work/ab0995/a270088/port_jax/runs/forca20_hindcast_v2/monthly \
  /work/ab0995/a270088/port_jax/runs/forca20_hindcast_v2/daily
echo "=== $(date) done exit=$? ==="
