#!/bin/bash
# Backfill the new CF `time[1]` coordinate onto already-written core2_hindcast_v2 /
# forca20_hindcast_v2 monthly+daily output (parses each store's existing calendar_date/
# calendar_month attr -- no need to recompute from run state). Additive + re-consolidates
# metadata; safe to run alongside the live self-chaining production jobs. Skips each
# root's most-recently-modified dir as a safety margin against the live writer.
#SBATCH --job-name=backfill_time
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/backfill_time.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/backfill_time.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
echo "=== $(date) backfill_time ==="
"$PY" /scratch/a/a270088/tmp/claude-24253/-home-a-a270088-port-jax/4aa79da0-783b-409a-9145-10c843e84004/scratchpad/backfill_time.py
echo "=== $(date) done exit=$? ==="
