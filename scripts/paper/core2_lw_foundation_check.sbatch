#!/bin/bash
# Long-window Task A1 — Foundation drift verdict. CPU-only analysis of the §0 foundation
# all-on 3-yr climate (job 25740424): monthly netCDF + ALL 36 monthly snapshots on $WORK.
# Reads ~68 GB of snapshots (one at a time, scalars only) — runs on a compute node, not login.
#SBATCH --job-name=lw_a1_found_check
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/lw_a1_found_check.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/lw_a1_found_check.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
echo "=== $(hostname) $(date) — A1 foundation drift verdict (full snapshot sweep) ==="
"$PY" scripts/paper/core2_lw_foundation_check.py --snap-stride 1 --years 3 --start-year 1958
echo "exit: $? ($(date))"
