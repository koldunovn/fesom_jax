#!/bin/bash
# Phase 8b — cache PHC winter IC for NG5 (7.4M) -> /work. The numpy extrapolation over 7.4M nodes
# may be slow (dars 3.16M took 670s); generous time + exclusive node (full RAM). One-time prep.
#SBATCH --job-name=cache_ic_ng5
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/cache_ic_ng5.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/cache_ic_ng5.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
echo "=== $(hostname) $(date) ==="
"$PY" scripts/cache_phc_ic.py --mesh-dir "$M" --out-dir "$M"
echo "exit: $? ($(date))"; ls -la "$M"/T_ic.npy 2>/dev/null
