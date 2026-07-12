#!/bin/bash
# Phase 8b B.1 — cache the PHC winter IC for farc → /work (the slow numpy extrapolation runs once).
#SBATCH --job-name=cache_ic_farc
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/cache_ic_farc.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/cache_ic_farc.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_farc
echo "=== node $(hostname)  $(date) ==="
"$PY" scripts/cache_phc_ic.py --mesh-dir "$M" --out-dir "$M"
echo "exit: $?  ($(date))"
ls -la "$M"/T_ic.npy "$M"/S_ic.npy 2>/dev/null