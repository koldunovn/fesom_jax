#!/bin/bash
#SBATCH --job-name=cache_ic_dars
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/cache_ic_dars.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/cache_ic_dars.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
echo "=== $(hostname) $(date) ==="
"$PY" scripts/cache_phc_ic.py --mesh-dir "$M" --out-dir "$M"
echo "exit: $? ($(date))"; ls -la "$M"/T_ic.npy 2>/dev/null
