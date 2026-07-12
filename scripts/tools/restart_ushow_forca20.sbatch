#!/bin/bash
#SBATCH --job-name=ushow_forca20
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/restart_ushow_forca20.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/restart_ushow_forca20.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
R=/work/ab0995/a270088/port_jax/runs/forca20_3yr/restart
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
echo "=== $(date) convert FORCA20 restart -> ushow zarr ==="
"$PY" scripts/tools/restart_to_ushow.py "$R" "$M" "${R%/restart}/restart_ushow.zarr"
echo "exit=$? ; wrote: $(ls -d ${R%/restart}/restart_ushow.zarr 2>/dev/null)"
