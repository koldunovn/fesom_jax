#!/bin/bash
# Build the dist_864 partition-faithful CORE2 PHC IC (data/ic_core2_dist864) — to fairly
# compare JAX-zstar vs c_zstar_2yr (which was run on 864 ranks, a different partition than the
# dist_16 IC). The 864-rank GS-fill simulation is the cost; generous time.
#SBATCH --job-name=ic_dist864
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/rebuild_ic_dist864.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/rebuild_ic_dist864.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname)  $(date) ==="
$PY scripts/tools/rebuild_ic_dist864.py
echo "exit: $?"
