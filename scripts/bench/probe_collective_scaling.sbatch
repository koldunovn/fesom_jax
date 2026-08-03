#!/bin/bash
# Per-collective cost vs rank count on JAX-CPU: is gloo O(P) (ring) or O(log P) (tree)?
# A ring would cap decomposition exactly where CORE2 stops improving (32 ranks), while
# Fortran runs the same mesh on 512 ranks.
#SBATCH --job-name=coll_scaling
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/coll_scaling.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/coll_scaling.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu JDIST_CPU=1 FESOM_CPU_COLL=gloo OMP_NUM_THREADS=8
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
for n in 2 4 8 16 32 64; do
  # spread across nodes the way the campaign does (16 ranks/node)
  mpirun -n $n --map-by ppr:16:node:pe=8 --bind-to core \
    "$PY" scripts/bench/probe_collective_scaling.py 2>&1 | grep -E "^\[coll\]|rror" | head -6
done
echo "=== done $(date) ==="
