#!/bin/bash
# Does CGPOLY MOVE THE SCALING CEILING? The campaign found CORE2 saturating at 2 nodes and
# getting WORSE at 4 (990 -> 1293 ms). The mechanism measured in job 26493370 says why: the
# CG's all-reduce is O(P), so each added rank makes 254 collectives/step dearer while
# shrinking each rank's work. A high-degree Chebyshev preconditioner cuts the ITERATION count
# ~4x, so it removes most of those collectives — if that diagnosis is right, the 4-node point
# should now beat the 2-node one, and the ceiling has moved.
#SBATCH --job-name=cgpoly_scaling
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cgpoly_scaling.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cgpoly_scaling.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu JDIST_CPU=1 FESOM_CPU_COLL=gloo OMP_NUM_THREADS=8
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --ic-dir $PWD/data/ic_core2 \
   --steps 8 --warmup 2 --dt 1800 --full 1 --year 1958 --halo coloured"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
# push the degree further at 32 ranks
for k in 8 12; do
  echo "#### degree=$k npes=32 ####"
  FESOM_CG_CHEB=$k mpirun -n 32 --map-by ppr:16:node:pe=8 --bind-to core \
    "$PY" scripts/bench/bench_forward_scaling.py $C --npes 32 2>&1 \
    | grep -E "^\[bench\] mesh|^\[bench-finite\]" | tail -2
done
# THE question: does the ceiling move? degree 6 at 1, 2, 4 nodes
for n in 16 32 64; do
  echo "#### degree=6 npes=$n ($((n/16)) node(s)) ####"
  FESOM_CG_CHEB=6 mpirun -n $n --map-by ppr:16:node:pe=8 --bind-to core \
    "$PY" scripts/bench/bench_forward_scaling.py $C --npes $n 2>&1 \
    | grep -E "^\[bench\] mesh|^\[bench-finite\]" | tail -2
done
echo "=== done $(date) ==="
