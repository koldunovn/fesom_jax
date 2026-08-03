#!/bin/bash
# CGPOLY DEGREE sweep on CPU — trade the expensive collective for the cheap one.
#
# In-graph costs at scale (job 26493370): a scalar psum grows with rank count, while a
# ppermute round stays flat. The CG pays 2 psums + 2 halos per iteration, so on CPU the
# psums are the expensive part. A degree-k Chebyshev preconditioner does k SpMV+halo per
# apply and NO dot products, cutting the ITERATION count — i.e. it trades psums (dear) for
# halo rounds (cheap). Degree 1 already gave -15.6%; on CPU the optimum should sit HIGHER
# than on GPU, where the balance is the other way round.
#SBATCH --job-name=cgpoly_degree
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cgpoly_degree.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cgpoly_degree.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu JDIST_CPU=1 FESOM_CPU_COLL=gloo OMP_NUM_THREADS=8
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --ic-dir $PWD/data/ic_core2 \
   --npes 32 --steps 8 --warmup 2 --dt 1800 --full 1 --year 1958 --halo coloured"
RUN="mpirun -n 32 --map-by ppr:16:node:pe=8 --bind-to core"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
for k in 0 1 2 3 4 6; do
  echo "#### CGPOLY degree=$k ####"
  FESOM_CG_CHEB=$k $RUN "$PY" scripts/bench/bench_forward_scaling.py $C 2>&1 \
    | grep -E "^\[bench\] mesh|^\[bench-finite\]|^\[bench-cheb\]|rror|Abort" | tail -3
done
# iteration counts (2 steps each) — the mechanism, not just the outcome
for k in 0 1 2 4; do
  echo "#### iterations at degree=$k ####"
  FESOM_CG_CHEB=$k FESOM_CG_ITERS=1 $RUN "$PY" scripts/bench/bench_forward_scaling.py $C \
    --steps 2 --warmup 1 2>&1 | grep -E "cg-iters" | head -2
done
echo "=== done $(date) ==="
