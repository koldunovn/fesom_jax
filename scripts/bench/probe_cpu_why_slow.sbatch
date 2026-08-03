#!/bin/bash
# WHY is CORE2 only ~5 SYPD on CPU? Separate the candidate causes instead of guessing.
#
# Established (2026-07-13 + tonight): the gap vs Fortran factors as ~2.7x kernel efficiency
# x ~2.5x parallel efficiency. Tonight localised the parallel half to INTRA-RANK thread
# scaling: one rank gains only 1.4x from 16 -> 128 cores (8686 -> 6124 ms), i.e. ~17%
# efficiency, which is why splitting the node into ranks beats fattening the thread pool.
#
# The untested multiplier is COLLECTIVE LATENCY. The SSH CG solver runs ~127 iterations per
# step, each with a halo exchange plus two dot-product all-reduces — order 400 collectives per
# step. On gloo/TCP each one costs far more than on NCCL, which would make CORE2 latency-bound
# and explain why it saturates at 2 nodes while Fortran runs 512 ranks happily.
#
# Legs, all at the campaign's best CORE2 config (2 nodes x 16 ranks x 8 cores, coloured):
#   A  baseline full model                        -> the 990 ms reference
#   B  CGPOLY (FESOM_CG_CHEB=1): ~3x fewer CG iterations. If step time falls with the
#      COLLECTIVE COUNT, latency dominates; if it barely moves, it does not.
#   C  base ocean only (no ice/KPP/GM)            -> how much is the CG/ocean core
#   D  CG iteration count, to know the multiplier rather than assume 127
#   E  raw collective latency at 32 ranks (small psum = latency, 1 MB psum = bandwidth)
#SBATCH --job-name=why_cpu_slow
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/why_cpu_slow.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/why_cpu_slow.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JDIST_CPU=1
export FESOM_CPU_COLL=gloo
export OMP_NUM_THREADS=8

POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --npes 32 --steps 8 \
   --warmup 2 --dt 1800 --ic-dir $PWD/data/ic_core2 --year 1958 --halo coloured"
RUN="mpirun -n 32 --map-by ppr:16:node:pe=8 --bind-to core"

echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="

echo "#### A baseline: full model ####"
$RUN "$PY" scripts/bench/bench_forward_scaling.py $C --full 1 2>&1 \
  | grep -E "^\[bench\]|rror" | tail -3

echo "#### B CGPOLY (3x fewer CG iterations) ####"
FESOM_CG_CHEB=1 $RUN "$PY" scripts/bench/bench_forward_scaling.py $C --full 1 2>&1 \
  | grep -E "^\[bench\]|cheb|rror" | tail -4

echo "#### C base ocean only (no ice/KPP/GM) ####"
$RUN "$PY" scripts/bench/bench_forward_scaling.py $C --full 0 --ice 0 --kpp 0 --gm 0 2>&1 \
  | grep -E "^\[bench\]|rror" | tail -3

echo "#### D CG iteration count (2 steps) ####"
FESOM_CG_ITERS=1 $RUN "$PY" scripts/bench/bench_forward_scaling.py $C --full 1 --steps 2 \
  --warmup 1 2>&1 | grep -E "cg-iters|^\[bench\]" | head -8

echo "#### E raw collective latency at 32 ranks ####"
$RUN "$PY" scripts/bench/probe_collective_latency.py 2>&1 | grep -E "psum|halo|rror" | head -8

echo "=== done $(date) ==="
