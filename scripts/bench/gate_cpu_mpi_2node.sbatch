#!/bin/bash
# GATE: is multi-node CPU alive via MPI collectives (MPItrampoline + MPIwrapper)?
#
# The 2026-07-13 CPU campaign found JAX-CPU effectively SINGLE-NODE on jaxlib 0.10.1:
# gloo-over-TCP made 2 nodes 6x SLOWER than the same rank count on one node (12.9 s/step),
# and the fake-device variants crashed. jaxlib 0.10.1 does ship xla::cpu::MpiCollectives
# behind MPItrampoline (verified in libjax_common.so), so FESOM_CPU_COLL=mpi routes the
# collectives through Levante's OpenMPI instead. This job answers ONE question:
#   does a rank crossing the node boundary still cost 6x, or does it now scale?
#
# Legs: (0) 2-node psum correctness, both launchers  (1) 1 node x 8 procs, MPI
#       (2) 2 nodes x 16 procs, MPI  (3) 2 nodes x 16 procs, gloo  <- the A/B
#SBATCH --job-name=gate_cpu_mpi
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/gate_cpu_mpi.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/gate_cpu_mpi.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
SP=unused
cd /home/a/a270088/port_jax

export MPITRAMPOLINE_LIB=/work/ab0995/a270088/mpiwrapper-ompi412/lib64/libmpiwrapper.so
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JDIST_CPU=1
export OMP_NUM_THREADS=16

POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --steps 8 --warmup 2 \
   --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958 --halo padded"

echo "=== nodes: $SLURM_JOB_NODELIST  $(date) ==="

# ---- leg 0: cross-node correctness, 1 rank per node, both launchers -------------
echo "#### GATE0a psum 2 nodes via srun --mpi=pmix ####"
FESOM_CPU_COLL=mpi srun --mpi=pmix -N 2 -n 2 --ntasks-per-node=1 -c 16 \
    "$PY" scripts/bench/gate_cpu_collectives.py 2>&1 | grep -E "procs|psum|gsum|GATE|rror" | head -8
echo "srun-gate-exit=$?"

echo "#### GATE0b psum 2 nodes via mpirun ####"
FESOM_CPU_COLL=mpi mpirun -n 2 --map-by ppr:1:node:pe=16 --bind-to core \
    "$PY" scripts/bench/gate_cpu_collectives.py 2>&1 | grep -E "procs|psum|gsum|GATE|rror" | head -8
echo "mpirun-gate-exit=$?"

# ---- leg 1: CORE2, 1 node x 8 procs x 16 cores, MPI collectives (the baseline) --
echo "#### LEG1 core2 1 node x 8 procs, MPI ####"
FESOM_CPU_COLL=mpi srun --mpi=pmix -N 1 -n 8 --ntasks-per-node=8 -c 16 \
    "$PY" scripts/bench/bench_forward_scaling.py $C --npes 8 2>&1 | tail -20

# ---- leg 2: CORE2, 2 nodes x 16 procs x 16 cores, MPI collectives (THE number) --
echo "#### LEG2 core2 2 nodes x 16 procs, MPI ####"
FESOM_CPU_COLL=mpi srun --mpi=pmix -N 2 -n 16 --ntasks-per-node=8 -c 16 \
    "$PY" scripts/bench/bench_forward_scaling.py $C --npes 16 2>&1 | tail -20

# ---- leg 3: same, gloo — the A/B against the 2026-07-13 dead end ----------------
echo "#### LEG3 core2 2 nodes x 16 procs, GLOO (control) ####"
FESOM_CPU_COLL=gloo srun --mpi=pmix -N 2 -n 16 --ntasks-per-node=8 -c 16 \
    "$PY" scripts/bench/bench_forward_scaling.py $C --npes 16 2>&1 | tail -20

echo "=== done $(date) ==="
