#!/bin/bash
# GATE v2: what actually revives multi-node CPU — the collectives BACKEND, the halo
# TRANSPORT, or both? They are orthogonal knobs, so this is a 2x2 at the node boundary.
#
#   backend:   gloo (TCP, the 2026-07-13 default) vs mpi (MPItrampoline -> Levante OpenMPI)
#   transport: padded (dense all_to_all) vs coloured (K ppermute rounds, Phase 8d — never
#              tested on CPU; its message pattern is the one that should suit a slow fabric)
#
# v1 (job 26488736) established: cross-node MPI collectives WORK under JAX (2-node psum PASS),
# but srun --mpi=pmix cannot launch this OpenMPI (no SLURM PMI support) — mpirun only.
# v1's one full-model number: gloo+padded, 2 nodes x 16 procs = 2933 ms/step.
#SBATCH --job-name=gate_cpu_mpi2
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/gate_cpu_mpi2.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/gate_cpu_mpi2.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax

export MPITRAMPOLINE_LIB=/work/ab0995/a270088/mpiwrapper-ompi412/lib64/libmpiwrapper.so
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JDIST_CPU=1
export OMP_NUM_THREADS=16

POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --steps 8 --warmup 2 \
   --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958"

echo "=== nodes: $SLURM_JOB_NODELIST  $(date) ==="
echo "--- mpirun slot check ---"; mpirun -n 8 --map-by ppr:8:node:pe=16 --bind-to core hostname 2>&1 | sort | uniq -c | head

leg () {   # leg <backend> <transport> <nprocs> <label>
  echo "#### LEG $4: core2 npes=$3 backend=$1 halo=$2 ####"
  FESOM_CPU_COLL=$1 mpirun -n "$3" --map-by ppr:8:node:pe=16 --bind-to core \
      "$PY" scripts/bench/bench_forward_scaling.py $C --npes "$3" --halo "$2" 2>&1 \
      | grep -vE "^\[Gloo\]|^$" | tail -12
  echo "--- leg $4 done $(date +%T) ---"
}

# one-node baselines (the 2026-07-13 optimum topology: 8 procs x 16 cores)
leg gloo padded   8  1A
leg mpi  padded   8  1B
leg gloo coloured 8  1C

# the node boundary — the 2x2 that decides whether a CPU campaign is possible
leg gloo padded   16 2A
leg mpi  padded   16 2B
leg gloo coloured 16 2C
leg mpi  coloured 16 2D

echo "=== done $(date) ==="
