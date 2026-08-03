#!/bin/bash
# GATE v3: does the coloured transport keep scaling past 2 CPU nodes — and is it reliable?
#
# v2 (job 26488829) found the transport, not the collectives backend, was the whole story:
#   1 node  x  8 procs  gloo+padded   1258.8 ms   (baseline)
#   2 nodes x 16 procs  gloo+padded   2889.0 ms   (the anti-scaling on record)
#   2 nodes x 16 procs  gloo+COLOURED 1004.5 ms   <- faster than the 1-node baseline
#   every mpi-backend leg              FAILED     (XLA MpiCollectives is single-thread-only)
#   1 node  x  8 procs  gloo+coloured  FAILED     (gloo collective-permute, connection reset)
#
# So: (a) does 1004 ms reproduce, (b) does 4 nodes / 32 procs go faster still, (c) was the
# 1-node coloured failure flaky or real? Two repeats of each load-bearing point.
#SBATCH --job-name=gate_cpu_col4
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/gate_cpu_col4.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/gate_cpu_col4.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JDIST_CPU=1
export FESOM_CPU_COLL=gloo
export OMP_NUM_THREADS=16

POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --steps 8 --warmup 2 \
   --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958"

echo "=== nodes: $SLURM_JOB_NODELIST  $(date) ==="

leg () {   # leg <transport> <nprocs> <label>
  echo "#### LEG $3: core2 npes=$2 halo=$1 ####"
  mpirun -n "$2" --map-by ppr:8:node:pe=16 --bind-to core \
      "$PY" scripts/bench/bench_forward_scaling.py $C --npes "$2" --halo "$1" 2>&1 \
      | grep -vE "^\[Gloo\]|^$" | tail -10
  echo "--- leg $3 done $(date +%T) ---"
}

leg coloured  8 "A1 1node-8"        # was the v2 failure — flaky or real?
leg coloured 16 "B1 2node-16"       # repeat of the 1004 ms point
leg coloured 32 "C1 4node-32"       # does it keep scaling?
leg padded   32 "D  4node-32-pad"   # padded control at 4 nodes
leg coloured 32 "C2 4node-32"       # repeat
leg coloured  8 "A2 1node-8"        # repeat

echo "=== done $(date) ==="
