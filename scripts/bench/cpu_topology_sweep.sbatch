#!/bin/bash
# CPU TOPOLOGY SWEEP — how should a 128-core node be split into ranks, now that the halo
# transport has changed?
#
# The 8 procs x 16 cores optimum on record (2026-07-13) was measured with the PADDED
# transport, whose cost grows with rank count — which is what punished thin ranks then.
# Coloured costs K=Delta ppermute rounds instead (independent of P), so the optimum may
# have moved. Fixed hardware (2 nodes = 256 cores), vary the split:
#
#   4 procs/node x 32 cores  -> npes 8      16 procs/node x 8 cores -> npes 32
#   8 procs/node x 16 cores  -> npes 16     32 procs/node x 4 cores -> npes 64
#
# Runs on 2 nodes because coloured at 1 node x 8 procs fails (gloo collective-permute,
# reproduced twice) — legs S5-S7 characterise where on the 1-node axis that starts.
#SBATCH --job-name=cpu_topo_sweep
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cpu_topo_sweep.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cpu_topo_sweep.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JDIST_CPU=1
export FESOM_CPU_COLL=gloo

POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --steps 8 --warmup 2 \
   --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958"

echo "=== nodes: $SLURM_JOB_NODELIST  $(date) ==="

leg () {   # leg <transport> <nprocs> <procs-per-node> <cores-per-proc> <label>
  echo "#### LEG $5: core2 npes=$2 halo=$1  ($3 procs/node x $4 cores) ####"
  OMP_NUM_THREADS=$4 mpirun -n "$2" --map-by "ppr:$3:node:pe=$4" --bind-to core \
      "$PY" scripts/bench/bench_forward_scaling.py $C --npes "$2" --halo "$1" 2>&1 \
      | grep -vE "^\[Gloo\]|^$" | tail -8
  echo "--- leg $5 done $(date +%T) ---"
}

# --- the sweep: 2 nodes, 256 cores, four ways of splitting them ---
leg coloured  8  4 32 "S1 2node 4x32"
leg coloured 16  8 16 "S2 2node 8x16"
leg coloured 32 16  8 "S3 2node 16x8"
leg coloured 64 32  4 "S4 2node 32x4"
# --- padded control at the thin end (expected to lose: its cost grows with P) ---
leg padded   32 16  8 "S5 2node 16x8 padded"

# --- 1-node characterisation: where does the coloured gloo failure start? ---
leg coloured  2  2 64 "S6 1node 2x64"
leg coloured  4  4 32 "S7 1node 4x32"
leg coloured  8  8 16 "S8 1node 8x16"

echo "=== done $(date) ==="
