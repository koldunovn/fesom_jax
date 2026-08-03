#!/bin/bash
# Does the GPU's topology — ONE process holding MANY devices — fix the CPU memory wall?
#
# On CPU the device IS the host, so our 16-processes-per-node launch makes 16 copies of the
# global state in one node's RAM (integrate_sharded.py:54: "each process builds the full global
# host array and places only its local shards"). On GPU one process drives all 4 GPUs, so a node
# pays for ONE copy — which is why an 80 GB A100 is never the constraint.
#
# In-process CPU devices were measured 1.7x slower on 2026-07-13 — but with the PADDED
# transport, before coloured existed, i.e. the same stale-verdict trap as the "single-node"
# claim. Retest with coloured. If it holds up, dars escapes its 2-fat-ranks/node cage.
#SBATCH --job-name=inproc_dev
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/inproc_dev.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/inproc_dev.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export FESOM_CPU_COLL=gloo
export OMP_NUM_THREADS=8
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
W=/work/ab0995/a270088/fesom_jax_meshes

echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
free -g | head -2

echo "#### A core2 npes=16: ONE process, 16 in-process devices, 1 node ####"
XLA_FLAGS="--xla_force_host_platform_device_count=16" \
  mpirun -n 1 --map-by ppr:1:node:pe=128 --bind-to none \
  "$PY" scripts/bench/bench_forward_scaling.py --name core2 --mesh-dir $PWD/data/mesh_core2 \
  --dist-dir $POOL/core2 --ic-dir $PWD/data/ic_core2 --npes 16 --steps 8 --warmup 2 --dt 1800 \
  --full 1 --year 1958 --halo coloured 2>&1 | grep -E "^\[bench|rror|Abort" | tail -4
free -g | head -2

echo "#### B core2 npes=32: 1 process/node x 16 devices, 2 nodes ####"
JDIST=1 GPUS_PER_NODE=16 XLA_FLAGS="--xla_force_host_platform_device_count=16" \
  mpirun -n 2 --map-by ppr:1:node:pe=128 --bind-to none \
  "$PY" scripts/bench/bench_forward_scaling.py --name core2 --mesh-dir $PWD/data/mesh_core2 \
  --dist-dir $POOL/core2 --ic-dir $PWD/data/ic_core2 --npes 32 --steps 8 --warmup 2 --dt 1800 \
  --full 1 --year 1958 --halo coloured 2>&1 | grep -E "^\[bench|rror|Abort" | tail -4
free -g | head -2

echo "#### C dars npes=32: 1 process/node x 16 devices, 2 nodes (vs 16 nodes x 2 ranks) ####"
JDIST=1 GPUS_PER_NODE=16 XLA_FLAGS="--xla_force_host_platform_device_count=16" \
  mpirun -n 2 --map-by ppr:1:node:pe=128 --bind-to none \
  "$PY" scripts/bench/bench_forward_scaling.py --name dars --mesh-dir $W/mesh_dars \
  --dist-dir $POOL/dars --ic-dir $W/mesh_dars --npes 32 --steps 5 --warmup 2 --dt 180 \
  --full 1 --year 1958 --halo coloured 2>&1 | grep -E "^\[bench|rror|Abort|Killed" | tail -4
free -g | head -2
echo "=== done $(date) ==="
