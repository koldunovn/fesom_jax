#!/bin/bash
# Is the coloured/gloo collective-permute defect DETERMINISTIC or PROBABILISTIC?
#
# §3b of the findings claims "structural, per (mesh, npes) — not flaky", on the strength of
# core2 npes=8 failing 4/4, farc npes=32 3/3, dars npes=64 2/2. But farc npes=128 PASSED at
# job 26489019 (6205 ms) and FAILED at job 26494414 with only the CG preconditioner changed.
# One of those claims is wrong. Five repeats of the same config settle it.
#SBATCH --job-name=gloo_defect_rate
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/gloo_defect_rate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/gloo_defect_rate.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu JDIST_CPU=1 FESOM_CPU_COLL=gloo OMP_NUM_THREADS=16
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
W=/work/ab0995/a270088/fesom_jax_meshes
C="--name farc --mesh-dir $W/mesh_farc --dist-dir $POOL/farc --ic-dir $W/mesh_farc \
   --npes 128 --steps 5 --warmup 2 --dt 900 --full 1 --year 1958 --halo coloured"
RUN="mpirun -n 128 --map-by ppr:8:node:pe=16 --bind-to core"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
for rep in 1 2 3; do
  echo "#### farc npes=128 degree=6 rep $rep ####"
  FESOM_CG_CHEB=6 $RUN "$PY" scripts/bench/bench_forward_scaling.py $C 2>&1 \
    | grep -E "^\[bench\] mesh|Gloo collective permute failed" | head -2
done
for rep in 1 2; do
  echo "#### farc npes=128 degree=0 rep $rep ####"
  $RUN "$PY" scripts/bench/bench_forward_scaling.py $C 2>&1 \
    | grep -E "^\[bench\] mesh|Gloo collective permute failed" | head -2
done
echo "=== done $(date) ==="
