#!/bin/bash
# GATE + A/B for FESOM_SSH_HALO — running the CG's 2-D node halo on a DIFFERENT transport
# from the 3-D fields.
#
# Rationale: the transports were ranked on the 3-D exchange, which is bandwidth-bound and
# favours coloured's K neighbour rounds. The CG halo is a 2-D field (~32 KB/rank on CORE2)
# exchanged ~2x per iteration x ~127 iterations, so it is LATENCY-bound — and coloured pays K
# sequential ppermutes (6+ on CORE2) where padded pays one all_to_all. If the halo is a real
# share of the step, swapping just this one exchange should show up plainly.
#
# CORRECTNESS FIRST: the transports are forward bit-exact, so the three modes must agree
# array-for-array. A perf number from a mis-wired map is worthless.
#SBATCH --job-name=ssh_halo_ab
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/ssh_halo_ab.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/ssh_halo_ab.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
OUT=/scratch/a/a270088/ssh_halo_gate
mkdir -p $OUT

echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="

# ---------- part 1: bit-exactness, single process x 4 in-process devices ----------
# npes=4 (npes=8 is the one CORE2 partition the coloured/gloo defect kills).
G="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --ic-dir $PWD/data/ic_core2 \
   --npes 4 --steps 3 --warmup 1 --dt 1800 --full 1 --year 1958 --halo coloured"
for mode in default padded allgather; do
  echo "#### GATE $mode ####"
  env_set=""; [ "$mode" != "default" ] && env_set="FESOM_SSH_HALO=$mode"
  env $env_set OMP_NUM_THREADS=32 XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    "$PY" scripts/bench/bench_forward_scaling.py $G --state-out $OUT/$mode.npz 2>&1 \
    | grep -E "^\[bench" | tail -3
done
"$PY" - <<EOF
import numpy as np
ref = np.load("$OUT/default.npz")
for m in ("padded", "allgather"):
    got = np.load("$OUT/%s.npz" % m)
    same = all(np.array_equal(ref[k], got[k]) for k in ref.files)
    worst = max(float(np.max(np.abs(ref[k] - got[k]))) for k in ref.files)
    print("[gate] ssh_halo=%-9s bit-identical=%s  max|diff|=%.3e" % (m, same, worst), flush=True)
EOF

# ---------- part 2: the A/B that matters, 2 nodes x 16 ranks (the campaign best) ----------
export JDIST_CPU=1 FESOM_CPU_COLL=gloo OMP_NUM_THREADS=8
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --ic-dir $PWD/data/ic_core2 \
   --npes 32 --steps 8 --warmup 2 --dt 1800 --full 1 --year 1958 --halo coloured"
RUN="mpirun -n 32 --map-by ppr:16:node:pe=8 --bind-to core"

for mode in coloured padded allgather; do
  echo "#### A/B ssh_halo=$mode (main halo coloured) ####"
  FESOM_SSH_HALO=$mode $RUN "$PY" scripts/bench/bench_forward_scaling.py $C 2>&1 \
    | grep -E "^\[bench\]|^\[bench-finite\]|rror|Abort" | tail -3
done

echo "#### A/B ssh_halo=padded + CGPOLY (stack the two levers) ####"
FESOM_SSH_HALO=padded FESOM_CG_CHEB=1 $RUN "$PY" scripts/bench/bench_forward_scaling.py $C 2>&1 \
  | grep -E "^\[bench\]|^\[bench-finite\]|rror|Abort" | tail -3

echo "=== done $(date) ==="
