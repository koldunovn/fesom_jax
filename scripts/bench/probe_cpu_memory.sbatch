#!/bin/bash
# Where does CPU host memory actually go? dars OOMs at BOTH 8 ranks/node x 16 cores (232 GB)
# and 4 ranks/node x 32 cores (227 GB) — nearly identical per-node totals, which says the
# footprint tracks the NODE (i.e. a per-core term), not the rank count. If that is right,
# per-rank RSS = A + B x cores, and the fix for big meshes is fewer CORES per rank, not fewer
# ranks. Two single-rank legs at 16 and 128 cores separate A from B.
#
# npes=1 uses the synth-serial partition (no halo, no collectives) — this measures setup and
# the step working set only, which is exactly the thing that OOMs.
#SBATCH --job-name=probe_cpu_mem
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/probe_cpu_mem.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/probe_cpu_mem.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu

W=/work/ab0995/a270088/fesom_jax_meshes
probe () {   # probe <mesh> <meshdir> <icdir> <dt> <cores>
  echo "#### PROBE mesh=$1 cores=$5 (1 rank, npes=1) ####"
  OMP_NUM_THREADS=$5 XLA_FLAGS="--xla_force_host_platform_device_count=1" \
  /usr/bin/time -v taskset -c 0-$(( $5 - 1 )) \
    "$PY" scripts/bench/bench_forward_scaling.py --name "$1" --mesh-dir "$2" \
      --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/"$1" --ic-dir "$3" \
      --npes 1 --steps 2 --warmup 1 --dt "$4" --full 1 --year 1958 2>&1 \
    | grep -E "^\[bench\]|Maximum resident|Elapsed|rror|Killed" | tail -6
  echo "--- probe $1 cores=$5 done $(date +%T) ---"
}

echo "=== $(hostname) $(date) ==="
free -g | head -2
probe core2 "$PWD/data/mesh_core2" "$PWD/data/ic_core2" 1800 16
probe core2 "$PWD/data/mesh_core2" "$PWD/data/ic_core2" 1800 128
probe dars  "$W/mesh_dars"         "$W/mesh_dars"        180  16
probe dars  "$W/mesh_dars"         "$W/mesh_dars"        180  128
echo "=== done $(date) ==="
