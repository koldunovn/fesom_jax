#!/bin/bash
# CORE2 COMPLETE-model per-step bench on ONE CPU NODE (paper §5 CPU-execution paragraph).
# Same harness + protocol as bench_core2_full.sbatch (compile-once, warm reused executable,
# real JRA1958 + PHC IC), but JAX_PLATFORMS=cpu, single process, npes=1 (synth-serial
# partition, zero halo). XLA-CPU parallelizes within the node via its thread pool — this IS
# the JAX CPU story: one node, no multi-node CPU collectives (see paper text).
#SBATCH --job-name=bench_core2_cpu
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_core2_cpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_core2_cpu.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-128}
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --npes 1 --steps 10 \
   --warmup 2 --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958"
echo "=== node $(hostname)  $(date)  cores=${SLURM_CPUS_PER_TASK} ==="
echo "#### full (ice+kpp+gm) CPU ####"; "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 1 --kpp 1 --gm 1
echo "#### oce (base) CPU ####";        "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 0 --kpp 0 --gm 0
echo "=== done $(date) ==="
