#!/bin/bash
# S.7 part 3.2 — KPP forced-path exchange gate on a COMPUTE node:
#   (1) the forced KPP sharded gates (npes==1 byte-id + npes==2 owned: Kv/Av at the clean
#       floor prove the blmc-smoother per-sweep + viscA exchanges; also exercises reductions),
#   (2) single-device byte-identity of the eos.smooth_nod3D (exch=None) + KPP changes.
#SBATCH --job-name=test_kpp_shard
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_kpp_shard.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_kpp_shard.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname) ==="

echo "### (1) KPP forced sharded gates (4 fake-devices) ###"
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -q -p no:cacheprovider -rs -k "kpp" -s
rc1=$?

echo "### (2) single-device byte-identity (eos smooth + KPP assembled step) ###"
JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_eos.py fesom_jax/tests/test_kpp_step.py \
  -q -p no:cacheprovider -rs
rc2=$?

echo "exit: kpp_sharded=$rc1 single_device=$rc2"
