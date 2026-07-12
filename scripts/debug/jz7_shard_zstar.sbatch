#!/bin/bash
# JZ.7 — sharded N-vs-1 with ale_cfg ON (zstar). The full 4-config CORE2 model (KPP+GM+ice+
# zstar) under shard_map: npes=1 byte-id vs dense + npes=2 owned-match. Needs CPU fake-devices
# (--xla_force_host_platform_device_count=4). WARM hbar seed ⇒ the live geometry is stretched.
# Usage: sbatch scripts/debug/jz7_shard_zstar.sbatch
#SBATCH --job-name=jz7_shard
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/jz7_shard.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/jz7_shard.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== node $(hostname)  $(date) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    $PY -m pytest fesom_jax/tests/test_step_sharded.py \
    -k "zstar_serial_sharded or zstar_assembled_sharded" \
    -q -p no:cacheprovider -s
RC=$?
echo "=== JZ.7 shard-zstar gate rc=$RC ==="
[ $RC -eq 0 ] && echo "JZ.7-SHARD GREEN" || echo "JZ.7-SHARD FAILURE (see above)"
