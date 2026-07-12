#!/bin/bash
# Phase 9b JT.5 — TKE sharded N-vs-1 (CPU ×4 fake-devices). Runs ONLY the TKE sharded tests
# (serial byte-id + npes=2 owned-match), isolating them from the slow test_gradient_sharded
# group that wall-times the full suite. The owned-match Av is the direct proof the internal
# node-tke_Av exchange (_wire_kv_av) is correct; the generic field loop covers the new tke field.
#SBATCH --job-name=tke_jt5_shard
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/tke_jt5_shard.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/tke_jt5_shard.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname)  $(date) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -v -p no:cacheprovider \
  -k "tke_serial_sharded or tke_sharded_step_owned"
echo "exit: $?"
