#!/bin/bash
# Phase 9c JM.5 — FOCUSED mEVP sharded N-vs-1 (just the 2 mEVP tests, strict u_ice/v_ice gate),
# generous time so it completes (the broad regression job walled at 50 min). Compute node, CPU
# 4 fake-devices.
#SBATCH --job-name=mevp_shard_foc
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/mevp_jm5_shard_focused.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/mevp_jm5_shard_focused.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date) — mEVP sharded N-vs-1 (strict gate, focused) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -m pytest -v -p no:cacheprovider -rA \
    fesom_jax/tests/test_step_sharded.py::test_mevp_serial_sharded_step_matches_dense \
    "fesom_jax/tests/test_step_sharded.py::test_mevp_sharded_step_owned_matches[2]"
echo "=== rc=$? ==="
