#!/bin/bash
# Phase 9c JM.5 — mEVP sharded N-vs-1 (CPU 4 fake-devices) on a COMPUTE node. The full-model
# shard_map compiles are too heavy for the login node (the suite-sbatch sharding group takes
# ~30 min). Runs the whole test_step_sharded.py (my new mEVP serial + owned-match N-vs-1 tests
# + the existing ice/gm/kpp/zstar N-vs-1 gates — a regression check on the JM.5 edit).
#SBATCH --job-name=mevp_jm5_shard
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:50:00
#SBATCH -o /home/a/a270088/port_jax/scripts/mevp_jm5_shard.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/mevp_jm5_shard.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== node $(hostname)  $(date) — mEVP sharded N-vs-1 (4 fake-devices) ==="
# the two new mEVP tests first (the JM.5 gate), then the whole file (regression).
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -v -p no:cacheprovider \
    -k "mevp or (ice and sharded) or serial_sharded" -rA
echo "=== mevp-focused rc=$? ==="
echo ""
echo "=== full test_step_sharded.py (regression) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -q -p no:cacheprovider
echo "=== full-file rc=$? ==="
