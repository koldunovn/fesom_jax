#!/bin/bash
# JT.0 targeted sharded check: confirm the new prognostic State.tke field flows through
# the sharded N-vs-1 path (partition_state + fold/unfold + reconstruction) — the generic
# dataclasses.fields(State) loops in test_step_sharded.py now also compare tke (0==0 on the
# tke_cfg=None path). Selects ONLY the step N-vs-1 tests (serial + owned-match + reconstruct),
# skipping the slow test_gradient_sharded group that times out the full suite. Debug QOS.
#SBATCH --job-name=tke_jt0_shard
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/tke_jt0_shard.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/tke_jt0_shard.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== node $(hostname)  $(date) ==="
echo "=== test_step_sharded N-vs-1 (tke field-loop coverage; gradient tests excluded) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  $PY -m pytest fesom_jax/tests/test_step_sharded.py -v -p no:cacheprovider \
  -k "serial or reconstruction or owned_matches or teacher_forced"
RC=$?
echo ""
echo "=== exit $RC  $(date) ==="
exit $RC
