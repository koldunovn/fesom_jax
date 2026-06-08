#!/bin/bash
# S.7 part 3.3 (re-gate, FIXED) + 3.5 — ice forced-path (a_ice exchanged before the
# stress_surf gather) + the multi-step scan / teacher-forced gates, on a COMPUTE node:
#   (1) ice forced sharded gates (npes==1 byte-id + npes==2 owned: now uv clean) + the
#       multi-step scan (lowers/finite) + teacher-forced per-step (clean except FCT),
#   (2) single-device byte-identity (ice_evp / ice_adv / ice_step exch=None).
#SBATCH --job-name=test_ice_ms
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=90G
#SBATCH --time=00:55:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_ice_ms.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_ice_ms.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname) ==="

echo "### (1) ice (fixed) + multi-step gates (4 fake-devices) ###"
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -q -p no:cacheprovider -rs \
  -k "(ice and not assembled) or multistep" -s
rc1=$?

echo "### (2) single-device byte-identity (ice modules) ###"
JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_ice_evp.py fesom_jax/tests/test_ice_adv.py \
  fesom_jax/tests/test_ice_step.py -q -p no:cacheprovider -rs
rc2=$?

echo "exit: ice_ms=$rc1 single_device=$rc2"
