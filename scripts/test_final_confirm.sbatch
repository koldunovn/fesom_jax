#!/bin/bash
# S.7 part 3 final confirmation on a COMPUTE node:
#   (1) the multistep scan test (range-check line fixed) — clean green,
#   (2) the OCEAN single-device suite (pi + Phase-5 CORE2) byte-identical to v1.0 (proves the
#       step.py changes — boundary_node/_red_mask/gm-kpp-ice exch threading — are dead branches).
#SBATCH --job-name=test_final
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_final.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_final.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname) ==="

echo "### (1) multistep scan test (fixed) ###"
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -q -p no:cacheprovider -rs \
  -k "multistep_scan" -s
rc1=$?

echo "### (2) OCEAN single-device byte-identity (pi + Phase-5 CORE2) ###"
JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_step_pi.py fesom_jax/tests/test_core2_step.py \
  -q -p no:cacheprovider -rs
rc2=$?

echo "exit: multistep=$rc1 ocean_single=$rc2"
