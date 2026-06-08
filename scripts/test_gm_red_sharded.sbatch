#!/bin/bash
# S.7 part 3 — GM npes==2 gates + reductions routing, on a COMPUTE node:
#   (1) the per-kernel GM-exchange gate (gm_diagnostics owned-match — proves fer_uv/
#       slope_tapered/Ki correct) + the full GM step owned-match (loosened FCT budget),
#   (2) the reductions routing gate (sss_runoff_fluxes owned-sum+psum == single-device),
#   (3) single-device reduction byte-identity (test_sss_runoff.py — owned_mask=None ⇒ v1.0).
#SBATCH --job-name=test_gm_red
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_gm_red.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_gm_red.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname) ==="

echo "### (1) GM npes==2 gates (per-kernel gm_diagnostics + full step) ###"
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -q -p no:cacheprovider -rs \
  -k "gm and owned" -s
rc1=$?

echo "### (2) reductions routing (sss_runoff_fluxes owned-sum+psum, 4 fake-devices) ###"
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_reductions_sharded.py -q -p no:cacheprovider -rs -s
rc2=$?

echo "### (3) single-device reduction byte-identity (sss_runoff) ###"
JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_sss_runoff.py -q -p no:cacheprovider -rs
rc3=$?

echo "exit: gm=$rc1 reductions=$rc2 sss_single=$rc3"
