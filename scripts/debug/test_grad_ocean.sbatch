#!/bin/bash
# S.8 — the AD gradient gate (OCEAN, no forcing). Each grad test runs in its OWN fresh
# process (explicit ::name, NOT a -k expression — `-k "a or b"` over-matched all 5, and
# running them in one process accumulates compiled-executable memory → OOM). Per-process
# isolation also localizes any failure. The grad compile is heavier than the forward.
#SBATCH --job-name=test_grad_ocean
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_grad_ocean.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_grad_ocean.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
T=fesom_jax/tests/test_gradient_sharded.py
echo "=== node $(hostname) ==="
for name in test_grad_ocean_param_matches_dense test_grad_ocean_fd_self_consistent \
            test_grad_multistep_scan_backward test_grad_ocean_ic_field_and_masked_nan; do
  echo "######## $name ########"
  XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
    "$PY" -m pytest "$T::$name" -q -p no:cacheprovider -rs -s
  echo ">>>> $name exit: $?"
done
