#!/bin/bash
# S.8 — the AD gradient gate (FORCED: KPP + GM/Redi + prognostic ice + bulk/SSS). The
# heaviest backward in the port: it gates the ice EVP 120-subcycle jax.checkpoint'd lax.scan
# whose BACKWARD recompute re-runs the in-scan u_ice/v_ice all_gather (transpose = reverse
# exchange), the CG custom_linear_solve transpose_solve sharded, and the KPP/GM/reductions
# psum transposes — all firing. d/d(k_gm) is now active (GM on). Budget --time/--mem large.
#SBATCH --job-name=test_grad_forced
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_grad_forced.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_grad_forced.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname) ==="
# The assembled T0 grad (KPP+GM+ice EVP scan backward) is the port's heaviest adjoint —
# 2 fake devices (the test is npes=2), large --mem (OCEAN's lighter T0 grad OOM'd at 110G).
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_gradient_sharded.py -q -p no:cacheprovider -rs \
  -k "assembled" -s
echo "exit: $?"
