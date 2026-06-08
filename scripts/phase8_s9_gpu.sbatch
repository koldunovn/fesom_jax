#!/bin/bash
# S.9 — the REAL multi-GPU gate. Runs the headline S.7 forward + S.8 gradient gates on a
# real 4×A100 node (NCCL, not CPU fake-devices) — the first time the sharded model touches
# real GPUs. Uses jax.devices() (the 4 A100s); npes=2 tests use devices[:2]. NO
# --xla_force_host_platform_device_count, NO JAX_PLATFORMS=cpu. Each test runs in its OWN
# fresh process (the S.8 OOM lesson: multiple heavy grad compiles in one process accumulate).
#SBATCH --job-name=phase8_s9_gpu
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/phase8_s9_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/phase8_s9_gpu.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null || true
"$PY" -c "import jax; jax.config.update('jax_enable_x64', True); print('jax', jax.__version__, 'devices:', jax.devices())"
# S.9(a) forward: npes==1 byte-identity + the assembled KPP+GM+ice N-vs-1 on real GPUs.
# S.9(b) gradient: the OCEAN param grad (k_ver/a_ver vs dense) + the FORCED assembled grad
#   (the ice-EVP backward) on real GPUs. (npes=2 → 2 of the 4 A100s; npes=4 is a follow-up run.)
for t in \
  "test_step_sharded.py::test_serial_sharded_step_matches_dense" \
  "test_step_sharded.py::test_assembled_kpp_gm_ice_sharded_owned_matches" \
  "test_gradient_sharded.py::test_grad_ocean_param_matches_dense" \
  "test_gradient_sharded.py::test_grad_assembled_forced_backward" ; do
  echo "######## $t ########"
  "$PY" -m pytest "fesom_jax/tests/$t" -q -p no:cacheprovider -rs -s
  echo ">>>> $t exit: $?"
done
echo "=== done $(date) ==="
