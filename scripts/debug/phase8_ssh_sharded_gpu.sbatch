#!/bin/bash
# S.6 EARLY GPU confirmation (the formal multi-GPU gate is S.9): run the distributed
# SSH CG test on a REAL 4×A100 node — confirms the collective-inside-while_loop-inside-
# custom_linear_solve path lowers + matches on real NCCL (not just CPU fake-devices).
# Uses jax.devices() (4 real GPUs) — NO --xla_force_host_platform_device_count.
#SBATCH --job-name=phase8_ssh_gpu
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/phase8_ssh_sharded_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/phase8_ssh_sharded_gpu.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null || true
"$PY" -c "import jax; print('jax devices:', jax.devices())"
# real GPUs: the test reads jax.devices() (4 A100s) — npes=2 and npes=4 both run.
"$PY" -m pytest fesom_jax/tests/test_ssh_sharded.py -q -p no:cacheprovider -rs
echo "exit: $?"
