#!/bin/bash
# Verify the SHARDED all_gather adjoint for the TKE + NN-of-TKE path == the dense gradient, on 2 real
# A100s (single node ⇒ JAX sees both GPUs, no jax.distributed needed). This is the gate for the
# multi-GPU NN twin: test_gradient_sharded proves all_gather==dense for PP/KPP/GM ocean params but
# NEVER for tke_cfg+the NN leaf. use_ragged=False (the autodiff-correct halo). Fast on GPU (~min).
#SBATCH --job-name=verify_shard_tke
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/verify_sharded_tke_grad.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/verify_sharded_tke_grad.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.92}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

echo "=== $(hostname) $(date)  sharded TKE+NN adjoint verify on 2 A100 ==="
nvidia-smi -L 2>/dev/null || true
"$PY" scripts/verify_sharded_tke_grad.py
echo "=== done $(date) (exit $?) ==="
