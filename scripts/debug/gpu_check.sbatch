#!/bin/bash
# Task 0.1 GPU verification: confirm JAX sees an A100 and runs a float64 jit.
# Submit:  sbatch scripts/debug/gpu_check.sbatch    (from the repo root)
#SBATCH -J fesom_gpu_check
#SBATCH -A ab0995_gpu
#SBATCH -p gpu-devel
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH --mem=16G
#SBATCH -o /home/a/a270088/port_jax/scripts/gpu_check.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/gpu_check.%j.out

set -uo pipefail
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python

echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L || echo "(no nvidia-smi)"
echo
echo "=== jax smoke test ==="
"$PY" /home/a/a270088/port_jax/scripts/debug/gpu_smoke.py
rc=$?
echo "=== exit rc=$rc ==="
exit $rc
