#!/bin/bash
# Task 3.1 GPU gate: N=200 backward-pass memory sanity (checkpointed lax.scan).
# Submit:  sbatch scripts/debug/phase3_grad_memory.sbatch    (from the repo root)
#SBATCH -J fesom_phase3_mem
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH --mem=32G
#SBATCH -o /home/a/a270088/port_jax/scripts/phase3_grad_memory.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/phase3_grad_memory.%j.out

set -uo pipefail
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax || exit 2

echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L || echo "(no nvidia-smi)"

echo; echo "=== GATE: N=200 backward WITH checkpointing (must fit + finite) ==="
"$PY" scripts/debug/phase3_grad_memory.py --n 200 --checkpoint
rc=$?

echo; echo "=== reference: N=200 backward WITHOUT checkpointing (expect more mem / OOM) ==="
"$PY" scripts/debug/phase3_grad_memory.py --n 200 --no-checkpoint || true

echo "=== gate exit rc=$rc ==="
exit $rc
