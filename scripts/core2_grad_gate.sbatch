#!/bin/bash
# Task 5.8 (GATE 5): CORE2-slice gradient gate on an A100 — the AD↔FD plateau through
# the CG (tight-forward) + the assembled-CORE2 masked-NaN probe d(loss)/d(T0) + the
# checkpointed N-step backward peak memory (CORE2 ≈ 40× pi nodes → must fit the A100-40).
# Submit:  sbatch scripts/core2_grad_gate.sbatch   (from the repo root)
#SBATCH -J fesom_core2_grad
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --mem=32G
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_grad_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_grad_gate.%j.out

set -uo pipefail
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax || exit 2

echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L || echo "(no nvidia-smi)"

echo; echo "=== GATE 5: CORE2-slice gradient (N=20) ==="
"$PY" scripts/core2_grad_gate.py --n 20 --year 1958
rc=$?

echo "=== gate exit rc=$rc ==="
exit $rc
