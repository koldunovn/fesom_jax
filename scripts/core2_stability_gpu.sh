#!/bin/bash
# Task 5.7 — JAX CORE2 forward stability run on an A100 (1-day + multi-day).
# Runs the assembled CORE2 model (PHC IC + JRA55 bulk + SSS/runoff + shortwave +
# the static ice mask) jitted, monitoring per-step + per-day stability (NaN, SST
# range, |SSH|, max|vel|, the Aleutian watch node). 1728 steps = 10 model days at
# dt=500. Submit:  sbatch scripts/core2_stability_gpu.sh   (from the repo root)
#SBATCH -J fesom_core2_stab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --mem=32G
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_stability_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_stability_gpu.%j.out

set -uo pipefail
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax || exit 2

echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L || echo "(no nvidia-smi)"

echo; echo "=== JAX CORE2 stability: 10 model days (1728 steps, dt=500) ==="
# --every 36 aligns the JAX report steps with the C arbiter's FESOM_PRINT_EVERY=36
# monitor lines (steps 36/72/108/144...) for a direct trajectory comparison.
"$PY" scripts/core2_stability_run.py --steps 1728 --year 1958 --every 36
rc=$?
echo "=== gate exit rc=$rc ==="
exit $rc
