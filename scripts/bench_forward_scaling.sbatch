#!/bin/bash
# Phase 8b forward-scaling benchmark on real A100s — ragged vs all_gather halo, npes 2 & 4.
# Validates the multistep ragged FORWARD lowers + runs on GPU (the scan + ragged collective,
# untested by the 1-step B.0c gate), and gives the first timing comparison. CORE2 is small
# (weak absolute scaling signal) — the real numbers come from farc/dars/NG5 (B.1+, needs the
# mesh-build step). Each (npes, halo) combo in its own process.
#SBATCH --job-name=bench_fwd_scaling
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_fwd_scaling.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_fwd_scaling.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null || true
for npes in 2 4; do
  for r in 0 1; do
    echo "######## npes=$npes ragged=$r ########"
    "$PY" scripts/bench_forward_scaling.py --npes $npes --steps 100 --ragged $r
  done
done
echo "=== done $(date) ==="
