#!/bin/bash
# CORE2 all-on per-step cost DECOMPOSITION (model-paper perf investigation). Short (1 node, 4 GPU,
# 25 min ⇒ backfill-friendly, won't delay the reserved R1 slot). Times the production kernel with one
# physics component toggled off at a time to locate the ~520 ms/step all-on cost vs the 87 ms bench.
#SBATCH --job-name=c2_decomp
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_perf_decomp.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_perf_decomp.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" PYTHONUNBUFFERED=1
# The old ~520 ms/step all-on number was partly the per-chunk-recompile bug — always
# re-run this decomposition with the executable reuse cache ON (the post-fix protocol).
export FESOM_REUSE_EXE=1
echo "=== $(date) core2 perf decomposition (dist_4/1node) ==="
srun --ntasks=1 --cpu-bind=none "$PY" scripts/bench/core2_perf_decomp.py
echo "=== done $(date) ==="
