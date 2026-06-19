#!/bin/bash
# A6: run ONE config-driven model segment (load restart/IC -> N steps -> write restart -> exit).
# Multi-segment campaigns chain these with scripts/chain_submit.sh (SLURM --dependency=afterok).
#
# Usage:
#   sbatch scripts/run_from_config.sbatch CONFIG.yaml [--restart-in DIR] [--restart-out DIR] [--steps N]
# Multi-node (NG5/dars) via jax.distributed: set JDIST=1 and --nodes>1 (1 proc/node x4 GPU).
#
#SBATCH --job-name=fesom_run
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/fesom_run.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/fesom_run.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# the C1 OOM lesson: disable CUDA-graph command buffers + reuse one jitted forward
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

CONFIG="${1:?usage: sbatch run_from_config.sbatch CONFIG.yaml [overrides]}"
shift

if [ "${JDIST:-0}" = "1" ]; then
  # multi-node: 1 process per node, each owning its 4 local GPUs (jax.distributed)
  srun --ntasks-per-node=1 $PY scripts/run_from_config.py "$CONFIG" "$@"
else
  $PY scripts/run_from_config.py "$CONFIG" "$@"
fi
