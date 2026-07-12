#!/bin/bash
# Phase 8b B.3 STEP 1 — multi-node jax.distributed bring-up sanity on 2 GPU nodes (8 A100).
# 1 task/node, 4 GPUs/task; srun launches the per-node processes; jax.distributed.initialize()
# auto-detects SLURM. Confirms global devices + cross-process collectives before the real run.
#SBATCH --job-name=mn_sanity
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/mn_sanity.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/mn_sanity.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export GPUS_PER_NODE=4
echo "=== $(date)  nodes=$SLURM_JOB_NODELIST ntasks=$SLURM_NTASKS ==="
srun --cpu-bind=none "$PY" scripts/debug/multinode_sanity.py
echo ">>>> exit: $?  ($(date))"
