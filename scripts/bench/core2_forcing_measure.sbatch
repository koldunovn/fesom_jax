#!/bin/bash
#SBATCH --job-name=c2_fmeas
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_forcing_measure.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_forcing_measure.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" PYTHONUNBUFFERED=1
# single-threaded BLAS ⇒ the thread-parallelism probe measures true core scaling (1 core/thread)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "=== $(date) CORE2 forcing cost measurement ==="
srun --ntasks=1 --cpu-bind=none "$PY" scripts/bench/core2_forcing_measure.py
echo "=== done $(date) ==="
