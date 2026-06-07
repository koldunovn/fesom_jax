#!/bin/bash
# S.6 fixture capture — a realistic CORE2 ssh_rhs from the real assembled KPP+GM+ice
# step (2 eager steps, dt=1800) + the single-device CG iteration-count / margin report.
# Cheap (~2 min) CPU job on a compute node (the eager assembled step needs RAM).
#SBATCH --job-name=capture_ssh_rhs
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/capture_core2_ssh_rhs.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/capture_core2_ssh_rhs.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== node $(hostname)  $(date) ==="
"$PY" scripts/capture_core2_ssh_rhs.py
echo "exit: $?"
