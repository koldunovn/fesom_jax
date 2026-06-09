#!/bin/bash
# Isolation repro for the JAX ragged_all_to_all autodiff bug (docs/JAX_RAGGED_A2A_BUG.md):
# does the BARE lax.ragged_all_to_all violate the adjoint identity (→ JAX bug) or only our
# composition (→ our side)? 1 GPU node (4 A100), npes 2 & 4. 30-min walltime for backfill.
#SBATCH --job-name=ragged_adjoint
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/ragged_adjoint.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/ragged_adjoint.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== node $(hostname) $(date) ==="
nvidia-smi -L 2>/dev/null | head -4 || true
"$PY" scripts/ragged_a2a_adjoint_repro.py
echo ">>>> exit: $? ($(date))"
