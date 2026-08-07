#!/bin/bash
# The adjoint-knob tests, on CPU. Fast group only — pi mesh and the closure called directly.
#
# ⚠️ Output is NOT piped. Piping pytest through `tail` buffers everything until it exits, so a
# job that hits its time limit prints NOTHING and you learn only that it was too slow. That
# cost one 30-minute job here. Stream it, and let `tail` be the reader's problem afterwards.
#SBATCH --job-name=nn_adjtests
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00
#SBATCH -o scripts/nn_eddy/adjtests.%j.out
#SBATCH -e scripts/nn_eddy/adjtests.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export JAX_PLATFORMS=cpu
echo "=== $(hostname) $(date) — adjoint-knob tests (CPU group) ==="
"$PY" -m pytest -v -p no:randomly \
  fesom_jax/tests/test_adjoint_stab.py \
  fesom_jax/tests/test_adjoint_config.py \
  fesom_jax/tests/test_adjoint_config_tke.py \
  fesom_jax/tests/test_gradient.py
echo "=== done $(date) rc=$? ==="
