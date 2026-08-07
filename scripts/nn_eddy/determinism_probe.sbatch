#!/bin/bash
# Which XLA setting, if any, makes this model bit-reproducible on GPU, and what does it cost?
# Four settings, cheapest first. Short forward (6 steps) so the strict flags still compile.
#SBATCH --job-name=nn_detprobe
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/detprobe.%j.out
#SBATCH -e scripts/nn_eddy/detprobe.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
STEPS=${STEPS:-6}
probe () {
  echo; echo "##### $1 #####"
  ( export XLA_FLAGS="$2"; ${3:+export $3}; timeout 2400 \
    "$PY" scripts/nn_eddy/determinism_probe.py --snap "$SNAP" --steps "$STEPS" --runs 3 )
  echo "  (exit $?)"
}
echo "=== $(hostname) $(date) — bit-reproducibility probe, $STEPS forward steps ==="
probe "default GPU"                 "--xla_gpu_enable_command_buffer="
probe "deterministic reductions"    "--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_reductions=true"
probe "deterministic ops (strict)"  "--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true"
probe "CPU reference"               ""  "JAX_PLATFORMS=cpu"
echo; echo "=== done $(date) ==="
