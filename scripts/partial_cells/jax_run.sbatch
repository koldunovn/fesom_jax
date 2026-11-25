#!/bin/bash
# One JAX CORE2 run of the partial-cells validation (plan Task 7/8) on one A100: cold start from IC, N steps,
# monthly-mean output (a run shorter than a month writes the partial period: a 48-step run = the day-1 mean).
#   cd <worktree> && sbatch --export=ALL,CFG=experiments/partial_cells/core2_zstar_pc.yaml,STEPS=48,OUT=<dir>,IC=<ic dir> \
#        scripts/partial_cells/jax_run.sbatch
#SBATCH --job-name=pc_jax
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/partial_cells/jax/logs/run.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
mkdir -p "$OUT"
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) CFG=$CFG STEPS=$STEPS OUT=$OUT IC=$IC ==="
$PY -c "import fesom_jax; print('fesom_jax from', fesom_jax.__file__)"
$PY scripts/run_from_config.py "$CFG" --steps "$STEPS" --ic-dir "$IC" --year 1958 \
    --monthly-out "$OUT/monthly" --restart-out "$OUT/restart" --progress
echo "=== rc=$? $(date) ==="
