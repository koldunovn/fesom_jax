#!/bin/bash
# Albedo: the sharded REcoM gradient (scripts/recom/core2_bgc_adj.py) on the four A100 of one gpu node.
#   sbatch scripts/albedo/run_core2_bgc_adj.sbatch --config ... --restart ... --dist-dir ... --days 1 --out ...
#SBATCH --job-name=fj_bgcadj
#SBATCH -p gpu
#SBATCH --gpus=a100:4
#SBATCH -A clidyn.clidyn
#SBATCH --qos=12h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}"
source scripts/albedo/env.sh
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
echo "=== $(hostname) $(date) $(git rev-parse --short HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/recom/core2_bgc_adj.py "$@"
echo "=== exit $? $(date) ==="
