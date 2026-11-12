#!/bin/bash
# First job on Levante's A100 partition: the GPU env check and a cross-machine gate in one. JAX must see the GPU, then
# the 1-day pi gradient of surface Chl from the 5-year spin-up (pool restarts/pi_spin5) runs with adjoint against
# finite differences. Compare its |dJ/dx|, dJ/dlog p and the adjoint/FD ratios with the same job on Albedo
# (job 49000368, bgc_gradient_horizon/levante_gate_albedo; plan 20261105 D3: the forward noise differs by machine).
#   sbatch -A <project>_gpu scripts/levante/check_gpu.sbatch          (from the checkout)
#SBATCH --job-name=fj_check_gpu
#SBATCH --partition=gpu
#SBATCH --account=bk1415_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH -o /work/bk1415/a270044/fesom_jax_pool/runs/slurm/%x.%j.out
set -u
WT=${WT:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$WT/scripts/levante/env_gpu.sh"
cd "$WT" || exit 2
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} $(git rev-parse --short HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -c "import jax, fesom_jax; print('jax', jax.__version__, jax.devices()); assert jax.devices()[0].platform == 'gpu'" || exit 3
"$PY" scripts/recom/bgc_gradient_horizon.py --config configs/levante/pi_recom_spectral_spin.yaml --year 1958 \
    --restart "$POOL/restarts/pi_spin5" --days 1 --cost chl --repeats 2 --out "$RUNS/horizon/levante_gate"
rc=$?
echo "=== exit $rc $(date) ==="
exit $rc
