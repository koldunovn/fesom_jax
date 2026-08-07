#!/bin/bash
#SBATCH --job-name=nn_adjrepro
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -o scripts/nn_eddy/adjrepro.%j.out
#SBATCH -e scripts/nn_eddy/adjrepro.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_repro_results.jsonl
echo "=== $(hostname) $(date) — is dL/dm reproducible? identical passes, same process ==="
for N in 1 2; do
  echo "----- N=$N d, 5 identical passes -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days $N \
      --chunk-steps 48 --repeat 5 --fd-h "" --run-start-year 1958 --results "$RES" \
      --tag reproducibility
done
echo "=== done $(date) ==="
