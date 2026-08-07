#!/bin/bash
# Same reproducibility probe as adjoint_repro.sbatch, but with XLA's deterministic-ops mode
# on. The unstructured FEM operators this model is built from are scatter-adds, which XLA
# implements with GPU atomics: the summation order varies run to run, so the forward is not
# bit-reproducible, and the backward amplifies that difference by ~1e9 over a single day.
# If this run comes back bit-identical across passes while the non-deterministic one does
# not, the nondeterminism is the atomics and this flag is the fix.
#SBATCH --job-name=nn_adjrepro_det
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -o scripts/nn_eddy/adjrepro_det.%j.out
#SBATCH -e scripts/nn_eddy/adjrepro_det.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true"
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_repro_results.jsonl
echo "=== $(hostname) $(date) — reproducibility WITH --xla_gpu_deterministic_ops=true ==="
echo "XLA_FLAGS=$XLA_FLAGS"
for N in 1 2; do
  echo "----- N=$N d, 5 identical passes, deterministic ops -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH-det --n-days $N \
      --chunk-steps 48 --repeat 5 --fd-h "" --run-start-year 1958 --results "$RES" \
      --tag reproducibility-deterministic
done
echo "=== done $(date) ==="
