#!/bin/bash
# The ice configuration we would actually ship, at a real window length.
#
# Everything measured so far used the ice switched off entirely in the backward. The 8-step
# dig showed that is heavier-handed than necessary: only the mEVP momentum solve needs to be
# frozen (3.4e-05 vs 2.0e+39), while ice thermodynamics, advection, cut-off and the
# ocean-to-ice coupling all differentiate fine. That intermediate has never been run past 8
# steps, so this measures it at 5 and 10 days against the fully-frozen-ice reference.
#
# What we want to see: the same growth rate and the same reproducibility as fully-frozen ice,
# with a gradient that DIFFERS from it — the difference is the recovered ice pathway, which is
# the whole point. Identical numbers would mean the ice contributes nothing and the extra
# differentiation is not worth its cost.
#SBATCH --job-name=nn_icerheo
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o scripts/nn_eddy/icerheo.%j.out
#SBATCH -e scripts/nn_eddy/icerheo.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_horizon_results.jsonl
go () { # go <ndays> <ice> <tag>
  echo; echo "----- N=$1 d  ice=$2 -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days "$1" \
      --chunk-steps 48 --repeat 2 --fd-h 1e-3 --run-start-year 1958 --ice "$2" \
      --freeze mix --tag "$3" --results "$RES"
  echo "  (exit $?)"
}
echo "=== $(hostname) $(date) — ice with ONLY the rheology frozen, at 5 and 10 days ==="
go 5  rheology icerheo
go 5  frozen   icerheo-ref
go 10 rheology icerheo
go 10 frozen   icerheo-ref
rc=$?
echo; echo "=== done $(date) rc=$rc ==="
