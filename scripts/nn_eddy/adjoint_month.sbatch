#!/bin/bash
# A one-month adjoint window with the mixing coefficients frozen.
#
# Ten days came back clean: growth 1.0007 per timestep, a factor 1.40 over the window,
# reproducible. A month is 1440 timesteps; at that rate it should grow by about 2.7, which
# would still be an entirely usable gradient. This measures it instead of extrapolating.
#
# Costs, from the measured per-day rates (~170 s forward, ~440 s backward per 5 days):
#   forward ~17 min, backward ~44 min per pass, so ~2 h for two passes plus the reference run.
#   host memory ~1.9 GB per saved day => ~59 GB for 31 boundary states. Fine on this node.
#   GPU peak stays at one chunk, ~45 GB, regardless of window length.
#
# ⚠️ --truth-days 30 builds a NEW 30-day reference trajectory (~10 min) in its own cache file.
# Its per-level variance weights differ from the 10-day one, so the loss and gradient VALUES
# here are not comparable with the earlier table; the growth rate and reproducibility are.
#SBATCH --job-name=nn_adjmonth
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH -o scripts/nn_eddy/adjmonth.%j.out
#SBATCH -e scripts/nn_eddy/adjmonth.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_horizon_results.jsonl
go () { # go <ndays> <repeat> <freeze> <tag>
  echo; echo "----- N=$1 d  repeat=$2  freeze=${3:-none} -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days "$1" \
      --truth-days 30 --chunk-steps 48 --repeat "$2" --fd-h "" --run-start-year 1958 \
      ${3:+--freeze "$3"} --tag "$4" --results "$RES"
  echo "  (exit $?)"
}
echo "=== $(hostname) $(date) — one-month adjoint window, mixing frozen ==="
# Shortest first: a 30-day reference is built on the first call and cached, and if the long
# runs die we still have the shorter rungs on the SAME reference for a like-for-like curve.
go 10 1 mix month-ref-10d
go 20 1 mix month-ref-20d
go 30 2 mix month
go 30 1 ""  month-unfrozen-contrast
echo; echo "=== done $(date) ==="
