#!/bin/bash
# Can we afford to differentiate the sea ice now?
#
# The ice has been excluded from the backward from the start (IceConfig adjoint_mode="frozen"):
# the forward runs full modified elastic-viscous-plastic ice, the backward skips the whole ice
# update, so no gradient flows through ice at all. The reason was that the rheology — 120
# stress-relaxation subcycles per step — was assumed to be the dominant blow-up source. That
# assumption was never tested against the alternative, and the alternative turned out to be
# the true one: the turbulence closure was the amplifier.
#
# So: same window, same everything, mixing frozen in both, ice frozen vs ice differentiated.
# Watch the growth per timestep and the peak memory — 120 subcycles of tape per step is the
# part that could make this unaffordable even if it is numerically well behaved.
#SBATCH --job-name=nn_exactice
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH -o scripts/nn_eddy/exactice.%j.out
#SBATCH -e scripts/nn_eddy/exactice.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_horizon_results.jsonl
go () { # go <ndays> <ice> <freeze> <tag>
  echo; echo "----- N=$1 d  ice=$2  freeze=${3:-none} -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days "$1" \
      --chunk-steps 48 --repeat 2 --fd-h 1e-3 --run-start-year 1958 --ice "$2" \
      ${3:+--freeze "$3"} --tag "$4" --results "$RES"
  echo "  (exit $?)"
}
echo "=== $(hostname) $(date) — exact sea ice in the adjoint ==="
# The winter seed is 1960-02-20: peak Northern Hemisphere ice, so the ice adjoint has the
# most to do here. If it survives this it survives anywhere in the year.
go 2 exact  mix exactice        # the question
go 2 frozen mix frozenice-ref   # the reference, rerun in this job so the comparison is clean
go 5 exact  mix exactice        # does it hold over a longer window?
go 2 exact  ""  exactice-nomix  # and out of curiosity: exact ice with the closure live too
echo; echo "=== done $(date) ==="
