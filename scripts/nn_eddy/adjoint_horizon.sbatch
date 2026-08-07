#!/bin/bash
# NN-GM Task 2 follow-on — adjoint blow-up attribution + stabilizer sweep.
#
# Runs adjoint_horizon.py once per configuration as a SEPARATE python invocation (fresh
# process => clean per-config peak memory, the horizon_scan.sbatch pattern), each appending
# a JSON line to the results file.
#
# The default sweep answers, in order:
#   1. base @ N=1 d      — VALIDATION. Chunked reverse accumulation is exact, and with one
#                          48-step chunk it is the same graph horizon_scan.py built, so this
#                          must reproduce dL/dm = -2.6949679934915594. If it does not, the
#                          chunking is wrong and nothing below means anything.
#   2. base @ N=2 d      — the first blown window, with the per-chunk cotangent-norm trace:
#                          smooth growth = chaos, concentrated growth = a switch.
#   3. one knob @ N=2 d  — which term's state-dependence in the backward is the amplifier.
#
# Submit from the port_jax worktree root:
#   sbatch scripts/nn_eddy/adjoint_horizon.sbatch
#   KNOBS="bv mix slope" NDAYS=3 sbatch scripts/nn_eddy/adjoint_horizon.sbatch
#   CHUNK=12 KNOBS="" NDAYS=2 sbatch scripts/nn_eddy/adjoint_horizon.sbatch   # fine 6-h trace
#SBATCH --job-name=nn_adjhorizon
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH -o scripts/nn_eddy/adjhorizon.%j.out
#SBATCH -e scripts/nn_eddy/adjhorizon.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
WORK=/work/ab0995/a270088/port_jax
SNAP=${SNAP:-$WORK/foundation_snaps/snap_step037440.pkl}
SEASON=${SEASON:-winterNH}
NDAYS=${NDAYS:-2}
CHUNK=${CHUNK:-48}
# The knob order is a bisection, most-informative first. `bv` cuts every N^2 feedback at its
# source, so if it does not tame the growth then `mix`, `convect` and `slope` (all downstream
# of N^2) cannot either and can be skipped. `fct` is the independent second family: numerics
# rather than closure.
KNOBS=${KNOBS:-"bv fct mix convect slope tkestate visc"}
# FD costs two extra full forwards per h. The sign is what the verdict uses, so one h is
# enough while sweeping many knobs; widen it for the configuration you end up trusting.
FDH=${FDH:-1e-3}
# Identical passes in the SAME process, to separate the gradient's value from its noise.
REPEAT=${REPEAT:-1}
RES=${RES:-scripts/nn_eddy/adjoint_horizon_results.jsonl}
run () {  # run <label> <extra args...>
  echo "----- $1 -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season "$SEASON" \
      --chunk-steps "$CHUNK" --fd-h "$FDH" --repeat "$REPEAT" --run-start-year 1958 \
      --results "$RES" "${@:2}"
  echo "  ($1 exit $?)"
}
echo "=== $(hostname) $(date) — adjoint attribution seed=$SEASON N=$NDAYS d chunk=$CHUNK ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"

# 1. validation against horizon_scan.py's N=1 d number
run "validate N=1 d (expect dL/dm = -2.694968e+00)" --n-days 1 --tag validate-vs-horizon-scan

# 2. the unstabilized 2-day trace
run "base N=$NDAYS d" --n-days "$NDAYS" --tag base

# 3. one freeze knob at a time
for K in $KNOBS; do
  run "freeze=$K N=$NDAYS d" --n-days "$NDAYS" --freeze "$K" --tag attrib
done
echo "=== done $(date) ==="
