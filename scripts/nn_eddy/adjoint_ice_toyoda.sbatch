#!/bin/bash
# Does the Toyoda/Lyu approximation make the sea-ice adjoint work here?
#
# The exact rheology adjoint returns NaN immediately. The reason is arithmetic, not chaos: the
# mEVP viscosity is P/(Delta+delta_min) with delta_min=1e-11, so its derivative with respect to
# the deformation rate is ~1e22 in rigid ice and ~1e16 in ordinary deforming ice. float64
# overflows after 14 to 19 of the 120 subcycles inside a SINGLE ice step.
#
# Toyoda et al. (2019), as used by Lyu et al. (2023, Ocean Sci. 19, 305), remove exactly that
# term: the viscosities are held constant with respect to the strain rate in the adjoint. Over
# a one-year window that gave them a 40.2% cost reduction against 32.3% for a free-drift
# adjoint, at 1.2x the cost. Here it is one backward-only freeze; the forward is untouched.
#SBATCH --job-name=nn_icetoyoda
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH -o scripts/nn_eddy/icetoyoda.%j.out
#SBATCH -e scripts/nn_eddy/icetoyoda.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_horizon_results.jsonl
go () { # go <ndays> <ice> <freeze> <tag>
  echo; echo "----- N=$1 d  ice=$2  freeze=$3 -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days "$1" \
      --chunk-steps 48 --repeat 2 --fd-h 1e-3 --run-start-year 1958 --ice "$2" \
      --freeze "$3" --tag "$4" --results "$RES"
  echo "  (exit $?)"
}
echo "=== $(hostname) $(date) — Toyoda/Lyu viscosity freeze for the ice adjoint ==="
# The control FIRST this time. The previous attempt at it was destroyed by a library edit I
# made while that job was running; do not edit fesom_jax/ while this one is in the queue.
go 2 frozen mix          frozenice-control
go 2 exact  mix,icevisc  toyoda-icevisc
go 5 exact  mix,icevisc  toyoda-icevisc
echo; echo "=== done $(date) ==="
