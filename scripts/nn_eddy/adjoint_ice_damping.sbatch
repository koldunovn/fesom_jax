#!/bin/bash
# The only published stabilization that KEEPS EVP rheology in the adjoint instead of removing
# it: a Newtonian damping term in the linearized stress equations, with a damping timescale
# T_N in subcycle sweeps (Kauker-line parameter-optimization work: T_N=7 generally, 3 in thick
# strongly converging ice, over 400 subcycles).
#
# Prediction for our 120-sweep configuration, from the measured 1.1795 amplification per sweep:
#   T_N   net/sweep   per model step   over a 48-step day
#     4      0.8846        4.1e-07              ~0  (annihilated)
#     5      0.9436        9.4e-04              ~0
#     6      0.9829          0.127         7.9e-44
#   6.57     1.0000            1.00            1.00  (exact cancellation)
#     7      1.0110            3.72         2.3e+27  (blown)
#    10      1.0615            1297        overflow
# So the whole usable range sits between 6 and 7 sweeps — an 8% tuning tolerance. The point of
# measuring is that 1.1795 is a GLOBAL AVERAGE and the real amplification is localized in
# converging ice, so a uniform constant will over-damp most of the domain while under-damping
# the part that actually blows up. That is presumably why the published version varies T_N in
# space. If no single value works here, that is the result.
#
# Reference for judging the answer: the rheology-frozen gradient, which is the same
# configuration with the rheology contribution set to zero instead of damped.
#SBATCH --job-name=nn_icedamp
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o scripts/nn_eddy/icedamp.%j.out
#SBATCH -e scripts/nn_eddy/icedamp.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_horizon_results.jsonl
go () { # go <ice> <T_N> <tag>
  echo; echo "----- 1 day  ice=$1  T_N=${2:-off} sweeps -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days 1 \
      --chunk-steps 48 --repeat 2 --fd-h 1e-3 --run-start-year 1958 --ice "$1" \
      --ice-damping "${2:-0}" --freeze mix --tag "$3" --results "$RES"
  echo "  (exit $?)"
}
echo "=== $(hostname) $(date) — Newtonian damping of the mEVP adjoint, 1-day window ==="
go rheology 0    icedamp-ref-rheologyfrozen
for TN in 4 5 6 6.57 7 8 10; do go exact "$TN" "icedamp-TN$TN"; done
rc=$?
echo; echo "=== done $(date) rc=$rc ==="
