#!/bin/bash
# Does the TKE closure really differentiate better than KPP, in OUR model?
#
# Martin Losch's experience with MITgcm is that the adjoint behaves better with GGL90 (the
# same Gaspar 1990 TKE family this model uses) than with KPP. The mechanism is visible in the
# two schemes' formulations. KPP hangs its diffusivity profile off a boundary-layer depth
# located by searching for where a bulk Richardson number crosses a critical value: a discrete
# index, discontinuous in the ocean state. fesom_jax/kpp.py stop_gradients that index and
# several other discrete selections (the caseA branch, the velocity-scale bin) because they
# have no useful derivative. TKE instead advances a prognostic energy equation with a smooth
# implicit solve and needs no such cuts: fesom_jax/tke.py contains no stop_gradient at all.
#
# Prediction, therefore: KPP should amplify at least as much as TKE, and its gradient should
# be further from the finite-difference reference because of the derivative terms that were
# deliberately dropped. Same window, same seed, same loss, one variable changed.
#SBATCH --job-name=nn_kppvstke
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH -o scripts/nn_eddy/kppvstke.%j.out
#SBATCH -e scripts/nn_eddy/kppvstke.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_horizon_results.jsonl
go () { # go <mixing> <freeze> <tag>
  echo; echo "----- mixing=$1  freeze=${2:-none} -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days 2 \
      --chunk-steps 48 --repeat 3 --fd-h 1e-2,1e-3 --run-start-year 1958 --mixing "$1" \
      ${2:+--freeze "$2"} --tag "$3" --results "$RES"
  echo "  (exit $?)"
}
echo "=== $(hostname) $(date) — KPP vs TKE differentiability, 2-day window ==="
# ⚠️ The truth trajectory is cached per snapshot and was built with TKE. The KPP runs use it
# unchanged ON PURPOSE: the target is held fixed so the only thing that differs between the
# two rows is which closure the STUDENT and its adjoint use. The KPP loss value is therefore
# not comparable to the TKE loss value; the growth rate and the reproducibility are.
go tke ""    kppvstke-tke-exact
go kpp ""    kppvstke-kpp-exact
go tke mix   kppvstke-tke-frozen
go kpp mix   kppvstke-kpp-frozen
echo; echo "=== done $(date) ==="
