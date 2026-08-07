#!/bin/bash
# Does freezing the vertical mixing coefficients in the backward actually open up the
# horizon? The attribution sweep says it collapses the per-step amplification from ~1.2-1.5
# to 1.007, which is the difference between growing by 1e8 over two days and growing by 2.
# Two questions remain: is the resulting gradient REPRODUCIBLE (repeat passes), and does the
# horizon extend to windows that were previously unreachable (5 and 10 days, now possible
# because the reverse pass is accumulated chunk by chunk).
#SBATCH --job-name=nn_mixfrozen
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o scripts/nn_eddy/mixfrozen.%j.out
#SBATCH -e scripts/nn_eddy/mixfrozen.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_horizon_results.jsonl
go () { # go <ndays> <repeat> <freeze> <tag>
  echo "----- N=$1 d  repeat=$2  freeze=${3:-none}  -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days "$1" \
      --chunk-steps 48 --repeat "$2" --fd-h 1e-2,1e-3 --run-start-year 1958 \
      ${3:+--freeze "$3"} --tag "$4" --results "$RES"
  echo "  (exit $?)"
}
echo "=== $(hostname) $(date) — mixing-frozen backward: reproducibility and reach ==="
go 2  3 mix mixfrozen-repeat      # is the fixed gradient reproducible?
go 2  1 fct fct-attrib            # the one knob the earlier sweep never reached
go 5  2 mix mixfrozen-reach       # windows that used to run out of GPU memory
go 10 2 mix mixfrozen-reach
go 5  1 ""  base-reach            # unfrozen, for contrast, now that memory allows it
echo "=== done $(date) ==="
