#!/bin/bash
# NN-GM Task 2 — gradient-horizon scan (GATE HORIZON_OK). Loops window length N (days)
# as SEPARATE python invocations (fresh process per N => clean per-N peak memory, the
# longwindow C1 pattern), each appending a JSON line to the results file.
# Submit from the port_jax worktree root:
#   sbatch scripts/nn_eddy/horizon_scan.sbatch                             # winter seed, end-marks
#   SNAP=.../snap_step044640.pkl SEASON=summerNH sbatch scripts/nn_eddy/horizon_scan.sbatch
#   MARKS=daily sbatch scripts/nn_eddy/horizon_scan.sbatch                 # dense-marks variant
#SBATCH --job-name=nn_horizon
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH -o scripts/nn_eddy/horizon.%j.out
#SBATCH -e scripts/nn_eddy/horizon.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export FESOM_FCT_STOPGRAD_AE=1     # docs/LIMITER_GRADIENTS.md option (c): limiter factors constant in the backward
WORK=/work/ab0995/a270088/port_jax
SNAP=${SNAP:-$WORK/foundation_snaps/snap_step037440.pkl}
SEASON=${SEASON:-winterNH}
MARKS=${MARKS:-end}
NLIST=${NLIST:-"1 2 3 5 7 10"}   # shortest first: each N is a fresh process, so an OOM at long N does not lose the short-N results
RES=${RES:-scripts/nn_eddy/horizon_results_stopgrad.jsonl}
echo "=== $(hostname) $(date) — Task-2 horizon scan seed=$SEASON marks=$MARKS N=[$NLIST] d ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
for N in $NLIST; do
  echo "----- N=$N d -----"
  "$PY" scripts/nn_eddy/horizon_scan.py --snap "$SNAP" --season "${SEASON}-stopgrad" --n-days "$N" \
      --marks "$MARKS" --run-start-year 1958 --results "$RES"
  echo "  (N=$N exit $?)"
done
echo "=== done $(date) ==="
