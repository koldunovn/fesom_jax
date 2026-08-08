#!/bin/bash
# NN-GM child-plan M1 GATE on one A100: stability (10 d, perturbed NN, vs a gm_nn=None
# control) then the 5-day AD-vs-FD gradient check (bar: ≤10%).
#
# The gate requires the SAME numbers from TWO separate jobs to 3 significant figures, so
# submit this twice and compare the two JSONL records — a gradient that is not reproducible
# across processes is not a gradient (the horizon scan's own standard).
#   sbatch scripts/nn_eddy/seam_gate.sbatch
#   sbatch scripts/nn_eddy/seam_gate.sbatch      # second, independent job
# Override the seed state with SNAP=..., the network seed with SEED=...
#SBATCH --job-name=nn_seam_gate
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/seamgate.%j.out
#SBATCH -e scripts/nn_eddy/seamgate.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=${SNAP:-/work/ab0995/a270088/nn_eddy/student_free_1958/snap_day150.pkl}
SEED=${SEED:-17}
RESULTS=${RESULTS:-/work/ab0995/a270088/nn_eddy/seam_gate/seam_gate.jsonl}
echo "=== $(hostname) $(date)  commit $(git rev-parse --short HEAD) ==="
echo "=== SNAP=$SNAP  SEED=$SEED  job=$SLURM_JOB_ID ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"

echo ""
echo "=== (a) stability: 10 d perturbed NN vs gm_nn=None control ==="
"$PY" scripts/nn_eddy/seam_gate.py --snap "$SNAP" --seed "$SEED" --mode stability \
    --results "$RESULTS"
RC1=$?

echo ""
echo "=== (b) gradient: 5-day AD vs FD on the last-layer scaling probe ==="
"$PY" scripts/nn_eddy/seam_gate.py --snap "$SNAP" --seed "$SEED" --mode gradient \
    --results "$RESULTS"
RC2=$?

echo ""
echo "=== done $(date)  stability rc=$RC1  gradient rc=$RC2 ==="
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ] && echo "M1 GATE JOB COMPLETE (read the verdict tokens above)" \
    || echo "M1 GATE JOB FAILED"
exit $(( RC1 | RC2 ))
