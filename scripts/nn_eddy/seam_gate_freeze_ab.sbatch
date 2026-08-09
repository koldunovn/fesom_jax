#!/bin/bash
# Is the seam's gradient a function of its inputs? The A/B that docs/ADJOINT_HORIZON.md 1.2
# ran for the vertical mixing closure, repeated for the LEARNED closure.
#
# Background. Two identical jobs (26800197/8) returned dL/dalpha = +5.93e-02 and +2.30e-03 —
# a factor of 26 apart, opposite in sign to finite differences, while the LOSS agreed to three
# digits. That is the signature in 1.2: the GPU's atomic scatter-adds make the forward
# nondeterministic at ~1e-6 relative (measured here as loss(alpha=1) = 2.1e-6 / 2.8e-6 where
# it should be 0), and any closure whose coefficients depend on the ocean state re-injects
# that difference every step for the backward to amplify.
#
# `freeze_mixing=True` covers the VERTICAL mixing closure and is already on. It cannot cover
# `gm_nn`, which is a NEW state-dependent diffusivity built 2026-08-08, after those flags were
# measured. `AdjointConfig.freeze_nn_features` is the same ECCO move for the network: use the
# multipliers it computes, omit their dependence on the state. dL/d(weights) is PRESERVED.
#
# ⚠️ The horizon result does not settle this. There the diffusivity was a constant times a
# scalar — no state dependence, nothing to inject — so "clean to 30 days" says nothing about
# a network reading the state at every node and level.
#
# Read the two `[repeat]` lines. 1.2's contrast was 21x-33x spread WITH SIGN FLIPS unfrozen,
# against six significant figures frozen. Anything less than ~3 significant figures frozen
# means the diagnosis is wrong and the next step is measurement, not another patch.
#SBATCH --job-name=nn_freeze_ab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/freezeab.%j.out
#SBATCH -e scripts/nn_eddy/freezeab.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/nn_eddy/port_jax || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=${SNAP:-/work/ba1583/a270088/nn_eddy/student_free_1958/snap_day150.pkl}
RES=${RES:-/work/ba1583/a270088/nn_eddy/seam_gate/seam_gate.jsonl}
REPEAT=${REPEAT:-2}
echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"

echo ""
echo "=== A: CONTROL — network inputs differentiated (expect the 1.2 signature) ==="
"$PY" scripts/nn_eddy/seam_gate.py --snap "$SNAP" --seed 17 --mode gradient \
    --repeat "$REPEAT" --results "$RES"
RCA=$?

echo ""
echo "=== B: freeze_nn_features — network inputs treated as constants in the BACKWARD ==="
"$PY" scripts/nn_eddy/seam_gate.py --snap "$SNAP" --seed 17 --mode gradient \
    --repeat "$REPEAT" --freeze-nn-features --results "$RES"
RCB=$?

echo ""
echo "=== done $(date)  control rc=$RCA  frozen rc=$RCB ==="
exit $(( RCA | RCB ))
