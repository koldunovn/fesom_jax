#!/bin/bash
# NN-GM child-plan M1 GATE, gradient half: 30-day AD-vs-FD on the last-layer scaling probe,
# by CHUNKED REVERSE ACCUMULATION.
#
# ⚠️ TWO THINGS IN HERE ARE THE RESULT OF A DAY OF FAILED JOBS. Do not "simplify" either.
#
# 1. --days 30, not 5. At 5 days the finite-difference comparison is not a measurement: the
#    loss moves 1.7e-6 between two identical passes (GPU atomic scatter-adds), while the FD
#    numerator at the best step size is 1.4e-6 — under the noise — and at larger h truncation
#    error dominates (35.8% at h=1e-2). There is no usable h. The AD signal grows with window
#    length and the noise floor does not: the horizon ladder measured agreement 4.2% at 5 d,
#    5.4% at 10, 2.4% at 20, 0.50% at 30. seam_gate.py now REFUSES to present an agreement
#    number whose FD numerator does not clear the loss noise it measured in the same process.
#
# 2. --snap is the FOUNDATION snapshot, not student_free_1958/snap_day150.pkl. The adjoint's
#    per-step amplification is a property of the seed STATE: 1.0008/step here against
#    1.0539/step from the student snapshot, measured with the network entirely ABSENT in both
#    (26803145 vs 26801647). Over 240 steps that is 1.2x against 3e5. Seeded from the student
#    snapshot this gate returned a positive AD against a negative FD and did not reproduce.
#    The foundation snapshot's date is 1960-02-19, hence --year 1960; a snapshot run against
#    the wrong forcing year is a different experiment.
#
# Attempts 1-3 (26798462, 26798463, 26798871) OOM'd because they built the whole reverse pass
# as ONE device computation. That does not fit an 80 GB A100 for ANY --seg-mult: a CORE2 State
# is 1.93 GB and _run_steps(segments=S) stores S+N/S of them. run_gradient is chunked now —
# device peak flat in window length, ~43 GB at any length, boundaries on the host.
#
# --repeat 2 is not optional: the other half of this gate is that the gradient is a function
# of its inputs, and one sample of a quantity that once moved 2.877x with a sign flip is not
# a measurement.
#
# Cost: ~2.5-3 h. Forward ~450 s, backward ~2050 s, two probe passes plus the self-check pass,
# plus 2 forwards per h in the sweep.
#SBATCH --job-name=nn_seam_grad
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH -o scripts/nn_eddy/seamgrad.%j.out
#SBATCH -e scripts/nn_eddy/seamgrad.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/nn_eddy/port_jax || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=${SNAP:-/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl}
echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
"$PY" scripts/nn_eddy/seam_gate.py \
    --snap "$SNAP" --year "${YEAR:-1960}" \
    --seed 17 --mode gradient --days "${DAYS:-30}" --repeat "${REPEAT:-2}" \
    --h-sweep "${HS:-1e-2,1e-3}" \
    --chunk-steps "${CHUNK:-48}" --seg-mult "${SEGM:-1.0}" \
    --results /work/ba1583/a270088/nn_eddy/seam_gate/seam_gate.jsonl
RC=$?
echo "=== done $(date)  gradient rc=$RC ==="
exit $RC
