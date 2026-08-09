#!/bin/bash
# M1 GRADIENT GATE, the ROUTINE check: <J.v, w> = <v, J^T.w> at the 5-day training window.
#
# Forward-mode AD on the left, this repo's hand-rolled chunked reverse accumulation on the
# right. No step size, no noise floor — both sides are exact derivatives, so the only thing
# separating them is that they are two different compilations on a GPU whose finite-element
# assembly uses atomic scatter-adds. Bar 1%, against a model that reproduces to ~6e-4 on the
# loss between identical runs. NOT a round-off bar; a 1e-9 bar was set here once and no
# correct result could have met it.
#
# Why this and not finite differences at 5 days: a central difference needs a step size h, and
# h has no valid value here. Large h brings truncation error (35.8% at h=1e-2); small h makes
# the two losses nearly equal, and their difference drowns in the 1.7e-6 the loss moves between
# two identical passes. The largest attainable numerator is 1.4e-6 — under the noise even at
# the best h. A 30-day window does fix that (0.50% agreement) and is worth running ONCE, but it
# costs four extra full forwards, 2.5-3 h of an A100, against ~25 min here.
#
# Two cotangents, one shared forward-mode sweep: 'loss' is the one training actually uses, so
# its row is directly comparable with --mode gradient; 'random' probes the whole Jacobian, so a
# defect in a state component the loss happens to be insensitive to still shows up.
#
# ⚠️ FOUNDATION seed, --year 1960 to match its date (1960-02-19). The identity itself holds for
# any seed — it tests the implementation, not the conditioning — but the amplification screen
# reported alongside is only meaningful on a state whose adjoint is usable, and the student
# snapshot's is not (1.0539/step against 1.0008).
#SBATCH --job-name=nn_dotprod
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH -o scripts/nn_eddy/dotprod.%j.out
#SBATCH -e scripts/nn_eddy/dotprod.%j.out
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
    --snap "$SNAP" --year "${YEAR:-1960}" --seed 17 \
    --mode dotprod --days "${DAYS:-5}" \
    --dot-cotangents "${COTS:-loss,sst,heat}" \
    --chunk-steps "${CHUNK:-48}" --seg-mult "${SEGM:-1.0}" \
    --results /work/ba1583/a270088/nn_eddy/seam_gate/seam_gate.jsonl
RC=$?
echo "=== done $(date)  dotprod rc=$RC ==="
exit $RC
