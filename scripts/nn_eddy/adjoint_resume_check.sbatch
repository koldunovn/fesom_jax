#!/bin/bash
# Does a reverse sweep split across PROCESSES give the same gradient as one that ran straight
# through? Measured on a short window before anything long depends on it.
#
# Three steps, one job:
#   1. reference: a 10-day window in one shot, --repeat 2. The repeat is not decoration — it
#      MEASURES the floor this test is allowed to be judged against. Two identical passes of
#      this model differ by ~3e-5 on the gradient (GPU atomic scatter-adds in the FE assembly),
#      so a restart that agrees to 3e-5 is agreeing as well as the model agrees with ITSELF.
#      Setting a round-off bar here would fail every correct result; that mistake is on record.
#   2. the same window with --ckpt-dir and a 5-minute budget, re-invoked until it exits 0.
#      Each leg is a FRESH PROCESS: new compile, boundaries read back from disk, cotangent
#      reloaded from its slot. That is the thing being tested, not an in-process loop.
#   3. compare the two dL/dm against the floor from step 1.
#
# ⚠️ Each leg pays its own XLA compile (a few minutes). That is the real cost of chaining and
# it is why --stop-after-min should be minutes below the wall clock, not seconds: legs that are
# too short spend all their time compiling.
#
# Nothing here deletes anything. The checkpoint alternates between two fixed cotangent slots
# and the manifest names the live one, so an interrupted write lands in the slot that is NOT
# named and the previous checkpoint survives intact. (Workspace house rule: no unattended
# deletion, ever.)
#SBATCH --job-name=nn_resume_chk
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/resumechk.%j.out
#SBATCH -e scripts/nn_eddy/resumechk.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/nn_eddy/port_jax || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=${SNAP:-/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl}
WEIGHTS=/work/ba1583/a270088/nn_eddy/horizon_truth/snap_step037440_m1.25_30d.npz
RES=/work/ba1583/a270088/nn_eddy/resume_check/results.jsonl
CKPT=/work/ba1583/a270088/nn_eddy/resume_check/ckpt_$SLURM_JOB_ID
N=${N:-10}
MINS=${MINS:-5}
mkdir -p "$(dirname "$RES")"
echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"

COMMON=(--snap "$SNAP" --season winterNH --n-days "$N" --truth-days 30 --chunk-steps 48
        --freeze mix --weights-from "$WEIGHTS" --run-start-year 1958 --results "$RES"
        --fd-h "")

echo ""; echo "===== 1. reference: one shot, two passes (measures the floor) ====="
"$PY" scripts/nn_eddy/adjoint_horizon.py "${COMMON[@]}" --repeat 2 --tag "resume-ref-N${N}"
echo "  (reference exit $?)"

echo ""; echo "===== 2. the same window, checkpointed, ${MINS}-minute legs ====="
LEG=0
RESUME=""
while : ; do
  LEG=$((LEG+1))
  if [ "$LEG" -gt "${MAXLEGS:-16}" ]; then echo "TOO MANY LEGS — giving up"; exit 1; fi
  echo ""; echo "----- leg $LEG -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py "${COMMON[@]}" \
      --ckpt-dir "$CKPT" --ckpt-every 2 --stop-after-min "$MINS" $RESUME \
      --tag "resume-split-N${N}"
  RC=$?
  echo "  (leg $LEG exit $RC)"
  [ "$RC" -eq 0 ] && break
  [ "$RC" -ne 42 ] && { echo "LEG FAILED rc=$RC"; exit "$RC"; }
  RESUME="--resume"
done
echo "  finished after $LEG legs"

echo ""; echo "===== 3. verdict ====="
"$PY" - "$RES" "$N" <<'EOF'
import json, sys
res, N = sys.argv[1], int(sys.argv[2])
ref = split = None
for line in open(res):
    r = json.loads(line)
    if r.get("n_days") != N or r.get("grad") is None:
        continue
    if r.get("tag") == f"resume-ref-N{N}":
        ref = r
    elif r.get("tag") == f"resume-split-N{N}":
        split = r
if ref is None or split is None:
    print("RESUME_CHECK_INCONCLUSIVE — missing a record"); raise SystemExit(1)
a, b = ref["grad"], split["grad"]
rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
# the floor the reference measured on ITSELF, not a bar chosen in advance
rg = ref.get("rep_grad") or []
floor = (max(abs(x) for x in rg) - min(abs(x) for x in rg)) / max(abs(a), 1e-300) if len(rg) > 1 else None
print(f"  one shot   dL/dm = {a:+.10e}   loss {ref['loss']:.8e}")
print(f"  resumed    dL/dm = {b:+.10e}   loss {split['loss']:.8e}")
print(f"  difference {rel:.3e} relative")
if floor is None:
    print("  NO FLOOR MEASURED — the reference did not repeat; cannot judge"); raise SystemExit(1)
print(f"  the model's own spread over {len(rg)} identical passes: {floor:.3e} relative")
ok = rel <= max(10 * floor, 1e-6)
print(f"  amplification: ref median {ref.get('amp_median'):.5f}/step, "
      f"resumed {split.get('amp_median'):.5f}/step, log-spread "
      f"{ref.get('amp_log_spread'):.4f} vs {split.get('amp_log_spread'):.4f}")
print("RESUME_CHECK_" + ("PASS" if ok else "FAIL")
      + f" (bar = 10x the measured floor = {max(10*floor,1e-6):.3e})")
raise SystemExit(0 if ok else 1)
EOF
RC=$?
echo "=== done $(date) rc=$RC  (checkpoint left in $CKPT — prune by hand) ==="
exit $RC
