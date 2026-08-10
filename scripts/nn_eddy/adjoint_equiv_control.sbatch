#!/bin/bash
# What is the BETWEEN-PROCESS spread of dL/dm for one unchanged version?
#
# `adjoint_equivalence.sbatch` (job 26842065) compared the pre-extraction code against the
# library version and reported, at a ONE-CHUNK window, forward losses of 5.686157e-04 and
# 5.645529e-04 — 7.2e-3 apart, against a WITHIN-process repeat spread of 5.1e-5 and 6.0e-6. At
# two chunks the same comparison was 2.8e-4 apart against a within-process spread of ~2e-4, i.e.
# indistinguishable.
#
# A systematic difference in the code cannot shrink 25-fold when the window doubles. So the
# suspect is the control, not the extraction: `--repeat` measures two passes through ONE compiled
# executable in ONE process, and its spread is only the atomic scatter-adds. Two separate
# processes RECOMPILE, and XLA chooses kernels by timing them at compile time, so a second process
# can select different algorithms and produce a different — equally correct — rounding.
#
# ⚠️ That makes `--repeat` the WRONG yardstick for a two-process comparison, and using it is how a
# correct extraction gets called a regression. This job measures the right one: the SAME script,
# the same arguments, four separate processes, interleaved so drift in machine load cannot align
# with the version being tested.
#
# It answers one question. If ref-vs-ref is of the same order as ref-vs-new, the extraction is not
# distinguishable from recompiling. If ref-vs-ref is tiny and ref-vs-new is not, the extraction
# changed a number and Task 5 is wrong.
#
#   sbatch scripts/nn_eddy/adjoint_equiv_control.sbatch
#SBATCH --job-name=nn_adjctl
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH -o scripts/nn_eddy/adjctl.%j.out
#SBATCH -e scripts/nn_eddy/adjctl.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

WORK=/work/ab0995/a270088/port_jax
SNAP=${SNAP:-$WORK/foundation_snaps/snap_step037440.pkl}
SEASON=${SEASON:-winterNH}
ND=${ND:-1}
OUT=${OUT:-scripts/nn_eddy/adjoint_equiv_control_results.jsonl}

# ⚠️ It MUST live in scripts/nn_eddy/. The script derives the repo root as
# `Path(__file__).resolve().parents[2]` and puts it first on sys.path; from anywhere else that
# resolves to the wrong tree and `fesom_jax` comes from the MAIN checkout instead of this
# worktree. Its own guard catches that and refuses to report ("code that did not run") — which is
# exactly what happened when this was written to /scratch (job 26843938, all ref passes lost).
# It is untracked and removed on exit; `test_private_api_boundary.py` only inspects TRACKED files
# so this copy cannot be mistaken for research code that kept a private import.
REF=scripts/nn_eddy/adjoint_horizon_ref.py
git show "${REF_COMMIT:-e800118}:scripts/nn_eddy/adjoint_horizon.py" > "$REF" || exit 2
trap 'rm -f "$REF"' EXIT
grep -q "from fesom_jax.integrate import _run_steps" "$REF" || {
    echo "ABORT: the reference copy is not the pre-extraction code."; exit 2; }

echo "=== node $(hostname)  $(date)  N=${ND}d ==="

# Interleaved ref,new,ref,new — NOT ref,ref,new,new. If the node warms up, or another job lands
# on it, a blocked layout would put that drift entirely on one version.
for I in 1 2; do
    for SIDE in ref new; do
        SCRIPT=$([ "$SIDE" = ref ] && echo "$REF" || echo scripts/nn_eddy/adjoint_horizon.py)
        echo ""
        echo "=== $SIDE pass $I ==="
        "$PY" "$SCRIPT" --snap "$SNAP" --season "$SEASON" --n-days "$ND" \
            --boundary-stride 1 --freeze mix,slope --ice rheology \
            --repeat 1 --fd-h "" --tag "ctl-${SIDE}-${I}" --results "$OUT" \
            || echo "  ($SIDE pass $I exited $?)"
    done
done

echo ""
echo "=== VERDICT ==="
"$PY" - "$OUT" <<'PYEOF'
import json, sys, itertools
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = {}
for r in rows:
    if str(r.get("tag", "")).startswith("ctl-"):
        by[r["tag"]] = r

def g(tag):
    r = by.get(tag)
    return None if r is None else r.get("grad")

def rel(a, b):
    return abs(b - a) / max(abs(a), 1e-300)

ref = [g(f"ctl-ref-{i}") for i in (1, 2)]
new = [g(f"ctl-new-{i}") for i in (1, 2)]
if any(x is None for x in ref + new):
    print("MISSING runs:", {t: g(t) for t in
          ("ctl-ref-1", "ctl-ref-2", "ctl-new-1", "ctl-new-2")})
    print("CONTROL_INCONCLUSIVE"); raise SystemExit(0)

for name, xs in (("ref", ref), ("new", new)):
    for i, x in enumerate(xs, 1):
        print(f"  {name} pass {i}: dL/dm = {x:+.17e}")
within_ref, within_new = rel(ref[0], ref[1]), rel(new[0], new[1])
cross = [rel(a, b) for a, b in itertools.product(ref, new)]
print(f"\n  between-process, SAME version:   ref {within_ref:.2e}   new {within_new:.2e}")
print(f"  between-version (4 pairings):    min {min(cross):.2e}   max {max(cross):.2e}")
floor = max(within_ref, within_new)
print(f"\n  the yardstick is {floor:.2e} — the largest difference the SAME code produced across")
print("  two processes. Nothing compiled twice can be asked to beat it.")
if max(cross) <= max(floor, 3e-5):
    print("CONTROL_OK — the extraction is not distinguishable from recompiling")
else:
    print(f"CONTROL_FAILED — between-version {max(cross):.2e} exceeds the same-version "
          f"floor {floor:.2e}; the extraction changed a number")
PYEOF
