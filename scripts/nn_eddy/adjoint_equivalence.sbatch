#!/bin/bash
# End-to-end sanity check for the extracted library adjoint on the assembled CORE2 model.
#
# ⚠️ THIS IS NOT THE EQUIVALENCE GATE, though it was written to be. The gate is
# `test_scan_steps_builds_the_same_graph_as_the_call_it_replaces` in
# fesom_jax/tests/test_adjoint_chunked.py, which proves the migrated call sites lower to a
# BYTE-IDENTICAL jaxpr at chunk lengths 6, 48 and 50 (job 26845302) — including 48, the length
# production runs, where `_run_steps` takes its separate remainder branch. Identical graph ⇒ no
# run can differ for any reason attributable to the change, on any backend, at any window length.
#
# This job could not establish that, and the reason is worth keeping. On an A100 the model is not
# reproducible between processes: two processes recompile, XLA picks kernels by timing them, and
# scatter-adds on an unstructured mesh are not deterministic. Measured on the UNCHANGED code
# across processes (job 26844357): dL/dm spreads of 2.9e-5 and 1.0e-4, with the two versions'
# values INTERLEAVING and one run a clear outlier in the forward loss. The differences being
# hunted are smaller than the instrument's scatter, and no number of samples fixes an instrument.
#
# What this job is still good for: proving the assembled model — CORE2 mesh, ice, TKE, the freeze
# monkeypatches, the checkpointer — runs end to end through the library function and returns a
# finite, informative gradient. Run it when the seam changes. Do not read its `rel` column as a
# verdict on numerical equivalence.
#
# `adjoint_horizon.py`'s reverse sweep used to be six closures inside `main()`
# (`fwd_chunk`/`bwd_chunk`/`forward`/`rebuild_span`/`chunk_sfs` and the reverse loop). They are
# now `fesom_jax.adjoint.chunked_grad` / `chunked_forward`. Nothing about the scheme changed, so
# nothing about `dL/dm` may change either — and "nothing changed" is a claim about numbers, not a
# claim about a diff.
#
# ⚠️ The comparison is OLD CODE vs NEW CODE in ONE JOB, on the same seed, window and flags — not
# the new code against a row in the results file. Rows in that file were produced across weeks of
# changing defaults; `chunk_steps`, `boundary_stride`, `ice` and `mixing` were only added to the
# provenance later, so a historical row cannot say what configuration produced it. A comparison
# whose control is under-specified would pass or fail for reasons that have nothing to do with the
# extraction.
#
# ⚠️ THE BAR IS THE MEASURED NOISE FLOOR, NOT A CONSTANT — this project's own rule, "never set a
# round-off bar; measure the floor and judge against that". The plan proposed 3e-5, which is the
# agreement recorded across boundary strides 1/2/3. But on an A100 the model is not deterministic
# run to run (atomic scatter-adds on an unstructured mesh), and the FIRST run of this job measured
# the same code against itself at **2.3e-4** across two repeats in one process — an order of
# magnitude above 3e-5. A fixed 3e-5 would therefore fail a bit-identical extraction, and passing
# it would be luck rather than evidence.
#
# So each side runs --repeat 2 and the verdict compares the between-version difference against the
# larger of 3e-5 and the within-version spread the run itself measured. The claim being tested is
# "swapping the implementation changes dL/dm no more than running the same implementation twice
# does" — which is the strongest claim the hardware permits. Do not replace it with a constant.
#
# Three windows, chosen so each exercises something the previous one cannot:
#   N=1, stride 1   ONE chunk. No boundary is crossed at all — isolates the chunk advance and
#                   the loss seed from the reverse walk.
#   N=2, stride 1   TWO chunks: the first configuration in which a cotangent crosses a boundary.
#   N=2, stride 2   the span-rebuild path, which recomputes rather than stores.
#
#   sbatch scripts/nn_eddy/adjoint_equivalence.sbatch
#SBATCH --job-name=nn_adjequiv
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o scripts/nn_eddy/adjequiv.%j.out
#SBATCH -e scripts/nn_eddy/adjequiv.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

WORK=/work/ab0995/a270088/port_jax
SNAP=${SNAP:-$WORK/foundation_snaps/snap_step037440.pkl}
SEASON=${SEASON:-winterNH}
OUT=${OUT:-scripts/nn_eddy/adjoint_equivalence_results.jsonl}

# The REFERENCE is the committed pre-extraction file, written out at run time rather than kept in
# the tree: it must be the code that actually ran before, not a copy someone maintained alongside.
#
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
    echo "ABORT: the reference copy does not import _run_steps, so it is not the pre-extraction"
    echo "code and this job would compare the new implementation against itself."; exit 2; }
grep -q "from fesom_jax.adjoint import" scripts/nn_eddy/adjoint_horizon.py || {
    echo "ABORT: adjoint_horizon.py does not use fesom_jax.adjoint — nothing to compare."; exit 2; }

echo "=== node $(hostname)  $(date) ==="
echo "=== reference: $(git rev-parse --short "${REF_COMMIT:-e800118}") ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --repeat 2 on both sides: the within-process spread is the yardstick the between-version
# difference is read against. --fd-h with a single h keeps the two extra forwards per run cheap
# while still exercising chunked_forward, which is the path the FD check now takes.
for CFG in "1 1" "2 1" "2 2"; do
    set -- $CFG; ND=$1; ST=$2
    for SIDE in ref new; do
        SCRIPT=$([ "$SIDE" = ref ] && echo "$REF" || echo scripts/nn_eddy/adjoint_horizon.py)
        echo ""
        echo "=== $SIDE  N=${ND}d  stride=${ST} ==="
        "$PY" "$SCRIPT" --snap "$SNAP" --season "$SEASON" --n-days "$ND" \
            --boundary-stride "$ST" --freeze mix,slope --ice rheology \
            --repeat 2 --fd-h 1e-3 --tag "equiv-${SIDE}-N${ND}-s${ST}" --results "$OUT" \
            || echo "  ($SIDE N=${ND} stride=${ST} exited $?)"
    done
done

echo ""
echo "=== VERDICT ==="
"$PY" - "$OUT" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if str(r.get("tag", "")).startswith("equiv-")]
by = {}
for r in rows:                       # last write wins, so a re-run supersedes
    by[r["tag"]] = r
bad = 0
FLOOR = 3e-5          # the plan's figure; used only when the run measures a SMALLER noise floor
print(f"{'window':>16} {'ref dL/dm':>24} {'new dL/dm':>24} {'rel':>9} {'noise':>9} {'bar':>9}")
for nd, st in ((1, 1), (2, 1), (2, 2)):
    a, b = by.get(f"equiv-ref-N{nd}-s{st}"), by.get(f"equiv-new-N{nd}-s{st}")
    if not a or not b:
        print(f"  N={nd}d stride={st}: MISSING ({'ref' if not a else ''}{' new' if not b else ''})")
        bad += 1
        continue
    ga, gb = a.get("grad"), b.get("grad")
    if ga is None or gb is None:
        print(f"  N={nd}d stride={st}: a side produced no gradient "
              f"(ref={a.get('error')} new={b.get('error')})")
        bad += 1
        continue
    rel = abs(gb - ga) / max(abs(ga), 1e-300)
    # grad_spread is max|g|/min|g| over --repeat passes: the SAME code, the SAME process, the same
    # compiled executable. Whatever it is, it is the smallest difference this hardware can call
    # meaningful, and neither side can be asked to beat it.
    noise = max((a.get("grad_spread") or 1.0) - 1.0, (b.get("grad_spread") or 1.0) - 1.0)
    bar = max(FLOOR, noise)
    flag = "" if rel <= bar else "   <-- OVER"
    bad += rel > bar
    print(f"  N={nd}d s={st}  {ga:>24.17e} {gb:>24.17e} {rel:>9.2e} {noise:>9.2e} {bar:>9.2e}{flag}")
print("")
print("bar = max(3e-5, the within-version repeat spread this run measured). A fixed 3e-5 would")
print("fail a bit-identical extraction whenever the hardware's own noise exceeds it, which it did")
print("here; see the header. If `rel` is at or below `noise`, swapping the implementation moved")
print("dL/dm no more than running the same implementation twice does.")
print("EQUIVALENCE_OK" if not bad else "EQUIVALENCE_FAILED")
PYEOF
