#!/bin/bash
# Finite-difference validation of the LONG windows — the one check the horizon claim is missing.
#
# Why. The 2/5/10-day rows were validated against finite differences (4.2% at 5 d, 5.4-5.8% at
# 10 d). The 20- and 30-day rows were not: adjoint_month.sbatch ran them with `--fd-h ""`, so
# everything we know about them is that they are reproducible in-process and that the cotangent
# grows smoothly. That is exactly the evidence class that made the ORIGINAL 1-day gate wrong —
# it too was self-consistent by its own criterion, and it disagreed with FD by 411%. Until an
# independent check exists past 10 days, a training curriculum that reaches for 30-day windows
# is resting on the same kind of evidence that already failed once.
#
# Also: the headline 20/30-day rows ran with the ice fully frozen in the backward (the script
# default), NOT with `--ice rheology`, which is what Task 9 locks into the training config. The
# third run below closes that gap at the window length that matters.
#
# The 30-day reference trajectory is already cached (horizon_truth/snap_step037440_m1.25_30d.npz,
# built by job 26768058), so nothing here rebuilds truth.
#
# Cost: --fd-h takes ONE value; each value costs two extra full forwards (central difference).
# At 30 d a forward is ~480 s, so an FD point adds ~16 min to a ~45 min gradient. Cheap.
#SBATCH --job-name=nn_fdlong
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o scripts/nn_eddy/fdlong.%j.out
#SBATCH -e scripts/nn_eddy/fdlong.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

echo "=== $(hostname) $(date) — FD check at 20 and 30 days ==="
echo "=== checkout: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="
# The editable install resolves fesom_jax to the MAIN checkout ~/port_jax; the worktree only
# wins because the script puts its own root on sys.path. Assert it rather than trust it.
"$PY" - <<'EOF' || exit 1
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import fesom_jax
here, got = pathlib.Path.cwd().resolve(), pathlib.Path(fesom_jax.__file__).resolve()
print(f"[import] fesom_jax <- {got}")
if here not in got.parents:
    sys.exit(f"FATAL: fesom_jax imported from {got}, not from the checkout under test {here}")
EOF

SNAP=/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl
RES=scripts/nn_eddy/adjoint_horizon_results.jsonl

go () { # go <ndays> <ice> <tag>
  echo; echo "----- N=$1 d  ice=$2  freeze=mix  fd-h=1e-3 -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season winterNH --n-days "$1" \
      --truth-days 30 --chunk-steps 48 --repeat 1 --fd-h 1e-3 --run-start-year 1958 \
      --ice "$2" --freeze mix --tag "$3" --results "$RES"
  echo "  (exit $?)"
}

# Shortest first, so a time-cut still leaves the 20-day answer on the same reference.
go 20 frozen   fdlong-20d            # validates the headline 20 d row
go 30 frozen   fdlong-30d            # validates the headline 30 d row — the one that sets H
go 30 rheology fdlong-30d-rheology   # the config Task 9 actually locks in
rc=$?
echo; echo "=== done $(date) rc=$rc ==="
exit $rc
