"""Start (or resume) an OMIP cycle experiment as a chain of jobs on one dolpung GH200 node (fesom3-jax's
tools/submit.py, adapted to fesom-jax).

    submit.py EXPERIMENT_DIR ROOT [--time 08:00:00] [--resubmit 100] [--s-per-year 150] [--dry-run]

EXPERIMENT_DIR is experiments/<name> (experiment.toml); ROOT is the run root (a new directory, or an existing run to
resume). The jobs run a frozen snapshot of this checkout's HEAD (/work/ab0995/a270088/fesom_jax_dev/wt_omip_<sha>,
a detached worktree with the data symlink; uncommitted changes to tracked files are refused).

What it writes into ROOT (an existing file is kept: a resume never replaces the run's own copies):
  experiment.toml      the run's copy; every job reads it at its start (cycles)
  tools/run_cycles.py  the planner the jobs call (frozen with the run)
  chain.sbatch         the job: whole years while they fit its time limit, then it submits itself again
  logs/                the jobs' logs; cycle<N>/run/<segment>.log the full run_from_config.py output of a segment

dolpung: at most two nodes of this user at a time over all sessions. This script and every resubmission count the
user's other dolpung jobs (running and pending) and, at two or more, queue the job behind one of them
(--dependency=afterany) instead of next to them.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1]
DEV = Path("/work/ab0995/a270088/fesom_jax_dev")
DATA = Path("/work/ab0995/a270088/port_jax/data")
DIST_DIR = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2"
ARM_ENV_SH = "/home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh"
ARM_EXTRA = DEV / "envs" / "arm_extra" / "site"

# The other dolpung jobs of this user (running or pending, this job excluded) and, at two or more, a dependency on
# the first running one (else the first pending one). Shared by submit.py and the chain's resubmission.
DOLPUNG_GUARD = r"""dolpung_dep() {
    local mine=${1:-} jobs n first
    jobs=$(squeue -h -p dolpung -u "$USER" -t R,PD -o "%i %t" | awk -v m="$mine" '$1 != m' | sort -k2,2r)
    n=$(printf '%s\n' "$jobs" | grep -c .)
    if [ "$n" -ge 2 ]; then
        first=$(printf '%s\n' "$jobs" | head -1 | cut -d' ' -f1)
        echo "--dependency=afterany:$first"
    fi
}"""


def snapshot(checkout):
    """A detached worktree of the checkout's HEAD under fesom_jax_dev (created once per commit)."""
    def git(*a):
        return subprocess.run(["git", "-C", str(checkout), *a], check=True, capture_output=True, text=True).stdout
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise SystemExit(f"submit: uncommitted changes to tracked files in {checkout}:\n{dirty}")
    sha = git("rev-parse", "HEAD").strip()
    snap = DEV / f"wt_omip_{sha[:7]}"
    if not snap.exists():
        git("worktree", "add", "--detach", str(snap), sha)
        (snap / "data").symlink_to(DATA)
    elif git("-C", str(snap), "rev-parse", "HEAD").strip() != sha:
        raise SystemExit(f"submit: {snap} exists at another commit")
    return snap, sha


def chain_script(e, root, snap, sha, time_limit, resubmit):
    log = root / "logs" / "chain-%j.out"
    return f"""#!/bin/bash
#SBATCH --job-name=omip-{root.name}
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --output={log}
# {e['name']}: {e.get('description', '')}
# scripts/omip/submit.py, fesom-jax {sha[:12]} ({snap})
set -u
ROOT={shlex.quote(str(root))}
SNAP={shlex.quote(str(snap))}
source {ARM_ENV_SH}                     # sets PY (the aarch64 venv); unsets PYTHONPATH
export PYTHONPATH=$SNAP:{ARM_EXTRA}
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${{JAX_COMPILATION_CACHE_DIR+x}}" ]; then echo "FATAL: JAX_COMPILATION_CACHE_DIR is set"; exit 2; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1
export XLA_FLAGS=""                     # CMDBUF on: XLA's command buffers kept (the fast-set measurements)
RESUBMIT=${{RESUBMIT:-{resubmit}}}
S_PER_YEAR=${{S_PER_YEAR:-150}}          # wall seconds a model year took in the last segment
MARGIN=${{MARGIN:-900}}                  # seconds kept free at the end of the job
{DOLPUNG_GUARD}
cd "$SNAP" || exit 2
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$SNAP/') else 4)" \\
    || {{ echo "FATAL: fesom_jax does not import from $SNAP"; exit 4; }}
end=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
echo "chain job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: root $ROOT, snapshot $SNAP, ends $(date -d @"$end" +%FT%T)"
eval "$("$PY" "$ROOT/tools/run_cycles.py" env "$ROOT")"      # CONFIG DT PART CYCLES RYEARS RUN_OPTIONS
while :; do
    budget=$(( end - $(date +%s) - MARGIN ))
    line=$("$PY" "$ROOT/tools/run_cycles.py" next "$ROOT" --budget-s "$budget" --s-per-year "$S_PER_YEAR" \\
        --tag "$SLURM_JOB_ID")
    echo "[$(date +%FT%T)] next: $line (budget $budget s, $S_PER_YEAR s a year)"
    set -- $line
    case $1 in
        done) echo "all $CYCLES cycles done"; exit 0 ;;
        wait) break ;;
        stop) echo "the chain stops: $*"; exit 3 ;;
    esac
    c=$2; rin=$3; steps=$4; out=$5; arch=$6; cyc=$7
    extra=()
    [ "$rin" != - ] && extra+=(--restart-in "$rin")
    [ "$cyc" = 1 ] && extra+=(--cycle-start)
    mkdir -p "$ROOT/cycle$c/run"
    seglog="$ROOT/cycle$c/run/$(basename "$out").log"
    t0=$(date +%s)
    "$PY" scripts/run_from_config.py "$CONFIG" --dist-dir {DIST_DIR} --partition "$PART" --steps "$steps" \\
        --yearly-out "$out" --restart-archive-out "$arch" --restart-archive-period year \\
        --restart-archive-length "$RYEARS" --diagnostics --progress "${{extra[@]}}" $RUN_OPTIONS > "$seglog" 2>&1
    rc=$?
    grep -E "cycle_start|precision=|split_explicit|CFL|NG5_R0|RUN_DRIVER_OK|Error|error:|Traceback" "$seglog" \\
        | grep -v cuda_vmm | cut -c1-300 | head -40
    if [ "$rc" -ne 0 ]; then echo "run_from_config.py failed (exit $rc, $seglog): the chain stops"; exit "$rc"; fi
    t=$(( $(date +%s) - t0 ))
    S_PER_YEAR=$(( t * 86400 * 365 / (steps * DT) + 1 ))
    echo "[$(date +%FT%T)] segment done: $steps steps in $t s ($S_PER_YEAR s a model year)"
done
if [ "$RESUBMIT" -gt 0 ]; then
    dep=$(dolpung_dep "$SLURM_JOB_ID")
    echo "cycles left: resubmitting $ROOT/chain.sbatch ($((RESUBMIT - 1)) more after it) ${{dep:-}}"
    sbatch $dep --export=ALL,RESUBMIT=$((RESUBMIT - 1)),S_PER_YEAR="$S_PER_YEAR" "$ROOT/chain.sbatch"
else
    echo "cycles left and no resubmissions left: sbatch $ROOT/chain.sbatch"
fi
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("--time", default="08:00:00")
    ap.add_argument("--resubmit", type=int, default=100)
    ap.add_argument("--s-per-year", type=int, default=150, help="the first job's guess of wall seconds a model year")
    ap.add_argument("--dry-run", action="store_true", help="write ROOT/chain.sbatch, do not submit")
    a = ap.parse_args(argv)
    exp = a.experiment.resolve()
    e = tomllib.loads((exp / "experiment.toml").read_text())
    root = a.root.resolve()
    if str(root).startswith("/work/ab0995/a270088/port_jax/runs/"):
        raise SystemExit(f"submit: {root} is under the production runs")
    snap, sha = snapshot(CHECKOUT)
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    for src, dst in ((exp / "experiment.toml", root / "experiment.toml"),
                     (snap / "scripts" / "omip" / "run_cycles.py", root / "tools" / "run_cycles.py")):
        if dst.exists():
            print(f"submit: keeping the run's {dst.relative_to(root)}")
        else:
            shutil.copy2(src, dst)
    chain = root / "chain.sbatch"
    chain.write_text(chain_script(e, root, snap, sha, a.time, a.resubmit))
    print(f"submit: {chain} (fesom-jax {sha[:12]}, snapshot {snap}, {a.time})")
    r = subprocess.run([sys.executable, str(root / "tools" / "run_cycles.py"), "next", str(root)], cwd=snap,
                       capture_output=True, text=True)
    print(f"submit: the first segment would be: {(r.stdout or r.stderr).strip()}")
    if a.dry_run:
        return
    dep = subprocess.run(["bash", "-c", DOLPUNG_GUARD + "\ndolpung_dep"], capture_output=True, text=True).stdout.split()
    env = dict(os.environ, S_PER_YEAR=str(a.s_per_year))
    r = subprocess.run(["sbatch", "--parsable", *dep, "--export=ALL", str(chain)], capture_output=True, text=True,
                       env=env)
    if r.returncode != 0:
        raise SystemExit(f"submit: sbatch: {r.stderr.strip()}")
    print(f"submit: job {r.stdout.strip().split(';')[0]} {' '.join(dep)}; logs in {root / 'logs'}")


if __name__ == "__main__":
    main()
