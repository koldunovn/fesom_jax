#!/bin/bash
# One round of a multi-job TKE-constant calibration (scripts/calib/tke_lbfgs.py mjstep / winjob):
#
#   scripts/calib/mj_round.sh <run_dir> <months> [tke_lbfgs options, WITHOUT --months/--out]
#
# 1. runs the coordinator `tke_lbfgs.py mjstep` (no model: it replays the optimiser log and sums
#    the per-window result files; seconds, runs where this script runs);
# 2. if the next evaluation still misses windows, submits one CPU job per missing month
#    (scripts/adjoint/job_cpu.sbatch ... winjob --months <m> --eval <k>), recording each job id
#    in <run_dir>/eval_<k>/job_<MM>.txt.
# Call it again once those jobs have ended. A window whose job ended without its result file is
# resubmitted once; a second failure stops with exit code 3. Jobs on a user hold are set aside
# (the window is submitted again), so a round can be re-queued with other limits. Every job gets the same options, so
# the cost, controls and window are those of the coordinator. Nothing is deleted or overwritten.
# MJ_TIME (default 00:30:00: on this system short jobs start first) is the window jobs' limit;
# MJ_MULTI=1 submits the round as ONE multi-node job (scripts/calib/mj_round.sbatch, one window per
# node), which counts once against the per-user running-job limit.
set -euo pipefail
WT=${WT:-$(cd "$(dirname "$0")/../.." && pwd)}
RUN=$1
MONTHS=$2
shift 2
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export PYTHONPATH=$WT
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then
    echo "FATAL: JAX_COMPILATION_CACHE_DIR is set; no persistent XLA cache"
    exit 2
fi

out=$(JAX_PLATFORMS=cpu "$PY" "$WT/scripts/calib/tke_lbfgs.py" mjstep --months "$MONTHS" \
      --out "$RUN" "$@" | tail -1)
echo "$out"
field() { echo "$out" | "$PY" -c "import json,sys; r=json.loads(sys.stdin.read()); v=r['$1']; print(' '.join(map(str, v)) if isinstance(v, list) else v)"; }
[ "$(field status)" = "need" ] || exit 0
k=$(field eval)
d=$(printf "%s/eval_%03d" "$RUN" "$k")
mkdir -p "$d" "$RUN/logs"
todo=()
for m in $(field missing); do
    mm=$(printf "%02d" "$m")
    jf="$d/job_$mm.txt"
    if [ -f "$jf" ]; then
        # active = running, or pending without a user hold; held jobs are set aside (not
        # failures); only jobs that ENDED without the result file count toward the limit
        active=""
        ended=0
        # a line "#reset" (appended by hand after fixing a bug) restarts the failure count
        for j in $(awk '/^#reset/{n=NR} {a[NR]=$0} END{for (i=n+1; i<=NR; i++) print a[i]}' "$jf"); do
            st=$(sacct -j "$j" -X -n -P -o State | head -1)
            case "$st" in
                RUNNING|REQUEUED|CONFIGURING|COMPLETING) active=$j ;;
                PENDING|"")
                    reason=$(squeue -j "$j" -h -o %r 2>/dev/null || true)
                    [ "$reason" = "JobHeldUser" ] || active=$j ;;
                *) ended=$((ended + 1)) ;;
            esac
        done
        if [ -n "$active" ]; then
            echo "month $m: job $active active"
            continue
        fi
        if [ "$ended" -ge 2 ]; then
            echo "month $m: jobs $(tr '\n' ' ' < "$jf")ended without a result"
            exit 3
        fi
    fi
    todo+=("$m")
done
[ "${#todo[@]}" -gt 0 ] || exit 0
if [ "${MJ_MULTI:-0}" = "1" ]; then
    # one multi-node job for the whole round (one entry against the per-user job limit)
    ms=$(IFS=:; echo "${todo[*]}")                 # colons: --export splits on commas
    jid=$(sbatch --parsable --job-name="mjr_e${k}" --nodes="${#todo[@]}" \
          --export=ALL,WT="$WT",MJ_MONTHS="$ms",MJ_EVAL="$k" --time="${MJ_TIME:-00:30:00}" \
          -o "$RUN/logs/%x.%j.out" "$WT/scripts/calib/mj_round.sbatch" --out "$RUN" "$@")
    for m in "${todo[@]}"; do
        echo "$jid" >> "$(printf "%s/job_%02d.txt" "$d" "$m")"
    done
    echo "months $ms: submitted $jid (${#todo[@]} nodes)"
    exit 0
fi
for m in "${todo[@]}"; do
    mm=$(printf "%02d" "$m")
    jid=$(sbatch --parsable --job-name="mj_e${k}_m${mm}" --export=ALL,WT="$WT" --mem=200G \
          --time="${MJ_TIME:-00:30:00}" -o "$RUN/logs/%x.%j.out" \
          "$WT/scripts/adjoint/job_cpu.sbatch" "$WT/scripts/calib/tke_lbfgs.py" winjob \
          --months "$m" --eval "$k" --out "$RUN" "$@")
    echo "$jid" >> "$d/job_$mm.txt"
    echo "month $m: submitted $jid"
done
