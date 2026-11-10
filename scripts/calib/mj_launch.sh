#!/bin/bash
# Start a multi-job TKE-constant calibration with persistent window workers
# (scripts/calib/tke_lbfgs.py winloop):
#
#   scripts/calib/mj_launch.sh <run_dir> <months> [tke_lbfgs options, WITHOUT --months/--mj-months/--out]
#
# One CPU job per month (scripts/adjoint/job_cpu.sbatch, --mem=200G). Each worker compiles its
# window once and loops over the evaluations: it waits for <run_dir>/eval_<k>/u.json, writes its
# eval_<k>/win_<MM>.json, and the worker that completes an evaluation runs the coordinator, which
# posts the next point (L-BFGS or Adam, from the same options). Every worker gets the SAME
# options. A worker that ends early (time limit) can be started again with the same command:
# it skips the evaluations it already wrote. scripts/calib/mj_round.sh drives a run by hand
# instead (one job per window and evaluation). Job ids go to <run_dir>/workers.txt.
# MJ_TIME (default 08:00:00) and MJ_DEP (an sbatch --dependency value) are optional.
set -euo pipefail
WT=${WT:-$(cd "$(dirname "$0")/../.." && pwd)}
RUN=$1
MONTHS=$2
shift 2
mkdir -p "$RUN/logs"
dep=()
if [ -n "${MJ_DEP:-}" ]; then
    dep=(--dependency="$MJ_DEP")
fi
for m in ${MONTHS//,/ }; do
    mm=$(printf "%02d" "$m")
    jid=$(sbatch --parsable "${dep[@]}" --job-name="mjw_$(basename "$RUN")_m$mm" \
          --export=ALL,WT="$WT" --mem=200G --time="${MJ_TIME:-08:00:00}" \
          -o "$RUN/logs/%x.%j.out" "$WT/scripts/adjoint/job_cpu.sbatch" \
          "$WT/scripts/calib/tke_lbfgs.py" winloop --months "$m" --mj-months "$MONTHS" \
          --out "$RUN" "$@")
    echo "$m $jid" | tee -a "$RUN/workers.txt"
done
