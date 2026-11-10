#!/bin/bash
# Submit ONE chain step of the sea-ice calibration (scripts/icecal/icecal.py chain-step) for one
# window to one dolpung GH200, through the dolpung wrapper:
#
#   scripts/icecal/job_chain_gh200.sh <run_dir> <window>
#
# Prints the job id (last line) and appends it to <run_dir>/jobs.txt and to the ledger
# ($ICECAL_LEDGER, default runs/icecal/submitted.txt; one tab-separated line: date, job id, task,
# purpose, GPU-h estimate). Used by chain-step for the next evaluation, and by hand to start or
# resume a chain.
#
# Inside a dolpung job (a chain step submitting the next one) the new job waits for this one
# (--dependency=afterany:$SLURM_JOB_ID): the chain hands its GPU on, the account's GPU count does not
# grow, and no cap check is needed. By hand, it refuses (nothing submitted) with exit 5 if the
# account already holds >= 4 dolpung GPUs (the cap is for ALL sessions together: running jobs plus
# pending jobs not waiting on a dependency; a pending afterany job takes over its predecessor's
# GPU). Exit 3 if <run_dir>/STOP exists, 6 if squeue cannot be read; any other non-zero status is
# sbatch's. Never deletes, moves or edits anything.
#
# Environment: WT (the checkout; default: this script's), ICECAL_WRAPPER (default the dolpung
# wrapper), ICECAL_LEDGER, ICECAL_MEM (default 200G; 300G if a measured VmHWM exceeds 180 GB),
# ICECAL_TIME (default 07:50:00).
set -u
R=${1:?usage: job_chain_gh200.sh <run_dir> <window>}
WIN=${2:?usage: job_chain_gh200.sh <run_dir> <window>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WT=${WT:-$(cd "$HERE/../.." && pwd)}
WRAP=${ICECAL_WRAPPER:-/work/ab0995/a270088/fesom_jax_dev/runs/dolpung/job_gh200.sbatch}
LEDGER=${ICECAL_LEDGER:-/work/ab0995/a270088/fesom_jax_dev/runs/icecal/submitted.txt}
MEM=${ICECAL_MEM:-200G}
WALL=${ICECAL_TIME:-07:50:00}
MAX_GPU=4

if [ -e "$R/STOP" ]; then
    echo "job_chain_gh200: $R/STOP exists; not submitting"
    exit 3
fi
if [ ! -f "$R/run_config.json" ]; then
    echo "job_chain_gh200: $R/run_config.json missing (run chain-init first)"
    exit 2
fi
dep=()
if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_PARTITION:-}" = dolpung ]; then
    dep=(--dependency="afterany:$SLURM_JOB_ID")
else
    q=$(squeue -p dolpung -u a270088 -h -o "%T %r") || { echo "job_chain_gh200: squeue failed"; exit 6; }
    n=$(printf '%s\n' "$q" | awk '$1 == "RUNNING" || ($1 == "PENDING" && $2 !~ /^Dependency/)' | grep -c .)
    if [ "$n" -ge "$MAX_GPU" ]; then
        echo "job_chain_gh200: $n dolpung GPUs already held or eligible for a270088 (cap $MAX_GPU); not submitting"
        exit 5
    fi
fi
mkdir -p "$R/logs"
out=$(sbatch --parsable "${dep[@]}" --export=ALL,WT="$WT" --time="$WALL" --mem="$MEM" \
             --job-name="ic_ch$WIN" -o "$R/logs/chain_${WIN}.%j.out" \
             "$WRAP" scripts/icecal/icecal.py chain-step --run-dir "$R" --window "$WIN")
rc=$?
if [ $rc -ne 0 ]; then
    echo "job_chain_gh200: sbatch failed (rc=$rc): $out"
    exit $rc
fi
jid=${out%%;*}
echo "$jid" >> "$R/jobs.txt"
printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%F)" "$jid" "S2/S3" \
    "icecal chain-step $R window $WIN (dolpung GH200, one evaluation)" "~3.8 GPU-h" >> "$LEDGER"
echo "$jid"
