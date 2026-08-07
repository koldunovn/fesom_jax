#!/bin/bash
# Append-only sampler for the NG5 teacher chain's step rate.
#
# WHY THIS EXISTS. The open question on the teacher's compute budget is what fraction of the
# wall time is output rather than physics — the previous session measured ~64% and could not tell
# whether deflate was most of it. The chain's own log cannot answer it after the fact: FESOM
# prints "FESOM step: N day: D" every 960 steps but stamps no wallclock on anything, so once a
# month has finished, its step rate is unrecoverable. Sampling (now, last-step-in-log) as it runs
# recovers the whole timeline, including the stalls at each 5-day snapshot write and across the
# compression_level 1 -> 0 change at month 03 and the dt 120 -> 240 switch at month 03.
#
# Read-only with respect to everything that matters: it greps the chain's logs and appends one
# CSV line per sample to its OWN file under monitoring/. No rm, no mv, no truncation — the file
# is opened in append mode and nothing else is ever touched (house rule).
#
# Usage:  nohup bash scripts/nn_eddy/sample_teacher_rate.sh [interval_s] >>monitoring/sampler.log 2>&1 &
set -o pipefail
RUN=/work/ab0995/a270088/nn_eddy/ng5_teacher
MON=/work/ab0995/a270088/nn_eddy/monitoring
CSV=$MON/teacher_rate.csv
INT=${1:-60}
mkdir -p "$MON"
[ -s "$CSV" ] || echo "epoch,iso,jobid,jobstate,elapsed,log,mstep,day,year,clock_day,archive_kb,spd" >> "$CSV"

while true; do
    epoch=$(date +%s); iso=$(date -Is)
    read -r jobid jobstate elapsed <<< "$(squeue -u "$USER" -h -n ng5_teacher -o '%i %T %M' 2>/dev/null | head -1)"
    log=$(ls -t "$RUN"/logs/fesom.*.out 2>/dev/null | head -1)
    mstep=""; day=""; year=""
    if [ -n "$log" ]; then
        # last "FESOM step: N day: D year: Y" line — the model's own progress marker
        read -r mstep day year <<< "$(grep -aE 'FESOM step:' "$log" 2>/dev/null | tail -1 \
            | sed -E 's/.*step: *([0-9]+) *day: *([0-9]+) *year: *([0-9]+).*/\1 \2 \3/')"
    fi
    clock_day=$(awk 'NR==2{print int($2)}' "$RUN/fesom.clock" 2>/dev/null)
    # apparent size, so a sparse or striped file is counted as its data, not its blocks
    archive_kb=$(du -sk --apparent-size /work/ba1583/a270088/nn_eddy/ng5_teacher 2>/dev/null | cut -f1)
    spd=$(awk -F'[=!]' '/^ *step_per_day/{gsub(/ /,"",$2); print $2}' "$RUN/namelist.config" 2>/dev/null)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$epoch" "$iso" "${jobid:-none}" "${jobstate:-none}" "${elapsed:-}" \
        "$(basename "${log:-none}")" "${mstep:-}" "${day:-}" "${year:-}" \
        "${clock_day:-}" "${archive_kb:-}" "${spd:-}" >> "$CSV"
    sleep "$INT"
done
