#!/bin/bash
# Freeze a commit of this repository for a job: git archive into /work/ba1583/a270088/awicm/ocn/snap/<sha>
# (once per commit; an existing snapshot is reused). Prints the snapshot directory.
set -eu
c=$(git -C "$(dirname "$0")/../.." rev-parse --short=10 "${1:-HEAD}")
d=/work/ba1583/a270088/awicm/ocn/snap/$c
if [ ! -e "$d/SNAPSHOT_COMMIT" ]; then
    mkdir -p "$d"
    git -C "$(dirname "$0")/../.." archive "$c" | tar -x -C "$d"
    echo "$c" > "$d/SNAPSHOT_COMMIT"
fi
echo "$d"
