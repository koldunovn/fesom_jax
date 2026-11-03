#!/bin/bash
# Link every segment's monthly stores of a chain (scripts/albedo/chain_core2_gpu.sh ROOT) into ROOT/monthly/<YYYY_MM>.
# A month two segments split (a 5840-step half year ends 1.5 days into July) is merged, sample-weighted, into a real
# store there instead (merge_monthly.py); a stale link of such a month is replaced.
set -euo pipefail
ROOT=${1:?ROOT}; mkdir -p "$ROOT/monthly"
PY=${PY:-/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
for name in $(ls -d "$ROOT"/seg*/monthly/*/ 2>/dev/null | xargs -n1 basename | sort -u); do
    parts=("$ROOT"/seg*/monthly/"$name")
    if [ ${#parts[@]} -gt 1 ]; then
        [ -L "$ROOT/monthly/$name" ] && rm "$ROOT/monthly/$name"
        [ -e "$ROOT/monthly/$name" ] || "$PY" "$HERE/merge_monthly.py" "$ROOT/monthly/$name" "${parts[@]}"
    else
        [ -e "$ROOT/monthly/$name" ] || ln -s "${parts[0]}/" "$ROOT/monthly/$name"
    fi
done
ls "$ROOT/monthly"
