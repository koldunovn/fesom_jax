#!/bin/bash
# Link every segment's monthly stores of a chain (scripts/albedo/chain_core2_gpu.sh ROOT) into ROOT/monthly/<YYYY_MM>.
set -euo pipefail
ROOT=${1:?ROOT}; mkdir -p "$ROOT/monthly"
for m in "$ROOT"/seg*/monthly/*/; do
    [ -d "$m" ] || continue; name=$(basename "$m")
    [ -e "$ROOT/monthly/$name" ] || ln -s "$m" "$ROOT/monthly/$name"
done
ls "$ROOT/monthly"
