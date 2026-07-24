#!/bin/bash
# Wrap SLURM process 0 in `nsys profile`; all other ranks exec the command directly.
# Usage (inside srun): nsys_wrap.sh <cmd...>  with NSYS_OUT set to the report path.
if [ "${SLURM_PROCID:-0}" = "0" ] && [ -n "${NSYS_OUT:-}" ] && command -v nsys >/dev/null; then
  mkdir -p "$(dirname "$NSYS_OUT")"
  exec nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none -f true \
       -o "$NSYS_OUT" "$@"
fi
exec "$@"
