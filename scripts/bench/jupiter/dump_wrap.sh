#!/bin/bash
# Arm XLA HLO dumping on SLURM process 0 only, then exec the real command.
# Usage (inside srun):  dump_wrap.sh <cmd...>   with HLO_DUMP_DIR set in the env.
# All ranks compile the same SPMD module, so one rank's dump is complete; dumping
# from all 32+ ranks would write the same multi-100MB text files N times over NFS.
if [ "${SLURM_PROCID:-0}" = "0" ] && [ -n "${HLO_DUMP_DIR:-}" ]; then
  mkdir -p "$HLO_DUMP_DIR"
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$HLO_DUMP_DIR"
fi
exec "$@"
