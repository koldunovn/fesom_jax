#!/bin/bash
# Long-window Task C1 — frozen-ice clean-gradient horizon + sign-flip sweep at ONE seed.
# Loops N as SEPARATE python invocations (a FRESH process per N ⇒ clean per-N peak memory,
# the A7 pattern), each appending a JSON line. Single-GPU, a100_80 (the adjoint job tier).
# Submit once per seed (winter / summer) via the SNAP/SEASON env vars:
#   SNAP=$WORK/foundation_snaps/snap_step037440.pkl SEASON=winterNH sbatch scripts/paper/core2_lw_signflip.sbatch
#   SNAP=$WORK/foundation_snaps/snap_step044640.pkl SEASON=summerNH sbatch scripts/paper/core2_lw_signflip.sbatch
#SBATCH --job-name=lw_signflip
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/lw_signflip.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/lw_signflip.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
WORK=/work/ab0995/a270088/port_jax
SNAP=${SNAP:-$WORK/foundation_snaps/snap_step037440.pkl}
SEASON=${SEASON:-winterNH}
NLIST=${NLIST:-"12 24 48 96 192"}
RES=${RES:-scripts/lw_signflip_results.jsonl}
echo "=== $(hostname) $(date) — C1 sign-flip sweep seed=$SEASON ($SNAP) N=[$NLIST] ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
for N in $NLIST; do
  echo "----- N=$N -----"
  "$PY" scripts/paper/core2_lw_signflip.py --snap "$SNAP" --season "$SEASON" --n "$N" \
      --run-start-year 1958 --results "$RES"
  echo "  (N=$N exit $?)"
done
echo "=== done $(date) ==="
