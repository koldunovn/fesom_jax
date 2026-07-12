#!/bin/bash
# Long-window Task D1 — ensemble-averaged adjoint d(10-yr-mean MLD)/d(c_k). Single-GPU, a100_80.
# Default = the full run on the 10-yr reference; override SNAPDIR/K/N/RSY for the smoke:
#   SNAPDIR=$WORK/foundation_snaps K=6 RSY=1958 sbatch scripts/paper/core2_lw_avgadj.sbatch   # plumbing smoke
#   (full, after B1):  sbatch scripts/paper/core2_lw_avgadj.sbatch
#SBATCH --job-name=lw_avgadj
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/lw_avgadj.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/lw_avgadj.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
WORK=/work/ab0995/a270088/port_jax
SNAPDIR=${SNAPDIR:-$WORK/longwindow/ref10_snaps}
K=${K:-200}
N=${N:-48}
RSY=${RSY:-1963}
TARGETS=${TARGETS:-"mld_ck t100_kgm"}
MODES=${MODES:-"adjoint"}        # adjoint | tlm | "adjoint tlm" — each mode×target is ~3.4 h, prefer separate jobs
RES=${RES:-scripts/lw_avgadj_results.jsonl}
echo "=== $(hostname) $(date) — D1 avg-sens  snaps=$SNAPDIR  K=$K N=$N start_yr=$RSY modes=[$MODES] targets=[$TARGETS] ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
mkdir -p "$WORK/longwindow/avgadj"
for MODE in $MODES; do
  for TGT in $TARGETS; do
    echo "----- mode=$MODE target=$TGT -----"
    "$PY" scripts/paper/core2_lw_avgadj.py --mode "$MODE" --target "$TGT" --snap-dir "$SNAPDIR" --K "$K" --n "$N" \
        --run-start-year "$RSY" --results "$RES"
    echo "  (mode=$MODE target=$TGT exit $?)"
  done
done
echo "exit: $? ($(date))"
