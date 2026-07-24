#!/bin/bash
# FESOM2-JAX strong-scaling on JUPITER (GH200).  **VERIFIED ON THE MACHINE 2026-07-23**
# (replaced the untested template).  Drives ONE mesh across several node counts inside ONE
# allocation via sub-allocation sruns — so every rung of a curve sits on the SAME hardware
# (the Levante campaign burned a session chasing an "allocation fluke" at 128; same-alloc
# rungs are how you avoid re-running that).  Overlap one rung between allocations when a
# curve has to be split across jobs, to calibrate any drift.
#
#   MESH_NAME=ng5 NODES_LIST="8 16" sbatch -N16 scripts/bench/jupiter/bench_scaling_jupiter.sbatch
#
# -N must be >= max(NODES_LIST).  MaxNodes=UNLIMITED on booster; 128-256-node jobs start
# within minutes (Kokkos twin).
#
#SBATCH --job-name=jax_scale
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72                  # 72-core Grace per superchip
#SBATCH --time=04:00:00
#SBATCH --gpu-bind=none                     # jax maps its local GPUs itself
#SBATCH -o scripts/logs/jupiter/bench_%x_%j.out
#SBATCH -e scripts/logs/jupiter/bench_%x_%j.out
set -uo pipefail                            # NOT -e: a failed rung must not kill the curve
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh

MESH_NAME="${MESH_NAME:-core2}"
NPES_LIST="${NPES_LIST:-}"                  # GPU counts, e.g. "1 2 4 8" (preferred)
NODES_LIST="${NODES_LIST:-}"                # legacy: node counts, npes = N*GPUS_PER_NODE
STEPS="${STEPS:-150}"
REPS="${REPS:-2}"
HALO_LIST="${HALO_LIST:-auto}"              # "auto" = best-per-point; "padded coloured" = A/B
AB_MAX_NPES="${AB_MAX_NPES:-100000}"        # only A/B at or below this GPU count; above it,
                                            # fall back to the best-per-point pick (running a
                                            # transport that is near-certain to lose at 256
                                            # GPUs costs 64 nodes for a foregone conclusion)
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"          # seconds per single (npes, halo, rep) run
if [ -z "$NPES_LIST" ]; then
  for N in ${NODES_LIST:-1 2}; do NPES_LIST="$NPES_LIST $((N * GPUS_PER_NODE))"; done
fi

# Per-mesh production physics + production dt (SYPD = dt/(s_step*365.25)).
# GM is core2-only (paper's per-mesh production model).  dars cold-starts at dt=120 for
# stability; per-step cost is dt-independent, so SYPD still uses the production dt (240).
RAW_VAR="RAW_${MESH_NAME}"
MDIR="$MESH/mesh_${MESH_NAME}"; IC="$MDIR"; DDIR="${!RAW_VAR:?no RAW_$MESH_NAME}"
case "$MESH_NAME" in
  core2) DT=1800; DT_PROD=1800; PHYS="--mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1" ;;
  farc)  DT=900;  DT_PROD=1200; PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
  dars)  DT=120;  DT_PROD=240;  PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
  ng5)   DT=180;  DT_PROD=240;  PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
  *) echo "unknown MESH_NAME=$MESH_NAME" >&2; exit 2 ;;
esac

C="--name $MESH_NAME --mesh-dir $MDIR --dist-dir $DDIR --ic-dir $IC --steps $STEPS \
   --dt $DT --full 1 $PHYS --year 1958"

echo "=== $(date) JUPITER scaling $MESH_NAME nodes=[$NODES_LIST] steps=$STEPS reps=$REPS"
echo "=== alloc: $SLURM_NNODES nodes | $SLURM_JOB_NODELIST"
echo "=== dt=$DT dt_prod=$DT_PROD phys=$PHYS"
for NP in $NPES_LIST; do
  N=$(( (NP + GPUS_PER_NODE - 1) / GPUS_PER_NODE )); [ "$N" -lt 1 ] && N=1
  # Sub-node rungs (1/2 GPU) run inside a 4-GPU node: the model takes jax.devices()[:npes].
  if [ "$N" -gt 1 ]; then export JDIST=1; else unset JDIST || true; fi
  if [ "$HALO_LIST" = "auto" ] || [ "$NP" -gt "$AB_MAX_NPES" ]; then
    HALOS=$([ "$NP" -le 16 ] && echo padded || echo coloured)      # best transport per point
  else HALOS="$HALO_LIST"; fi
  for TR in $HALOS; do
    for REP in $(seq 1 "$REPS"); do
      echo "#### $MESH_NAME ${NP}GPU (${N}N) halo=$TR rep=$REP dt_prod=$DT_PROD ####"
      # Per-run timeout: one hung collective must not eat the whole allocation.  The Levante
      # campaign lost a job to exactly this (the padded halo at 64 GPUs, job 26228433), where
      # a stall is indistinguishable from a slow trace/compile.  Generous enough for the NG5
      # compile (~1-3 min) + setup + 150 steps; kills a genuine hang.
      timeout --kill-after=120 "$RUN_TIMEOUT" \
        srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
          $PY scripts/bench/bench_forward_scaling.py $C --npes "$NP" --halo "$TR"
      rc=$?
      [ $rc -eq 0 ] || echo "#### $MESH_NAME ${NP}GPU halo=$TR rep=$REP rc=$rc $([ $rc -ge 124 ] && \
        echo '(TIMEOUT after '"$RUN_TIMEOUT"'s — hang or pathological compile)') ####"
    done
  done
done
echo "=== done $(date) ==="
