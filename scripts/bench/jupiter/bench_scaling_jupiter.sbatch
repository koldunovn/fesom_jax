#!/bin/bash
# FESOM2-JAX strong-scaling on JUPITER (GH200) — TEMPLATE. See docs/JUPITER_SCALING.md.
# Drives ONE mesh across several node counts in one allocation via sub-allocation sruns.
# ⚠️VERIFY every #SBATCH line (partition/account/qos) on the machine before submitting.
#
#   MESH_NAME=ng5 NODES_LIST="8 16 32" sbatch scripts/bench/jupiter/bench_scaling_jupiter.sbatch
#
#SBATCH --job-name=jax_scale_jupiter
#SBATCH --account=CHANGEME                 # ⚠️VERIFY: your JSC compute project
#SBATCH --partition=booster                # ⚠️VERIFY the GH200 partition name
#SBATCH --nodes=32                          # set to max(NODES_LIST); MaxNodes=UNLIMITED per Kokkos run
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72                  # 72-core Grace per superchip
#SBATCH --time=02:00:00
#SBATCH --gpu-bind=none                     # jax maps local GPUs itself (or use --gpus-per-task=1)
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/bench/jupiter/env_jupiter.sh

MESH_NAME="${MESH_NAME:-core2}"
NODES_LIST="${NODES_LIST:-1 2}"            # core2 bootstrap default
STEPS="${STEPS:-150}"
REPS="${REPS:-2}"

# per-mesh production physics + dt + transport (best-per-point: padded <=16 GPU, coloured beyond)
case "$MESH_NAME" in
  core2)   MDIR="data/mesh_core2";        DDIR="$DIST/core2";    IC="data/ic_core2"; DT=1800; PHYS="--mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1" ;;
  farc)    MDIR="$MESH/mesh_farc";        DDIR="$DIST/farc";     IC="$MESH/mesh_farc"; DT=900;  PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
  dars)    MDIR="$MESH/mesh_dars";        DDIR="$DIST/dars";     IC="$MESH/mesh_dars"; DT=120;  PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
  forca20) MDIR="$MESH/mesh_forca20";     DDIR="$DIST/forca20";  IC="$MESH/mesh_forca20"; DT=240; PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
  ng5)     MDIR="$MESH/mesh_ng5";         DDIR="$DIST/ng5";      IC="$MESH/mesh_ng5"; DT=180;  PHYS="--mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1" ;;
  *) echo "unknown MESH_NAME=$MESH_NAME" >&2; exit 2 ;;
esac

C="--name $MESH_NAME --mesh-dir $MDIR --dist-dir $DDIR --ic-dir $IC --steps $STEPS \
   --dt $DT --full 1 $PHYS --year 1958"

echo "=== $(date) JUPITER scaling $MESH_NAME nodes=[$NODES_LIST] on $SLURM_JOB_NODELIST ==="
for N in $NODES_LIST; do
  NP=$((N * GPUS_PER_NODE))
  TR=$([ "$NP" -le 16 ] && echo padded || echo coloured)      # best transport per point
  if [ "$N" -gt 1 ]; then export JDIST=1; else unset JDIST; fi
  for REP in $(seq 1 "$REPS"); do
    echo "#### $MESH_NAME ${NP} GPU ($N node) $TR rep=$REP ####"
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/bench_forward_scaling.py $C --npes "$NP" --halo "$TR" \
      || echo "#### ${NP} GPU rep=$REP rc=$? (read the [bench-finite] line before trusting anything) ####"
  done
done
echo "=== done $(date) ==="
