#!/bin/bash
# One-time JUPITER data prep for ONE mesh: raw FESOM ASCII -> the JAX `.npy` export, then the
# cached PHC3.0 winter IC in the same dir (so `--full 1` runs never interpolate the global PHC
# nc live).  The repo ships only the tiny `pi` mesh — on Levante `data/` was a symlink to
# /work, so every JUPITER mesh export is generated HERE.  VERIFIED 2026-07-23.
#
#   MESH_NAME=ng5 sbatch scripts/bench/jupiter/prepare_mesh_jupiter.sbatch
#
# Costs (measured): core2 6.5 s / 41 MB.  The (nod2D, nl) float64 arrays dominate, so the
# export scales as nod2D*nl: farc 638k*48, dars 3.16M*57, ng5 7.4M*70 (~20 GB, the big one).
# Run on a compute node, not the login node — peak host RAM is several x the export size.
#
#SBATCH --job-name=jax_prep
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH -o scripts/logs/prep_%x_%j.out
#SBATCH -e scripts/logs/prep_%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/bench/jupiter/env_jupiter.sh
unset JDIST                                   # single process, no jax.distributed here

MESH_NAME="${MESH_NAME:?set MESH_NAME=core2|farc|dars|ng5}"
RAW_VAR="RAW_${MESH_NAME}"; RAW="${!RAW_VAR:?no RAW_$MESH_NAME in env_jupiter.sh}"
OUT="$MESH/mesh_${MESH_NAME}"

echo "=== $(date) prep $MESH_NAME: $RAW -> $OUT on $(hostname) ==="
mkdir -p "$OUT"

if [ -f "$OUT/meta.txt" ]; then
  echo "[prep] mesh export already present, skipping prepare_mesh"
else
  time $PY scripts/prepare_mesh.py "$RAW" "$OUT"
fi

if [ -f "$OUT/T_ic.npy" ]; then
  echo "[prep] PHC IC already cached, skipping"
else
  time $PY scripts/cache_phc_ic.py --mesh-dir "$OUT" --out-dir "$OUT"
fi

echo "=== done $(date) ==="
du -sh "$OUT"; ls "$OUT" | wc -l; cat "$OUT/meta.txt"
