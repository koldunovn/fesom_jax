#!/bin/bash
# The every-step column dump around burst nodes (scripts/recom/core2_column_dump.py) on one A100 node of Levante:
# the forward of scripts/levante/run_bgc_adj_gpu.sbatch (same config, restart, dist_4 partition), no gradient.
#   RESTART=$POOL/restarts/w28sl1/seg3 sbatch -J fj_core2_cols_jan scripts/levante/run_column_dump_gpu.sbatch \
#       --days 30 --node 88529 108568 --out $RUNS/bgc_adj/core2_guard/h30_jan/columns
# Environment: CONFIG, RESTART, YEAR, DIST. All flags go to the driver.
#SBATCH --job-name=fj_core2_cols
#SBATCH --partition=gpu
#SBATCH --account=bk1415_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_80:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH -o /work/bk1415/a270044/fesom_jax_pool/runs/slurm/%x.%j.out
set -u
WT=${WT:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$WT/scripts/levante/env_gpu.sh"
export FESOM_BGC_LAYOUT=${FESOM_BGC_LAYOUT:-loop}
CONFIG=${CONFIG:-$WT/configs/levante/core2_recom_spectral_w28sl1.yaml}
RESTART=${RESTART:-$POOL/restarts/w28sl1/seg4}
YEAR=${YEAR:-1998}
DIST=${DIST:-$POOL/fesom_jax_data/core2_mesh_ic/mesh_core2_part}
cd "$WT" || exit 2
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" || { echo "FATAL: fesom_jax not from $WT"; exit 4; }
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} $(git rev-parse --short HEAD) CONFIG=$CONFIG RESTART=$RESTART DIST=$DIST ==="
"$PY" scripts/recom/core2_column_dump.py --config "$CONFIG" --year "$YEAR" --restart "$RESTART" --dist-dir "$DIST" \
    --npes 4 "$@"
rc=$?
echo "=== exit $rc $(date) ==="
exit $rc
