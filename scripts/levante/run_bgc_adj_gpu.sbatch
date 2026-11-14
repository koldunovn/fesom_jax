#!/bin/bash
# The sharded REcoM gradient (scripts/recom/core2_bgc_adj.py) on one whole A100 node of Levante (gpu, 4x A100 80GB, NVLink).
# CORE2 by default: the configs/levante config of the FESOM 2.8 reference pair, a w28sl1 restart and the dist_4 partition
# from the pool, and the settings that fit four A100 80GB on Albedo (plan 20261105, job 49013169: --remat-blocks,
# FESOM_BGC_LAYOUT=loop, 2-step chunks; 64.7 GiB on the first GPU, 102 GB of boundaries on the host per model day).
#   sbatch scripts/levante/run_bgc_adj_gpu.sbatch --days 1 --out $RUNS/bgc_adj/core2_1d
# Environment: CONFIG, RESTART, YEAR, DIST (the mesh directory holding dist_4), REMAT (default --remat-blocks; REMAT=
# drops it), FESOM_BGC_LAYOUT (default major: with the sharded placement of adjoint._placer it fits, 57.4 GiB, and halves
# the reverse, 423 against 865 s a CORE2 day, 28972957; loop is the low-memory fallback). All flags go to the driver.
#SBATCH --job-name=fj_bgcadj_gpu
#SBATCH --partition=gpu
#SBATCH --account=bk1415_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_80:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH -o /work/bk1415/a270044/fesom_jax_pool/runs/slurm/%x.%j.out
set -u
WT=${WT:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$WT/scripts/levante/env_gpu.sh"
export FESOM_BGC_LAYOUT=${FESOM_BGC_LAYOUT:-major}
CONFIG=${CONFIG:-$WT/configs/levante/core2_recom_spectral_w28sl1.yaml}
RESTART=${RESTART:-$POOL/restarts/w28sl1/seg4}
YEAR=${YEAR:-1998}
DIST=${DIST:-$POOL/fesom_jax_data/core2_mesh_ic/mesh_core2_part}
cd "$WT" || exit 2
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" || { echo "FATAL: fesom_jax not from $WT"; exit 4; }
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} $(git rev-parse --short HEAD) CONFIG=$CONFIG RESTART=$RESTART DIST=$DIST REMAT=${REMAT---remat-blocks} FESOM_BGC_LAYOUT=$FESOM_BGC_LAYOUT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" scripts/recom/core2_bgc_adj.py --config "$CONFIG" --year "$YEAR" --restart "$RESTART" --dist-dir "$DIST" \
    --npes 4 ${REMAT---remat-blocks} --chunk-steps 2 "$@"
rc=$?
echo "=== exit $rc $(date) ==="
exit $rc
