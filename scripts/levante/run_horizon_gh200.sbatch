#!/bin/bash
# The BGC gradient horizon (scripts/recom/bgc_gradient_horizon.py) on one dolpung GH200 node, CORE2 by default:
# the configs/levante config of the FESOM 2.8 reference pair and a w28sl1 restart from the pool. All flags after the
# script name go to the driver, e.g. the 1-day smoke that did not fit an Albedo A100 (89-187 GiB, plan 20261105):
#   sbatch scripts/levante/run_horizon_gh200.sbatch --days 1 --cost chl --repeats 2 --remat-blocks --out $RUNS/horizon/core2_smoke
# Environment: CONFIG (default the CORE2 w28sl1 config), RESTART (default the 2000-07-01 restart, seg4), YEAR (the
# start year of the run RESTART comes from, 1998). The driver is single-device: it uses the first of the four GPUs.
#SBATCH --job-name=fj_hor_gh200
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH -o /work/bk1415/a270044/fesom_jax_pool/runs/slurm/%x.%j.out
set -u
WT=${WT:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$WT/scripts/levante/env_gh200.sh"
CONFIG=${CONFIG:-$WT/configs/levante/core2_recom_spectral_w28sl1.yaml}
RESTART=${RESTART:-$POOL/restarts/w28sl1/seg4}
YEAR=${YEAR:-1998}
cd "$WT" || exit 2
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" || { echo "FATAL: fesom_jax not from $WT"; exit 4; }
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} $(git rev-parse --short HEAD) CONFIG=$CONFIG RESTART=$RESTART ==="
env | grep -E "^FESOM_" | sort
"$PY" scripts/recom/bgc_gradient_horizon.py --config "$CONFIG" --year "$YEAR" --restart "$RESTART" "$@"
rc=$?
echo "=== exit $rc $(date) ==="
exit $rc
