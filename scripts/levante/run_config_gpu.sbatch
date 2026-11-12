#!/bin/bash
# A forward run (scripts/run_from_config.py) on one GPU of Levante's A100 partition (gpu, x86, A100 80GB), e.g. continuing the CORE2 reference pair:
#   sbatch scripts/levante/run_config_gpu.sbatch configs/levante/core2_recom_spectral_w28sl1.yaml \
#       --year 1998 --restart-in $POOL/restarts/w28sl1/seg5 --steps 11680 --restart-out $RUNS/w28sl1/seg6/restart \
#       --monthly-out $RUNS/w28sl1/seg6/monthly
# All arguments go to run_from_config.py; the account is one GPU account of yours (`sbatch -A <project>_gpu`); for
# four GPUs `--gres=gpu:a100_80:4`. A sharded run over the four GPUs needs a dist_4 partition of the mesh
# (`partition:` in the config, docs/recom/RUNNING_ALBEDO.md); with `partition: null` it runs on one GPU.
#SBATCH --job-name=fj_run_gpu
#SBATCH --partition=gpu
#SBATCH --account=bk1415_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH -o /work/bk1415/a270044/fesom_jax_pool/runs/slurm/%x.%j.out
set -u
WT=${WT:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$WT/scripts/levante/env_gpu.sh"
cd "$WT" || exit 2
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" || { echo "FATAL: fesom_jax not from $WT"; exit 4; }
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} $(git rev-parse --short HEAD) args: $* ==="
"$PY" scripts/run_from_config.py --progress "$@"
rc=$?
echo "=== exit $rc $(date) ==="
exit $rc
