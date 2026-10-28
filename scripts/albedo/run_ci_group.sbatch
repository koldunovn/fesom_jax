#!/bin/bash
# Albedo: the data-free `ci` test group (from scripts/runs/suite_groups.txt) on one CPU compute node.
#   sbatch scripts/albedo/run_ci_group.sbatch            (from the repo root)
#SBATCH --job-name=fj_ci
#SBATCH -p mpp
#SBATCH -A clidyn.clidyn
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH -o /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/slurm/%x.%j.out
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-$HOME/REcoM_in_FESOM-JAX/fesom_jax_recom}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date) ==="
FILES=$(awk 'NF && $1 !~ /^#/ && $1=="ci" {print $2}' scripts/runs/suite_groups.txt)
echo "$FILES" | wc -l | xargs echo "ci group files:"
$PY -m pytest -q -p no:cacheprovider --durations=10 $FILES
echo "=== exit $? $(date) ==="
