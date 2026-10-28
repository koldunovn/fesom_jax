#!/bin/bash
# scripts/perf/ab_time.sbatch on one whole GH200 node of dolpung (aarch64). At most two dolpung nodes for the
# whole user at a time (the user's rule): check `squeue -p dolpung -u a270088` and chain with --dependency.
#SBATCH --job-name=fj_ab_gh200
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
export ARM_ENV_SH=${ARM_ENV_SH:-/home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh}
exec bash "$WT/scripts/perf/ab_time.sbatch" "$@"
