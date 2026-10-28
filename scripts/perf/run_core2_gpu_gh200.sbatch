#!/bin/bash
# scripts/perf/run_core2_gpu.sbatch on one whole GH200 node of Levante's dolpung partition (aarch64 hosts).
#   sbatch --export=ALL,WT=<checkout>,OUT=<new dir>,CONFIG=<yaml> scripts/perf/run_core2_gpu_gh200.sbatch [run flags]
# dolpung footprint: at most two nodes from this work at a time (the user's rule).
#SBATCH --job-name=fj_run_gh200
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
export ARM_ENV_SH=${ARM_ENV_SH:-$WT/scripts/perf/env_dolpung.sh}
exec bash "$WT/scripts/perf/run_core2_gpu.sbatch" "$@"
