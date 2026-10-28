#!/bin/bash
# scripts/perf/gpu_profile.sbatch on one whole GH200 node of Levante's dolpung partition (aarch64 hosts):
# the aarch64 Python of runs/dolpung/job_gh200.sbatch (ARM_ENV_SH) and the aarch64 nsys, then the same steps.
#   sbatch --export=ALL,WT=<checkout>,P=4,LABEL=gh200 scripts/perf/gpu_profile_gh200.sbatch [run flags]
# dolpung footprint: at most one node from this work at a time (the account's limit of two is shared).
#SBATCH --job-name=fj_perf_gh200
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
export ARM_ENV_SH=${ARM_ENV_SH:-/home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh}
export NSYS_BIN=${NSYS_BIN:-/sw/spack-levante-0.23.1/linux-rhel9-neoverse_v2/nvhpc-25.7-dsupkck/Linux_aarch64/25.7/profilers/Nsight_Systems/bin/nsys}
exec bash "$WT/scripts/perf/gpu_profile.sbatch" "$@"
