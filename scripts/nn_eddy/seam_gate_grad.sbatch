#!/bin/bash
#SBATCH --job-name=nn_seam_grad
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o scripts/nn_eddy/seamgrad.%j.out
#SBATCH -e scripts/nn_eddy/seamgrad.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/nn_eddy/port_jax || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
# The gradient half only. Big arrays now enter the jitted functions as ARGUMENTS rather than
# closures: the previous version embedded 4.58 GB + 3.77 GB of captured constants across three
# executables and OOM'd at 49.53 GiB (26798462/3). --seg-mult 2 stores fewer checkpoint carries
# and recomputes more, for headroom on top of that.
"$PY" scripts/nn_eddy/seam_gate.py \
    --snap /work/ba1583/a270088/nn_eddy/student_free_1958/snap_day150.pkl \
    --seed 17 --mode gradient --seg-mult "${SEGM:-2}" \
    --results /work/ba1583/a270088/nn_eddy/seam_gate/seam_gate.jsonl
