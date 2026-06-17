#!/bin/bash
# Multi-GPU (+ multi-node) NN-of-TKE twin — the SHARDED adjoint for long continuous windows.
# shard_map over P devices distributes the per-step VJP tape ~25/P per device (the all_gather halo
# only replicates the ~15 communicated fields, ~2 GB) ⇒ P=4-16 GPUs unlock a continuous N≫10 window.
# all_gather halo (use_ragged=False = the autodiff-correct path). Single node (P≤4) runs directly;
# multi-node sets JDIST=1 + srun (jax.distributed auto-detects SLURM). Submit:
#   1 node  P=4 : sbatch --nodes=1 scripts/core2_paper_nn_twin_sharded.sbatch 4 48
#   2 nodes P=8 : sbatch --nodes=2 scripts/core2_paper_nn_twin_sharded.sbatch 8 96
#   4 nodes P=16: sbatch --nodes=4 scripts/core2_paper_nn_twin_sharded.sbatch 16 192
#SBATCH --job-name=core2_nntwin_shard
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_nn_twin_sharded.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_nn_twin_sharded.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.92}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
NPES=${1:-4}              # P devices (must equal SLURM_NNODES × 4 and have /pool/.../dist_<P>)
N=${2:-48}               # continuous window steps
CONFIG=${3:-all3}
ITERS=${4:-100}
S0=${S0-/work/ab0995/a270088/port_jax/nn_twin_S0_1yr.pkl}
FROM_S0=""
[ -n "$S0" ] && [ -f "$S0" ] && FROM_S0="--from-s0 $S0"
EXTRA=${EXTRA:-}

echo "=== $(hostname) $(date)  P=$NPES N=$N config=$CONFIG nodes=$SLURM_NNODES ==="
nvidia-smi -L 2>/dev/null | head -4 || true

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    export JDIST=1 GPUS_PER_NODE=4
    srun --cpu-bind=none "$PY" scripts/core2_paper_nn_twin_sharded.py \
        --npes "$NPES" --n "$N" --config "$CONFIG" --iters "$ITERS" $FROM_S0 $EXTRA
else
    "$PY" scripts/core2_paper_nn_twin_sharded.py \
        --npes "$NPES" --n "$N" --config "$CONFIG" --iters "$ITERS" $FROM_S0 $EXTRA
fi
echo "=== done $(date) (exit $?) ==="
