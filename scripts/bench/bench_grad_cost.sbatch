#!/bin/bash
# Gradient-cost benchmark (paper §5): forward vs value_and_grad per step, CORE2, 1 GPU.
# Attempt 3. #1 (PREALLOCATE=false): forwards OK, grads OOM on the 40-49GiB carry-stack
# alloc (fragmentation). #2 (MEM_FRACTION=0.92): arena starved the CUDA module loader
# ("failed to load in-memory CUBIN") before anything ran. Now: default preallocation at
# 0.75 (~60GB arena on the 80GB A100, ~20GB left for CUBIN/driver) + remat_blocks=1.
# Ladder, likely-fitting first:
#   1) n=12, per-step checkpointing   (~25 GiB carry stack — safe, guarantees a number)
#   2) n=48, O(sqrt N) checkpointing  (~40 GiB stack — the long-window setting)
#   3) n=24, per-step checkpointing   (~49 GiB stack + temps: borderline in a 60GB arena)
# READ-ONLY inputs; writes only this log.
#SBATCH --job-name=bench_grad_cost
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_grad_cost.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_grad_cost.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
# Attempt 4: attempt 3 landed on a 40GB card (Levante mixes a100_40/a100_80!) — even
# n=12's ~25GiB carry stack cannot fit there. Pin the 80GB nodes via --constraint.
# Attempt 5: on the 80GB card the FORWARD runs but the GRADIENT executable still fails
# at LOAD ("in-memory CUBIN ... CUDA_ERROR_OUT_OF_MEMORY") — load-time device-code/graph
# memory, not an HLO buffer. Disable XLA command buffers (CUDA-graph instantiation of
# the huge backward program), exactly as run_perf.sbatch does by default.
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
C="--mesh-dir $PWD/data/mesh_core2 --ic-dir $PWD/data/ic_core2 --year 1958 --dt 1800 --remat-blocks 1"
echo "=== node $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
echo "#### grad n=12 seg=0 ####";  "$PY" scripts/bench/bench_grad_cost.py $C --n 12 --seg 0
echo "#### grad n=48 seg=-1 ####"; "$PY" scripts/bench/bench_grad_cost.py $C --n 48 --seg -1
echo "#### grad n=24 seg=0 ####";  "$PY" scripts/bench/bench_grad_cost.py $C --n 24 --seg 0
echo "=== done $(date) ==="
