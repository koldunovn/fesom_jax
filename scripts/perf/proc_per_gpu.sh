#!/bin/bash
# One process per GH200 under `srun --ntasks-per-node=4`: local task i drives GPU i (CUDA_VISIBLE_DEVICES) from the
# cores of Grace i (taskset 72i..72i+71; a GH200 node is 4 x 72 Grace cores), as fesom3_jax's `gpu-procs` preset
# (its docs/PERFORMANCE.md, "One process per GH200": -3 % from separate processes, -2-3 % from the pinning).
#   srun --ntasks-per-node=4 --cpus-per-task=72 --cpu-bind=none bash scripts/perf/proc_per_gpu.sh <command> ...
set -eu
i=${SLURM_LOCALID:?run under srun}
export CUDA_VISIBLE_DEVICES=$i GPUS_PER_NODE=1
lo=$((72 * i))
exec taskset -c "$lo-$((lo + 71))" "$@"
