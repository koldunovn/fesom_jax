#!/bin/bash
# Fast-adjoint plan, the GPU-options step: scripts/perf/adj_profile.py on one whole GH200 node of dolpung — the timing
# of one adjoint chunk's forward and reverse programs, then nsys over one of each (kernels inside CUDA graphs traced
# node by node) with the optimised HLO dumped, and the kernels mapped to model areas (hlo_kernel_map.py).
#   sbatch --export=ALL,WT=<checkout>[,LABEL=..][,GPU_XLA_FLAGS=..][,FESOM_...=..] scripts/perf/adj_profile_gh200.sbatch [adj_profile flags]
# dolpung footprint: one node.
#SBATCH --job-name=adj_prof
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/perf/adj_prof.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
LABEL=${LABEL:-base}
source /home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh          # sets PY; unsets PYTHONPATH
export PYTHONPATH=$WT:/work/ab0995/a270088/fesom_jax_dev/envs/arm_extra/site
NSYS_BIN=${NSYS_BIN:-/sw/spack-levante-0.23.1/linux-rhel9-neoverse_v2/nvhpc-25.7-dsupkck/Linux_aarch64/25.7/profilers/Nsight_Systems/bin/nsys}
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp CUDA_CACHE_PATH=/scratch/a/a270088/fesom_jax_dev_tmp/nv_gh200
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: no persistent XLA cache"; exit 2; fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 PYTHONUNBUFFERED=1
export XLA_FLAGS="$(echo --xla_gpu_enable_command_buffer= ${GPU_XLA_FLAGS:-})"
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" || { echo "FATAL: import"; exit 4; }
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-adj-$LABEL
mkdir -p "$OUT/nsys"
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} WT=$WT out=$OUT XLA_FLAGS=$XLA_FLAGS flags: $* ==="
env | grep -E "^FESOM_" | sort
"$PY" scripts/perf/adj_profile.py time "$OUT" "$@" 2>&1 | grep -v cuda_vmm | tail -n 3
XLA_FLAGS="$(echo $XLA_FLAGS --xla_dump_to=$OUT/hlo --xla_dump_hlo_as_text)" \
"$NSYS_BIN" profile -t cuda,nvtx -s none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node -o "$OUT/nsys/prof" -f true "$PY" scripts/perf/adj_profile.py prof "$OUT" "$@" \
    > "$OUT/nsys/prof.log" 2>&1
echo "nsys exit $?"
"$NSYS_BIN" stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv --output "$OUT/nsys/prof" \
    "$OUT/nsys/prof.nsys-rep" > "$OUT/nsys/stats.log" 2>&1
echo "stats exit $?"
HLO=$(ls -S "$OUT"/hlo/*after_optimizations.txt 2>/dev/null | head -n 1)
K=$("$PY" -c "import json; print(json.load(open('$OUT/time.json'))['chunk_steps'])")
if [ -n "$HLO" ] && [ -f "$OUT/nsys/prof_cuda_gpu_kern_sum.csv" ]; then
    "$PY" scripts/perf/hlo_kernel_map.py "$HLO" "$OUT/nsys/prof_cuda_gpu_kern_sum.csv" --steps $((K * 4)) --by area \
        --top 40 > "$OUT/areas.txt" 2>&1
    "$PY" scripts/perf/hlo_kernel_map.py "$HLO" "$OUT/nsys/prof_cuda_gpu_kern_sum.csv" --steps $((K * 4)) --by line \
        --top 40 > "$OUT/lines.txt" 2>&1
    head -n 50 "$OUT/areas.txt"
fi
echo "=== done $(date) ==="
