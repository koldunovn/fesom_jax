#!/bin/bash
# Nsight Compute over one short chunk of the production run path on one dolpung GH200 node: every kernel's time and
# DRAM bytes on GPU 0 (`gpu_profile.py prof` opens the cudaProfilerApi range at its timed chunk), then the bytes per
# source line and their bandwidth (scripts/perf/hlo_kernel_map.py --ncu) against the HLO dump of the same compile.
# Command buffers off, so ncu sees every kernel as its own launch. P=1 (the default) runs the model on ONE GPU
# (`--partition 1`): with several GPUs ncu's kernel replay of an NCCL kernel on GPU 0 waits forever for the peers
# it does not replay (job 28455372 hung there).
#
#   sbatch --export=ALL,WT=<checkout>[,CONFIG=configs/core2_gpu_recom.yaml][,K=4][,LABEL=x] scripts/perf/ncu_gh200.sbatch \
#       [run flags, e.g. --restart-in DIR --precision single]
#
# Results: runs/perf/<job>-ncu-<label>/ (ncu.csv, lines.txt, areas.txt, kernels.txt, hlo/).
#SBATCH --job-name=fj_ncu_gh200
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/slurm/%x.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
CONFIG=${CONFIG:-configs/core2_gpu_recom.yaml}
K=${K:-4}
P=${P:-1}
if [ "$P" = 1 ]; then PART=1; else PART=dist_$P; fi
LABEL=${LABEL:-recom_dp}
export ARM_ENV_SH=${ARM_ENV_SH:-/home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh}
source "$ARM_ENV_SH"                    # sets PY; unsets PYTHONPATH
export PYTHONPATH=$WT:/work/ab0995/a270088/fesom_jax_dev/envs/arm_extra/site
NCU=${NCU:-/sw/spack-levante-0.23.1/linux-rhel9-neoverse_v2/nvhpc-25.7-dsupkck/Linux_aarch64/25.7/profilers/Nsight_Compute/ncu}
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp
export CUDA_CACHE_PATH=$TMPDIR/nv_$(uname -m)
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: no persistent XLA cache"; exit 2; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-ncu-$LABEL
mkdir -p "$OUT"
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_dump_to=$OUT/hlo --xla_dump_hlo_as_text"
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" \
    || { echo "FATAL: fesom_jax does not import from $WT"; exit 4; }
cfg="$OUT/config.yaml"
# the forcing read-ahead runs in-process (forcing_procs 0): ncu's frontend crashed when the forked read-ahead
# processes connected (job 28455235, exit 139 with --target-processes all)
sed -e 's/^\(  output_async:\) *true/\1 false/' -e 's/^\(  restart_async:\) *true/\1 false/' \
    -e 's/^\(  forcing_procs:\) *[0-9]*/\1 0/' "$CONFIG" > "$cfg"
echo "=== $(hostname) job ${SLURM_JOB_ID:-local} $CONFIG K=$K P=$P flags: $* ($(date +%T))"
"$NCU" --profile-from-start off --devices 0 --target-processes application-only --cache-control none \
    --clock-control none --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    -o "$OUT/prof" -f \
    "$PY" scripts/perf/gpu_profile.py prof "$OUT" --chunk-steps "$K" --steps "$K" -- "$cfg" \
    --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --partition "$PART" "$@" > "$OUT/prof.log" 2>&1
echo "ncu exit $? ($(date +%T))"; tail -n 3 "$OUT/prof.log"
"$NCU" --import "$OUT/prof.ncu-rep" --csv --page raw > "$OUT/ncu.csv"
HLO=$(ls -S "$OUT"/hlo/*jit_body*after_optimizations.txt 2>/dev/null | head -n 1)
[ -n "$HLO" ] || HLO=$(ls -S "$OUT"/hlo/*after_optimizations.txt | head -n 1)
for by in area line kernel; do
    "$PY" scripts/perf/hlo_kernel_map.py "$HLO" "$OUT/ncu.csv" --ncu --steps "$K" --by "$by" --top 40 > "$OUT/${by}s.txt"
done
head -n 25 "$OUT/lines.txt"
echo "=== done $(date +%T)"
