#!/bin/bash
# ROOT-CAUSE CONFIRMATION + FIX test for the ng5 x P=128 cliff (HANDOFF-20260724).
# nsys (job 1036137) found it: ONE kernel — input_add_multiply_reduce_select_fusion, the
# 1,425-op kInput fusion that XLA builds ONLY at P=128 shapes by merging the
# sss_runoff._area_mean balance reduce into the vmapped ice-thermodynamics producer —
# runs 383 ms ONCE PER STEP = the entire excess.  (zstar x ice: the dataflow
# ice thermo -> water_flux/virtual_salt -> balance mean only exists with both on.)
# Each leg changes exactly ONE thing vs the campaign env:
#
#   ctrl       unchanged (expect ~640-660 ms — revalidates the cliff on this allocation)
#   barrier    FESOM_BALANCE_BARRIER=1 — lax.optimization_barrier on the _area_mean
#              input (model-side surgical fix; byte-identical numerics). Expect ~240 ms.
#   mofoff     --xla_disable_hlo_passes=multi_output_fusion (compiler-side confirmation:
#              the monster is a 6-output MOF product)
#   cbuf       XLA command buffers at DEFAULT (independent launch-path test, cheap)
#   barrier64  the fix at P=64 (16N) — no-regression guard on a healthy config
#
#   sbatch scripts/bench/jupiter/knobs_ng5_128.sbatch
#
#SBATCH --job-name=knobs128
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:45:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/knobs128_%j.out
#SBATCH -e scripts/logs/jupiter/knobs128_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"
BASEXLA="--xla_gpu_enable_command_buffer="   # the campaign value from env_jupiter.sh

PROD="--full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1"

echo "=== $(date) knob scan (ng5 P=128 prod) on $SLURM_JOB_NODELIST ==="
echo "=== other jobs of ours (want: none) ==="; squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true

leg () {   # leg <tag> <npes> [ENV=VAL ...] — one srun per knob, everything else identical
  local tag="$1" NP="$2"; shift 2
  local N=$(( (NP + 3) / 4 ))
  echo "#### knob ng5 ${NP}GPU (${N}N) [$tag] env: $* ####"
  timeout --kill-after=120 "$RUN_TIMEOUT" \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      env "$@" $PY scripts/bench/bench_forward_scaling.py \
        --name ng5 --mesh-dir "$MESH/mesh_ng5" --dist-dir "$RAW_ng5" \
        --ic-dir "$MESH/mesh_ng5" \
        --npes "$NP" --steps 150 --dt 180 --year 1958 --halo coloured $PROD
  local rc=$?
  [ $rc -eq 0 ] || echo "#### [$tag] rc=$rc ####"
}

leg ctrl      128
leg barrier   128 FESOM_BALANCE_BARRIER=1
leg mofoff    128 XLA_FLAGS="$BASEXLA --xla_disable_hlo_passes=multi_output_fusion"
leg cbuf      128 XLA_FLAGS=
leg barrier64 64  FESOM_BALANCE_BARRIER=1

echo "=== done $(date) ==="
echo "=== barrier at ~240 ms + ctrl at ~650 ms = root cause CONFIRMED and FIXED ==="
