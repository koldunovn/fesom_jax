#!/bin/bash
# Experiment (2) of HANDOFF-20260724-ng5-128-cliff.md: WHERE does the extra time go at
# (ng5, P=128, zstar)?  nsys timeline on rank 0 for prod P=64 vs prod P=128 vs
# zstar-OFF P=128.  The ablation (job 1035822) localised the cliff to zstar
# (649.5 -> 222.2 ms at P=128 when zstar goes off); this discriminates whether the
# zstar-x-P=128 extra 400 ms/step is kernels, NCCL collectives, or gaps (launch/sync
# bubbles) — which no measurement so far has done.
#
# 10 steps: enough steady-state CG iterations (~10 x ~99) to profile; the nsys overhead
# and the AB2 first step make these timings NOT campaign numbers.
#
#   sbatch scripts/bench/jupiter/profile_ng5_128.sbatch
#
#SBATCH --job-name=nsysng5
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/nsysng5_%j.out
#SBATCH -e scripts/logs/jupiter/nsysng5_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
module load Nsight-Systems 2>/dev/null || true
export JDIST=1
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"
NSYS_ROOT="$ESTA/nsys/$SLURM_JOB_ID"
WRAP="scripts/bench/jupiter/nsys_wrap.sh"
chmod +x "$WRAP"

PROD="--full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1"
NOZ="--full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 0"
ICEOFF="--full 1 --ice 0 --tke 1 --kpp 0 --gm 0 --zstar 1"
STEPS="${PROF_STEPS:-10}"

echo "=== $(date) nsys profile on $SLURM_JOB_NODELIST -> $NSYS_ROOT ==="
echo "=== other jobs of ours (want: none) ==="; squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true
nsys --version || echo "WARN: nsys not found — legs will run unprofiled"

leg () {   # leg <tag> <npes> <config flags...>
  local tag="$1" NP="$2"; shift 2
  local N=$(( (NP + 3) / 4 ))
  echo "#### nsys ng5 ${NP}GPU (${N}N) halo=coloured steps=$STEPS [$tag] ####"
  NSYS_OUT="$NSYS_ROOT/${tag}_p${NP}" \
  timeout --kill-after=120 "$RUN_TIMEOUT" \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      "$WRAP" $PY scripts/bench/bench_forward_scaling.py \
        --name ng5 --mesh-dir "$MESH/mesh_ng5" --dist-dir "$RAW_ng5" \
        --ic-dir "$MESH/mesh_ng5" \
        --npes "$NP" --steps "$STEPS" --dt 180 --year 1958 --halo coloured "$@"
  local rc=$?
  [ $rc -eq 0 ] || echo "#### [$tag p$NP] rc=$rc ####"
}

leg prod   64  $PROD
leg prod   128 $PROD
leg iceoff 128 $ICEOFF
leg noz    128 $NOZ

echo "=== reports ==="
ls -la "$NSYS_ROOT" 2>/dev/null
for r in "$NSYS_ROOT"/*.nsys-rep; do
  [ -e "$r" ] || continue
  echo "== stats: $r =="
  nsys stats --force-export=true -r cuda_gpu_kern_sum "$r" 2>/dev/null | head -40
  nsys stats -r cuda_api_sum "$r" 2>/dev/null | head -25
done
echo "=== done $(date) ==="
