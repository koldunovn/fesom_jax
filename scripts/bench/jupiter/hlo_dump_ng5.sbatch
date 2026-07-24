#!/bin/bash
# Experiment (1) of HANDOFF-20260724-ng5-128-cliff.md: dump the OPTIMIZED HLO of the
# production step program at ng5 P=64/128/256 and diff them.  Facts (a)+(b) of the
# handoff say the *program* differs at (ng5, P=128) while the mathematics does not —
# compile is host-side XLA work and is ~2x inflated at exactly the configuration that
# runs 2.65x slower.  So: look at the program.
#
# UPDATE mid-ablation (job 1035822): the cliff needs ZSTAR x ICE — at P=128 in the same
# allocation: baseline-prod 649.5 ms (cliff, true compile ~58 s), zstar-OFF 222.2 ms
# (no cliff, compile ~34 s), ice-OFF 193.4 ms (NO cliff — but compile ~38 s, still ~2x
# its P=64's 18 s).  Compile inflation follows zstar alone; the runtime cliff needs
# zstar AND ice together.  Legs therefore: prod (sick) at 64/128/256, plus TWO P=128
# controls — noz (healthy runtime + healthy compile) and iceoff (healthy runtime +
# INFLATED compile).  Features present in prod_128 AND iceoff_128 but not noz_128 track
# the compile inflation; features ONLY in prod_128 track the runtime cliff.
#
# --steps 5 keeps each leg short WITHOUT changing the graph: run_steps_sharded builds
# step-1 eager + lax.scan(length=n_steps-1, xs=None), so for any n>=2 the HLO is
# structurally identical (only the scan trip-count constant differs).  Verified in
# integrate_sharded.py:548-551.  Timing rows from this job are NOT campaign numbers
# (dump I/O inflates compile=; per_step over 5 steps carries the AB2 first-step);
# the [bench] lines here are for leg bookkeeping only.
#
# Dumps land in $ESTA/hlo_dumps/$SLURM_JOB_ID/<leg>/ from process 0 only (dump_wrap.sh).
#
#   sbatch scripts/bench/jupiter/hlo_dump_ng5.sbatch
#
#SBATCH --job-name=hlodump
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=64
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/hlodump_%j.out
#SBATCH -e scripts/logs/jupiter/hlodump_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"
DUMP_ROOT="$ESTA/hlo_dumps/$SLURM_JOB_ID"
WRAP="scripts/bench/jupiter/dump_wrap.sh"
chmod +x "$WRAP"

# production physics — the exact config of the anomaly (ablate_ng5_128.sbatch PROD)
PROD="--full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1"
STEPS="${HLO_STEPS:-5}"

echo "=== $(date) HLO dump on $SLURM_JOB_NODELIST -> $DUMP_ROOT ==="
echo "=== other jobs of ours (want: none) ==="; squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true

leg () {   # leg <tag> <npes> <config flags...>
  local tag="$1" NP="$2"; shift 2
  local N=$(( (NP + 3) / 4 ))
  echo "#### hlodump ng5 ${NP}GPU (${N}N) halo=coloured steps=$STEPS [$tag] ####"
  HLO_DUMP_DIR="$DUMP_ROOT/${tag}_p${NP}" \
  timeout --kill-after=120 "$RUN_TIMEOUT" \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      "$WRAP" $PY scripts/bench/bench_forward_scaling.py \
        --name ng5 --mesh-dir "$MESH/mesh_ng5" --dist-dir "$RAW_ng5" \
        --ic-dir "$MESH/mesh_ng5" \
        --npes "$NP" --steps "$STEPS" --dt 180 --year 1958 --halo coloured "$@"
  local rc=$?
  [ $rc -eq 0 ] || echo "#### [$tag p$NP] rc=$rc ####"
  echo "#### dump files: $(ls "$DUMP_ROOT/${tag}_p${NP}" 2>/dev/null | wc -l) ####"
}

NOZ="--full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 0"
ICEOFF="--full 1 --ice 0 --tke 1 --kpp 0 --gm 0 --zstar 1"

# the decisive three (zstar ON), then the two P=128 controls
leg prod   64  $PROD
leg prod   128 $PROD
leg prod   256 $PROD
leg noz    128 $NOZ
leg iceoff 128 $ICEOFF

echo "=== done $(date); dumps in $DUMP_ROOT ==="
du -sh "$DUMP_ROOT"/* 2>/dev/null
