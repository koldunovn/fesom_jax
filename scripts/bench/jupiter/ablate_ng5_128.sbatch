#!/bin/bash
# WHERE does the ng5 x P=128 time actually go?  Component ablation, P=64 vs P=128.
#
# Everything STRUCTURAL has been checked and is clean at P=128: partition balance
# (pad 1.013/1.026/1.016), Lmax per kind (smooth, monotonic), coloured buffer volume (8736,
# BELOW P=64's 10794), colouring rounds K (14, vs 15 at the FAST P=256), and — measured
# directly by collective_scan.py at the exact device counts — the collectives themselves:
#     us @256KB/dev      64 GPU   128 GPU   256 GPU
#     ppermute             64.5      65.6      70.1    <- coloured's primitive: FLAT
#     allreduce           135.7     161.3     254.8    <- monotonic, no 128 spike
#     allgather           502.3    1083.9    2388.3    <- monotonic, no 128 spike
# And dars at P=128 (same 32 nodes, same device count) is entirely normal.  So it is neither
# the fabric nor the device count nor the shapes.
#
# WHAT HAS NEVER BEEN MEASURED: where the time goes INSIDE the step.  The SSH solve is a
# data-dependent `lax.while_loop` (ssh.py:562) whose trip count is NOT fixed, and under zstar
# the stiffness operator is re-assembled PER CG ITERATION.  So a configuration that needs more
# CG iterations pays a multiplied cost — a mechanism invisible to every structural check above,
# and partition-dependent WITHOUT being partition-imbalanced.
#
# Each leg is run at BOTH P=64 and P=128; the diagnostic is the RATIO P128/P64 per leg:
#   baseline (production)  -> expect ~2.7x  (236 -> 643 ms, the anomaly)
#   zstar OFF (linfs)      -> if the ratio COLLAPSES to ~1, the cliff is the zstar x CG
#                             per-iteration assembly
#   full OFF (uniform)     -> if the ratio is still ~2.7x, it is in the BASE ocean+SSH solve,
#                             not the physics package
#   ice OFF / tke OFF      -> narrows further if the two above disagree
#
#   sbatch scripts/bench/jupiter/ablate_ng5_128.sbatch
#
#SBATCH --job-name=ablate
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/ablate_%j.out
#SBATCH -e scripts/logs/jupiter/ablate_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"

BASE="--name ng5 --mesh-dir $MESH/mesh_ng5 --dist-dir $RAW_ng5 --ic-dir $MESH/mesh_ng5 \
   --steps 150 --dt 180 --year 1958 --halo coloured"

echo "=== $(date) ng5 P=128 ablation on $SLURM_JOB_NODELIST ==="
echo "=== other jobs of ours (want: none) ==="; squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true

leg () {   # leg <label> <npes> <extra flags...>
  local label="$1" NP="$2"; shift 2
  local N=$(( (NP + 3) / 4 ))
  echo "#### ng5 ${NP}GPU (${N}N) halo=coloured [$label] ####"
  timeout --kill-after=120 "$RUN_TIMEOUT" \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/bench_forward_scaling.py $BASE --npes "$NP" "$@"
  local rc=$?
  [ $rc -eq 0 ] || echo "#### [$label] rc=$rc ####"
}

PROD="--full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1"

for NP in 64 128; do
  leg "baseline-prod"  $NP $PROD
  leg "zstar-OFF"      $NP --full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 0
  leg "full-OFF"       $NP --full 0
  leg "ice-OFF"        $NP --full 1 --ice 0 --tke 1 --kpp 0 --gm 0 --zstar 1
  leg "tke-OFF"        $NP --full 1 --ice 1 --mevp 1 --tke 0 --kpp 0 --gm 0 --zstar 1
done

echo "=== done $(date) ==="
echo "=== compare the P128/P64 ratio PER LEG; the leg whose ratio collapses to ~1 carries"
echo "=== the cliff. Remember compile= embeds the 150-step first run. ==="
