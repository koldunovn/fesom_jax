#!/bin/bash
# Root-cause probe for the ng5 x P=128 cliff: scan the model's actual collectives at the
# EXACT device counts that bracket the anomaly (64 / 128 / 256 GPUs = 16 / 32 / 64 nodes).
#
# The gap this closes: fabric_probe.py only reached 8 nodes / 32 GPUs and only measured
# all-reduce, so the 32-node / 128-GPU configuration where the cliff lives was never probed.
#
# Runtime inflation at ng5/P=128 orders by BYTES MOVED (coloured 2.65x < ragged ~5x < padded
# 5.95x), which is a communication signature.  If a collective is pathological at 128 devices
# for the model's message sizes (~0.4-0.8 MB), it shows up here directly — and NCCL_DEBUG on
# the 128-GPU leg records which algorithm/protocol NCCL picked, since a bad
# algorithm/protocol choice at one (size, nranks) point is exactly the kind of thing that is
# catastrophic at 128 ranks and fine at 64 and 256.
#
#   sbatch scripts/bench/jupiter/collective_scan.sbatch
#
#SBATCH --job-name=cscan
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=64
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/cscan_%j.out
#SBATCH -e scripts/logs/jupiter/cscan_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1

echo "=== $(date) collective scan on $SLURM_JOB_NODELIST ==="
echo "=== other jobs of ours (want: none) ==="; squeue -u "$USER" -o "%.10i %.10j %.9T %.6D" || true

for N in 16 32 64; do
  echo "######## $N nodes / $((N*4)) GPUs ########"
  timeout --kill-after=60 900 \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/jupiter/collective_scan.py || echo "rc=$?"
done

# Which NCCL algorithm/protocol is chosen at 128 ranks vs 64/256?  A switch here would be the
# mechanism.  Captured separately so the timing runs above stay clean.
for N in 16 32 64; do
  echo "######## NCCL algo/proto selection at $N nodes / $((N*4)) GPUs ########"
  timeout --kill-after=60 900 \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      --export=ALL,NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,TUNING \
      $PY scripts/bench/jupiter/collective_scan.py 2>&1 \
    | grep -aE "^\[cscan\]|Channel|algorithm|protocol|nChannels|NET/IB :|treeThreshold|tuning" \
    | head -60 || echo "rc=$?"
done

echo "=== done $(date) ==="
