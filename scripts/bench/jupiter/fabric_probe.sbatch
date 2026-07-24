#!/bin/bash
# Direct inter-node fabric measurement (see fabric_probe.py for why).  Runs the probe at 1, 2,
# 4 and 8 nodes so the intra-node (NVLink) baseline and the inter-node (NDR IB) step are
# separable: 4 GPUs = one node = NVLink only; 8 GPUs = the first inter-node hop.
#
#   sbatch scripts/bench/jupiter/fabric_probe.sbatch
#
#SBATCH --job-name=fabric
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:40:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/fabric_%j.out
#SBATCH -e scripts/logs/jupiter/fabric_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/bench/jupiter/env_jupiter.sh

echo "=== $(date) fabric probe on $SLURM_JOB_NODELIST ==="
scontrol show res 2>/dev/null | grep -c "ReservationName=emergency_maintenance" \
  | sed 's/^/[fabric] emergency_maintenance reservations present: /'
sinfo -p booster -o "%D %t" --noheader | sed 's/^/[fabric] booster state: /'

echo "######## 1 node (4 GPU) — NVLink only, no fabric ########"
unset JDIST
srun -N1 -n1 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/jupiter/fabric_probe.py

export JDIST=1
for N in 2 4 8; do
  echo "######## $N nodes ($((N*4)) GPU) — inter-node NDR IB ########"
  srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
    $PY scripts/bench/jupiter/fabric_probe.py || echo "rc=$?"
done
echo "=== done $(date) ==="
