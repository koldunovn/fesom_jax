#!/bin/bash
# DIAGNOSTIC: why is core2/8-GPU 190 ms in the campaign job but 46 ms in the gates job?
# Same mesh, same physics, same transport, both 2 nodes, placement verified distinct.
# The two candidate differences are (a) --steps 150 vs 50, (b) the `timeout` wrapper the
# campaign sbatch puts around srun.  This isolates BOTH, on ONE allocation so node-to-node
# variation cannot confound it.  Also dumps stray-process / clock state per leg.
#
#   sbatch scripts/bench/jupiter/diag_2node.sbatch
#
#SBATCH --job-name=diag2n
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:50:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/diag2n_%j.out
#SBATCH -e scripts/logs/jupiter/diag2n_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1

C="--name core2 --mesh-dir $MESH/mesh_core2 --dist-dir $RAW_core2 --ic-dir $MESH/mesh_core2 \
   --dt 1800 --full 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1 --year 1958 --npes 8 --halo padded"

echo "=== $(date) diag on $SLURM_JOB_NODELIST ==="
srun -N2 -n2 --ntasks-per-node=1 nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader

echo "######## A: steps=50, bare srun (the GATES recipe -> expect ~46 ms) ########"
srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py $C --steps 50

echo "######## B: steps=150, bare srun (isolates STEP COUNT) ########"
srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py $C --steps 150

echo "######## C: steps=150, timeout-wrapped srun (isolates the TIMEOUT WRAPPER) ########"
timeout --kill-after=120 2400 \
  srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py $C --steps 150

echo "######## D: steps=50 again, bare srun (drift check within the allocation) ########"
srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py $C --steps 50

echo "######## E: steps=150, coloured (does the transport change the picture?) ########"
srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py \
  $C --steps 150 --halo coloured 2>/dev/null || \
  srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py \
    --name core2 --mesh-dir $MESH/mesh_core2 --dist-dir $RAW_core2 --ic-dir $MESH/mesh_core2 \
    --dt 1800 --full 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1 --year 1958 --npes 8 \
    --halo coloured --steps 150

echo "=== done $(date) ==="
