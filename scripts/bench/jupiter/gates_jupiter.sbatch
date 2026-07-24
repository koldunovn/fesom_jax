#!/bin/bash
# JUPITER sanity gates J1 + J2 (docs/JUPITER_SCALING.md §3) on the cheap CORE2 mesh.
#   J1  single-node 4-GPU  -> intra-node NVLink collectives + both transports run
#   J2  TWO-node 8-GPU     -> the LOAD-BEARING one: NCCL must use the NDR InfiniBand
#                             fabric, not TCP.  A hang or absurd numbers here means the
#                             fabric, and every multi-node number after it is worthless.
# J3 (bench-finite on every row) is enforced by reading the [bench-finite] lines below.
#
#   sbatch scripts/bench/jupiter/gates_jupiter.sbatch
#
#SBATCH --job-name=jax_gates
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/gates_%j.out
#SBATCH -e scripts/logs/jupiter/gates_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh

C="--name core2 --mesh-dir $MESH/mesh_core2 --dist-dir $RAW_core2 --ic-dir $MESH/mesh_core2 \
   --steps 50 --dt 1800 --full 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1 --year 1958"

echo "=== $(date) gates on $SLURM_JOB_NODELIST ==="

echo "######## J1a: 1 node, 4 GPU, padded ########"
unset JDIST
srun -N1 -n1 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py \
  $C --npes 4 --halo padded || echo "J1a rc=$?"

echo "######## J1b: 1 node, 4 GPU, coloured ########"
srun -N1 -n1 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py \
  $C --npes 4 --halo coloured || echo "J1b rc=$?"

echo "######## J2a: 2 nodes, 8 GPU, padded, NCCL_DEBUG=INFO (prove the fabric) ########"
export JDIST=1
srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none \
  --export=ALL,NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,NET \
  $PY scripts/bench/bench_forward_scaling.py $C --npes 8 --halo padded 2>&1 \
  | tee scripts/logs/jupiter/nccl_debug_$SLURM_JOB_ID.txt | grep -viE "^jpbo.*(Channel|Connected|comm 0x)" \
  || echo "J2a rc=$?"

echo "---- NCCL transport verdict (want IB/RoCE, NOT 'Socket' for the inter-node ring) ----"
grep -hoE "NET/(IB|Socket)[^ ]*" scripts/logs/jupiter/nccl_debug_$SLURM_JOB_ID.txt | sort | uniq -c
grep -hE "NCCL INFO NET/IB : |Using network|nRanks" scripts/logs/jupiter/nccl_debug_$SLURM_JOB_ID.txt | head -6

echo "######## J2b: 2 nodes, 8 GPU, padded, clean timing ########"
srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py \
  $C --npes 8 --halo padded || echo "J2b rc=$?"

echo "######## J2c: 2 nodes, 8 GPU, coloured ########"
srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none $PY scripts/bench/bench_forward_scaling.py \
  $C --npes 8 --halo coloured || echo "J2c rc=$?"

echo "=== done $(date) ==="
