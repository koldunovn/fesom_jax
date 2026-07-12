#!/bin/bash
# CORE2 COMPLETE-model per-step bench on ONE GPU (paper §5: the communication-free
# single-device baseline that anchors ideal scaling / parallel efficiency). Same harness +
# protocol as bench_core2_full.sbatch; npes=1 => synth-serial partition, zero halo, the
# exchange maps are identities.
#SBATCH --job-name=bench_core2_1gpu
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_core2_1gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_core2_1gpu.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --npes 1 --steps 25 \
   --warmup 5 --dt 1800 --full 1 --ic-dir $PWD/data/ic_core2 --year 1958"
echo "=== node $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
# ⚠️ Levante mixes a100_40 (cell09) and a100_80 (cell13) — ALWAYS pin --constraint
# (job 26037293 silently landed on a 40GB card; every other scaling row is a100_80).
echo "#### full (ice+kpp+gm) 1 GPU ####"; "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 1 --kpp 1 --gm 1
echo "#### oce (base) 1 GPU ####";        "$PY" scripts/bench/bench_forward_scaling.py $C --ragged 0 --ice 0 --kpp 0 --gm 0
echo "=== done $(date) ==="
