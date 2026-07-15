#!/bin/bash
# Phase 8b B.3 STEP 4 — NG5 (7.4M x nl70) FULL model multi-node — the HEADLINE scaling goal.
# dist_16 = 4 GPU NODES (16 A100), real JRA1958 + PHC IC + prognostic ice, dt=180. RAGGED only:
# all_gather is known to OOM multi-node (dars-8 proved it) AND its OOM hangs the NCCL rendezvous
# to the time limit, wasting the 4-node allocation — so we run the halo-only ragged path that
# multi-node scaling needs. 1 process/node x 4 local GPUs (jax.distributed).
#
# Sizing (from the dars-8 lesson): NG5 dist_8 (926k nod/GPU) would OOM (~103 GiB/GPU, JAX's
# step working set is heavier than Kokkos'); dist_16 (463k nod/GPU) ~52 GiB/GPU fits 80 GB.
# Host: NG5 global State ~162 GB + partitioned copy ~325 GB peak/process < 515 GB node RAM (--mem=0).
# vs Kokkos M524 CUDA NG5: 2N 2.335 / 4N 1.273 / 8N 0.810 / 16N 0.492 s/step (dt180, fixed GPU halo).
#SBATCH --job-name=bench_ng5_mn
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:35:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_ng5_mn.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_ng5_mn.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
C="--name ng5 --mesh-dir $M --dist-dir $POOL/ng5 --npes 16 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958 --ragged 1"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### NG5 FULL 16GPU (4 nodes) ragged ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
echo ">>>> ng5 ragged exit: $? ($(date))"
echo "=== done $(date) ==="
