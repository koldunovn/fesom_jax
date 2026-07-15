#!/bin/bash
# Phase 8b B.3 STEP 4 — NG5 (7.4M x nl70) FULL model at dist_32 = 8 GPU NODES (32 A100).
# dist_16 (4N) OOM'd: per-GPU model peak ~140 GiB working set + 83.6 GB I/O ("exceeds base
# limit") — NOT host-build (peak_gpu_after_setup=21.43 GiB OK) and NOT load imbalance (Lmax/mean
# =1.00). dist_32 has 231k nod/GPU (16.2M node-lev/GPU) — FEWER than the working dars-8 (22.5M),
# so if the per-GPU memory scales with devices it fits; if the 83.6 GB I/O is a GLOBAL fold
# signature that does NOT scale, it OOMs again (→ needs per-subdomain / I/O-reduction). Decisive.
# vs Kokkos M524 CUDA NG5: 8N (dist_32) = 0.810 s/step (dt180). ragged only (allgather OOMs +hangs).
#SBATCH --job-name=bench_ng5_d32
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/bench_ng5_d32.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/bench_ng5_d32.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
ZD=/work/ab0995/a270088/fesom_jax_meshes/ng5_out.zarr
# WITH the NG5 placement fix (_to_global_sharded → make_array_from_callback, no global array on
# one device) + sharded Zarr output (--out-zarr: each GPU writes its shard in parallel, no gather).
# The earlier dist_32 (25448673) OOM'd in a 125.81 GiB jit__identity_fn (global folded on one GPU);
# this run tests the fix lets NG5 RUN past it AND writes output.
C="--name ng5 --mesh-dir $M --dist-dir $POOL/ng5 --npes 32 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958 --ragged 1 --out-zarr $ZD"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### NG5 FULL 32GPU (8 nodes) ragged + zarr output ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
echo ">>>> ng5 d32 ragged exit: $? ($(date))"
echo "=== done $(date) ==="
