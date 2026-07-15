#!/bin/bash
# Phase 8d — dars (3.16M) at dist_32 (8 nodes, 32 A100): coloured vs padded vs ragged.
#
# THE decisive test. dars-32 is where padded's pad factor (15.3x) first loses to ragged
# (319.6 vs 349.1 ms, +9.2%, job 26228413) — so it is the smallest, cheapest configuration
# in which the coloured transport's whole reason to exist can actually pay off. The volume
# model says coloured ships 1.3x ragged's lanes here vs padded's 15.3x (0.83 vs 9.4 MB per
# device per exchange). If coloured beats padded anywhere, it beats it here; if it also
# reaches ragged, then the AD-correct transport is the fast one at EVERY scale and ragged's
# forward-only niche closes.
#
# dt=180 from cold PHC goes non-finite within 25 steps BY DESIGN (timing unaffected; the
# finiteness gate prints nbad — production dars is dt=120+w_split). Ragged runs LAST so the
# two new-transport numbers are banked first (the 26228433 lesson).
#SBATCH --job-name=bench_dars_halo4_32
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
# .log, NOT .out: keep transport A/Bs out of the paper's bench_*.out glob (see the other
# halo sbatch scripts).
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_dars_halo4_32.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_dars_halo4_32.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --npes 32 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
for H in coloured padded ragged; do
  echo "#### dars-32 $H ####"
  srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo $H
done
echo "=== done $(date) ==="
