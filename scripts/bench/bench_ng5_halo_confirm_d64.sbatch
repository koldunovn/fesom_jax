#!/bin/bash
# Phase 8d CONFIRMATION — NG5-64: is coloured's 9% win over ragged real, or node noise?
#
# Job 26232225 measured coloured 543.28 ms vs ragged 592.07 vs padded 741.85 at NG5/dist_64 --
# the coloured transport WINS at the largest configuration, which would make the AD-correct
# transport also the fastest one there. Before that changes the guidance AND the paper, it needs
# a second sample, because we have just learned how noisy ragged is: at dars-32 it swung 8.6%
# between allocations (319.56 ms job 26228413 -> 346.92 ms job 26232224) while padded reproduced
# to 0.1%. Coloured's NG5-64 margin (9.0%) is the SAME SIZE as that swing.
#
# Design: INTERLEAVED, two samples each, one job/one allocation -- so within-job variance is
# measured directly rather than assumed, and a slow node cannot alias into a transport. padded is
# dropped (741.85 ms, and it reproduced to 0.1% at dars-32; nothing left to learn).
# Prior ragged samples at NG5-64: 583.84 ms (job 25453667, June) and 592.07 (26232225) => ragged
# itself IS reproducible here to ~1.4%, which is why the win is probably real.
#SBATCH --job-name=bench_ng5_confirm
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:15:00
# .log, NOT .out — keep transport A/Bs out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_ng5_confirm.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_ng5_confirm.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
C="--name ng5 --mesh-dir $M --dist-dir $POOL/ng5 --npes 64 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
for REP in 1 2; do
  for H in ragged coloured; do
    echo "#### NG5-64 $H (rep $REP) ####"
    srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo $H
  done
done
echo "=== done $(date) ==="
