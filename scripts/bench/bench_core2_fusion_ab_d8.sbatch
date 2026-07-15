#!/bin/bash
# M7-levers fusion attribution A/B — CORE2-8 (2 nodes), main vs perf/kokkos-m7-levers,
# SAME allocation, interleaved x2 (the bench_ng5_halo_confirm_d64 pattern).
#
# Measures the DEVICE-side effect of the two collective fusions on this branch
# (halo.exchange_pair: 240->120 ice exchanges/step; reductions.global_dot_pair: 3->2
# psums/CG-iter). bench_forward_scaling times the warm jitted chunk only — host forcing
# prep is outside the timed region, so the jra55 clamp fix CANNOT contaminate this Δ.
# mEVP ON (--mevp 1) or the ice fusion is under-exercised. padded transport both legs
# (reproduced to 0.1% at dars-32 — lowest within-job noise of the four transports).
#
# Leg isolation: fesom_jax is an EDITABLE install pointing at /home/a/a270088/port_jax,
# so each leg pins PYTHONPATH to its own tree (verified: PYTHONPATH precedes the
# editable finder). main worktree: /home/a/a270088/port_jax_main_ab @ ceea184.
#SBATCH --job-name=jax_fusion_ab_core2
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80; one slow node paces the job
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:15:00
# .log, NOT .out — keep A/Bs out of the paper's bench_*.out glob (20260714 gotcha 1).
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/fusion_ab_core2_d8.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/fusion_ab_core2_d8.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
BRANCH_TREE=/home/a/a270088/port_jax
MAIN_TREE=/home/a/a270088/port_jax_main_ab
C="--name core2 --mesh-dir $BRANCH_TREE/data/mesh_core2 --dist-dir $POOL/core2 \
   --npes 8 --steps 25 --warmup 5 --dt 1800 --full 1 --mevp 1 \
   --ic-dir $BRANCH_TREE/data/ic_core2 --year 1958 --halo padded"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST  CORE2-8 fusion A/B ==="
echo "main   = $(git -C $MAIN_TREE rev-parse --short HEAD)  ($MAIN_TREE)"
echo "branch = $(git -C $BRANCH_TREE rev-parse --short HEAD)  ($BRANCH_TREE)"
for REP in 1 2; do
  for TREE in "$MAIN_TREE" "$BRANCH_TREE"; do
    LEG=$([ "$TREE" = "$MAIN_TREE" ] && echo main || echo branch)
    echo "#### CORE2-8 mevp padded LEG=$LEG rep=$REP  sha=$(git -C $TREE rev-parse --short HEAD) ####"
    (cd "$TREE" && PYTHONPATH="$TREE" srun --cpu-bind=none "$PY" \
        "$TREE/scripts/bench/bench_forward_scaling.py" $C)
  done
done
echo "=== done $(date) ==="
