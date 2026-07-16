#!/bin/bash
# M7-levers fusion attribution A/B — dars-32 (8 nodes), main vs perf/kokkos-m7-levers,
# SAME allocation, interleaved x2. The comm-matters companion to
# bench_core2_fusion_ab_d8.sbatch — see that script's header for the full design
# (device-only timing, PYTHONPATH leg isolation, padded both legs, mEVP ON).
#
# dt=60, NOT 180 (learned from job 26274791): at dt=180 the dars cold start blows up
# INSIDE the 25-step window (bench-finite: 118M non-finite T, both legs) — with a NaN
# state the SSH CG collapses, removing exactly the psum traffic the fusion targets, and
# the A/B read a meaningless 0.0% tie. dt=60 is the known-stable dars bench regime
# (configs/dars_kpp_evp_dt60.yaml); the ice exchange fusion is dt-independent (fixed 120
# subcycles/step), the CG share is somewhat smaller at dt=60 — note it when quoting.
#SBATCH --job-name=jax_fusion_ab_dars
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80; one slow node paces the job
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
# .log, NOT .out — keep A/Bs out of the paper's bench_*.out glob (20260714 gotcha 1).
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/fusion_ab_dars_d32.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/fusion_ab_dars_d32.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
BRANCH_TREE=/home/a/a270088/port_jax
MAIN_TREE=/home/a/a270088/port_jax_main_ab
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
# DT overridable: sbatch --export=ALL,DT=120 …  (dars usually recovers at dt=120 per the
# Kokkos session's finding; dt=60 is the conservative floor. bench-finite decides validity.)
C="--name dars --mesh-dir $M --dist-dir $POOL/dars \
   --npes 32 --steps 25 --warmup 5 --dt ${DT:-60} --full 1 --mevp 1 \
   --ic-dir $M --year 1958 --halo padded"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST  dars-32 fusion A/B ==="
echo "main   = $(git -C $MAIN_TREE rev-parse --short HEAD)  ($MAIN_TREE)"
echo "branch = $(git -C $BRANCH_TREE rev-parse --short HEAD)  ($BRANCH_TREE)"
for REP in 1 2; do
  for TREE in "$MAIN_TREE" "$BRANCH_TREE"; do
    LEG=$([ "$TREE" = "$MAIN_TREE" ] && echo main || echo branch)
    echo "#### dars-32 mevp padded LEG=$LEG rep=$REP  sha=$(git -C $TREE rev-parse --short HEAD) ####"
    (cd "$TREE" && PYTHONPATH="$TREE" srun --cpu-bind=none "$PY" \
        "$TREE/scripts/bench/bench_forward_scaling.py" $C)
  done
done
echo "=== done $(date) ==="
