#!/bin/bash
# CGPOLY A/B at NG5-64 (16 nodes) — THE judge point for the Chebyshev CG preconditioner
# (M7 lever, port_kokkos Task E.3 transferred). The lever cuts CG ITERATIONS (gated ≥1.5×
# in test_ssh_sharded CHEB_ITERS) ⇒ psums/solve drop with the ratio while SpMV+halo work
# rises ~(1+k)/2-fold per iteration — pays where Allreduce latency dominates, i.e. HERE
# (16 nodes, the fusions' regime map from session 2). Same-code env A/B (FESOM_CG_CHEB),
# interleaved ×2, coloured transport (fastest at NG5-64, Phase 8d).
# NOTE the bench's --warmup flag is DEAD (parsed, never consumed) — standard cold
# 25-step protocol, comparable with the fusion A/Bs and the NG5 halo confirms.
#SBATCH --job-name=jax_cgpoly_ab_ng5
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
# .log NOT .out — A/Bs stay out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cgpoly_ab_ng5_d64.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cgpoly_ab_ng5_d64.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
C="--name ng5 --mesh-dir $M --dist-dir $POOL/ng5 --npes 64 --steps 25 \
   --dt 180 --full 1 --halo coloured --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST  CGPOLY A/B ng5-64 (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  for LEG in off on; do
    unset FESOM_CG_CHEB; [ "$LEG" = on ] && export FESOM_CG_CHEB=3
    echo "#### ng5-64 coloured LEG=$LEG rep=$REP ####"
    srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
  done
done
echo "=== done $(date) ==="
