#!/bin/bash
# CGPOLY A/B at CORE2-8 (2 nodes) — the SMALL-SHARD/latency point of the lever's regime map.
# Per the E.3 spec a FLAT result here is EXPECTED, not a kill signal (the polynomial adds k
# SpMV-halos per apply while cutting iterations ⇒ roughly halo-neutral at small scale; the
# judge point is NG5-64, bench_cgpoly_ab_ng5_d64.sbatch). Same-code env A/B: the legs differ
# ONLY by FESOM_CG_CHEB (no worktrees needed — the lever is opt-in on one commit).
# --warmup 30: the CG iteration count settles ~step 30 from cold (the Kokkos 300-step
# protocol note) — timing before that UNDERSTATES an iteration-count lever.
#SBATCH --job-name=jax_cgpoly_ab_core2
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
# .log NOT .out — A/Bs stay out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cgpoly_ab_core2_d8.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cgpoly_ab_core2_d8.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir data/mesh_core2 --dist-dir $POOL/core2 --npes 8 \
   --steps 25 --warmup 30 --dt 1800 --full 1 --mevp 1 --halo padded \
   --ic-dir data/ic_core2 --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST  CGPOLY A/B core2-8 (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  for LEG in off on; do
    unset FESOM_CG_CHEB; [ "$LEG" = on ] && export FESOM_CG_CHEB=3
    echo "#### core2-8 mevp padded LEG=$LEG rep=$REP ####"
    srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
  done
done
echo "=== done $(date) ==="
