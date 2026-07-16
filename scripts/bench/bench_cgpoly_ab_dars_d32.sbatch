#!/bin/bash
# CGPOLY A/B at dars-32 (8 nodes) — the COMPUTE-DOMINATED anchor of the regime map (99k nodes/GPU; the fusions washed here). dt=120 = the bench-finite-clean dars regime (26299153).
# Per the E.3 spec a FLAT result here is EXPECTED, not a kill signal (the polynomial adds k
# SpMV-halos per apply while cutting iterations ⇒ roughly halo-neutral at small scale; the
# judge point is NG5-64, bench_cgpoly_ab_ng5_d64.sbatch). Same-code env A/B: the legs differ
# ONLY by FESOM_CG_CHEB (no worktrees needed — the lever is opt-in on one commit).
# NOTE the bench's --warmup flag is DEAD (parsed, never consumed — timing is the 2nd call
# of one compiled N-step run from cold), so this is the standard cold 25-step protocol,
# comparable with every other A/B in the campaign. The settled-regime iteration ratio is
# pinned separately by the pytest gate (CHEB_ITERS on the captured real rhs).
#SBATCH --job-name=jax_cgpoly_ab_dars
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
# .log NOT .out — A/Bs stay out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cgpoly_ab_dars_d32.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cgpoly_ab_dars_d32.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name dars --mesh-dir /work/ab0995/a270088/fesom_jax_meshes/mesh_dars --dist-dir $POOL/dars --npes 32 \
   --steps 25 --dt 120 --full 1 --mevp 1 --halo padded \
   --ic-dir /work/ab0995/a270088/fesom_jax_meshes/mesh_dars --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST  CGPOLY A/B dars-32 dt120 (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  for LEG in off on; do
    unset FESOM_CG_CHEB; [ "$LEG" = on ] && export FESOM_CG_CHEB=3
    echo "#### dars-32 mevp padded LEG=$LEG rep=$REP ####"
    srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C
  done
done
echo "=== done $(date) ==="
