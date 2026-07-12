#!/bin/bash
# dars timing-validity A/B (paper §5 data integrity; docs/memo/DARS_INSTABILITY_FINDING.md):
# the existing dars scaling rows were timed at dt=180 from a cold start — a configuration
# that goes NaN within ~60 steps (the 1.2km refined-patch CFL; w_split off). Per-step time
# should be dt-independent, but a degrading state can shift the CG early-stop iteration
# count. This job re-times dars/dist_16 at (a) dt=180 (the historical row, now WITH the
# finiteness gate) and (b) dt=60 (stable per the memo — the healthy-state reference).
# If (a)~(b) in s/step, the historical rows stand as timing; the gate lines document state
# health either way. READ-ONLY inputs; writes only this log.
#SBATCH --job-name=bench_dars_ab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/dars_ab.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/dars_ab.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --npes 16 --steps 25 --warmup 5 \
   --full 1 --ic-dir $M --year 1958 --ragged 1"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
echo "#### dars 16GPU dt=180 (historical bench config + finiteness gate) ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --dt 180
echo "#### dars 16GPU dt=60 (stable reference) ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --dt 60
echo "#### dars 16GPU dt=120 zstar+wsplit+mEVP (production-like; KPP stands in for TKE"
echo "#### to dodge the known TKE multi-node compile hang) ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --dt 120 \
    --zstar 1 --wsplit 1 --mevp 1 --kpp 1 --gm 0
echo "=== done $(date) ==="
