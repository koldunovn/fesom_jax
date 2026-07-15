#!/bin/bash
# Phase 8c — CORE2 FULL model, all THREE halo transports on GPU, one job per topology.
# The padded (slot-padded dense all_to_all, AD-correct, Phase 8c) transport has CPU numbers
# (docs/PARALLELISM.md) but no GPU ones; this prices it against ragged/allgather on the same
# node(s) and day, same harness/flags as bench_core2_full.25441203 / bench_core2_2n.25456963.
# Submit:
#   sbatch --nodes=1 --gres=gpu:4 --export=ALL,NPES=4 scripts/bench/bench_core2_halo3.sbatch
#   sbatch --nodes=2 --gres=gpu:4 --export=ALL,NPES=8 scripts/bench/bench_core2_halo3.sbatch
#SBATCH --job-name=bench_core2_halo3
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
# .log, NOT .out: the paper reducer globs scripts/logs/bench_*.out and would pull these
# A/B re-runs into scaling.csv (figure takes best-per-ngpu, macros take first-in-glob-order
# => they silently disagree). Keep transport A/Bs out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_core2_halo3.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_core2_halo3.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
NPES=${NPES:?set NPES=4 or 8}
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
C="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --npes $NPES --steps 25 \
   --warmup 5 --dt 1800 --full 1 --ice 1 --kpp 1 --gm 1 --ic-dir $PWD/data/ic_core2 --year 1958"
RUN="$PY scripts/bench/bench_forward_scaling.py"
if [ "$NPES" -gt 4 ]; then export JDIST=1 GPUS_PER_NODE=4; RUN="srun --cpu-bind=none $RUN"; fi
echo "=== node(s) $SLURM_JOB_NODELIST  npes=$NPES  $(date) ==="
# padded first (the missing number), then the two references re-run for same-day control
echo "#### full padded ####";    $RUN $C --halo padded
echo "#### full ragged ####";    $RUN $C --halo ragged
echo "#### full allgather ####"; $RUN $C --halo allgather
echo "=== done $(date) ==="
