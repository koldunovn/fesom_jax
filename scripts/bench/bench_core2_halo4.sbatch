#!/bin/bash
# Phase 8d — CORE2 FULL model, all FOUR halo transports on GPU, one job per topology.
# Extends bench_core2_halo3.sbatch (identical harness/flags/paths — do NOT reinvent them)
# with the coloured-ppermute transport.
#
# CORE2 is where coloured is EXPECTED TO STRUGGLE, which is why it is tested first: padded
# wins here (80.1 ms @4 GPU) precisely BECAUSE the halo is tiny and latency-bound — one fused
# dense collective beats bookkeeping. Coloured trades that single collective for K=3 ppermute
# rounds. And the DOMINANT per-step comm is the CG's 2-D ssh exchange (~2 x ~127 iterations):
# small payloads, so collective COUNT matters more than bytes there. The volume model says
# coloured ships 1.0x ragged's lanes vs padded's 1.8-3.4x — but volume is not what CORE2 is
# paying for. This job is the honest test of the idea's weak end.
# Submit:
#   sbatch --nodes=1 --gres=gpu:4 --export=ALL,NPES=4 scripts/bench/bench_core2_halo4.sbatch
#   sbatch --nodes=2 --gres=gpu:4 --export=ALL,NPES=8 scripts/bench/bench_core2_halo4.sbatch
#SBATCH --job-name=bench_core2_halo4
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
# .log, NOT .out: the paper reducer globs scripts/logs/bench_*.out and would pull these
# A/B re-runs into scaling.csv (figure takes best-per-ngpu, macros take first-in-glob-order
# => they silently disagree). Keep transport A/Bs out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_core2_halo4.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_core2_halo4.%j.log
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
# coloured first (the new number), then the three references re-run for same-day control
echo "#### full coloured ####";  $RUN $C --halo coloured
echo "#### full padded ####";    $RUN $C --halo padded
echo "#### full ragged ####";    $RUN $C --halo ragged
echo "#### full allgather ####"; $RUN $C --halo allgather
echo "=== done $(date) ==="
