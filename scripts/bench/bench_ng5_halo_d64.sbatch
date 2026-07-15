#!/bin/bash
# Phase 8c — NG5 (7.4M x nl70) at dist_64 (16 nodes, 64 A100): padded vs ragged.
# The last decision point for "padded everywhere": it wins/ties ragged at <=16 GPUs
# (CORE2 4/8, dars 16), loses by 9% at dars-32, and NG5-64 is where the pad factor is
# largest. No zarr output (pure step timing). Tabulated ragged reference: 0.584 s/step.
#
# RERUN of job 26228433, which hit a 40-min wall AFTER "host setup done" with the padded
# run and produced NOTHING — not even the ragged control. Three changes:
#   1. ragged FIRST, so the control is banked (~10 min) before padded can misbehave;
#   2. padded capped by `timeout`, so a stall costs 30 min of 16 nodes, not the wall clock;
#   3. bench_forward_scaling now prints trace/compile stage lines, so a repeat stall is
#      localised (slow trace vs slow compile vs hung collective) instead of just silent.
# scripts/bench/probe_pad_factor.py says why padded is expected to lose here: it ships
# P*pad_slot = 42k node lanes (24 MB/exch f64) against ragged's 1.9k (1.1 MB) — a 22x pad
# factor at P=64, vs 5.4x at dars-16 (tie) and 15.3x at dars-32 (ragged +9%).
#SBATCH --job-name=bench_ng5_halo64
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80   # gpu partition mixes a100_40/a100_80 (~25 % HBM apart); one slow node paces the whole job — pin for comparable numbers
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
# .log, NOT .out: the paper reducer globs scripts/logs/bench_*.out and would pull these
# A/B re-runs into scaling.csv (figure takes best-per-ngpu, macros take first-in-glob-order
# => they silently disagree). Keep transport A/Bs out of the paper's bench_*.out glob.
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_ng5_halo64.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_ng5_halo64.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
C="--name ng5 --mesh-dir $M --dist-dir $POOL/ng5 --npes 64 --steps 25 --warmup 5 \
   --dt 180 --full 1 --ic-dir $M --year 1958"
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ==="
# ragged first = the control, banked before anything else can misbehave (see header).
echo "#### NG5-64 ragged ####"
srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo ragged
echo "=== ragged done $(date) ==="
# COLOURED (Phase 8d) — the prize: the volume model says 1.51 MB/exch here against padded's
# 23.7 (a 16x cut) and ragged's 1.08. If it lands near ragged, the AD-correct transport is
# also the fast one at 64 GPUs and ragged's forward-only niche closes at every scale.
echo "#### NG5-64 coloured ####"
timeout --signal=INT 1800 srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo coloured
echo "=== coloured done $(date) (rc=$?) ==="
# 30-min cap: if padded stalls again, the stage prints show WHERE and we stop paying. This
# still answers the open question the paper implicitly asserts — is padded merely SLOW at 64
# GPUs, or does it not complete at all?
echo "#### NG5-64 padded (timeout-capped 30 min) ####"
timeout --signal=INT 1800 srun --cpu-bind=none "$PY" scripts/bench/bench_forward_scaling.py $C --halo padded
rc=$?; [ $rc -eq 124 ] && echo "!!!! padded HIT THE 30-MIN CAP — see the last [bench] stage line above"
echo "=== done $(date) (padded rc=$rc) ==="
