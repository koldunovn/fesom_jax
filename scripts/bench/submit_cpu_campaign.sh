#!/bin/bash
# Submit the CPU strong-scaling campaign — one job per (mesh, node count) rung.
#
#   ./submit_cpu_campaign.sh <core2|farc|dars|ng5> [only_first_n_rungs]
#
# RANKS PER NODE IS SET BY HOST MEMORY, NOT BY SPEED, on the big meshes. Every rank calls
# load_mesh on the whole global export AND builds the global State + PHC IC on the host before
# sharding, so per-rank RSS is ~independent of npes:
#
#   core2  ~2 GB/rank    farc ~5 GB/rank    dars ~29 GB/rank (measured)   ng5 ~60 GB/rank (est)
#
# Measured the hard way (job 26488934): dars at 8 ranks/node peaked at 232 GB and was OOM-killed
# on a 256 GB node; ng5 at 4 ranks/node was killed too (job 26488935). Hence 4 ranks/node for
# dars and 2 for ng5 — the cores stay fully used (128/ranks each), but the fatter thread pools
# are less efficient, which is itself a campaign result.
#
# The topology sweep (job 26488917, CORE2, 2 nodes, coloured) ranked the splits:
#   16x8 = 992 ms  <  8x16 = 1012 ms  <  32x4 = 1206 ms  ;  4x32 = npes-8 defect, see below.
# 16 ranks/node also keeps every rung clear of the npes=8 coloured/gloo failure.
set -euo pipefail
cd /home/a/a270088/port_jax
MESH=${1:?usage: submit_cpu_campaign.sh <core2|farc|dars|ng5> [first_n_rungs]}
LIMIT=${2:-99}
LEG=scripts/bench/cpu_campaign_leg.sbatch

case "$MESH" in
  core2) PPN=16; NODES="1 2 4 8";     TIME=00:30:00 ;;
  farc)  PPN=8;  NODES="1 2 4 8 16";  TIME=00:40:00 ;;
  dars)  PPN=4;  NODES="4 8 16 32";   TIME=01:00:00 ;;
  ng5)   PPN=2;  NODES="8 16 32";     TIME=02:00:00 ;;
  *) echo "unknown mesh $MESH"; exit 2 ;;
esac

CORES=$((128 / PPN))
i=0
for n in $NODES; do
  i=$((i + 1)); [ "$i" -gt "$LIMIT" ] && break
  npes=$((n * PPN))
  sbatch --nodes=$n --ntasks-per-node=$PPN --cpus-per-task=$CORES --time=$TIME \
         --job-name="${MESH}_n${n}_col" \
         --export=ALL,CMESH=$MESH,CNPES=$npes,CPPN=$PPN,CHALO=coloured,CSTEPS=5,CWARM=2 \
         $LEG
  # padded anchor at the smallest rung only: it anti-scales past one node, and the transports
  # are forward bit-exact, so one anchor is enough to place the coloured curve.
  if [ "$i" -eq 1 ]; then
    sbatch --nodes=$n --ntasks-per-node=$PPN --cpus-per-task=$CORES --time=$TIME \
           --job-name="${MESH}_n${n}_pad" \
           --export=ALL,CMESH=$MESH,CNPES=$npes,CPPN=$PPN,CHALO=padded,CSTEPS=5,CWARM=2 \
           $LEG
  fi
done
echo "submitted $MESH: nodes {$NODES} (first $LIMIT) x ${PPN} ranks/node x ${CORES} cores"
