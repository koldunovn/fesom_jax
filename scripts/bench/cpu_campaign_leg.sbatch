#!/bin/bash
# ONE RUNG of the CPU strong-scaling campaign — parameterised, so every mesh and node count
# runs the identical code path. Submit with the allocation on the command line:
#
#   sbatch --nodes=4 --ntasks-per-node=8 --cpus-per-task=16 --time=00:30:00 \
#          --export=ALL,CMESH=dars,CNPES=32,CPPN=8 scripts/bench/cpu_campaign_leg.sbatch
#
# CNPES must equal nodes x CPPN, and a dist_<CNPES> partition must exist. Cores per rank are
# derived as 128/CPPN so the node is always fully occupied (see the topology sweep for why
# the split matters more than the core count).
#
# Transport defaults to coloured: on CPU it is the only one that crosses the node boundary
# without anti-scaling (2 nodes, CORE2: coloured 1004 ms vs padded 2889 ms, job 26488829).
#SBATCH --job-name=cpu_camp
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cpu_camp_%x.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cpu_camp_%x.%j.out
source /sw/etc/profile.levante 2>/dev/null
module load openmpi/4.1.2-gcc-11.2.0 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax

: "${CMESH:?set CMESH}"; : "${CNPES:?set CNPES}"; : "${CPPN:?set CPPN}"
CHALO=${CHALO:-coloured}
CORES=$((128 / CPPN))
CSTEPS=${CSTEPS:-8}
CWARM=${CWARM:-2}

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JDIST_CPU=1
export FESOM_CPU_COLL=gloo
export OMP_NUM_THREADS=$CORES

POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
W=/work/ab0995/a270088/fesom_jax_meshes
case "$CMESH" in
  core2) M=$PWD/data/mesh_core2; D=$POOL/core2; IC=$PWD/data/ic_core2; DT=1800 ;;
  farc)  M=$W/mesh_farc;  D=$POOL/farc;  IC=$M; DT=900 ;;
  dars)  M=$W/mesh_dars;  D=$POOL/dars;  IC=$M; DT=180 ;;
  ng5)   M=$W/mesh_ng5;   D=$POOL/ng5;   IC=$M; DT=180 ;;
  *) echo "unknown CMESH=$CMESH"; exit 2 ;;
esac

NODES=${SLURM_JOB_NUM_NODES:-?}
echo "=== CPU CAMPAIGN  mesh=$CMESH npes=$CNPES nodes=$NODES  ${CPPN}procs/node x ${CORES}cores"
echo "===   halo=$CHALO dt=$DT steps=$CSTEPS  nodelist=$SLURM_JOB_NODELIST  $(date) ==="
free -g | head -2

# Full output goes to its own file — a tail -N here once truncated away the actual error
# message and left only its stack trace, which cost a debugging round.
FULL=/home/a/a270088/port_jax/scripts/logs/full_${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out
mpirun -n "$CNPES" --map-by "ppr:$CPPN:node:pe=$CORES" --bind-to core \
    "$PY" scripts/bench/bench_forward_scaling.py \
    --name "$CMESH" --mesh-dir "$M" --dist-dir "$D" --ic-dir "$IC" \
    --npes "$CNPES" --steps "$CSTEPS" --warmup "$CWARM" --dt "$DT" --full 1 --year 1958 \
    --halo "$CHALO" > "$FULL" 2>&1
echo "full log: $FULL"
grep -vE "^\[Gloo\]|^$|^\s+@ " "$FULL" | tail -16

echo ">>>> exit=$? mesh=$CMESH npes=$CNPES nodes=$NODES"
free -g | head -2
echo "=== done $(date) ==="
