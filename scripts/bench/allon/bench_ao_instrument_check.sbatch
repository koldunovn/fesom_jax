#!/bin/bash
# ALL-ON INSTRUMENT CHECK (core2 dist_4, OFF legs only): my ao core2-4 baseline measured
# 239 ms/step where the session-2 ondev A/B measured 112.5 on the SAME config+instrument —
# and ao dars-8 ran at 100 s/step (26315973 TIMEOUT, chunk2 == chunk1 so not compile).
# Four legs isolate the two env differences vs the certified template:
#   noflag_nopre   : no transport flag (allgather), PREALLOCATE=false   (= session-2 exactly)
#   padded_nopre   : --padded,                    PREALLOCATE=false   (= my ao harness)
#   noflag_pre     : no transport flag,           preallocation ON (default)
#   padded_pre     : --padded,                    preallocation ON
#SBATCH --job-name=ao_instr_check
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/ao_instr_check.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/ao_instr_check.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" PYTHONUNBUFFERED=1 FESOM_REUSE_EXE=1
CONFIG=configs/core2_full_bench.yaml
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
run_leg() { # $1=label $2=prealloc(0/1) $3... extra flags
  local LEG=$1 PRE=$2; shift 2
  if [ "$PRE" = 0 ]; then export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  else unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION; fi
  echo "#### ao-check npes=4 LEG=$LEG $(date +%T) ####"
  srun --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
      --dist-dir "$POOL" --ic-dir data/ic_core2 --partition dist_4 \
      --steps 240 --chunk-steps 48 --progress "$@"
  echo "#### npes=4 LEG=$LEG rc=$? $(date +%T) ####"
}
echo "=== $(date) node=$SLURM_JOB_NODELIST instrument check (sha $(git rev-parse --short HEAD)) ==="
run_leg noflag_nopre 0
run_leg padded_nopre 0 --padded
run_leg noflag_pre   1
run_leg padded_pre   1 --padded
echo "=== done $(date) ==="
