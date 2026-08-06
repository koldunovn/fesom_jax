#!/bin/bash
# Verification run for the RECALIBRATED test_forced_multistep_owned_matches[2]
# (free-running 3-step compare -> teacher-forced per-step compare, Decision 4).
# Runs the whole test_forcing_sharded.py file so the neighbouring gates are shown to be
# unaffected by the edit.
#SBATCH --job-name=forced_tf_gate
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/forced_tf_gate.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/forced_tf_gate.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
TREE=/home/a/a270088/port_jax
cd "$TREE"
echo "=== $(date) sha=$(git -C $TREE rev-parse --short HEAD) ==="
export JAX_PLATFORMS=cpu
PYTHONPATH="$TREE" XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -m pytest fesom_jax/tests/test_forcing_sharded.py -q -rA
echo "=== rc=$? done $(date) ==="
