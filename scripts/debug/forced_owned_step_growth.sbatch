#!/bin/bash
# Growth-curve + localization discriminator for the pre-existing
# test_forced_multistep_owned_matches[2] failure. See the module docstring of
# scripts/debug/forced_owned_step_growth.py for what each block decides.
#SBATCH --job-name=forced_growth
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/forced_growth.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/forced_growth.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
TREE=/home/a/a270088/port_jax
cd "$TREE"
echo "=== $(date) sha=$(git -C $TREE rev-parse --short HEAD) ==="
export JAX_PLATFORMS=cpu
PYTHONPATH="$TREE" XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -u scripts/debug/forced_owned_step_growth.py
echo "=== rc=$? done $(date) ==="
