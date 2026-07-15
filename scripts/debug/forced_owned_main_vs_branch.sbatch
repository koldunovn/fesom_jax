#!/bin/bash
# DISCRIMINATOR: test_forcing_sharded::test_forced_multistep_owned_matches[2] failed
# (uv owned max|Δ|=9e-3 > 1e-7) in the on-device forcing gate job 26275725 — but that
# test was NEVER run on branch perf/kokkos-m7-levers (the green job 26267782 covered
# jra55/transports/ssh/EVP only). Run it in BOTH trees:
#   main   (worktree /home/a/a270088/port_jax_main_ab @ ceea184)
#   branch (/home/a/a270088/port_jax @ perf/kokkos-m7-levers + on-device forcing WIP)
# main FAIL too ⇒ pre-existing (not the fusions); main PASS + branch FAIL ⇒ a branch
# fusion (CG psum pair / exchange_pair) breaks the npes=2 forced path — bisect next.
#SBATCH --job-name=forced_owned_ab
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/forced_owned_ab.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/forced_owned_ab.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
export JAX_PLATFORMS=cpu
T=fesom_jax/tests/test_forcing_sharded.py::test_forced_multistep_owned_matches
for TREE in /home/a/a270088/port_jax_main_ab /home/a/a270088/port_jax; do
  echo "=== $(date) TREE=$TREE sha=$(git -C $TREE rev-parse --short HEAD) ==="
  (cd "$TREE" && PYTHONPATH="$TREE" XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      "$PY" -m pytest "$T" -q)
  echo "=== rc=$? ==="
done
echo "=== done $(date) ==="
