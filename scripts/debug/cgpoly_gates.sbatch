#!/bin/bash
# CGPOLY gates (the M7 Chebyshev-precond CG lever, port_kokkos Task E.3 transferred):
#   * test_ssh_sharded.py — ALL existing S.6 gates (the OFF path must stay untouched:
#     cheb=None is the byte-identical default) + the new CGPOLY gates:
#     equal-tolerance iters ratio ≥1.5 on the captured REAL CORE2 ssh_rhs (k=2,3)
#     and sharded-vs-single-device iteration-count/owned-d_eta parity at k=3.
#SBATCH --job-name=cgpoly_gates
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/cgpoly_gates.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/cgpoly_gates.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
echo "=== $(date) ssh sharded + CGPOLY gates (4 fake devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 "$PY" -m pytest \
    fesom_jax/tests/test_ssh_sharded.py -q -s -x
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
