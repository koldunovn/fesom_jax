#!/bin/bash
# perf/kokkos-m7-levers gates, part 2: the sharded-step tail that job 26267782's 3 h wall
# cut off (21 instances had passed, 0 failures). The CG fused-psum integration was already
# covered by the passed instances (every sharded step solves the SSH CG); this tail covers
# the KPP/TKE/ice/mEVP/zstar sharded steps — ice/mEVP drive the fused u,v exchange_pair.
#SBATCH --job-name=m7levers_gates2
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/m7levers_gates2.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/m7levers_gates2.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
echo "=== $(date) sharded-step tail (4 fake devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 "$PY" -m pytest \
    fesom_jax/tests/test_step_sharded.py -q \
    -k "kpp or tke or ice or mevp or zstar or multistep or assembled"
rc=$?
echo "=== done $(date) rc=$rc ==="
exit $rc
