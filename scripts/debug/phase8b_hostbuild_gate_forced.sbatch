#!/bin/bash
# Phase 8b B.3 host-build rewrite GATE (FORCED — the heavier path): exercises _fold_forcing's
# host build + the forcing/boundary_node device_put in run_step_sharded, for the PRIMARY
# assembled KPP+GM+ice forced step, FORWARD and BACKWARD. These two tests are the most
# representative of the forced device_put surface; if both pass, the forced placement change
# is validated. Each runs in its OWN process (the forced-step shard_map compile is heavy +
# the grad compile accumulates executable memory). ~15-30 min compile EACH.
#SBATCH --job-name=p8b_hb_forced
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/phase8b_hostbuild_gate_forced.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/phase8b_hostbuild_gate_forced.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname)  $(date) ==="

echo "######## FORWARD: assembled KPP+GM+ice owned-match (4 fake-devices) ########"
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest "fesom_jax/tests/test_step_sharded.py::test_assembled_kpp_gm_ice_sharded_owned_matches" \
        -q -p no:cacheprovider -rs -s
RF=$?
echo ">>>> forward assembled exit: $RF"

echo ""
echo "######## BACKWARD: assembled forced grad (2 fake-devices) ########"
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
  "$PY" -m pytest "fesom_jax/tests/test_gradient_sharded.py::test_grad_assembled_forced_backward" \
        -q -p no:cacheprovider -rs -s
RB=$?
echo ">>>> backward assembled exit: $RB"

echo ""
echo "=== SUMMARY: forced-forward rc=$RF  forced-backward rc=$RB ==="
[ $RF -eq 0 ] && [ $RB -eq 0 ] && echo "FORCED HOST-BUILD GATE GREEN" || echo "FAILURES (see above)"
