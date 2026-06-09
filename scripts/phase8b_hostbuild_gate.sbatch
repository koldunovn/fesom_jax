#!/bin/bash
# Phase 8b B.3 host-build rewrite GATE (FAST): the host-build + always-device_put change is
# pure DATA PLACEMENT (host numpy → device_put-sharded, never the full global on GPU 0), so
# every value must be numerically identical. This job confirms:
#   (1) State (rest/zeros) + partition unchanged (1 device, default backend);
#   (2) the OCEAN sharded FORWARD (run_step_sharded host-fold; run_steps_sharded device_put
#       via the multistep tests) on 4 CPU fake-devices — the npes==1 byte-identity no-op +
#       npes==2 owned field-appropriate match.
# The forced device_put path (_fold_forcing + boundary_node) is the SEPARATE heavier gate
# (phase8b_hostbuild_gate_forced.sbatch). Grad: scripts/test_grad_ocean.sbatch.
#SBATCH --job-name=p8b_hostbuild
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=00:35:00
#SBATCH -o /home/a/a270088/port_jax/scripts/phase8b_hostbuild_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/phase8b_hostbuild_gate.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname)  $(date) ==="

echo "=== (1) State + partition (1 device, default backend) ==="
JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_state.py fesom_jax/tests/test_partition_state.py \
        fesom_jax/tests/test_ale.py fesom_jax/tests/test_tracers.py \
        -q -p no:cacheprovider -rs
R1=$?

echo ""
echo "=== (2) OCEAN sharded forward (4 CPU fake-devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_step_sharded.py \
        -k "not gm and not kpp and not ice and not assembled and not ragged" \
        -q -p no:cacheprovider -rs
R2=$?

echo ""
echo "=== SUMMARY: state/partition rc=$R1  ocean-sharded-forward rc=$R2 ==="
[ $R1 -eq 0 ] && [ $R2 -eq 0 ] && echo "HOST-BUILD GATE GREEN" || echo "FAILURES (see above)"
