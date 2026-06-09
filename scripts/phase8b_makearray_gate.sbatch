#!/bin/bash
# Phase 8b B.3 — gate the NG5 placement fix (_to_global_sharded: device_put → make_array_from_callback).
# Only run_steps_sharded calls _to_global_sharded, so the run_steps_sharded paths are the coverage:
#   (1) multistep FORWARD (test_multistep_scan_lowers_and_is_finite + teacher_forced) on 4 fake-devices
#   (2) multistep GRAD (test_grad_multistep_scan_backward) on 2 fake-devices
# Placement-only change ⇒ values identical; both must stay green. 30-min walltime = compute debug QOS.
#SBATCH --job-name=p8b_makearray
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/phase8b_makearray_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/phase8b_makearray_gate.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== node $(hostname)  $(date) ==="
echo "=== (1) multistep FORWARD (run_steps_sharded, 4 fake-devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  "$PY" -m pytest fesom_jax/tests/test_step_sharded.py -k "multistep" -q -p no:cacheprovider -rs
R1=$?
echo ""
echo "=== (2) multistep GRAD (run_steps_sharded backward, 2 fake-devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
  "$PY" -m pytest "fesom_jax/tests/test_gradient_sharded.py::test_grad_multistep_scan_backward" \
        -q -p no:cacheprovider -rs -s
R2=$?
echo ""
echo "=== SUMMARY: multistep-fwd rc=$R1  multistep-grad rc=$R2 ==="
[ $R1 -eq 0 ] && [ $R2 -eq 0 ] && echo "MAKEARRAY GATE GREEN" || echo "FAILURES (see above)"
