#!/bin/bash
# M1 seam tests (NN-GM child plan): the gm_nn wiring, its two bit-identity invariants, the
# FD gradient probe, the deployment path, and the sharded halo-placement gate.
#
# Three pytest processes, because the sharded test needs CPU fake-devices (the suite's own
# SHARDING-group trick) and the device count is fixed at process start:
#   1. the touched-library group (gm_nn + every GM/params/config consumer)
#   2. the seam file itself
#   3. the sharded placement gate with 4 fake devices
#SBATCH --job-name=nn_seam_tests
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=40:00
#SBATCH -o scripts/nn_eddy/seamtests.%j.out
#SBATCH -e scripts/nn_eddy/seamtests.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date) ==="
echo "=== checkout: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null) ==="

echo ""
echo "=== 1. network + the GM/params/config consumers the seam touched ==="
"$PY" -m pytest -q -p no:cacheprovider \
    fesom_jax/tests/test_gm_nn.py \
    fesom_jax/tests/test_gm_coeffs.py \
    fesom_jax/tests/test_gm_slopes.py \
    fesom_jax/tests/test_gm_redi.py \
    fesom_jax/tests/test_gm_bolus.py \
    fesom_jax/tests/test_gm_bolus_adv.py \
    fesom_jax/tests/test_gm_step.py \
    fesom_jax/tests/test_gm_eki.py \
    fesom_jax/tests/test_calibrate.py \
    fesom_jax/tests/test_calib_twin.py \
    fesom_jax/tests/test_config.py \
    fesom_jax/tests/test_run_config.py \
    fesom_jax/tests/test_run_entry.py
RC1=$?

echo ""
echo "=== 2. the seam gate ==="
"$PY" -m pytest -q -p no:cacheprovider fesom_jax/tests/test_gm_nn_seam.py
RC2=$?

echo ""
echo "=== 3. sharded halo placement (4 CPU fake-devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -m pytest -q -p no:cacheprovider fesom_jax/tests/test_gm_nn_sharded.py
RC3=$?

echo ""
echo "=== SUMMARY: consumers rc=$RC1  seam rc=$RC2  sharded rc=$RC3 ==="
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ] && [ $RC3 -eq 0 ] && echo "M1 TESTS GREEN" || echo "M1 TESTS FAILED"
