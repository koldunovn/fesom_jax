#!/bin/bash
# The M1 seam verdict on its own: the seam file plus the sharded halo-placement gate.
#
# Split out from gm_nn_seam_tests.sbatch because that job also re-runs four heavy assembled-CORE2
# consumer files (test_gm_step, test_gm_bolus_adv, test_calib_twin, test_run_entry) and the
# consumers must not be able to time-cut the verdict. Run both: this one answers "is the seam
# right", the other answers "did the seam disturb anything".
#SBATCH --job-name=nn_seam_only
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH -o scripts/runs/seamonly.%j.out
#SBATCH -e scripts/runs/seamonly.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== $(hostname) $(date)  commit $(git rev-parse --short HEAD) ==="

echo ""
echo "=== the seam gate (single device) ==="
# -s, not just -v: these tests PRINT the numbers that are the evidence (jaxpr equation counts,
# the multiplier bounds and spatial spread, the AD/FD agreement, the per-feature gradient rows).
# pytest captures stdout on a pass, so without -s a green run tells you nothing you can record.
"$PY" -m pytest -v -s -p no:cacheprovider fesom_jax/tests/test_gm_nn_seam.py
RC1=$?

echo ""
echo "=== the network's own suite (gm_nn.py was edited) ==="
"$PY" -m pytest -q -p no:cacheprovider fesom_jax/tests/test_gm_nn.py
RC2=$?

echo ""
echo "=== sharded halo placement (4 CPU fake-devices) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -m pytest -v -s -p no:cacheprovider fesom_jax/tests/test_gm_nn_sharded.py
RC3=$?

echo ""
echo "=== script-side tests (nn_data + eta product) ==="
"$PY" -m pytest -q -p no:cacheprovider scripts/tests/test_nn_data.py \
    scripts/tests/test_coarsen_eta.py fesom_jax/tests/test_suite_groups.py
RC4=$?

echo ""
echo "=== SUMMARY: seam rc=$RC1  network rc=$RC2  sharded rc=$RC3  scripts rc=$RC4 ==="
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ] && [ $RC3 -eq 0 ] && [ $RC4 -eq 0 ] \
    && echo "M1 SEAM GREEN" || echo "M1 SEAM FAILED"
