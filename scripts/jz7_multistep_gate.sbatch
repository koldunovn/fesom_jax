#!/bin/bash
# JZ.7 MULTI-STEP — assembled CORE2 zstar run for 3 steps vs the z2_cdump 3-step set.
# The REAL validation of the JZ.6 live-geometry re-points: at step 1 live==static, so the
# re-points only diverge from static at steps ≥2 (the stretched column). Heavy trace+compile
# (the full 4-config model, 3 chained steps) ⇒ a compute node.
# Usage: sbatch scripts/jz7_multistep_gate.sbatch
#SBATCH --job-name=jz7_ms
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/jz7_ms.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/jz7_ms.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== node $(hostname)  $(date) ==="
$PY -m pytest fesom_jax/tests/test_ale_zstar.py::test_jz7_assembled_zstar_steps123 \
    -q -p no:cacheprovider -s
RC=$?
echo "=== JZ.7 multi-step gate rc=$RC ==="
[ $RC -eq 0 ] && echo "JZ.7-MS GREEN" || echo "JZ.7-MS FAILURE (see above)"
