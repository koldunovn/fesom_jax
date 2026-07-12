#!/bin/bash
# JZ.7 — assembled CORE2 zstar step (KPP+GM+EVP+zstar) step-1 dump gate vs z2_cdump.
# The full 4-config model on the 126858-node CORE2 mesh: heavy trace+compile, so a
# compute node. Usage: sbatch scripts/debug/jz7_assembled_gate.sbatch
#SBATCH --job-name=jz7_gate
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/jz7_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/jz7_gate.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== node $(hostname)  $(date) ==="
$PY -m pytest fesom_jax/tests/test_ale_zstar.py::test_jz7_assembled_zstar_step1 \
    -q -p no:cacheprovider -s
RC=$?
echo "=== JZ.7 gate rc=$RC ==="
[ $RC -eq 0 ] && echo "JZ.7 GREEN" || echo "JZ.7 FAILURE (see above)"
