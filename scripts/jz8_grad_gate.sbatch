#!/bin/bash
# JZ.8 — zstar gradient gates (GATE 9a §4): masked-NaN d(SST)/d(T0) through the assembled
# KPP+GM+zstar step, d(SST)/d(k_ver) (PP+zstar), and the NEW d(SST)/d(hbar-IC) state path.
# N=1 backward on a compute node (CPU, no ice scan ⇒ fits RAM). Usage: sbatch scripts/jz8_grad_gate.sbatch
#SBATCH --job-name=jz8_grad
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/jz8_grad.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/jz8_grad.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== node $(hostname)  $(date) ==="
$PY -m pytest fesom_jax/tests/test_ale_zstar.py -k "jz8_grad" \
    -q -p no:cacheprovider -s
RC=$?
echo "=== JZ.8 grad gate rc=$RC ==="
[ $RC -eq 0 ] && echo "JZ.8-GRAD GREEN" || echo "JZ.8-GRAD FAILURE (see above)"
