#!/bin/bash
# JZ.2 forcing dump gate: the zstar ice-step forcing (water_flux, virtual_salt≡0, relax_salt,
# real_salt_flux) at CORE2 step 1 vs the C z2_cdump `forcing` tag. Heavy (126858 nodes + JRA55
# load + ice step) ⇒ a compute node.
#SBATCH --job-name=jz2_fgate
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH -o /home/a/a270088/port_jax/scripts/jz2_forcing_gate.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/jz2_forcing_gate.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== node $(hostname)  $(date) ==="
$PY -m pytest fesom_jax/tests/test_ale_zstar.py \
    -k "forcing_flip or forcing_dump_gate_zstar_diagnostic" \
    -q -p no:cacheprovider -s
echo "rc=$?"
date
