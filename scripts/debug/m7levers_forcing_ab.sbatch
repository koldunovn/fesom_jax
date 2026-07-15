#!/bin/bash
# Host-forcing A/B for perf/kokkos-m7-levers (CORE2 work -> compute node, not login).
#SBATCH --job-name=m7levers_forcing_ab
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/m7levers_forcing_ab.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/m7levers_forcing_ab.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
"$PY" scripts/bench/m7levers_forcing_ab.py
