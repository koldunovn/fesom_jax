#!/bin/bash
# Phase-9a JZ.0 sharded gate + diagnosis of the pre-existing sharded-gradient slowness.
# (1) Non-gradient sharded tests — the byte-identity gate for the ale_cfg threading added to
#     run_step_sharded/run_steps_sharded (these complete; they don't recompile-storm).
# (2) test_grad_ocean_fd_self_consistent — characterise the suite-blocking slowness with a wall
#     timeout (the Jun-8 baseline ran the WHOLE sharding group in 11:55; Phase 8b's ragged-halo
#     rewrite is the suspect — my ale_cfg=None is an inert closure constant).
#SBATCH --job-name=jz_shard_diag
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:55:00
#SBATCH -o /home/a/a270088/port_jax/scripts/jz_shard_diag.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/jz_shard_diag.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export XLA_FLAGS=--xla_force_host_platform_device_count=4

echo "=== node $(hostname)  $(date) ==="
echo "=== (1) NON-GRADIENT sharded tests — ale_cfg byte-identity gate ==="
$PY -m pytest fesom_jax/tests/test_halo.py fesom_jax/tests/test_step_sharded.py \
    fesom_jax/tests/test_ssh_sharded.py fesom_jax/tests/test_scatter_sharded.py \
    fesom_jax/tests/test_reductions_sharded.py -q -p no:cacheprovider
echo "rc(non-grad-shard)=$?"
date

echo ""
echo "=== (2) test_grad_ocean_fd (timeout 1200s) — characterise slow-vs-hang ==="
timeout 1200 $PY -m pytest \
    "fesom_jax/tests/test_gradient_sharded.py::test_grad_ocean_fd_self_consistent" \
    -q -p no:cacheprovider -s
echo "rc(grad-fd)=$?  (124 ⇒ timed out at 1200s ⇒ slow/hung)"
date
