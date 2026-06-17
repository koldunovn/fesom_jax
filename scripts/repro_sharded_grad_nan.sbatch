#!/bin/bash
# Bisect the multi-step SHARDED-grad NaN that blocks the §3 multi-GPU NN twin. The P=4 N=48 twin runs
# the FORWARD fine but jax.grad of the checkpointed sharded scan returns |g|=nan. The 1-step verify
# can't see it (TKE/NN has ZERO influence at step 1 — grad is structurally 0), so the NaN is
# MULTI-STEP-specific. This sweeps configs at NS=2 on CPU with 2 fake devices to localize which
# scheme (tke / gm / ice / interactions) and which Params/NN leaf carries the NaN. Cheap CPU job.
#SBATCH --job-name=repro_shard_nan
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/repro_sharded_grad_nan.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/repro_sharded_grad_nan.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_force_host_platform_device_count=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== $(hostname) $(date)  sharded-grad NaN bisection (CPU, 2 fake devices) ==="
# Most-informative order first (so a time-out still yields the decisive bisection):
#   all3   = the twin config (does it reproduce on CPU?)
#   tke    = NN-path NaN with TKE alone? (NaN intrinsic to TKE+core)
#   tkegm  = TKE x GM interaction?
#   gm/ice = does each scheme self-NaN (d/d(Params))?
for CFG in all3 tke tkegm gm ice; do
  echo ""
  echo "########## config=$CFG NS=2 ##########"
  "$PY" scripts/repro_sharded_grad_nan.py --nsteps 2 --config "$CFG" || echo "  (exit $?)"
done
echo ""
echo "=== done $(date) ==="
