#!/bin/bash
# Decisive GPU confirmation of the sharded multi-step grad-NaN FIX (cvmix_tke.py surf-flux denom).
# The CPU reproducer segfaults in XLA's CPU shard_map backend at the CORE2 mesh size, so verify on
# 2 real A100s — the platform where the §3 twin's sharded forward already runs. Pre-fix the twin
# returned |g|=nan; this runs the exact reproducer (sharded grad finite + == dense, NN-leaf check,
# relL2 match) on the FIXED code. config=tke isolates the TKE surf-flux fix; all3 = the twin config.
#SBATCH --job-name=repro_shard_nan_gpu
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/repro_sharded_grad_nan_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/repro_sharded_grad_nan_gpu.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.92}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

echo "=== $(hostname) $(date)  sharded-grad NaN repro on 2 A100 (FIXED cvmix_tke surf denom) ==="
nvidia-smi -L 2>/dev/null || true
for CFG in all3 tke; do
  echo ""
  echo "########## config=$CFG NS=2 ##########"
  "$PY" scripts/repro_sharded_grad_nan.py --nsteps 2 --config "$CFG" || echo "  (exit $?)"
done
echo ""
echo "=== done $(date) ==="
