#!/bin/bash
# Phase 8b B.0c — gate the halo-only ragged_all_to_all exchange on REAL A100s (NCCL).
# ragged_all_to_all is UNIMPLEMENTED on XLA:CPU, so this is the only place B.0 can be gated.
# (1) isolated primitive: ragged == all_gather forward (byte-id on valid) + grad (transpose);
# (2) step-level wiring: run_step_sharded(use_ragged=True) == (use_ragged=False) on owned.
# Each test in its own fresh process (the S.8 OOM lesson). Uses the real GPUs via jax.devices().
#SBATCH --job-name=phase8b_b0_gpu
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/phase8b_b0_gpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/phase8b_b0_gpu.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
echo "=== node $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null || true
"$PY" -c "import jax; jax.config.update('jax_enable_x64', True); print('jax', jax.__version__, jax.devices())"
for t in \
  "test_halo.py::test_ragged_primitive_matches_allgather" \
  "test_step_sharded.py::test_ragged_step_matches_allgather" ; do
  echo "######## $t ########"
  "$PY" -m pytest "fesom_jax/tests/$t" -q -p no:cacheprovider -rs -s
  echo ">>>> $t exit: $?"
done
echo "=== done $(date) ==="
