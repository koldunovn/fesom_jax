#!/bin/bash
# Compile-only memory probe for the §3 NN-twin backward (E1 OOM diagnosis + fix validation).
# memory_analysis() reads the XLA-computed peak from buffer assignment — NO execution, so this
# is a cheap GPU slot (minutes): it never allocates the 34 GiB temp, it just reports it. Used to
# (a) confirm forward vs backward peak on the REAL GPU layout and (b) validate the nested-remat
# fix (--remat blocks) BEFORE paying for a full twin run. Single A100; same XLA flags as the twin.
#SBATCH --job-name=core2_nntwin_memprobe
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_nntwin_memprobe.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_nntwin_memprobe.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
CONFIG=${1:-tkegm}
N=${2:-6}

echo "=== $(hostname) $(date)  memprobe config=$CONFIG N=$N ==="
nvidia-smi -L 2>/dev/null || true
echo ""
echo "----- BASELINE (no nested remat) -----"
"$PY" scripts/archive/core2_nn_twin_memprobe.py --config "$CONFIG" --n "$N" --remat off
echo ""
echo "----- FIX (nested in-step remat) -----"
"$PY" scripts/archive/core2_nn_twin_memprobe.py --config "$CONFIG" --n "$N" --remat blocks
echo ""
echo "=== done $(date) ==="
