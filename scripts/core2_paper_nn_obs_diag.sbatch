#!/bin/bash
# §3 / E2 DIAGNOSTIC — is the SLOW deployed (long-forward) MLD mixing-improvable at all? Deploy a
# UNIFORM multiplier c (a global c_k scaling, D2a-style) over the long forward and compare its
# MLD-vs-WOA to default (c=1). c=2.0 (more mixing, D2a's direction) vs c=0.7 (less, the NN's net
# direction). Isolates the fast/slow misalignment from the NN's spatial structure. Forward-only.
#SBATCH --job-name=core2_nn_obs_diag
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_paper_nn_obs_diag.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_paper_nn_obs_diag.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
NNPKL=/work/ab0995/a270088/port_jax/nn_obs_trained.pkl
RESULTS=scripts/nn_obs_diag.jsonl
: > "$RESULTS"
echo "=== $(hostname) $(date)  E2 deployment diagnostic: uniform-multiplier long forward ==="
for C in 2.0 0.7; do
    echo ""
    echo "===== uniform multiplier c=$C  (60-d forward)  $(date) ====="
    "$PY" scripts/core2_paper_nn_obs.py --mode validate --config all3 \
        --const-mult "$C" --long-steps 2880 --long-seg 240 \
        --nn-pkl "$NNPKL" --results "$RESULTS"
    echo "  (c=$C exit: $?)"
done
echo ""
echo "=== done $(date) ==="
