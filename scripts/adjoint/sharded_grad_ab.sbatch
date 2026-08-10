#!/bin/bash
# Does sharding the ADJOINT across GPUs pay, and is the gradient still right?
#
# ⚠️ ONE PROCESS PER CONFIGURATION. JAX does not hand device memory back when a jitted graph
# goes out of scope, so a dense backward run earlier in the SAME process leaves the card nearly
# full: 26818559 died in XLA autotuning trying to allocate 62 MiB, and 26803606 has the same
# signature. It also makes a benchmark whose timings depend on what ran before them. So the
# dense reference is computed ONCE into a JSON, and every (transport, npes) then gets a fresh
# process and a clean device.
#
# ⚠️ REAL GPUs, never fake host devices: ragged does not exist on XLA:CPU, the collectives
# behave differently, and a 40-minute "timeout" on 4 fake CPU devices (26807407) says nothing
# about the transport being measured.
#
# TWO transports, both AD-correct:
#   padded   — slot-padded dense all_to_all; transpose is another all_to_all. Fastest on GPU at
#              <=16 devices; volume grows ~linearly in P (1.8x@P4 ... 41x@P128).
#   coloured — K bipartite-coloured lax.ppermute rounds; transpose is the inverse ppermute. The
#              ONLY transport whose volume does not grow with P, fastest measured at NG5-64.
# `ragged` is excluded — its transpose is wrong by O(1), so it can never carry a backward, and
# the library refuses it with return_grad_fn=True. `all_gather` is excluded for memory.
#
# Correctness BEFORE timing, and the reference proves itself (run twice, self-spread printed).
#SBATCH --job-name=nn_shard_ab
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH -o scripts/adjoint/shardab.%j.out
#SBATCH -e scripts/adjoint/shardab.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/nn_eddy/port_jax || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
N=${N:-48}
# The reference carries a FORMAT tag in its name. It used to be a single scalar dL/dk_gm; it is
# now per-tunable gradients plus the final-state fields in a sibling .npz. The `[ -s "$REF" ]`
# guard below cannot tell a stale file from a fresh one, so a format change must change the name
# — otherwise a failed reference step silently falls through to last week's numbers. Bump the
# tag whenever the contents change; never delete the old one.
REF=/work/ba1583/a270088/nn_eddy/seam_gate/dense_ref_N${N}_perTunable.json
echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
nvidia-smi -L

echo ""; echo "########## dense reference (own process, then the card is released) ##########"
"$PY" scripts/adjoint/sharded_grad_ab.py --n "$N" --npes "" --dense-ref-out "$REF"
[ -s "$REF" ] || { echo "no dense reference written — stopping"; exit 1; }

for TR in ${TRANSPORTS:-padded coloured}; do
  # npes=1 is OUT of the default list, and not as a preference: the sharded reverse stores every
  # step's carry (no segment checkpointing in that runner), so at one device a 48-step window is
  # 48 x 1.93 GB = 93 GB of carries. XLA reported it could not rematerialize below 124 GB and
  # died in autotuning (26819176). Two devices halve it, four quarter it. Pass NPES="1 2 4" only
  # if you have shortened --n enough to fit.
  for P in ${NPES:-2 4}; do
    echo ""; echo "########## transport=$TR npes=$P (fresh process, clean device) ##########"
    "$PY" scripts/adjoint/sharded_grad_ab.py --n "$N" --npes "$P" --transport "$TR" \
        --dense-ref-in "$REF"
    echo "  (exit $?)"
  done
done
echo ""; echo "=== done $(date) ==="
