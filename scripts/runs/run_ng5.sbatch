#!/bin/bash
# NG5 (~5 km, 7.4 M nodes, 70 levels) — cold PHC spin-up, the flagship STABILITY ladder (Task B2).
# Modeled on run_b0_dars_seg0.sbatch (the proven multi-node recipe): host-build IC, FOLDED state
# end-to-end, --ragged halo, dist_64 (16 nodes x4 A100-80 = 64 GPU). Per-device load 115 k nod/GPU
# < dars@dist_16's 197 k ⇒ comfortably inside the proven memory envelope.
#
# R0 = cold dt=180 ~3 days (1440 steps), must be FINITE / CFL-stable (the headline risk — a prior
# JAX NG5 ran only 25 steps). The in-job --diagnostics flag reduces the final State to gather-free
# scalar health numbers and prints NG5_R0_FINITE / NG5_R0_NONFINITE. Gate R0 before R1.
#
# Parameterized (override at submit with --export=ALL,VAR=val): a cheap CANARY probe (STEPS=60) on
# the SAME dist_64 sharding de-risks setup/compile/early-finiteness before the full R0; then bump
# STEPS=1440 for the gate. CHUNK bounds the per-step forcing memory (~0.6 GB/step global; chunk 48
# = ~4.4 GB/device — small chunks for NG5). TAG names the restart subdir. RIN resumes a restart.
#   STEPS=${STEPS:-1440}  CHUNK=${CHUNK:-48}  TAG=${TAG:-r0}  RIN=${RIN:-}  (empty RIN ⇒ cold PHC IC)
#
#SBATCH --job-name=ng5_spin
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_ng5.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_ng5.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# CUDA-graph command buffers: disabled by default (the ADJOINT driver OOMed on accumulated graphs,
# PORTING_LESSONS). A FORWARD run has only 1-2 executables ⇒ safe + may cut kernel-launch overhead.
# CMDBUF=1 ⇒ leave them ON (XLA default) to MEASURE the speedup.
CMDBUF=${CMDBUF:-0}
if [ "$CMDBUF" = "1" ]; then export XLA_FLAGS=""; else export XLA_FLAGS="--xla_gpu_enable_command_buffer="; fi
export JDIST=1 GPUS_PER_NODE=4
export PYTHONUNBUFFERED=1   # flush python stdout to the .out file live (a walltime kill won't lose it)

STEPS=${STEPS:-1440}        # 3 days at dt=180 (= R0); probe with STEPS=60
CHUNK=${CHUNK:-48}          # per-chunk forcing steps (memory knob); 1440/48 = 30 chunks exactly
TAG=${TAG:-r0}
RIN=${RIN:-}               # restart-in (empty ⇒ cold PHC IC)
PROGRESS=${PROGRESS:-1}    # 1 ⇒ flushed per-phase/per-chunk timing (diagnose a long run live)
LOCALF=${LOCALF:-1}        # 1 ⇒ local-node forcing build (~npes× less host work; bit-identical)

MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5
OUT=/work/ab0995/a270088/port_jax/runs/ng5/$TAG

EXTRA=()
if [ -z "$RIN" ]; then
  rm -rf "$OUT"          # cold: clear a stale restart store (a prior dist_N's [P*Lmax] shape conflicts)
else
  EXTRA+=(--restart-in "$RIN")
fi
[ "$PROGRESS" = "1" ] && EXTRA+=(--progress)
[ "$LOCALF" = "1" ] && EXTRA+=(--local-forcing)
mkdir -p "$OUT"

echo "=== $(date) NG5 $TAG: nodes=$SLURM_JOB_NODELIST  dist_64 (64 GPU)  STEPS=$STEPS CHUNK=$CHUNK RIN=${RIN:-cold} ==="
srun --cpu-bind=none "$PY" scripts/run_from_config.py configs/ng5.yaml \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition dist_64 \
    --steps "$STEPS" --chunk-steps "$CHUNK" --ragged --diagnostics \
    --restart-out "$OUT" "${EXTRA[@]}"
R=$?
echo "ng5 $TAG exit=$R"
if [ $R -eq 0 ]; then
  echo "NG5_RUN_OK ($TAG: $STEPS steps dist_64; see the NG5_R0_FINITE/NONFINITE verdict above)"
else
  echo "NG5_RUN_FAILED ($TAG exit=$R)"
fi
exit $R
