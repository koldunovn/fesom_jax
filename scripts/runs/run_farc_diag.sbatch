#!/bin/bash
# farc DIAGNOSTIC: bisect the dars TKE+mEVP compile hang at 5x smaller scale (1 node, dist_4).
# Runs 3 short configs, each capped at 8 min by `timeout` — a pathological compile ⇒ exit 124.
#   tke_mevp  = the dars-hanging combo (reproduce?)   tke_evp = isolate TKE   kpp_mevp = isolate mEVP
# If tke_evp times out ⇒ TKE is the culprit; if kpp_mevp runs ⇒ mEVP is fine.
#SBATCH --job-name=farc_diag
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_farc_diag.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_farc_diag.%j.out

source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_farc
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/farc
OUT=/work/ab0995/a270088/port_jax/runs/farc_diag; rm -rf "$OUT"; mkdir -p "$OUT"

for CFG in farc_tke_evp farc_kpp_mevp farc_tke_mevp; do
  echo "============================================================"
  echo "=== $(date) TEST $CFG (8-min compile cap) ==="
  T0=$SECONDS
  timeout 480 srun --cpu-bind=none "$PY" scripts/run_from_config.py configs/$CFG.yaml \
      --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition dist_4 \
      --steps 10 --chunk-steps 10 --ragged --local-forcing --progress --diagnostics \
      --restart-out "$OUT/$CFG"
  R=$?
  echo ">>> $CFG exit=$R  elapsed=$((SECONDS-T0))s  $([ $R -eq 124 ] && echo 'TIMEOUT/HANG' || echo 'completed')"
done
echo "FARC_DIAG_DONE"
