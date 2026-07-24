#!/bin/bash
# JUPITER (GH200, aarch64) environment for FESOM2-JAX scaling — SOURCE this from the sbatch
# or an interactive shell.  **VERIFIED ON THE MACHINE 2026-07-23** (this file replaced the
# untested template; see docs/JUPITER_SCALING.md).
#
#   source scripts/bench/jupiter/env_jupiter.sh
#
# One-time env build (~3.5 min, pure wheels, no container, no compilation):
#   module load Stages/2026 GCCcore/14.3.0 Python/3.13.5
#   python -m venv $HOME/jaxenv && source $HOME/jaxenv/bin/activate
#   pip install -U pip wheel
#   pip install "jax[cuda12]==0.10.1"      # aarch64 cp313 wheels EXIST; 0.10.1 == the Levante
#                                          # campaign version, so zero API drift
#   pip install -e ".[dev,viz]"
#
# NOTE the JSC `jax/0.8.1` module is **CPU-ONLY** ("An NVIDIA GPU may be present ... but a
# CUDA-enabled jaxlib is not installed") — it cannot be used for this campaign.  The venv
# above is the supported path; no Apptainer/NGC container is needed on GH200.

# --- interpreter -------------------------------------------------------------------
# The venv is built on the module Python, so the module MUST be loaded before activating
# it (otherwise: "error while loading shared libraries: libpython3.13.so.1.0").
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5 >/dev/null 2>&1
export JAXENV="${JAXENV:-$HOME/jaxenv}"
source "$JAXENV/bin/activate"
export PY="${PY:-python}"

# --- XLA / device knobs (same as the Levante GPU benches) --------------------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1

# --- NCCL over NDR InfiniBand (jax.distributed collectives) ------------------------
# JUPITER: 4x200G NICs per node, one per GH200 superchip.  Leave NCCL to auto-detect
# (the Kokkos twin measured the JSC defaults winning outright over hand-tuning); set
# NCCL_DEBUG=INFO for one run if you need to prove the transport is IB, not TCP (gate J2).
# export NCCL_DEBUG=INFO

# --- data on JUPITER (all staged by the Kokkos campaign; VERIFIED to exist) ---------
export HCLIM=/e/scratch/hclimrep/koldunov1/meshes
export ESTA=/e/scratch/e-sta-destine/koldunov1
export FESOM_JRA_DIR="${FESOM_JRA_DIR:-$HCLIM/JRA55-do-v1.4.0}"
export FESOM_PHC_PATH="${FESOM_PHC_PATH:-$HCLIM/phc3.0/phc3.0_winter.nc}"
export FESOM_SSS_PATH="${FESOM_SSS_PATH:-$HCLIM/JRA55-do-v1.4.0/PHC2_salx.nc}"
export FESOM_RUNOFF_PATH="${FESOM_RUNOFF_PATH:-$HCLIM/JRA55-do-v1.4.0/runoff.nc}"  # NOT CORE2_runoff.nc here
export FESOM_CHL_PATH="${FESOM_CHL_PATH:-$HCLIM/Sweeney/Sweeney_2005.nc}"

# JAX `.npy` mesh exports — GENERATED ON JUPITER by scripts/bench/jupiter/prepare_meshes.sbatch
# (the repo ships only the tiny `pi` mesh; Levante's `data/` was a symlink to /work).
export MESH="${MESH:-$ESTA/fesom_jax_meshes}"
# FESOM dist_<N> partition dirs — raw meshes shared with the Kokkos twin, read in place.
export RAW_ng5="${RAW_ng5:-$HCLIM/ng5}"
export RAW_core2="${RAW_core2:-$ESTA/meshes/core2_private}"   # the private core2 (has dist_1)
export RAW_dars="${RAW_dars:-$ESTA/meshes/dars}"
export RAW_farc="${RAW_farc:-$ESTA/meshes/farc}"

# multi-node launch knobs (unset JDIST for a single-node run)
export JDIST=1
export GPUS_PER_NODE=4
export ACCOUNT="${ACCOUNT:-e-sta-destine}"
export PARTITION="${PARTITION:-booster}"

echo "[env_jupiter] $(python -c 'import jax;print("jax",jax.__version__)' 2>/dev/null) MESH=$MESH JRA=$FESOM_JRA_DIR"
