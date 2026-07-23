#!/bin/bash
# JUPITER (GH200, aarch64) environment for FESOM2-JAX scaling — SOURCE this from the
# sbatch. TEMPLATE: every ⚠️VERIFY line must be checked on the machine (see
# docs/JUPITER_SCALING.md §2). Not tested here (no JUPITER access from the porting host).

# --- pick ONE jaxlib path (§2 of the doc); export PY accordingly -------------------
# (a) NGC container (recommended on GH200):
#   module load Apptainer                                   # ⚠️VERIFY
#   JAXSIF=$HOME/jax.sif                                    # apptainer pull docker://nvcr.io/nvidia/jax:<tag>
#   PY="apptainer exec --nv -B $PWD:$PWD -B $DATA_ROOT:$DATA_ROOT $JAXSIF python"
# (b) pip venv with aarch64+CUDA wheels:
#   module load Python CUDA                                 # ⚠️VERIFY CUDA 12.x
#   source $HOME/jaxenv/bin/activate
#   PY="python"
export PY="${PY:-python}"

# --- XLA / device knobs (same as Levante GPU benches) ------------------------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1

# --- NCCL over NDR InfiniBand (jax.distributed collectives) — ⚠️VERIFY names --------
# Uncomment + set to the JSC-documented interface/HCA; run once with NCCL_DEBUG=INFO to
# confirm the transport is IB, not TCP (see gate J2 in the doc).
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_HCA=mlx5
# export NCCL_DEBUG=INFO

# --- data locations on JUPITER (⚠️VERIFY paths) ------------------------------------
export FESOM_JRA_DIR="${FESOM_JRA_DIR:?set to the JRA55-do-v1.4.0 copy on JUPITER}"
# export FESOM_SSS_PATH=...  FESOM_RUNOFF_PATH=...  FESOM_CHL_PATH=...  FESOM_PHC_PATH=...
export MESH="${MESH:-/e/scratch/CHANGEME/koldunov1/fesom_jax_meshes}"   # the .npy JAX exports
export DIST="${DIST:-/e/scratch/CHANGEME/koldunov1/meshes}"             # FESOM dist_<N> (shared w/ Kokkos)

# multi-node launch knobs (leave for single-node)
export JDIST=1
export GPUS_PER_NODE=4
echo "[env_jupiter] PY=$PY  MESH=$MESH  DIST=$DIST  JRA=$FESOM_JRA_DIR"
