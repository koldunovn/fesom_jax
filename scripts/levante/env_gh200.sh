# The run environment of fesom-jax on the dolpung GH200 nodes (aarch64). Source it INSIDE a dolpung job (logins are
# x86). As scripts/perf/env_dolpung.sh: no module loads, and the x86 spack tree that srun exports from the login
# environment is stripped from PATH and LD_LIBRARY_PATH. Sets PY to the venv of make_env_gh200.sbatch.
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$_here/gh200.conf"
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/sw/spack-levante/' | grep -v mambaforge | paste -sd:)
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/sw/spack-levante/' | grep -v mambaforge | paste -sd:)
export PATH LD_LIBRARY_PATH
unset PYTHONPATH PYTHONHOME CONDA_PREFIX
export PY=$VENV/bin/python
export TMPDIR=$SCRATCH_TMP CUDA_CACHE_PATH=$SCRATCH_TMP/nv_gh200
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$RUNS/slurm"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"      # command buffers off (docs/PORTING_LESSONS.md)
