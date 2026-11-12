# The run environment of fesom-jax on Levante's A100 nodes (partition gpu, x86). Source it inside a gpu job.
# Sets PY to the env of make_env_x86.sh. As the repo's Levante GPU jobs (scripts/runs/run_core2_hindcast.sbatch):
# no CUDA module (the CUDA 12 runtime comes as pip wheels), XLA command buffers off.
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$_here/gh200.conf"
unset PYTHONPATH PYTHONHOME
export PY=$VENV_X86/bin/python
export TMPDIR=$SCRATCH_TMP
mkdir -p "$TMPDIR" "$RUNS/slurm"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"      # command buffers off (docs/PORTING_LESSONS.md)
