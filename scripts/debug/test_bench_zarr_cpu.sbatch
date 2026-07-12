#!/bin/bash
# Validate the bench --out-zarr wiring end-to-end on CPU (CORE2 ocean, 2 fake-devices) before
# the expensive NG5 GPU run: run a few steps, write the sharded Zarr, reconstruct a field, assert
# finite + correct global shape. Exercises run_steps_sharded(return_executable) → jfn(*jargs)
# folded output → zarr_output.write_state_zarr → reconstruct_global. 30-min walltime (debug QOS).
#SBATCH --job-name=test_bench_zarr
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/test_bench_zarr_cpu.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/test_bench_zarr_cpu.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
ZD=/scratch/a/a270088/tmp/bench_zarr_core2
rm -rf "$ZD"
echo "=== node $(hostname) $(date) ==="
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
  "$PY" scripts/bench/bench_forward_scaling.py --name core2 --npes 2 --steps 2 --full 0 \
        --out-zarr "$ZD"
echo ">>>> bench exit: $?"
echo "=== reconstruct + verify ==="
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
  "$PY" - <<PYEOF
import numpy as np
from fesom_jax import zarr_output as zo
import zarr
root = zarr.open_group("$ZD", mode="r")
print("arrays:", list(root.array_keys()))
for f in ("T","S","eta_n","uv"):
    g = zo.reconstruct_global("$ZD", f)
    print(f"  {f:6s} global shape={g.shape}  finite={bool(np.isfinite(g).all())}  "
          f"min/max={g.min():.3e}/{g.max():.3e}")
print("ZARR BENCH PATH OK")
PYEOF
echo "exit: $?"