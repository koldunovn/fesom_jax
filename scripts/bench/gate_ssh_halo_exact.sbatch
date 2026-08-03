#!/bin/bash
# Bit-exactness of the FESOM_SSH_HALO override: the transports are forward bit-exact, so
# running the CG halo on a different one must not change a single value. (Perf verdict is
# separate — job 26493332 already showed padded/allgather are SLOWER; this checks the wiring
# is right, so that verdict rests on a correct implementation.)
#SBATCH --job-name=ssh_halo_exact
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/ssh_halo_exact.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/ssh_halo_exact.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=32
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
OUT=/scratch/a/a270088/ssh_halo_gate; mkdir -p $OUT
G="--name core2 --mesh-dir $PWD/data/mesh_core2 --dist-dir $POOL/core2 --ic-dir $PWD/data/ic_core2 \
   --npes 4 --steps 3 --warmup 1 --dt 1800 --full 1 --year 1958 --halo coloured"
for mode in default padded allgather; do
  echo "#### GATE ssh_halo=$mode ####"
  es=""; [ "$mode" != "default" ] && es="FESOM_SSH_HALO=$mode"
  env $es XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    "$PY" scripts/bench/bench_forward_scaling.py $G --state-out $OUT/$mode.npz 2>&1 \
    | grep -E "^\[bench|Error|error" | tail -4
done
"$PY" - <<'PYEOF'
import numpy as np
ref = np.load("/scratch/a/a270088/ssh_halo_gate/default.npz")
for m in ("padded", "allgather"):
    got = np.load("/scratch/a/a270088/ssh_halo_gate/%s.npz" % m)
    same = all(np.array_equal(ref[k], got[k]) for k in ref.files)
    worst = max(float(np.max(np.abs(ref[k] - got[k]))) for k in ref.files)
    print("[gate] ssh_halo=%-9s bit-identical=%s  max|diff|=%.3e" % (m, same, worst), flush=True)
PYEOF
echo "=== done $(date) ==="
