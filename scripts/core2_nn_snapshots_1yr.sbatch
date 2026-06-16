#!/bin/bash
# §3 / E2 — produce SEASONALLY-DIVERSE chunk-start states for the batched-window NN twin.
# The batched twin needs chunk starts from DIFFERENT months (winter deep convection vs summer
# stratification ⇒ very different N²/MLD/mixing), not 8 consecutive 5-h chunks (which are too similar
# and don't recover the NN field). This runs the all-on (zstar+TKE+mEVP+GM) model 1 year and pickles
# the State every ~30 days (snap_step<N>.pkl) → ~12 monthly snapshots spanning the seasonal cycle.
# Forward-only ⇒ single A100; same XLA flags as the spin-up. Snapshots on /work (each ~1.8 GB).
#SBATCH --job-name=core2_nn_snaps1yr
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o /home/a/a270088/port_jax/scripts/core2_nn_snapshots_1yr.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/core2_nn_snapshots_1yr.%j.out

source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAPDIR=${1:-/work/ab0995/a270088/port_jax/nn_twin_snaps}
EVERY=${2:-1440}          # ~30 d at dt=1800 → ~12 snapshots over 1 yr
OUT=/home/a/a270088/port_jax/data/nn_snaps_1yr

echo "=== $(hostname) $(date)  all3 1-yr snapshot spin-up every $EVERY steps -> $SNAPDIR ==="
nvidia-smi -L 2>/dev/null || true

"$PY" scripts/core2_kpp_climate_run.py --years 1 --start-year 1958 --dt 1800 \
    --ale on --tke on --mevp on --ic-dir data/ic_core2_dist864 \
    --out "$OUT" --every 1000 --snapshot-dir "$SNAPDIR" --snapshot-every "$EVERY"
rc=$?
echo "=== done $(date)  (exit $rc) ==="
[ "$rc" -eq 0 ] && ls -lah "$SNAPDIR"
exit $rc
