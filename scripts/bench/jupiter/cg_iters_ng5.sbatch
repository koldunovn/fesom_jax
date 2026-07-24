#!/bin/bash
# THE decisive measurement for the ng5 x P=128 cliff: how many PCG iterations does the SSH
# solve actually take, and does it STOP ON THE maxiter CAP instead of on the tolerance?
#
# Why this is the last hypothesis standing.  Nine structural explanations have been measured
# and eliminated (bad node; node/elem/edge balance; Lmax per kind; coloured buffer volume;
# colouring rounds K; SSH operator nnz_max, MAX/MEAN 1.011; device count — dars at P=128 is
# fine; raw collective performance at 64/128/256 GPUs — ppermute is FLAT; transport-
# specificity — all three transports collapse).  Everything static is clean.
#
# What is NOT static: `_pcg` is a `lax.while_loop` (ssh.py) with
#     cond = (resid >= rtol) & (it < maxiter),  MAXITER = 500 (config.py)
# and the CG halo refresh is documented as "the DOMINANT per-step comm
# (~2 exchanges x ~127 CG iterations)".  So if this configuration stops converging and runs
# to the 500 cap, EVERY per-iteration cost multiplies — which reproduces the whole observed
# signature: all transports hit, worst for the transport with the dearest exchange
# (padded 5.95x > ragged ~5x > coloured 2.65x), mesh-specific, and NON-MONOTONIC in P
# (a convergence threshold, not a scaling law).  500/190 ~ 2.6 vs the observed 2.65x.
#
# `hit_cap=1` would also be a CORRECTNESS finding, not just a performance one: the returned
# eta would not be solved to soltol at that configuration.
#
#   sbatch scripts/bench/jupiter/cg_iters_ng5.sbatch
#
#SBATCH --job-name=cgiters
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=64
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/cgiters_%j.out
#SBATCH -e scripts/logs/jupiter/cgiters_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
export JDIST=1
export FESOM_CG_ITERS=1          # arms the env-gated [cg-iters] print in ssh._pcg

# Few steps: we want the ITERATION COUNT, not a timing. Counts are a numerical property and
# are unaffected by machine load, so this is valid even if something else of ours is running.
C="--name ng5 --mesh-dir $MESH/mesh_ng5 --dist-dir $RAW_ng5 --ic-dir $MESH/mesh_ng5 \
   --steps 3 --dt 180 --full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 \
   --year 1958 --halo coloured"

echo "=== $(date) ng5 CG iteration counts on $SLURM_JOB_NODELIST ==="
for NP in 64 128 256; do
  N=$(( NP / 4 ))
  echo "######## ng5 ${NP}GPU (${N}N) ########"
  timeout --kill-after=120 1800 \
    srun -N"$N" -n"$N" --ntasks-per-node=1 --cpu-bind=none \
      $PY scripts/bench/bench_forward_scaling.py $C --npes "$NP" 2>&1 \
    | grep -aE "^\[cg-iters\]|^\[bench" | awk '!seen[$0]++' | head -25 \
    || echo "rc=$?"
done
echo "=== done $(date) ==="
echo "=== READ: it=<count> and hit_cap. hit_cap=1 at P=128 but 0 at P=64/256 => the cliff is"
echo "=== the PCG running to MAXITER=500, and eta is NOT converged there. ==="
