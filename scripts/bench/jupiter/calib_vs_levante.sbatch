#!/bin/bash
# GH200-vs-A100 CALIBRATION leg — the one run that makes a cross-machine ratio honest.
#
# docs/PARALLELISM.md records "JAX GPU: 53-57 SYPD, 1 node / 4xA100" for CORE2 in the LEGACY
# reference configuration: **KPP + GM/Redi + prognostic ice + real JRA55-1958 + PHC IC**,
# dt=1800.  The JUPITER campaign runs the paper's PRODUCTION physics instead (mEVP + TKE +
# zstar, no KPP) — a different model, so campaign ms/step must NOT be divided by that A100
# number.  This job reproduces the LEGACY config on GH200 so the machine ratio is measured
# on identical physics, and runs the production config beside it so the two are tied together.
#
#   sbatch scripts/bench/jupiter/calib_vs_levante.sbatch
#
#SBATCH --job-name=calib
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --gpu-bind=none
#SBATCH -o scripts/logs/jupiter/calib_%j.out
#SBATCH -e scripts/logs/jupiter/calib_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs/jupiter
source scripts/bench/jupiter/env_jupiter.sh
unset JDIST

B="--name core2 --mesh-dir $MESH/mesh_core2 --dist-dir $RAW_core2 --ic-dir $MESH/mesh_core2 \
   --dt 1800 --full 1 --year 1958"
# LEGACY = the docs/PARALLELISM.md reference model: KPP + GM + prognostic ice, linfs, EVP.
LEGACY="--ice 1 --kpp 1 --gm 1 --mevp 0 --tke 0 --zstar 0"
PROD="--mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1"

echo "=== $(date) GH200-vs-A100 calibration on $(hostname) ==="
echo "=== A100 reference (docs/PARALLELISM.md): CORE2 LEGACY, 4xA100, 53-57 SYPD "
echo "===   -> 86.5-93.0 ms/step  (SYPD = 1800/(s_step*365.25) = 4.928/s_step)"

for REP in 1 2; do
  echo "#### core2 4GPU (1N) halo=padded rep=$REP dt_prod=1800 [LEGACY kpp+gm] ####"
  $PY scripts/bench/bench_forward_scaling.py $B $LEGACY --npes 4 --halo padded --steps 150 \
    || echo "legacy rep=$REP rc=$?"
done
for REP in 1 2; do
  echo "#### core2 4GPU (1N) halo=padded rep=$REP dt_prod=1800 [PROD mevp+tke+zstar] ####"
  $PY scripts/bench/bench_forward_scaling.py $B $PROD --npes 4 --halo padded --steps 150 \
    || echo "prod rep=$REP rc=$?"
done
echo "=== done $(date) ==="
