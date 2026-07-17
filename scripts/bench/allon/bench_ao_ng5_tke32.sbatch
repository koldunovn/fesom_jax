#!/bin/bash
# NG5-32 all-ON under TKE (USER DECISION 2026-07-17: NG5 production mixing is TKE, not the
# stale ng5.yaml KPP — that config predates the TKE port). Three configs x2 reps decide the
# NG5 lever set AND disambiguate the tables regression (26322951 measured +36% under KPP):
#   off      --local-forcing                                    (production baseline)
#   tabcheb  --local-forcing --forcing-on-device --cheb-degree 3 (full lever set)
#   chebonly --local-forcing --cheb-degree 3                     (the KPP-era fallback)
# If tabcheb wins under TKE => KPP was the tables-regression culprit and NG5 gets the FULL
# lever set (no per-mesh exception); if tabcheb still regresses => local-tables/shard-size,
# keep chebonly and dig later (TABLES_NG5_DEVICE_REGRESSION).
#SBATCH --job-name=bench_ao_ng5tke
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_ao_ng5tke.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_ao_ng5tke.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" PYTHONUNBUFFERED=1 FESOM_REUSE_EXE=1
export JDIST=1 GPUS_PER_NODE=4
CONFIG=configs/ng5_full_bench.yaml
DIST=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/ng5
IC=/work/ab0995/a270088/fesom_jax_meshes/mesh_ng5
run_leg() { local LEG=$1; shift
  echo "#### ao-ng5tke npes=32 LEG=$LEG $(date +%T) ####"
  srun -N8 -n8 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
      --dist-dir "$DIST" --ic-dir "$IC" --partition dist_32 --coloured \
      --steps 240 --chunk-steps 48 --progress --local-forcing "$@"
  echo "#### npes=32 LEG=$LEG rc=$? $(date +%T) ####"
}
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ng5-32 TKE lever-set decision (sha $(git rev-parse --short HEAD)) ==="
for REP in 1 2; do
  run_leg off_$REP
  run_leg tabcheb_$REP --forcing-on-device --cheb-degree 3
  run_leg chebonly_$REP --cheb-degree 3
done
echo "=== done $(date) ==="
