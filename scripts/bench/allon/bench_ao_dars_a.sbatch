#!/bin/bash
# ALL-ON production-loop envelope — dars_a (see allon/README.md for the frozen protocol).
# Transport per point = the baseline envelope's banked winner (padded <=16 GPU, ragged
# mid-scale, coloured NG5) — run_from_config's default is the allgather ORACLE, which is
# unrepresentative at scale (and impractical at NG5).
#SBATCH --job-name=bench_ao_dars_a
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_ao_dars_a.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_ao_dars_a.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" PYTHONUNBUFFERED=1 FESOM_REUSE_EXE=1
CONFIG=configs/dars_full_bench.yaml
DIST=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
IC=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
EXTRA_BOTH=""
ON_FLAGS="--forcing-on-device --cheb-degree 3"
run_leg() { # $1=npes $2=srun-nodes $3=transport-flag $4=leg $5... extra
  local NP=$1 SN=$2 TR=$3 LEG=$4; shift 4
  echo "#### ao-dars_a npes=$NP LEG=$LEG $(date +%T) ####"
  if [ "$SN" -gt 1 ]; then export JDIST=1 GPUS_PER_NODE=4; else unset JDIST; fi
  srun -N$SN -n$SN --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
      --dist-dir "$DIST" --ic-dir "$IC" --partition dist_$NP $TR \
      --steps 240 --chunk-steps 48 --progress $EXTRA_BOTH "$@"
  echo "#### npes=$NP LEG=$LEG rc=$? $(date +%T) ####"
}
echo "=== $(date) nodes=$SLURM_JOB_NODELIST all-on envelope dars_a (sha $(git rev-parse --short HEAD)) ==="
run_leg 8 2 --padded off_1
run_leg 8 2 --padded on_1 $ON_FLAGS
run_leg 8 2 --padded off_2
run_leg 8 2 --padded on_2 $ON_FLAGS
run_leg 16 4 --ragged off_1
run_leg 16 4 --ragged on_1 $ON_FLAGS
run_leg 16 4 --ragged off_2
run_leg 16 4 --ragged on_2 $ON_FLAGS
run_leg 32 8 --ragged off_1
run_leg 32 8 --ragged on_1 $ON_FLAGS
run_leg 32 8 --ragged off_2
run_leg 32 8 --ragged on_2 $ON_FLAGS
echo "=== done $(date) ==="
