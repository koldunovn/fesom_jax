#!/bin/bash
# NG5-32 ON-leg REGRESSION DECOMPOSITION (26315977: ON = +33.6% ms/step while host share
# fell 13.2s->1.0s/chunk — the DEVICE time regressed 50->80 s/chunk). The ON bundle has two
# device-visible levers; this splits them at ng5-32, same protocol:
#   A  off            (--local-forcing only — the production baseline)
#   B  tables-only    (--local-forcing --forcing-on-device)      <- new local-tables path
#   C  cheb-only      (--local-forcing --cheb-degree 3)          <- CGPOLY at REAL iteration
#      counts (the kernel A/Bs ran 25 cold steps where CG has not spun up; a kappa=30
#      mis-fit to NG5's operator would hide there and explode here)
#SBATCH --job-name=bench_ao_ng5dec
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/bench_ao_ng5dec.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/bench_ao_ng5dec.%j.log
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
  echo "#### ao-ng5dec npes=32 LEG=$LEG $(date +%T) ####"
  srun -N8 -n8 --ntasks-per-node=1 --cpu-bind=none "$PY" scripts/run_from_config.py "$CONFIG" \
      --dist-dir "$DIST" --ic-dir "$IC" --partition dist_32 --coloured \
      --steps 240 --chunk-steps 48 --progress --local-forcing "$@"
  echo "#### npes=32 LEG=$LEG rc=$? $(date +%T) ####"
}
echo "=== $(date) nodes=$SLURM_JOB_NODELIST ng5-32 ON-regression decomposition ==="
run_leg off_1
run_leg tabonly_1 --forcing-on-device
run_leg chebonly_1 --cheb-degree 3
echo "=== done $(date) ==="
