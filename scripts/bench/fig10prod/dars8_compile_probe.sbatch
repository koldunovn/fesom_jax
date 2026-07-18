#!/bin/bash
# dars-8 XLA COMPILE-CLIFF BISECT: production physics at a 435k-lane shard never finishes
# compiling (>3 h, 26331913/26315973) while the same physics at <=280k lanes compiles in
# <150 s and the OLD model at the SAME shard compiled in 172 s. Four time-boxed legs
# (50 min each) find the guilty component — a leg that reaches "[bench] compile+1st run
# done" is below the cliff; a killed leg is above it:
#   full     : ice+mevp+tke+zstar (the production set — expected KILLED, the control)
#   nozstar  : ice+mevp+tke, linfs
#   notke    : ice+mevp+kpp+zstar
#   nomevp   : ice(EVP)+tke+zstar
#SBATCH --job-name=dars8_compile_probe
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:40:00
#SBATCH -o /home/a/a270088/port_jax/scripts/logs/dars8_compile_probe.%j.log
#SBATCH -e /home/a/a270088/port_jax/scripts/logs/dars8_compile_probe.%j.log
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
export JDIST=1 GPUS_PER_NODE=4
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
C="--name dars --mesh-dir $M --dist-dir $POOL/dars --steps 25 --full 1 --dt 120 \
   --ic-dir $M --year 1958 --halo padded --npes 8"
probe() { local LEG=$1; shift
  echo "#### dars8-probe LEG=$LEG $(date +%T) ####"
  timeout 50m srun -N2 -n2 --ntasks-per-node=1 --cpu-bind=none \
      "$PY" scripts/bench/bench_forward_scaling.py $C "$@"
  echo "#### LEG=$LEG rc=$? $(date +%T) ####    (rc=124 => KILLED at 50m = above the cliff)"
}
echo "=== $(date) nodes=$SLURM_JOB_NODELIST dars-8 compile bisect (sha $(git rev-parse --short HEAD)) ==="
probe full    --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1
probe nozstar --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 0
probe notke   --ice 1 --mevp 1 --tke 0 --kpp 1 --gm 0 --zstar 1
probe nomevp  --ice 1 --mevp 0 --tke 1 --kpp 0 --gm 0 --zstar 1
echo "=== done $(date) ==="
