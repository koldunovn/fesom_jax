#!/bin/bash
# Re-test dars (3.16M x nl57) FULL model on a SINGLE node (dist_4, 4 A100), WITH the
# make_array_from_callback placement fix. The earlier dars-4 (job 25446479) OOM'd BEFORE the
# fix: model working set ~49-55 GiB + a 29.32 GiB transient (likely the global-placement
# identity_fn the fix removes) -> ~78 GiB -> OOM on 80 GB. If that transient was placement,
# dars-4 should now fit (~49-55 GiB < 80). Single process, 4 local GPUs (NO jax.distributed).
# ragged, dt180. 30-min walltime for backfill.
#SBATCH --job-name=dars_1node
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/dars_1node.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/dars_1node.%j.out
source /sw/etc/profile.levante 2>/dev/null
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd /home/a/a270088/port_jax
POOL=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_dars
echo "=== node $(hostname) $(date) ==="; nvidia-smi -L 2>/dev/null | head -4
echo "#### dars FULL 4GPU (1 node) ragged — does it fit now? ####"
"$PY" scripts/bench/bench_forward_scaling.py --name dars --mesh-dir "$M" --dist-dir "$POOL/dars" \
    --npes 4 --steps 25 --warmup 5 --dt 180 --full 1 --ic-dir "$M" --year 1958 --ragged 1
echo ">>>> dars 1-node ragged exit: $? ($(date))"
