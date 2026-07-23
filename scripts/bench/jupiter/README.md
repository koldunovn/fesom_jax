# JUPITER (GH200) scaling — quick start

Full instructions: **`docs/JUPITER_SCALING.md`**. These are TEMPLATES — every `⚠️VERIFY` /
`CHANGEME` must be set on the machine (they were not tested from the porting host).

1. **Env** — get GPU JAX working on aarch64 first (the crux): edit `env_jupiter.sh`, pick
   a jaxlib path (NGC container recommended), pass gate **J0** (`jax.devices()` shows
   `CudaDevice`s).
2. **Data** — transfer the `.npy` JAX mesh exports + JRA55 forcing from Levante; reuse the
   Kokkos twin's `dist_<N>` partitions already on JUPITER. Set `MESH`/`DIST`/`FESOM_JRA_DIR`.
3. **Bootstrap** — CORE2 (in-repo mesh), validates env+launch+reduce:
   ```bash
   MESH_NAME=core2 NODES_LIST="1 2" sbatch scripts/bench/jupiter/bench_scaling_jupiter.sbatch
   ```
4. **Target** — the big meshes, past Levante's 64-GPU ceiling:
   ```bash
   MESH_NAME=ng5  NODES_LIST="8 16 32 64"  sbatch scripts/bench/jupiter/bench_scaling_jupiter.sbatch
   MESH_NAME=dars NODES_LIST="4 8 16 32"   sbatch scripts/bench/jupiter/bench_scaling_jupiter.sbatch
   ```
5. **Reduce** — feed `*.out` to `paper_jax/scripts/reduce/reduce_scaling.py`, or
   `SYPD = dt_prod/(sstep·365.25)` by hand.

Always read the `[bench-finite]` line before trusting a `per_step` number.
Kokkos twin's JUPITER reference: `port_kokkos/docs/plans/20260722-m7-JUPITER-scaling-PLAN.md`.
