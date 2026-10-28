# Running fesom-jax with REcoM on Albedo (AWI)

The Albedo counterpart of [RUNNING.md](RUNNING.md) (which is written for DKRZ Levante). Everything the repo's
defaults point at (`/work/ab0995/...`, `/pool/data/AWICM/...`, the Levante Python, `-A ab0995_gpu`) does not
exist here, so this page lists the Albedo substitutes, the scripts under `scripts/albedo/` and the configs
under `configs/albedo/` that use them. Set up and verified 2026-10-28 (account `sloza`).

## Environment

The conda env lives on `/work` (the home quota is 100 GB and full):

```bash
PY=/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python     # python 3.12, jax 0.10.1 + CUDA 12 wheels
```

It was built by `/albedo/work/user/sloza/fesom-jax-env/create_env.sh`:
`module load conda; mamba create -p .../fesom-jax python=3.12 pip; pip install -e "<repo>[cuda,dev,viz]" -c constraints.txt`
with `CONDA_PKGS_DIRS` / `PIP_CACHE_DIR` on `/work`. The editable install points at
`/albedo/home/sloza/REcoM_in_FESOM-JAX/fesom_jax_recom`; another checkout needs `PYTHONPATH=<checkout>`.

Two Albedo traps:

- **The login node has 256 cores and `ulimit -u 2048`.** XLA's CPU backend sizes its thread pools from the
  visible cores and dies at compile time with `pthread_create() failed` (docs/ENV.md). Run anything that
  compiles on the login node under `taskset -c 0-15 $PY ...`, or use a compute node.
- Importing JAX on the login node prints a CUDA `cuInit ... error 303` and falls back to CPU. Expected.

GPU jobs need `-A clidyn.clidyn -p gpu --gpus=N`. `gpu-001` has 2x A40 (46 GB), `gpu-002..005` 4x A100 (PCIe, 80 GB).
The GPUs are PCIe-attached (no NVLink), so the README's multi-GPU numbers do not transfer: run CORE2 on **one** GPU
(`partition: null`). Driver 535 / CUDA 12.2; the pip CUDA 12.9 wheels run on it (verified: `CudaDevice(id=0)`,
float64 matmul).

## Input data on Albedo

| input | Albedo location |
|---|---|
| CORE2 mesh bundle, PHC IC cache, partitions, JRA55-do 1958 (8 fields) | the Zenodo package, fetched to `/albedo/work/projects/p_specbic/sloza/fesom_jax_data/{core2_mesh_ic,core2_partitions,core2_forcing_1958}` by `scripts/fetch_data.py` |
| `psl` (sea-level pressure, FESOM2's `l_mslp`, **required by REcoM**) | not in the Zenodo package. `/albedo/work/projects/p_pool_fesom2/forcing/JRA55-do-v1.3.1/psl.{1958..2018}.nc` is readable; `.../fesom_jax_data/forcing_1958_mslp/` holds the 8 Zenodo fields + `psl.1958.nc` from v1.3.1 as one directory |
| JRA55-do v1.4.0, all years | `/albedo/pool/FESOM2/forcing/JRA55-do-v1.4.0` is a link into `/albedo/pool/FESOM/`, **not readable** by this account (ask for `p_pool_fesom` viewer access). v1.3.1 lacks `uas/vas/tas/rsds`; v1.5.0.1 (`/albedo/work/projects/p_pool_recom/forcing/JRA55-do-v1.5.0.1`) has 2020-2023 complete |
| SSS restoring, runoff, Sweeney chlorophyll | in `core2_mesh_ic/` of the Zenodo package (also `/albedo/work/projects/p_pool_fesom2/forcing/{CORE2,Sweeney}`) |
| REcoM CORE2 IC set (`fesom_jax.recom.ic.CORE2_IC_FILES`) | `/albedo/pool/FESOM2/recom/` (GLODAPv2 PI_TCO2 + TAlk mmol, WOA13 n00/i00, WOA18 o00 mmol, PISCES Fe) |
| Albani dust on the CORE2 nodes | `/albedo/pool/FESOM2/recom/CORE2/DustClimMonthlyAlbani.nc` |
| Albani dust on the pi nodes | FESOM2's `test/input/recom` is not on Albedo; `scripts/albedo/make_pi_dust.py` builds `fesom_jax_data/recom_pi/DustClimMonthlyAlbani_pimesh.nc` by nearest CORE2 node |
| raw CORE2 mesh (FESOM ASCII, for `prepare_mesh.py`) | `/albedo/pool/fesom2/core2` (dist_48 and up only) or the Zenodo `core2_mesh_ic/mesh_core2_raw` |
| REcoM's TEOS-10 SAAR table (`gsw_mod_saar_data.F90`, parsed at run time from `$FESOM_RECOM_SRC/mocsy/`) | copied from the Fortran fork into `fesom_jax_data/recom_src/mocsy/`; the REcoM 692f804 source tree is not on Albedo |
| the Levante 1958-2019 physics hindcast restarts | **not on Albedo**. CORE2 runs here cold-start from PHC (1958) unless a restart directory is copied over |

All of these are exported by `scripts/albedo/env.sh` (`FESOM_JRA_DIR`, `FESOM_IC_DIR`, `FESOM_RECOM_SRC`, ...);
the sbatch scripts source it, and an interactive session should too.

## Tests

```bash
sbatch scripts/albedo/run_ci_group.sbatch      # the data-free `ci` group on one CPU node (~10 min)
```

## pi + REcoM (CPU, 30 days)

`configs/albedo/pi_recom_bgc.yaml` = `configs/pi_recom_bgc.yaml` with the Albedo inputs (the CORE2 REcoM IC set is
interpolated onto pi because the 5-degree set is not here; the dust is the nearest-node file above).

```bash
sbatch scripts/albedo/run_pi_recom.sbatch
```

## CORE2 + REcoM (one GPU)

`configs/albedo/core2_recom.yaml` = `configs/core2_recom.yaml` on one GPU, double precision, cold start.

```bash
OUT=/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/core2_recom_1958      # must not exist yet
sbatch --export=ALL,OUT=$OUT,CONFIG=configs/albedo/core2_recom.yaml scripts/albedo/run_core2_gpu.sbatch \
    --steps 17520 --diagnostics --monthly-out $OUT/monthly --restart-out $OUT/restart
```

One year at dt 1800 is 17520 steps. `--precision single` is the opt-in faster mode (RUNNING.md). Output and
restarts are as in RUNNING.md ("What comes out"); the monthly zarr stores open with
`xarray.open_zarr(..., consolidated=False)`.

## Timings on Albedo (2026-10-28, cold start, 96 steps = 2 chunks of 48, compile excluded by the second chunk)

| run | node | s/step | SYPD | one year |
|---|---|---|---|---|
| CORE2 + REcoM, double precision (job 48595512) | 1x A100 80 GB PCIe (gpu-004) | 2.63 | 1.9 | ~13 h: split into two 12 h-QOS segments via `--restart-in` |
| CORE2 + REcoM, `--precision single` (job 48595567) | same | 1.75 | 2.8 | ~8.5 h |
| pi + REcoM, double precision, CPU (job 48595511) | 32 cores of an mpp node | ~0.9 | — | 30 days in ~15 min |

For comparison, Levante's 4x GH200 do the double-precision CORE2 + REcoM step in 0.060 s (RUNNING.md). A single
PCIe A100 is the wrong hardware for production hindcasts; it is enough for development, gates and short tests.
Both CORE2 runs ended `FINITE n_nonfinite=0` and wrote a restart.
