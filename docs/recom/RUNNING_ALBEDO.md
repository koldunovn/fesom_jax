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
sbatch scripts/albedo/run_pytest.sbatch fesom_jax/tests/test_recom_step_fork.py [-k expr]   # one test file on a CPU node
```

## pi + REcoM (CPU, 30 days)

`configs/albedo/pi_recom_bgc.yaml` = `configs/pi_recom_bgc.yaml` with the Albedo inputs (the CORE2 REcoM IC set is
interpolated onto pi because the 5-degree set is not here; the dust is the nearest-node file above).

```bash
sbatch scripts/albedo/run_pi_recom.sbatch
```

## pi + the spectral REcoM fork (CPU, 30 days)

The spectral fork (`sveta-loza/fesom2` `fesom2.6_recom_tra_diags_spectral`: 4 phytoplankton groups, CDOM, 13-band
light from OASIM; `docs/plans/20261028-recom-spectral-port.md`) runs from `configs/albedo/pi_recom_spectral.yaml`:
the fork's namelist set (`scripts/recom/fortran/namelists_albedo_spectral/`, 37 tracers), `recom.spectral: true`,
the optics `.dat` files (`recom.optics_dir`, `fesom_jax_data/recom_pi/optics`), the OASIM monthly climatology
(`forcing.oasim_dir`, `/albedo/work/projects/p_specbic/sloza/forcing_oasim_v2`) and the pi bundle built from the
fork's partition directory (`fesom_jax_data/mesh_pi_fork`, the levels the Fortran twin ran with).

```bash
sbatch scripts/albedo/run_pi_recom_spectral.sbatch
```

The job names `--ic-dir fesom_jax_data/ic_pi_fork`, the PHC physics IC built on that bundle
(`fesom_jax.phc_ic.build_and_cache_ic(load_mesh(.../mesh_pi_fork), None, .../ic_pi_fork)`): the packaged pi IC
(`fesom_jax/data/ic_pi`) has zero T and S in the three layers the fork's levels add at the polar nodes, and a run
from it goes non-finite within 32 steps (jobs 48602595, 48603145/46; the same config on the packaged pi mesh is
finite, job 48603144).

`recom.spectral_co2lim: true` applies the CO2 limitation in the spectral growth (the fork does not; the user's
decision of 2026-10-28). `recom.spectral: false` with the same namelist (`bgc_num 36`, no CDOM:
`namelists_albedo_spectral_nowb/`) is the fork's broadband build. The gates behind this run are the
`test_recom_*_fork.py`, `test_recom_spectral_light.py` and `test_recom_spectral_wiring.py` files against the twin
runs under `fesom_jax_runs/recom_ref/` (`docs/plans/20261028-recom-spectral-port.md`, progress log).

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
| pi + the spectral REcoM fork (37 tracers, 13-band light), CPU (job 48603201) | 32 cores of an mpp node | ~1.28 | — | 30 days in 21 min, `FINITE n_nonfinite=0` |
| CORE2 + the spectral REcoM fork, double precision (job 48603697, 96 steps) | 1x A100 80 GB PCIe | 2.50 | 2.0 | `FINITE n_nonfinite=0`; `configs/albedo/core2_recom_spectral.yaml` |

For comparison, Levante's 4x GH200 do the double-precision CORE2 + REcoM step in 0.060 s (RUNNING.md). A single
PCIe A100 is the wrong hardware for production hindcasts; it is enough for development, gates and short tests.
Both CORE2 runs ended `FINITE n_nonfinite=0` and wrote a restart.

## CORE2 + the spectral fork warm-started from the 1997 4-PFT Fortran restart (GPU)

The user's experiment setup: the spun-up 1997 state of `/albedo/work/projects/p_specbic/sloza/restart_1997_4pft`
(its README: the colleague's 2-PFT 1997 restart plus the cocco/phaeo and CDOM fields of the kdind15 run), the
JRA55-do repeat-year climatology `clim61` (FESOM2 `y_perpetual`), CORE2 runoff and PHC2 SSS from the FESOM2 pool,
a constant chlorophyll 0.1 for the shortwave penetration, the OASIM climatology.

```bash
# 1. the Fortran restart -> a fesom-jax restart store (a CPU node, ~2 min, 120 GB requested)
python scripts/recom/fortran_restart_to_jax.py --restart /albedo/work/projects/p_specbic/sloza/restart_1997_4pft \
    --year 1997 --config configs/albedo/core2_recom_spectral_warm1998.yaml \
    --out /albedo/work/projects/p_specbic/sloza/fesom_jax_runs/restart_1997_4pft_jax --date 1998-doy001
# 2. the run (one GPU), --year = the config's start_year (run_from_config defaults to 1958)
OUT=<new dir> CONFIG=configs/albedo/core2_recom_spectral_warm1998.yaml \
    sbatch --export=ALL,OUT,CONFIG scripts/albedo/run_core2_gpu.sbatch --steps 11680 --year 1998 \
    --restart-out $OUT/restart --monthly-out $OUT/monthly
```

`forcing.repeat_year: clim61` makes the JRA reader open `{var}.clim61.nc` for every model year and rebase its
365-day axis to that year (the 366th day of a leap year holds the last record, as FESOM2); `forcing.chl_const: 0.1`
replaces the Sweeney file. The converter maps the positional `tra00NN` files (positions 6/12/14/39 are `sf6`,
`cfc12`, `r14c`, `r39ar`) onto the config's layout, carries the AB2 histories (`_M1`, `urhs_AB`, `ssh_rhs_old`),
rebuilds the layer thicknesses from `hbar` (fesom-jax has no partial cells; the user's run has them), takes
`Alk_surf` from the IC set as REcoM does at setup, and fills the few cells below the Fortran bottom with the
column's deepest value (2 nodes on CORE2). `PROVENANCE.json` in the store records all of it. Note that the 1997
benthos pools are huge (median benthic C 1e11 mmol m-2 in the Fortran restart itself); the model carries them.

## A multi-year CORE2 run and the comparison with a Fortran run

```bash
scripts/albedo/chain_core2_gpu.sh configs/albedo/core2_recom_spectral_kdind15.yaml <ROOT> 4 5840 1958   # 2 years, 4 segments
scripts/albedo/link_monthly.sh <ROOT>                                                                     # ROOT/monthly/<YYYY_MM>
python scripts/recom/compare_fortran_jax.py --jax <ROOT>/monthly \
    --fortran /albedo/work/projects/p_specbic/sloza/tmp_recom-fesom_output/fesom2.6_spectral_kdind15 \
    --mesh /albedo/work/projects/p_specbic/sloza/fesom_jax_data/core2_mesh_ic/mesh_core2 --out <ROOT>/compare
```
One half-year segment of CORE2 + the spectral fork takes ~5.2 h on one PCIe A100 (3.2 s/step), inside the 12 h qos;
`chain_core2_gpu.sh` sequences the segments with `--dependency=afterok`. The comparison script prints a monthly table
(global NPP and CO2 flux [Pg C yr-1], surface Chl, pCO2, SST, ice area, the six inventories, pattern correlations)
and draws `compare_<YYYY_MM>.png` (surface Chl, NPP, CO2 flux: fesom-jax, Fortran, difference).
