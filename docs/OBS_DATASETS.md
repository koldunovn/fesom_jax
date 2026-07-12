# Observational datasets — paper obs-application targets (Task A1)

Staging for the obs-based half of the three capability pillars. **Staged by** `scripts/tools/stage_obs.sh`
**into** `/work/ab0995/a270088/port_jax/obs/` (large files live on `/work`, not `/home`).

## Why obs-based, NOT reanalyses (the methodology rule — Locked Decision #2)

The targets are **OMIP-style, observation-based products only**. OMIP-2 (Tsujino et al. 2020) and
FESOM2's own evaluations (Scholz et al. 2019/2022) deliberately avoid **data-assimilating
reanalyses** (ORAS5, GLORYS, SODA, C-GLORS, CORA2): a reanalysis has already ingested a model +
DA, so calibrating *our* model to it would tune toward another model's biases, not the observations.
EN4 is acceptable — it is an **objective analysis of profiles** (optimal interpolation of in-situ
T/S), not a coupled DA reanalysis. ⚠️ A C-GLORS MLD product *is* present in the Levante cmpitool
collection (`mlotst_C-GLORSv7_*`) — **do NOT use it**; it is a reanalysis. Use **de Boyer Montégut**
for MLD instead (below).

## What is staged

| Target (pillar) | Product | Levante path | Source | Status |
|---|---|---|---|---|
| **T/S, SST** (§0, §2 eddy, drift) | **WOA18** 1° annual T/S (decav) | `obs/woa18/woa18_decav_{t,s}00_01.nc` | NCEI download | ✅ downloaded (185 MB each) |
| T/S (higher-res option) | WOA23 1° annual T/S | `obs/woa23/` | NCEI download | ⏳ `--full` only |
| T/S seasonal cycle | WOA18 monthly T/S (t01..t12) | `obs/woa18/` | NCEI download | ⏳ `--full` only |
| **MLD** (§2 mixing, §3 NN) | **de Boyer Montégut** DR003 (**0.03 kg/m³**) | `obs/mld_dbm/mld_DR003_c1m_reg2.0.nc` | IFREMER cerweb | ⚠️ IFREMER server slow — retry `stage_obs.sh` |
| **T/S interannual** (§2 eddy) | **EN4** thetao/so (10/100/1000/4000 m, seasonal) | `obs/en4_cmpitool/` → cmpitool | symlink | ✅ symlinked (32 files) |
| **Sea ice** (§0) | **OSI-SAF** siconc (seasonal) | `obs/osisaf_cmpitool/` → cmpitool | symlink | ✅ symlinked (4 files) |

Symlink source: `/work/ab0995/a270301/cmpitool/obs` (a colleague's seasonally-averaged obs
collection — EN4 T/S + OSI-SAF sea-ice already on Levante; no re-download needed).

## Grids, units, and which model statistic matches each product

The `fesom_jax.obs_compare` / `obs_ice` operators consume regular lat/lon (and polar-stereo) grids.
**Match the model statistic to the obs product** (the A2 `aggregate_windows` spec):

- **WOA18/23 annual** (`t_an`/`s_an`, `[1, 102, 180, 360]` = time,depth,lat,lon; 1°; **°C**, psu) →
  compare against the model **annual mean**. 102 standard depths 0–5500 m. `t_an` is the objectively-
  analyzed climatological mean; `t_mn` is the statistical mean of observations. Use `t_an`/`s_an`.
- **WOA18 monthly** (`t01..t12`) → month-of-climatology comparison (the seasonal-cycle option).
- **de Boyer Montégut** `mld_DR003_c1m_reg2.0.nc` — monthly MLD climatology on a 2° grid, density
  threshold **0.03 kg/m³** (matches `obs_compare.mld_density_threshold`'s `dsigma=0.03`). Compare to
  the model MLD computed by the SAME diagnostic, aggregated to the same months/seasons.
- **EN4** `thetao`/`so` — seasonal (DJF/JJA/MAM/SON), 2° (91×180), at 10/100/1000/4000 m. ⚠️ **`thetao`
  is in KELVIN** — subtract 273.15 for °C before comparing. Use for the **interannual / seasonal**
  T/S spread (§2 eddy). EN4 version is HadOBS EN4.2.x (the cmpitool climatology).
- **OSI-SAF** `siconc` — seasonal, 2° (91×180), **percent (0–100)** — divide by 100 for a 0–1
  concentration before `obs_ice.ice_conc_misfit`. (The cmpitool product is on a regular lat/lon grid,
  not the native polar-stereo; `obs_ice`'s polar projection is for the raw NSIDC/OSI-SAF grids — for
  the cmpitool regular-grid product, bin with `obs_compare.to_obs_surface` instead.)

## Citations

- **WOA18**: Locarnini et al. (2018, temperature), Zweng et al. (2018, salinity), NOAA Atlas NESDIS.
- **WOA23**: Reagan et al. (2024).
- **de Boyer Montégut** et al. (2004), *JGR Oceans* 109, C12003 — MLD climatology; DR003 = 0.03 kg/m³
  density-threshold criterion. (A 2023 Argo-updated version exists on SEANOE doi:10.17882/91774; the
  `mld_DR003_c1m_reg2.0` classic product is the canonical 0.03 climatology used here.)
- **EN4**: Good et al. (2013), *JGR Oceans* 118 — HadOBS EN4 objective analysis.
- **OSI-SAF**: EUMETSAT OSI-SAF sea-ice concentration (OSI-450 / OSI-401).
- **OMIP rationale**: Tsujino et al. (2020), *GMD* 13; Griffies et al. (2016), *GMD* 9.

## Notes / gaps

- **dBM download**: the IFREMER `cerweb` host is slow/flaky (curl timeouts). `stage_obs.sh` retries 3×
  with a 30-min cap; re-run it if `obs/mld_dbm/` is empty. Alternative: the SEANOE 2023 landing page
  (doi:10.17882/91774) for the Argo-updated climatology.
- **Native NSIDC/OSI-SAF polar-stereo** (for `obs_ice`'s projection operator) was not located as raw
  files on Levante in this pass; the cmpitool OSI-SAF product (regular lat/lon, seasonal) covers the
  §0 sea-ice diagnostic via `obs_compare.to_obs_surface`. To exercise `obs_ice`'s polar projection,
  fetch a raw OSI-450 grid (EUMETSAT) — TODO if the polar-native comparison is wanted.
- The machinery (`obs_compare`, `obs_ice`) is **unit-tested against synthetic fixtures** and does not
  depend on these downloads; staging is for the GPU experiments (§0/§2/§3).
