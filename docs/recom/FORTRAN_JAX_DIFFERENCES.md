# fesom-jax vs the Fortran spectral fork: what differs in the CORE2 comparison runs

The CORE2 comparison (lane W of `docs/plans/20261028-recom-spectral-port.md`) sets fesom-jax against the user's
FESOM2-REcoM spectral fork runs. The two codes share the REcoM model (ported from the fork, gated to <= 2.4e-13 column
by column and <= 2.0e-5 over one full physics-matched step, `docs/recom/VALIDATION.md`), but not the ocean and sea-ice
model underneath: fesom-jax follows current upstream FESOM2, the fork sits on FESOM2 `cda35117` (2025-10-09) with
older physics options switched on. This page lists every difference known on 2026-10-30, so the comparison's
differences can be read as the sensitivity of the biology to the host physics.

## The runs

| | Fortran | fesom-jax |
|---|---|---|
| cold 1958-59 | `tmp_run_kdind15` -> `tmp_recom-fesom_output/fesom2.6_spectral_kdind15` | `configs/albedo/core2_recom_spectral_kdind15.yaml`, jobs 48621400-03, `fesom_jax_runs/core2_recom_spectral_kdind15` |
| warm 1998 (from the 1997 4-PFT restart) | `tmp_run_kdind15w` -> `fesom2.6_spectral_kdind15w` (1998-2000) | `configs/albedo/core2_recom_spectral_warm1998_noco2lim.yaml`, jobs 48626658-59, `fesom_jax_runs/core2_recom_spectral_warm1998_noco2lim` |
| code | `recom_tra_diags_spectral` (fork `e1e62886` for these outputs; the library was rebuilt at `4d96c3aa` on 10-29, after them) | branch `recom-spectral` |
| resources, CORE2 | 9 `mpp` nodes (864 ranks), ~48 min per model year | one PCIe A100, ~7.3 h per model year (2.3 s/step) |

Run directories under `/albedo/work/user/sloza/SPECTRAL/`, output under `/albedo/work/projects/p_specbic/sloza/`. The
cold and warm Fortran runs have identical `namelist.oce/tra/ice/dyn/forcing`.

## Biology: the same, except one switch

`namelist.recom` is identical apart from the input paths. The one difference is the CO2 limitation of the spectral
photosynthesis:

- The fork applies `CO2lim` only in the broadband (`#else`) branches of `recom_sms.F90` (2544, 2605, 2665, 2723).
  With `CO2lim = .true.` (both user runs) the spectral `Cphot` is still not CO2-limited.
- fesom-jax's `RecomConfig.spectral_co2lim` (default `True`, the user's stated deviation of 2026-10-28) applies it.
  The cold 1958-59 run had it on, so its comparison mixes this biology difference with the physics. The warm 1998
  run has `spectral_co2lim: false`, the fork's behaviour.

Stated fesom-jax deviations that remain (VALIDATION.md): the ballast reset reduces over the whole mesh, not per
rank; the first-call ballast reset at step 1 is skipped.

## Physics

| | Fortran (kdind15, kdind15w) | fesom-jax CORE2 configs | matchable how |
|---|---|---|---|
| vertical mixing | `mix_scheme 'KPP'` (Ricr 0.3, concv 1.6) | classical TKE (`tke: {}`, `kpp: null`) | Fortran `cvmix_TKE` + the 692f804 `namelist.cvmix` (done in the G4b twins); or fesom-jax `kpp: {}` (untested with REcoM) |
| Monin-Obukhov mixing | `use_momix .true.`, south of -50, 0.01 | none | Fortran `use_momix .false.` |
| instability mixing, Kv0 | `use_instabmix .true.` 0.1, `Kv0_const`, `K_ver 1e-5` | the same | already equal |
| GM | `Fer_GM .true.`, K_GM 2-2000, `K_GM_bvref 2`, `scaling_resolution` | current FESOM2 GM (ODM95 taper, depth scaling) | not matchable: the fork's `init_Redi_GM` is the FESOM 1.4-style builder. Off on both sides, or rebase the fork |
| Redi | `.true.`, `K_hor 3000` | current FESOM2 Redi with tapering | as GM |
| horizontal viscosity | `opt_visc 5` (easy backscatter, `visc_easybsreturn 1.5`) | `opt_visc 7` (biharmonic) only | Fortran `opt_visc 7` |
| sea-ice rheology | standard EVP (`whichEVP 0`, 120 subcycles) | mEVP (`whichEVP: 1`, alpha = beta = 250) | Fortran `whichEVP 1`; fesom-jax has standard EVP too (`ice: {whichEVP: 0}`) |
| ice diffusion | `ice_diff 0.0` | 10 (CORE2) | Fortran `ice_diff 10.0` |
| sea-ice code | the fork's 2025-10 ice | current FESOM2 ice | not by namelist; it sets the residual floor of the one-day pair (a_ice 3.2e-4) |
| partial cells | `use_partial_cell .true.` | none (layer thicknesses from `hbar`) | Fortran `.false.` |
| tracer advection weights | `tra_adv_ph/pv 1., 1.` | `0., 1.` only | Fortran `0., 1.` |
| FCT limiter bounds | without the thickness (before upstream 18346243 / 9afdae7b) | with `/hnode` | fesom-jax `legacy_fct_bounds` (a step kwarg, not yet a YAML key) |
| Redi K33 weights | before upstream #944 | current | fesom-jax `legacy_k33_weights` (step kwarg) |
| shortwave attenuation depth | absolute (before upstream #1116) | relative | fesom-jax `legacy_sw_depth` (step kwarg) |
| first forcing window winds | left unrotated on the rotated grid (fixed upstream in 66658357) | rotated | the fork with 66658357 cherry-picked (the G4b twins' executable); one forcing window only |

## Forcing and setup

Equal on both sides: JRA55-do repeat-year `clim61` (`y_perpetual`), CORE2 runoff, PHC2 SSS restoring (`surf_relax_S
1.929e-6`), constant chlorophyll 0.1 for the shortwave penetration, the Albani dust climatology, OASIM, `mslp`,
`dt` 2700 s, zstar, the CORE2 mesh. The warm run starts both sides from the same 1997 restart
(`scripts/recom/fortran_restart_to_jax.py`; the conversion rebuilds the layer thicknesses because fesom-jax has no
partial cells, and fills 2 nodes below the Fortran bottom).

## Reading the comparison

- With `spectral_co2lim: false` (the warm 1998 run) the remaining differences are the physics table above, so the
  Fortran - fesom-jax difference is the response of the biology to that set of physics changes taken together. It
  does not attribute the response to one of them.
- The one-day pair with the physics matched by namelist (G4b: TKE, mEVP, `ice_diff 10`, no momix, GM/Redi off, the
  three legacy flags, 66658357) agrees to 5.4e-5 in T and every BGC tracer to 2.0e-5. A CORE2 run of that pair
  would isolate what is left (the fork's ice code, chaos) and leave GM/Redi out of the comparison.
- Even with matched physics, two runs drift apart node by node within months (chaos). Compare multi-year runs by
  global NPP, CO2 flux, inventories, seasonal cycles and patterns, not pointwise.

## The cold 1958-59 run (co2lim on, all physics differences), for the record

`compare_fortran_jax.py`, `fesom_jax_runs/core2_recom_spectral_kdind15/compare`: pattern correlations 0.84-0.99;
NPP matches in Jan-Feb 1958 (77 vs 75, 155 vs 153 Pg C/yr) and is ~17 % low from mid-1958 (~98 vs ~118), the deficit
in the low-latitude gyres and the equatorial Atlantic and Indian Oceans, the Southern Ocean slightly higher; surface
Chl ~25 % low; CO2 uptake ~1-1.5 Pg C/yr weaker (pCO2 +5-8 uatm); SST within 0.07 degC; sea-ice area +1-2.5 Mkm2
from the first month; DIN, DSi, DIC, Alk inventories within ~1e-4, DFe higher in fesom-jax (0.802 vs 0.794 Pmol).
