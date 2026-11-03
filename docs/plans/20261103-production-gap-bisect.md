# The production-gap bisect: what makes kdind15w differ from the matched Fortran run

Written 2026-11-03 on Albedo. Option A of the closing note of `completed/20261028-recom-spectral-port.md`.

## Goal

The user's production warm-1998 run **kdind15w** and the physics-matched Fortran run **kdind15wjp** start from the
same 1997 4-PFT restart with the same forcing and `namelist.recom`, yet differ strongly over 1998
(`FORTRAN_JAX_DIFFERENCES.md`, "Result"; 1998 means from `scripts/recom/fortran/bisect_table.py`):

| 1998 mean | kdind15w (production) | kdind15wjp (matched) |
|---|---|---|
| NPP, small phytoplankton + diatoms + Phaeocystis [Pg C/yr] | 121.0 | 110.6 |
| diatom NPP [Pg C/yr] | 21.4 | 15.8 |
| surface Chl [mg m-3] | 0.362 | 0.283 |
| CO2 flux into the ocean [Pg C/yr] | 0.33 | -0.05 |
| sea-ice area north / south [Mkm2] | 11.92 / 8.75 | 12.11 / 9.69 |

fesom-jax reproduces kdind15wjp, so the gap is entirely on the Fortran side. Find which differences cause it, with
Fortran runs only, so the user knows how much of their production results depends on each.

## What differs (inventory 2026-11-03)

**Executable.**
- kdind15w output (28 Oct) was made by `recom_tra_diags_spectral` at `e1e62886`; that build (`tmp_run_kdind15w/fesom.x`
  and `libfesom.so` are links) has since been rebuilt at `21e0d02a` (2 Nov). `4d96c3aa` (linear mortality, `mort_phy
  = 0` by default) and `21e0d02a` (`RECOM_CO2LIM_SPECTRAL`, default `.false.`) should be inert: run B0 checks.
- kdind15wjp ran `recom_spectral_r692` `build_rel` = `e1e62886` + the 11 upstream REcoM commits (`1cc2d0d3` ballasting
  fix, the benthos deduplications `fd834980`/`663b467c`, allocation and NaN fixes, `9aaf96f7`) + the fesom-jax dump hooks
  (inert unless `FESOM_RECOM_DUMP_DIR`) + `d9e613f0` (1-rank I/O, inert at 864 ranks) + `63283bc9` (= upstream
  66658357: the first forcing window's winds rotated).
- **Compiler flags.** CMake "Debug" here is not unoptimised: with `FESOM_PLATFORM_STRATEGY=albedo` (the production
  build, set by `env.sh`) the flags are `-g ... -fp-model precise ... -march=core-avx2 -O3 -qopt-prefetch=5
  -unroll-aggressive`; the matched `build_rel` was configured without the strategy: `-O3 ... -fp-model precise ...
  -xHost`. `env/albedo/shell` alone does not set the strategy; pass `-DFESOM_PLATFORM_STRATEGY=albedo`.
- New build for the bisect: `recom_spectral_r692/build_alb` -> `install_alb/` (the r692 code at `63283bc9` with the
  production build's exact Fortran flags, checked against its `flags.make`).

**Physics namelists** (everything else identical, checked by `make_bisect_run.py --all-matched`):

| factor | production | matched |
|---|---|---|
| `gmredi` | `Fer_GM`, `Redi` `.true.`, `K_hor 3000.` | `.false.`, `0.` |
| `mix` | `mix_scheme 'KPP'` | `'cvmix_TKE'` + `namelist.cvmix` |
| `momix` | `use_momix .true.` | `.false.` |
| `adv` | tracer advection weights `1., 1.` | `0., 1.` |
| `evp` | `whichEVP 0` (EVP) | `1` (mEVP) |
| `icediff` | `ice_diff 0.0` | `10.0` |
| `visc` | `opt_visc 5` | `7` |
| `pcell` | `use_partial_cell .true.` | `.false.` |

## Method

`scripts/recom/fortran/make_bisect_run.py NAME --exe {prod,r692alb,r692rel} [FACTOR ...]` builds a run directory
under `/albedo/work/user/sloza/SPECTRAL/tmp_run_bisect/NAME` from the production 1998 namelists (config from
`namelist_config_backup_20261029_yr1998-99.txt`, 12 months) with the named factors at the matched values, frozen
copies of the executable, and `job_bisect` (9 `mpp` nodes x 96, as the production run); output
`tmp_recom-fesom_output/fesom2.6_bisect_NAME`. Checked: no factor reproduces the production namelists, all eight the
matched ones (comments, `ResultPath`, `run_length` aside). `scripts/recom/fortran/bisect_table.py OUT` tabulates every
run's 1998 means and its **position** per metric, (run - wjp) / (w - wjp): 1 = production, 0 = matched.

Noise: one year from the same restart, the global means of chaos-perturbed twins differ by ~0.1 % (the 2.6 and 2.8
fesom-jax pairs: three-group NPP <= 0.02 %); the gap is ~10 %. A factor that moves a metric by < ~0.03 of the gap is
indistinguishable from zero.

### Stage 1: code vs physics (4 runs)

| run | exe | namelists | answers |
|---|---|---|---|
| B0 `prod_repro` (1 month) | prod (`21e0d02a` now) | production | reproduces kdind15w January? (the rebuild is inert) |
| B1 `r692alb_prod` | r692alb | production | kdind15w -> B1: the REcoM commits + `63283bc9`, flags fixed |
| B2 `r692rel_prod` | r692rel | production | B1 -> B2: the compiler flags; B2 -> kdind15wjp: the physics, on the matched code |
| B3 `prod_matched` | prod | all eight matched | kdind15w -> B3: the physics on the production code; B3 -> kdind15wjp: the code. With B1/B2: additivity |

### Stage 2 (decided after stage 1)

- If the code dominates: split `1cc2d0d3` (the ballasting fix; the user runs `use_ballasting = .true.`) from the other
  ten REcoM commits and from `63283bc9` (cherry-pick onto `e1e62886` with the production flags, one at a time).
- If the physics dominates: one factor at a time from production toward matched (8 runs, on the code that stage 1
  shows matters less), then pairs where the singles do not add up (e.g. `gmredi` with `mix`; `evp` with `icediff`).
- Southern sea ice: the gap is mostly Antarctic (8.75 vs 9.69 Mkm2), which points at `evp`/`icediff`/`gmredi`/`mix`
  more than at the biology; check the per-hemisphere positions.

## Progress log

- 2026-11-03: inventory, `make_bisect_run.py`, `bisect_table.py` (with `compare_fortran_jax.Fortran.field` accepting
  runs shorter than 12 months). Submitted B0 48752912, B2 48752913, B3 48752914; B1 48752919 after `build_alb` (Fortran flags identical to the
  production build's `flags.make`).
