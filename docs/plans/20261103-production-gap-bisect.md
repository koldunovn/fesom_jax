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

`scripts/recom/fortran/make_bisect_run.py NAME --exe EXE [FACTOR ...]` builds a run directory
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
- 2026-11-05: stage 1 complete, all four jobs exit 0. `bisect_table.py` -> `fesom_jax_runs/production_gap_bisect/`.

## Stage 1 result (2026-11-05)

**B0:** the rebuilt production executable reproduces kdind15w January 1998: 133/133 output fields bit-identical.

Share of the gap, 1998 means: code = 1 - position(B1), physics = 1 - position(B3); additive if the sum is 1.

| metric | gap w - wjp | code (w -> B1) | physics (w -> B3) | sum |
|---|---|---|---|---|
| NPP, three groups | 10.35 Pg C/yr | 0.39 | 0.63 | 1.02 |
| small phytoplankton NPP | 4.63 Pg C/yr | 0.63 | 0.37 | 1.00 |
| diatom NPP | 5.53 Pg C/yr | 0.19 | 0.86 | 1.05 |
| surface Chl | 0.079 mg m-3 | 0.10 | 0.89 | 0.99 |
| CO2 flux | 0.375 Pg C/yr | 0.74 | 0.32 | 1.06 |
| surface pCO2 | -1.57 uatm | 0.87 | 0.23 | 1.10 |
| sea ice N / S | -0.19 / -0.94 Mkm2 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 |

- **Compiler flags: no effect** (B1 vs B2: NPP 0.001 Pg C/yr, CO2 flux and pCO2 about 0.003 of the gap, under the 0.03 noise).
- **Neither side dominates**, so both branches of stage 2 apply. Physics carries sea ice, diatoms and Chl; code carries most
  of the annual-mean carbon signal and small-phytoplankton NPP; total NPP is split.
- The code leaves sea ice and SST unchanged: it acts on the biogeochemistry, not on the ocean state.
- Phaeocystis (gap 0.19 Pg C/yr) and SST (gap 0.002 degC) positions are too small to read.
- **The gaps still grow in December**, so these shares describe year 1 and are not a stable split: the code effect on the CO2
  flux rises from 0.005 (January) to 0.55 Pg C/yr (December).
- **The annual CO2 split hides a seasonal sign change.** The code effect is one-signed; the physics effect is negative in
  August-October. December alone: CO2 flux 0.50 code, 0.55 physics; pCO2 0.95 code, 0.19 physics.
- By December surface DIC is 3.5 higher in B1 than in kdind15w and Alk 1.8 higher, consistent with the higher pCO2.
- Open: surface Alk falls 17-21 and DIC about 14 from January to December in kdind15w and kdind15wjp (not investigated).

## Stage 2a: the code split

`1cc2d0d3`'s own message says the old ballasting behaviour "results in large air-sea co2flux uptake", the direction of
the gap: the hypothesis to test, not a result. Variants on `e1e62886` with the production flags, production namelists,
worktrees `SPECTRAL/recom_bisect_<v>`, local branches `bisect_<v>`:

| run | variant | commit | content | answers |
|---|---|---|---|---|
| C0 `c_e1base` (1 month) | `e1base` | `e1e62886` | none | do these builds reproduce kdind15w January? (control) |
| C1 `c_ballast` | `ballast` | `046f4efd` | `1cc2d0d3` | the ballasting fix alone |
| C2 `c_ten` | `ten` | `97c4ace9` | the other ten REcoM commits | includes `9aaf96f7` (ErosionTOC2D) and the benthos deduplications |
| C3 `c_wind` | `wind` | `faa9e590` | `63283bc9` | the wind rotation alone |

Submitted 2026-11-05: C0 48831857, C1 48831858, C2 48831859, C3 48831860 (Fortran flags identical to the production
build's `flags.make`; namelists identical to B1's). C1 + C2 + C3 should add up to B1 if the commits are independent. `1cc2d0d3` conflicted on `e1e62886` in one place: it
deletes the `wFlux*` allocations and zeroing that `96137d06` added, leaving `wFluxPhy`/`wFluxDia = 0.d0` in `recom_init`,
the only uses of those arrays there; removed as the commit intends. `ten` differs from `_recom692` by exactly `1cc2d0d3`.

## The production code change of 2026-11-04 (`e4d41f52`)

`&spectral ice_light_transm` (light test L2), default 0, which keeps the old `(1 - a_ice)` expression. The user's check
137/137 fields bit-identical (`kdind15wco2sl2bit` vs `kdind15wco2s`, January 1998, job 48788274; the run output is deleted).
**No effect on this bisect:** the variants are built on `e1e62886`, which predates it, the kdind15w namelist does not set it,
and B0/B3 are frozen copies from 3 November. `make_bisect_run.py --exe prod` now takes B0's frozen copy rather than
following `recom_tra_diags_spectral`, which was rebuilt with this commit.
