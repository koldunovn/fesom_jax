# REcoM spectral light + 4-PFT ecosystem in fesom-jax

Branch `recom-spectral` off `main` (bdeb99e, "REcoM-2 biogeochemistry"). Written 2026-10-28 on Albedo (AWI), from
a read-only inventory of the Fortran fork; nothing below is ported yet.

## Goal

Run, in fesom-jax, the REcoM configuration of Svetlana Losa's FESOM2-REcoM fork
(`sveta-loza/fesom2`, branch `fesom2.6_recom_tra_diags_spectral`, checkout
`/albedo/work/user/sloza/SPECTRAL/recom_tra_diags_spectral`, HEAD e1e62886): the `bin_4p3z2d` ecosystem (4
phytoplankton groups: small phytoplankton, diatoms, coccolithophores, Phaeocystis; 3 zooplankton; 2 detritus
classes; CDOM; 37 BGC tracers) under spectrally resolved underwater light (13 wavebands, 400-700 nm, Darwin
"wavebands" heritage; OASIM surface spectral irradiance), reproducing the Fortran run and staying differentiable.

The reference Fortran run is `tmp_run_kdind15/` (CORE2, clim61 repeat-year forcing, 1958 cold start, OASIM v2
`darwin400`, optics `optics_phyto_recom_carbon_12_4pft_REFINED_v5.dat`, `RECOM_CALC_ACDOM/APART`, `RECOM_BMASS`).

## What has to be ported (inventory, from the fork's diff `cda35117..e1e62886`: 84 commits, +5096/-138 lines)

The JAX port (`fesom_jax/recom/`) is REcoM 692f804 in its `bin_2p3z2d` configuration: 2 phytoplankton, 3
zooplankton, 2 detritus, 31 tracers, broadband PAR (Beer-Lambert, `k_w + a_chl*Chl`), mocsy carbonate chemistry.
`fesom_jax/recom/params.py` `SUPPORTED` refuses `enable_coccos`. Three blocks of new physics sit on top:

### A. The 4-PFT ecosystem (already in the fork's BASELINE branch, not spectral-specific)

- Tracers 1029-1031 CoccoN/C/Chl, 1032-1034 PhaeoN/C/Chl (`enable_coccos`), `bgc_num` 36; the JAX slot table
  (`tracers.py` `BGC_IDS`) and `State.bgc` grow accordingly. ID layout: `recom_modules.F90:759-884`.
- Coccolithophore and Phaeocystis growth, quotas (`NCmax_p` 0.2), temperature functions (Blanchard, NaN guard
  `recom_sms.F90:1290`), grazing by the 3 zooplankton (`pzPhaeo`, `pzCocco`...), calcification by coccos (replaces
  the 2-PFT "calcifying fraction of small phytoplankton"), aggregation (`agg_fac_phaeo`, `recom_sms.F90:4254-4320`),
  Phaeocystis linear mortality (`:6212-6223`), iron-dependent Phaeocystis sinking incl. its benthic flux
  (`recom_sinking.F90:96-135, 593-648`), per-PFT seed quotas in `recom_init.F90`.
- Smaller biology switches (defaults bit-identical unless noted): Platt photoinhibition `beta_inh*` (both
  builds), `res_phy_const`, pseudo N fixation (`nfix_*`, `recom_sms.F90:2974-2995`), overflow DOC/respiration
  `f_ovf, ovf_to_cdom, res_ovf*` (`:6545-6585`), `k_din_calc`, `mort_phaeo`, `par_layer_mean` (default `.true.`,
  CHANGES the broadband light: exact layer mean instead of FESOM's layer top, `:1465-1484`).

### B. CDOM (spectral branch, `RECOM_CDOM`)

- Tracer 1037 CDOM (37 tracers with 3zoo2det + coccos). Source `fcdom`(0.02) x the DOC sources, DOC gets
  `(1-fcdom)`; sinks `rho_cdom*arrFunc` + photolysis `phot_cdom*min(PARave/kphot_cdom, 1)`
  (`recom_sms.F90:551, 5802-5807, 6485-6542`; constants `recom_modules.F90:2677-2680`). Feeds the CDOM absorption
  when `RECOM_CALC_ACDOM`.

### C. The spectral light model (`REcoM_spectral`, `recom_modules.F90:2356-4543`, ~2200 lines, all new)

1. Bands: `tlam = 13`, centres 400..700 by 25 nm, edges halfway (end bands 12.5 nm), `WtouEins` per band
   (`:2813-2910, 2852-2861`).
2. Surface irradiance: OASIM monthly climatology `Ed_<nnn>_*.nc`, `Es_<nnn>_*.nc` (1-degree grid, 12 records,
   W m-2 per band, `/albedo/work/projects/p_specbic/sloza/forcing_oasim_v2/`), horizontally interpolated to nodes,
   monthly HOLD (no time interpolation; the read triggers at month end, `gen_surface_forcing.F90:1800-1822`), NaN
   gap fill (`gen_modules_read_NetCDF.F90:117-146`), sea-ice shading `(1 - a_ice)` on the call
   (`recom_main.F90:429-434`). Without OASIM: `Edwsf = wb_width*sf*SurfSW` from `surfspec_13amt6.dat`, `Es = tiny`.
3. Inherent optical properties per level and band (`recom_forcing.F90:306-460, 596-667`): water (`abw25par.dat`,
   `bbw = 0.5 bw`); phytoplankton absorption `sum_PFT Chl_X * aphy_chl_X(lambda)` from `optics_phyto_*.dat`
   (4 blocks x 13 lines: lambda, ap, ap_ps, b, bb; columns b/bb per mmol C with `RECOM_BMASS`); CDOM
   (`aCDOM13amtVK2006.dat`, or `cdomcoeff*CDOM*exp(-0.021(lambda-450))` with `RECOM_CALC_ACDOM`); detritus
   (`optics_detritus_3bb.dat` per particle / `1.06e-13`, or the `RECOM_CALC_APART` formulas); optional background
   backscatter; `bb >= 2e-4`.
4. Vertical integration: default `RECOM_RADTRANS` = the 3-stream (Ed, Es, Eu) direct solver
   `MONOD_RADTRANS_DIRECT` (`:4069-4333`): Ed integrated exactly, Es/Eu amplitudes from a tridiagonal system
   (`SOLVE_TRIDIAGONAL_PIVOT`, `:4338-4419`), `rmud = min(1/cosAI, 1.5)`, `rmus = 1/0.83`, `rmuu = 2.5`, skipped where
   `Ed, Es < 1e-4`; scalar irradiance per band `sqrt(Etop*Ebot)` of `(rmud Ed + rmus Es + rmuu Eu)*WtouEins`
   (`:4298-4327`). Alternative `RECOM_RADTRANS=.false.`: per-band Beer-Lambert (`recom_forcing.F90:497-590`).
5. Growth: `alpha_I_X = 86400 * sum_lambda alphachl_nl_X(lambda) * PARw(lambda, k)`, `alphachl_nl_X = mQY_X *
   ap_ps_X(lambda)`, `mQY_X = sum(QYmax_X/13/WtouEins)` (`WAVEBANDS_INIT_VARI`, `:3607-3638`); `Cphot_X = pMax_X (1 -
   exp(-alpha_I_X * Chl2C_X / pMax_X))` (`recom_sms.F90:2501-2708`); Chl degradation and synthesis use `alpha_I`
   (`:2794-2890, 3079-3133`); `PARave = sum PARw / WtouEins` for photodamage, CDOM photolysis, `beta_inh`. The
   broadband `alfa*` are dead in a spectral build.
6. Diagnostics: `Edz/Esz/Euz/Eutop/Estop` x 13 bands (`io_meandata.F90`, `max_io_streams` 400).
7. Optional, NOT in the reference run: `RECOM_MARSHALL` (D1 photoinhibition tracers 1038-1041, `RECOM_APHYTO`),
   `DAR_NONSPECTRAL_BACKSCATTERING_RATIO`, `RECOM_CALC_REFLEC`, `MONOD_RADTRANS_ITER` (niter >= 0).

Two fork quirks to decide with the user before porting (they are not upstream REcoM):
- CO2 limitation (`CO2lim`) is applied only in the non-spectral `#else` branches (`recom_sms.F90:2544, 2605,
  2665, 2723`): the spectral `Cphot` never sees `PhyCO2` even with `CO2lim=.true.`.
- The N:C quota is clamped to `[NCmin_X, NCmax_X]` in the spectral build only (`:845-848, 903-906, 973-976,
  1015-1018`).
Decisions (user, 2026-10-28): mirror the N:C clamp; apply CO2 limitation in the spectral growth too (a stated
deviation from the fork, documented in the port and in the fork's issue list). The fork is first brought up to the
submodule REcoM by cherry-picking the 11 upstream commits (branch `fesom2.6_recom_tra_diags_spectral_recom692`,
worktree `/albedo/work/user/sloza/SPECTRAL/recom_spectral_r692`). The pi REcoM inputs come from upstream FESOM2's
committed `test/input/recom` (extracted to `fesom_jax_data/recom_pi/`).

### Version mismatch

The fork's REcoM is in-tree (`src/int_recom`), based on `cda35117` (2025-10-09), i.e. OLDER than the 692f804
submodule the JAX port tracks: 11 upstream commits (ballasting bug fixes, `ErosionTOC2D` NaN fix; ~7 files,
+264/-307, mainly `recom_sinking`/`recom_sms`) are in JAX but not in the fork. The Fortran twin for the gates
must therefore be the FORK built on Albedo, and the comparison tolerances must account for those 11 commits (or
the fork is first rebased onto the submodule REcoM, a Fortran task outside this plan).

## Lanes (each a branch off `recom-spectral`, each gated before the next starts)

- [ ] **Lane 0, Fortran reference on Albedo.** Build the fork with REcoM dumps (the env-gated dump hooks of
      `docs/recom/FORTRAN_REFERENCE.md`, ported to `src/int_recom` of the fork; `recom_in`/`recom_out` stages plus
      the new light arrays `PARw`, `Edz..Estop`, `oasim_ed2D/es2D`, `alpha_I_X`) on the pi mesh with
      `bin_4p3z2d` + spectral, 30 days, one rank, and the same run with `RECOM_WAVEBANDS=OFF` (the broadband
      4-PFT twin). Needs the pi REcoM inputs (`din5deg.nc` etc., FESOM2 `test/input/recom`, not on Albedo: fetch
      from the FESOM2 test-data server) and an OASIM climatology interpolated to pi (the reader does that).
      Partition: `mpp`, account `clidyn.clidyn`.
- [ ] **Lane P, parameters and tracers (A, B).** `params.py`: every `&spectral`, `&nam_rspecsbc` and new biology
      key with the fork's defaults; `SUPPORTED` gains `enable_coccos`, `RECOM_CDOM`, `RECOM_RADTRANS`, `OASIM`,
      `RECOM_BMASS`, `RECOM_CALC_ACDOM/APART`, refuses `RECOM_MARSHALL`, `RECOM_CALC_APHYT/REFLEC`,
      `DAR_NONSPECTRAL_BACKSCATTERING_RATIO`, niter >= 0. `tracers.py`: the 37-ID layout as a function of the
      switches (the fork's `initialize_tracer_indices`), IC file map, sinking/remin/bc_surface sets per layout.
      `State.bgc` sized by `NTR` from the config, restart reading tolerant of both layouts. Test: slot tables
      against `recom_modules.F90:759-884` for the 4 layouts.
- [ ] **Lane E, the 4-PFT sources and sinks (A, B).** `sms.py`: cocco/phaeo growth, quotas, grazing matrix,
      calcification, aggregation, mortality, CDOM, the new switches; `sinking.py`: Fe-dependent Phaeocystis sinking
      + benthic flux. Gate G2 on the broadband 4-PFT dumps (lane 0, `RECOM_WAVEBANDS=OFF`): updated tracers and
      diagnostics rel 1e-11. This lane is independent of the light model and is where most of the biology risk is.
- [x] **Lane L, the spectral light (C.1-C.4).** New `fesom_jax/recom/spectral.py`: band tables, the `.dat` readers
      (host, numpy), IOPs as array ops over `(nod2D, nl, 13)`, the 3-stream direct solver as a vectorised
      tridiagonal (`ops.tdma` over the column, batched over nodes and bands; `lax.scan`, no Python loops), the
      Beer-Lambert alternative, `PARw(nod2D, nl, 13)` and `PARave`. Gate: `PARw`/`Edz..Estop` against the lane-0
      spectral dumps at the probe columns (rel 1e-10; the solver is a direct method so this should be tight).
      Differentiability check: `jax.grad` of the column PAR w.r.t. Chl finite and matching FD.
- [x] **Lane F, OASIM forcing.** `fesom_jax/recom/oasim.py` + `surface_forcing.BgcForcing`: 26 monthly fields
      interpolated to nodes once on the host (the JRA bilinear machinery of `jra55.py`/`read_other_NetCDF` port),
      held per month in the per-step input tables like the dust (`didx` row), `(1 - a_ice)` shading in the step,
      NaN fill. Run-YAML keys `forcing.oasim_dir`, `forcing.oasim_pattern`.
- [x] **Lane W, wiring.** `stage1.py`/`wiring.py`: light before growth, `alpha_I` into the growth terms, the
      no-OASIM `surfspec` path, outputs (`edz3d` etc. in the mean streams, `recom.output`), budget terms for CDOM;
      configs `configs/pi_recom_spectral.yaml`, `configs/albedo/core2_recom_spectral.yaml`. Gates G4b/G5 on pi
      against the lane-0 spectral twin (one step, 30 days: inventories, CO2 flux, NPP per PFT); sharded = dense;
      gradient of a one-day NPP cost vs FD. Then the CORE2 run of `tmp_run_kdind15` on one Albedo GPU and the
      `scripts/recom/page` figures against it.

Rough size: lanes P+E are of the order of the original `recom-lane-sources` work (the sms grows by ~1000 Fortran
lines); lane L is ~2200 Fortran lines but structurally simple (columns x bands); lane 0 is a Fortran build and
dump exercise. Expect several sessions per lane; do not start lane E or L without lane 0's dumps.

## Progress log

- 2026-10-28, lane 0 setup: fork branch `fesom2.6_recom_tra_diags_spectral_recom692` = e1e62886 + the 11 upstream
  REcoM commits (cherry-picked; conflicts were the fork's `use REcoM_spectral` lines, its phaeo-ID keying and
  Fe-dependent Phaeocystis sinking, all resolved for the fork; the ballasting fix's detritus-only guard landed).
  Branch `..._recom692_dumps` on top: the fesom-jax dump hooks placed by hand (the patches assume FESOM2 e96a2b13;
  hunks applied with fuzz landed in wrong places), `recom_out` extended with the per-PFT diagnostics and the
  spectral planes `Edz3D..Estop3D`, `Ed0/Es0/Eu0_2D`, `oasim_ed2D/es2D`; `fesom_ini` accepts `n_part=1`
  (`fvom_init.F90`: a trivial partition, METIS skipped). Builds: `build/` (spectral ON) -> `bin/fesom.x`;
  `build_nowb/` (OFF) -> `install_nowb/bin/fesom.x`. Both are `CMAKE_BUILD_TYPE=Debug` as the user's runs.
  Worktree `/albedo/work/user/sloza/SPECTRAL/recom_spectral_r692`.
- pi inputs: `fesom_jax_data/recom_pi/` (upstream FESOM2's committed `test/input/recom`), `mesh_pi_part/` (the
  fork's `test/meshes/pi` + `dist_1`), `recom_pi/optics/` (the five `.dat` tables of `tmp_run_kdind15`). The
  fork's pi mesh equals the packaged pi to round-off (coordinates 9e-16 rad; `fesom_jax_data/mesh_pi_fork` is its
  prepared bundle, to be used by the JAX twin runs).
- Namelist set `scripts/recom/fortran/namelists_albedo_spectral/` = `tmp_run_kdind15` with the pi inputs, JRA55-do
  1958 (+psl), no partial cells, leap years, `opt_visc 7` (the one physics change: fesom-jax has no opt_visc 5),
  everything else the user's (KPP, EVP, GM 2000/bvref 2/order 1). The fork (FESOM 2.6) lacks the GM/ice/forcing
  keys of the JAX-physics set, so the physics floor of the connected gates will be measured by a physics-only
  pair, not matched by namelist. Job `scripts/albedo/run_pi_recom_fork.sbatch` (1 rank, `mpp`).
- Lane P done (commit 41013f0): `fesom_jax/recom/layout.py`, the fork keys in `params.py`, `RecomConfig.layout`.
  Finding: the fork has NO DICremin; its ID 1037 is CDOM (692f804's 1037 is DICremin), and its 2p3z2d is 30
  tracers (+1 CDOM = 1031). Lane E must thread `Layout` through `sms.py`, `sinking.py`, `transport.py`
  (`surface_bc`, `finish`), `wiring.py` (budget shape, output fields, `POC_IDS`) and `ic.py`, replacing the
  `tracers.py` constants (kept as the 692f804 default).

- 2026-10-28 evening, lane 0 debugging: the fork on ONE rank never reaches its first time step: the first forcing
  read (`getcoeffld`, `next_io_rank`) calls `next_io_rank_helper`, which recurses forever when rank 0 is the only
  rank (`io_netcdf_workaround_module.F90`, its own `todo` says so). Symptom: 99 % CPU, no I/O, the log ends after
  the nine `found calendar attr` lines; jobs 48596037 (Debug, 30 min), 48596399 (Release, 2 h), 48602196 all
  timed out there. Fixed on the dumps branch (d9e613f0: `partit%npes == 1` -> serial I/O on rank 0). Not the
  build type and not the spectral code. Along the way: `FORT_BUFFERED=false` in the job (unbuffered log),
  Release builds `install_rel/` and `install_nowb_rel/` (the Debug ones stay for the user's own runs), a `LAUNCH`
  override in the job script, and the non-spectral twin's namelist set `namelists_albedo_spectral_nowb/` (36
  tracers: a WAVEBANDS=OFF build has no CDOM, `validate_recom_tracers` stops otherwise; job 48602195).

- 2026-10-28 night, lane E part 1 DONE (commit on `recom-spectral`): `fesom_jax/recom/sms_fork.py` +
  `stage1.recom_sources_sinks_fork`; gate `test_recom_sms_fork.py` against the non-spectral twin `pi_30d_nowb`
  (job 48602409) <= 2.4e-13 at steps 1, 2, 3, 17 for every tracer, diagnostic, flux and cache field. Finding: the
  fork's `tiny_Si = tiny_C_d / SiCmax` (692f804 multiplies). The 30-day runs (48602409 broadband, 48602410
  spectral) write dumps at 224 and 960 too. Part 2 = the fork's sinking (VCocco/VPhaeo, the Fe-dependent
  Phaeocystis speed and benthic flux, PhyCalc with VCocco in the column but VPhy in the benthos) and stage 3.

- 2026-10-28 late, lane E part 2 DONE (commit 0f969cd + the stage-3 fix-up): the fork's sinking/benthos in
  `sinking.py` (layout-aware), `transport.recom_stage3(tracer_layout=...)`; gate `test_recom_sinking_fork.py`
  <= 4.6e-15. The stage-3 gate `test_recom_stage3_fork.py` first missed by 1e-3..5e-2 rel; three causes, all in
  the fork's FESOM2 base (merge base with upstream cda35117, 2025-10-09), none in REcoM:
  1. `namelist.tra` weights: the fork's run dirs use `tra_adv_ph/pv = 1., 1.`; the JAX kernels implement
     `0., 1.` only (`TracerConfig`). Both namelist sets now say `0., 1.` and the twins were rerun as
     `pi_30d_nowb_adv01` (job 48602452) and `pi_30d_spectral_adv01` (48602453); the `1., 1.` runs
     (48602409/48602410) stay valid for the column-local gates (G2, G3, the light solver) only.
  2. The FCT limiter bounds: the fork has `flux = fct_plus*dt/areasvol +- eps` without the thickness
     (FESOM2 added `/hnode` in 18346243, 2025-10-29, and `/hnode_new` in 9afdae7b, 2026-03-03): 3.6e-4 rel on T.
     `tracer_adv.zalesak_limit(legacy_bounds=True)`, threaded as `legacy_fct_bounds` through `advect_one_fct`,
     `recom.transport`, `recom.wiring.full_bgc_step` and `step()` next to `legacy_k33_weights` (also needed:
     the fork predates #944) — default off, `test_dp_lock`/`test_run_config` unchanged.
  3. The level structure: the fork's run read `mesh_pi_part/elvls.out` written by its own `fesom_ini`, which
     differs from the repo's `test/meshes/pi/elvls.out` (= the packaged pi) at elements 0, 1, 22 (22/20/22 vs
     19/18/19, node 11's `nlevels_nod2D` 22 vs 19): 2.6e-5 rel on T at nodes 0-11. `fesom_jax_data/mesh_pi_fork`
     is now `prepare_mesh.py` of `mesh_pi_part` (the old bundle kept as `mesh_pi_fork_repo_levels`).
  With the three: G4a on `pi_30d_nowb_adv01` <= 3.2e-14 at steps 1, 2, 3, 17 (`FESOM_RECOM_FORK_RUN` selects
  the run; all three fork gates default to the adv01 run). Steps 224/960 once the reruns write them.

- 2026-10-28 night, lane L DONE: `fesom_jax/recom/spectral.py` (band tables, the `.dat` readers, the IOPs, the
  direct 3-stream solver with the pivoting tridiagonal solve as two `lax.scan` sweeps over padded rows, the scalar
  irradiance, `sms_light`) and the spectral growth terms in `sms_fork.py` (`spectral=`, `spectral_co2lim=`; the
  N:C clamp `nc_clamp`). Gates on the spectral twin `pi_30d_spectral_adv01` (job 48602453): `test_recom_spectral_light`
  (Edz/Esz/Euz/Eutop/Estop x 13 bands, PAR3D) <= 1.5e-12 and `test_recom_sms_fork` with the spectral light (37
  tracers incl. CDOM, `spectral_co2lim=False`, TCphotLigLim_* skipped: the fork leaves them unset) <= 1.3e-13 at steps
  1, 2, 3, 17, 224. The three fork gates select the namelist set and the light by the run name (`FESOM_RECOM_FORK_RUN`);
  on the spectral twin sinking and stage 3 pass too (12 + 12). Step 224 exposed REcoM's whole-array ballast reset
  (`any(scaling_density <= tiny) -> 1` on the rank's array, recom_sinking.F90 K:858-861, upstream d5f9f8ac): mirrored
  for the fork layout in `sinking.ballast` (a per-device `any` over the wet layers; 692f804's programs unchanged).
  Open: steps 960 once the reruns finish; the first-call reset at step 1 stays a documented skip.

- 2026-10-28 late night, lanes F and W wired: `fesom_jax/recom/oasim.py` (the 12 x 13 OASIM fields through the
  `read_other_NetCDF` port of `sss_runoff`, gate: January vs `recom_out.oasim_ed2D/es2D` of the spectral twin
  <= 1e-12), held with the Albani dust in `BgcForcing` (`StepForcing.oasim_ed/es`, `ForcingTables.oasim_*`, the
  on-device combine, the local-forcing builder). `RecomConfig.spectral / optics_dir / spectral_co2lim`,
  `RunConfig.bgc_oasim()`, `ic_set: pi_present`; `wiring.full_bgc_step` is layout-aware and takes the fork's stage 1
  (`recom_sources_sinks_fork`) with the spectral light of the step, `nlmin` in `ForcingStatic.bgc_nlmin`, the 4-PFT
  NPP/Chl/export outputs; `check_supported` admits the fork's layouts. The cold start reproduces the twin's step-1
  tracers (`test_recom_spectral_wiring`): the fork's `recom_init` constants (`tracers.init_value(layout=)`: DiaN
  with `Chl2N_max_d`, 1023-1028 left at 0 when 3zoo2det and coccos are both on, CDOM `tiny`) and the fork's
  `DIC_PI` file override (`oce_setup_step.F90:1012`, removed upstream by e96a2b13: namelist.tra's `tco2_5deg.nc`
  is never read when `DIC_PI` is on). Config `configs/albedo/pi_recom_spectral.yaml`, job
  `scripts/albedo/run_pi_recom_spectral.sbatch` (first submission: job 48602583, 30 days of pi on one CPU node).

- 2026-10-28 night, the first JAX spectral run: job 48602595 (30 days of pi, 20 min) ended non-finite in the
  physics too; three 32-step smoke runs isolated it to the mesh bundle + IC pair, not the spectral code (the
  spectral config on the packaged pi mesh is finite, 48603144; the 692f804 config on the fork bundle is not,
  48603145): the packaged pi IC has T = S = 0 in the layers the fork's `elvls.out` adds at nodes 0/1/9/10/11.
  `fesom_jax_data/ic_pi_fork` (PHC on the fork bundle) fixes the IC; resubmitted as job 48603201.

- 2026-10-28 22:21, THE SPECTRAL REcoM RUNS IN FESOM-JAX: job 48603201, 30 days of pi on one CPU node, 960 steps,
  `FINITE n_nonfinite=0`, 1.28 s/step (the 692f804 configuration: 0.9), monthly output and a restart under
  `fesom_jax_runs/pi_recom_spectral/`. Lane W's run step is done; a CORE2 spectral smoke on one GPU
  (`configs/albedo/core2_recom_spectral.yaml`, 96 steps) submitted next.
- 2026-10-28 22:29, CORE2 + the spectral fork on one A100: job 48603697, 96 steps, `FINITE`, 2.50 s/step (the 692f804
  configuration: 2.63), from `configs/albedo/core2_recom_spectral.yaml` through `scripts/albedo/run_core2_gpu.sbatch`.
- 2026-10-28 22:45, the reruns finished (48602452 2h05, 48602453 2h15): every fork gate also passes at step 960 on
  both twins (broadband: sms, sinking, stage 3; spectral: those plus the light planes). The gate table is in
  `docs/recom/VALIDATION.md`.

- 2026-10-29, the fork's G4b: `fesom_jax/tests/test_recom_step_fork.py` on one-day physics-matched twins
  (`pi_1d_spectral_jaxphys` 48606105, `pi_1d_nowb_jaxphys` 48606106; namelist sets `namelists_albedo_spectral{,_nowb}_jaxphys/`
  with GM/Redi off on both sides, cvmix TKE + the 692f804 `namelist.cvmix`, mEVP, `ice_diff` 10; JAX
  `configs/albedo/pi_recom_spectral_g4b.yaml` + the three legacy step kwargs; the job scripts
  `scripts/albedo/run_pi_recom_fork.sbatch` now copies `namelist.cvmix`, `scripts/albedo/run_pytest.sbatch` runs a
  test file on a compute node). First result: T floor 3.5e-3 (jobs 48605353/48605354, kept as `_nofix`);
  `physics_floor.py --mesh --legacy --no-recom` showed the twin's winds 70 % off at step 1: the fork's base lacks
  upstream 66658357 (the first forcing window's wind coefficients never rotated on a rotated grid). Cherry-picked
  into the dump branch (63283bc9), release builds rebuilt, twins rerun: floor T 5.4e-5 / S 1.6e-5 / a_ice 3.2e-4,
  every BGC tracer <= 2.0e-5 on both twins (gate job 48606115). The residual is the fork's older sea ice
  (`docs/recom/VALIDATION.md`, `docs/PORTING_LESSONS.md` 2026-10-29). Lane W's gate G4b is done; G5 (30 days
  against a twin) would need a 30-day physics-matched twin and inherits the ice floor.

- 2026-10-29, the fork's sharded and gradient gates: `test_recom_sharded_fork.py` (pi dist_4 of the packaged pi,
  the fork bundle shares its nodes/elements/edges; 2 steps of the production spectral config from the 30-day
  restart; worst tracer 3.2e-13, physics 1.1e-14, job 48611814) and `test_recom_grad_fork.py` (G6: the 1-day NPP cost
  vs FD, expon_phy 4.4e-6, expon_cocco 6.6e-7, rho_N 8.2e-5, deg_Chl 6.0e-8, cdomcoeff 1.2e-7, graz_max numerically
  zero; job 48611817). Two fixes on the way: the ballast reset's `any` reduces over the whole mesh (was per
  device = Fortran's per-rank domain: DetC 2.6e-2 sharded vs dense; the stated deviation, `sinking.ballast(reduce_sum=)`),
  and `spectral._safe_sqrt` for the dark-cell `sqrt(0)` whose infinite derivative made every light-crossing gradient
  nan. `P_cm*`/`alfa*` are dead in the fork's 4-PFT spectral build (`Temp_*`, `alpha_I`). Lane W's gates are complete
  (`docs/recom/VALIDATION.md`). Open: `forcing.repeat_year`, the warm-start converter, the CORE2 comparison, the merge.

- 2026-10-29, lane W's tail: `forcing.repeat_year` (the JRA reader's file tag, `jra55.JRA55Reader(tag=)`, FESOM2's
  `y_perpetual`: the clim61 climatology's 365-day axis rebased to every model year; `test_repeat_year.py` on the
  Maresys files, job 48617080) and `forcing.chl_const`; `scripts/recom/fortran_restart_to_jax.py` converts the user's
  1997 4-PFT Fortran restart (positional tracer files, the AB2 histories, the ice, the benthos and HPlus; thicknesses
  rebuilt from hbar since fesom-jax has no partial cells; Alk_surf from the IC) into
  `fesom_jax_runs/restart_1997_4pft_jax` (job 48617062, 95 s); `configs/albedo/core2_recom_spectral_warm1998.yaml`
  runs from it with the user's forcing: 32 steps on one A100, job 48617081, 3.2 s/step. The 1997 benthos pools of
  the Fortran state are huge (median BenC 1e11 mmol m-2) and are carried as they are. Open: the CORE2 comparison
  figures against `tmp_run_kdind15`, the merge.

## Rules (as the parent REcoM port)

- Mirror the fork exactly, quirks included; anything not mirrored goes into the plan and `docs/PORTING_LESSONS.md`.
- Every new switch off by default; `recom_cfg=None` and the `bin_2p3z2d` namelist must leave the existing
  programs byte-identical (`test_dp_lock.py`, the pi twin gates in `docs/recom/VALIDATION.md` keep passing).
- Fortran `file:line` citations in docstrings; `jax.lax` control flow only; f64; no host-side branching on traced
  values; the light solver must be grad-safe (the point of the port).
- Cite the Albedo SLURM job ids as evidence in this plan.

## Inputs on Albedo (for lane 0 and the final run)

- Fortran fork: `/albedo/work/user/sloza/SPECTRAL/recom_tra_diags_spectral` (build via
  `/albedo/home/cvoelker/fesom2/env/albedo/shell`, CMake option `RECOM_WAVEBANDS`); run dirs `tmp_run_kdind15/`
  (reference), `tmp_run_cocco/`; optics `.dat` files next to the namelists (read as relative paths).
- OASIM: `/albedo/work/projects/p_specbic/sloza/forcing_oasim_v2/{Ed,Es}_{400..700}_oasim_clim1998-2012_darwin400.nc`.
- REcoM IC, dust, CO2: `/albedo/work/user/anbros001/Maresys/` and `/albedo/pool/FESOM2/recom/`.
- Atmosphere: clim61 repeat-year JRA55-do (`Maresys/*.clim61.nc`, `y_perpetual`); the JAX reader wants
  `{var}.{year}.nc` per year, so a `forcing.repeat_year` option (or symlinks per year) is part of lane W.
- Warm start: `/albedo/work/projects/p_specbic/sloza/restart_1997_4pft/` (Fortran restart layout; a converter to
  the JAX restart is lane W's last step if a warm start is wanted).
