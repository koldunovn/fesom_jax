# FESOM2 + REcoM Fortran reference (pi, double precision)

The reference the fesom-jax REcoM port is gated against (plan `docs/plans/completed/20261025-recom-port.md`, Tasks 3-4;
lane plan `docs/plans/20261025-recom-lane-fortran-ref.md`). Everything lives outside the repository:

| what | where |
|---|---|
| FESOM2 clone (instrumented) | `/work/ab0995/a270088/fesom2_recom`, local branch `recom-ref-dumps` |
| runs, dumps, logs | `/work/ab0995/a270088/recom_ref/` |
| pi mesh + 1-rank partition | `/work/ab0995/a270088/recom_ref/mesh_pi_part/` (copies of `test/meshes/pi` + `dist_1`) |
| uninstrumented binary | `/work/ab0995/a270088/recom_ref/exe_pristine/` |

In the repository: `scripts/recom/fortran/` (build script, run script, the namelist set, the instrumentation
patches, the comparison and diagnostics scripts), the reader `fesom_jax.io_dump.load_recom_dump` and its test
`fesom_jax/tests/test_recom_dump_io.py`.

## Source and build

- FESOM2 `e96a2b13` (upstream main of 2026-10-20) + the dump instrumentation `fc698a30` (local commit, branch
  `recom-ref-dumps`); REcoM submodule `692f8045e0d5bd296d9c0e8bd9ff4a2b5872f79d` + `b9e98a2` (local commit, the dump
  module). The instrumentation is in `scripts/recom/fortran/patches/` (0001 = FESOM2, 0002 = REcoM).
- Compiler: Intel oneAPI 2022.0.1 (`ifort` 2021.5.0) with OpenMPI 4.1.2, `env/levante.dkrz.de/shell`.
- CMake: FESOM2's `configure.sh levante -DRECOM_COUPLED=ON -DUSE_SINGLE_PRECISION=OFF`, `CMAKE_BUILD_TYPE=Release`,
  CVMix ON (FESOM2 default), no OpenMP. Definitions of the fesom target: `-DFESOM_PROFILING -DUSE_PRECISION=2
  -D__async_icebergs -D__cvmix -D__recom -D__usetp`. Flags (fesom): `-O3 -r8 -i4 -fp-model precise -no-prec-div
  -fimf-use-svml -init=zero -no-wrap-margin -fpe0 -fpp -fPIC -no-prec-sqrt -ip`; REcoM adds `-march=core-avx2
  -qopt-malloc-options=2 -qopt-prefetch=5 -unroll-aggressive`.

Rebuild recipe:

```bash
git clone https://github.com/FESOM/fesom2 /work/ab0995/a270088/fesom2_recom
cd /work/ab0995/a270088/fesom2_recom
git checkout -b recom-ref-dumps e96a2b13
git submodule update --init src/recom                      # 692f8045
git -C src/recom checkout -b recom-ref-dumps
git -C src/recom am <repo>/scripts/recom/fortran/patches/0002-*.patch       # src/recom/recom_ref_dump.F90
git apply --exclude=src/recom <repo>/scripts/recom/fortran/patches/0001-*.patch   # the FESOM2 hooks
git add src/fesom_module.F90 src/oce_ale_tracer.F90 src/oce_ale_vel_rhs.F90 src/recom_ref_dump_hooks.F90 src/recom
git commit -m "REcoM reference dumps"
<repo>/scripts/recom/fortran/build_fesom2_recom.sh         # -> bin/fesom.x + lib64/libfesom.so
```

The 1-rank partition: build `mesh_part` (`bash mesh_part/configure.sh`), copy the pi mesh files into a fresh
directory and run `fesom_meshpart` there with `&machine n_levels=1, n_part=1` (input:
`recom_ref/meshpart_run/namelist.config`). On one rank `myList_nod2D` is the identity (checked); the dumps write
`mesh.gid_nod` / `mesh.gid_elem` so a reader can verify it.

## Run setup

`scripts/recom/fortran/run_pi_recom.sbatch` (one MPI rank, `--hint=nomultithread`): it refuses an existing run
directory (except `CONTINUE=1`), writes `fesom.clock`, fills the namelist templates of
`scripts/recom/fortran/namelists/`, copies the executable and `libfesom.so` into `exe_seg<N>/` and records the
commits, md5 sums and the dump settings in `seg_<N>.info`. Cold start = `REcoM_restart = .false.` (BGC tracers
from the IC files); `CONTINUE=1` = restart from the run directory with `REcoM_restart = .true.`.

### The namelist set (for `configs/pi_recom.yaml`)

The `config/bin_2p3z2d` namelists with these changes (all of them in `scripts/recom/fortran/namelists/`):

| file | change from `bin_2p3z2d` | why |
|---|---|---|
| config | `step_per_day 32` (36), run length in days, mesh/clim/result paths, `n_part 1`, raw and bin restarts `off` (netCDF restart only), `logfile_outfreq 32` | fixed choices; one rank |
| dyn | `use_wsplit .true.` (`.false.`) | fixed choice |
| ice | `whichEVP 1` (0); `evp_rheol_steps 120` unchanged | fixed choice |
| tra | `tracer_init3d`: the 5-degree pi files of `test/input/recom` in the `bin_2p3z2d` order: T, S `woa18_netcdf_5deg.nc`; 1001 `din5deg.nc` (`n_an`); 1002 `GLODAPv2.2016b.PI_TCO2_fesom2_mmol_fix_z_Fillvalue.nc` (`PI_TCO2_mmol`, preindustrial as `bin_2p3z2d`); 1003 `talk5deg.nc`; 1018 `si5deg.nc`; 1019 `fe5deg.nc`; 1022 `oxy5deg.nc`; `t_insitu .true.` | pi inputs |
| forcing | `config/namelist.forcing.JRA` (JRA55-do v1.4.0) with `l_mslp .true.`, and `use_ocean_only_forcing .false.` (the template has `.true.`) | fixed choice; fesom-jax has no land-filled forcing reader |
| recom | `nm_fe_data_file 'DustClimMonthlyAlbani_pimesh.nc'` (read from `ClimateDataPath` = `test/input/recom/`), unused paths blanked, `REcoM_restart` from the job | pi inputs |
| io | daily means in double precision of sst, sss, ssh, ice, MLD1, the 2-D REcoM fields, temp, salt, otracers, PAR | output only |

The physics that results (everything not listed keeps the `bin_2p3z2d` value, which equals FESOM2's default
`config/` file where both set it):

- **config:** zstar, no partial cells, rotated grid (`force_rotation`, Euler 50/15/-90), leap years on, ice on,
  no cavities, no floating ice, `use_sw_pene .true.`, start 1958-01-01 00:00 (or 1958-07-01, day 182).
- **oce:** `C_d 0.0025`, `A_ver 1e-4`, `scale_area 5.8e9`, no SPP; GM `Fer_GM` with `K_GM_max 1000`, `K_GM_min 2`,
  `K_GM_bvref 1`, `K_GM_resscalorder 2`, `K_GM_cm 3`, `K_GM_cmin 0.1`, `scaling_resolution`, `scaling_GMzexp`
  (`zref 500`, `smin 0.6`), `scaling_ODM95` (`Scr 2e-3`, `Sd 1e-3`), no Ktaper; Redi on, `Redi_Ktaper`,
  `Redi_Kmax 0` (= `K_GM_max`), `Redi_Kmin 100`; **mixing `KPP` (FESOM-native KPP)**, `Ricr 0.3`, `concv 1.6`,
  `visc_sh_limit 5e-3`; no tides.
- **dyn:** `opt_visc 5` (`gamma0 0.003`, `gamma1 0.1`, `gamma2 0.285`, `easybsreturn 1.5`, `check_opt_visc`),
  implicit vertical viscosity, `momadv_opt 2`, no free slip, `use_wsplit` (`wsplit_maxcfl 1.0`), AB2, the implicit
  (CG) SSH solver (no split-explicit); solver tolerances are not set in `bin_2p3z2d` (code defaults).
- **ice:** mEVP (`whichEVP 1`, 120 subcycles, `alpha_evp = beta_evp = 250`), `Pstar 30000`, `ellipse 2`,
  `c_pressure 20`, `delta_min 1e-11`, `Cd_oce_ice 0.0055`, `ice_gamma_fct 0.5`, `ice_diff 0`, `theta_io 0`,
  `ice_ave_steps 1`; thermodynamics `Sice 4`, `h0 0.5`, emissivities 0.97, albedos 0.81/0.77/0.70/0.68/0.10,
  `con 2.1656`, `consn 0.31`; ice IC not from file.
- **tra:** 33 tracers (1, 2, 1001-1030, 1037), all `MFCT`/`QR4C`/`FCT` with weights `0., 1.`; no biharmonic
  filter, implicit vertical diffusion, AB2, `ltra_diag .false.`; `use_instabmix` (0.1), no momix, no windmix,
  `diff_sh_limit 5e-3`, `Kv0_const`, `K_ver 1e-5`, `K_hor 0`, no double diffusion, `surf_relax_T 0`,
  `surf_relax_S 1.929e-6` with local reference SSS, `clim_relax 0`.
- **forcing:** NCAR bulk formulae (z 10 m), `Ce = Ch = 1.75e-3`, `Cd_atm_oce 1e-3`, `Cd_atm_ice 1.2e-3`; JRA55-do
  `uas vas huss rsds rlds tas prra prsn psl`, `nm_nc_iyear 1900`, `nm_nc_freq 1`, `nm_nc_tmid 0`, not perpetual;
  runoff `CORE2_runoff.nc` (CORE2 climatology); SSS restoring `PHC2_salx.nc`; constant chlorophyll 0.1.
- **recom:** `bin_2p3z2d` as is: 2 phyto + 3 zoo + 2 detritus (`enable_3zoo2det`, no coccos), `bgc_num 31`,
  `biostep 1`, `Nmocsy 1`, ballasting with density and viscosity scaling, Albani dust (`dust_sol 0.02`),
  `constant_CO2` 278 ppm, alkalinity restoring (`surf_relax_Alk 3.2e-7`), `Diags .true.`, O2-dependent
  remineralisation, photodamage, CO2 limitation, variable grazing preference, grazing on detritus, no rivers,
  erosion, aeolian N, MEDUSA, ciso or atbox. `use_virt_tracers .true.` is set but inactive under zstar.

## Dumps

Env-gated in our clone only: `FESOM_RECOM_DUMP_DIR` switches them on (one MPI rank required),
`FESOM_RECOM_DUMP_STEPS` lists the steps as **`mstep`, the step number within the run segment** (FESOM2's `n`,
which restarts at 1 after a restart). Without the variable the hooks return at their first test; the 1-day
smoke runs with the instrumented binary with and without dumps and with the uninstrumented binary are
bitwise identical in all 493 variables of the restarts and outputs (jobs 28417235, 28417236, 28417240; the
30-day check is below).

Format (`recom_ref_dump` v1, read by `fesom_jax.io_dump.load_recom_dump(dirpath, step)`): one directory
`step_<mstep:06d>/` per dumped step with `fields.bin` (raw little-endian records in write order) and `index.txt`
(header `# recom_ref_dump v1 mstep= yearnew= daynew= timenew= month= ndpyr= dt= nod2D= elem2D= nl= npes=1`, then
`<name> <f8|i4> <ndim> <Fortran dims> <byte offset>`). The reader returns C-order arrays with the Fortran shape
reversed: a node field `a(nl-1, nod2D)` is `(nod2D, nl-1)`, `uv(2, nl-1, elem2D)` is `(elem2D, nl-1, 2)`,
`Benthos(nod2D, 4)` is `(4, nod2D)`. Rows are the owned nodes/elements, which on one rank are the global ones in
order. Levels are FESOM's (index 0 = the surface layer); entries below `nlevels_nod2D - 1` are whatever the model
array holds there (mostly 0). Each step is ~0.9 GB.

Catalogue (pi: nod2D 3140, elem2D 5839, nl 48; "tr<ID>" = every tracer 1, 2, 1001-1030, 1037 unless a subset is
named; units are the model's: tracers mmol/m3 except Fe umol/m3, T degC, S psu):

| stage (write order) | where | records |
|---|---|---|
| `mesh` | after `clock` | `gid_nod`, `gid_elem`, `ulevels_nod2D`, `nlevels_nod2D`, `ulevels`, `nlevels` (i4); `areasvol`, `area` (nod2D, nl), `elem_area`, `geo_coord_nod2D` (nod2D, 2; radians), `ocean_area` |
| `recom_in` | `fesom_module.F90`, just before `call recom` | scalars `dt`, `kappa`, `rad`, `timenew`; `calendar` (i4: daynew, month, mstep, ndpyr, yearold); `a_ice`, `m_ice`, `m_snow`, `press_air` [Pa], `u_wind`, `v_wind` [m/s], `shortwave` [W/m2]; `hnode`, `z_3d_n`, `zbar_3d_n`; `tr<ID>`; `Benthos` (4, nod2D) [mmol/m2]; `GloHplus` (holds the surface **pH**, the `flxco2` first guess), `Alk_surf`, `GloFeDust`, `GloNDust`, `AtmCO2` (12); the carbonate cache `CO23D pH3D pCO23D HCO33D CO33D OmegaC3D kspc3D rhoSW3D` |
| `recom_out` | just after `call recom` | `tr<ID>`; `Benthos`, `GlodecayBenthos` (4, nod2D) [mmol/m2/s]; `GloHplus`; the carbonate cache; `relax_alk`, `virtual_alk`, `AtmFeInput`, `AtmNInput`, `GloPCO2surf`, `GlodPCO2surf`, `GloCO2flux` [mmol/m2/d], `GloCO2flux_seaicemask`, `GloO2flux`, `GloO2flux_seaicemask` [mmol/m2/s, the bc_surface inputs], `PistonVelocity`, `alphaCO2`, `x_co2atm`, `PAR3D`; 2-D diagnostics `NPPn NPPd GPPn GPPd NNAn NNAd Chldegn Chldegd grazmeso_{tot,n,d,det,mic,det2} grazmacro_{tot,n,d,mes,det,mic,det2} grazmicro_{tot,n,d}`; 3-D diagnostics `aggn aggd docexn docexd respn respd NPPn3D NPPd3D calcdiss calcif respmeso respmacro respmicro TPhyCO2 TDiaCO2 TCphotLigLim_phyto TCphot_phyto TCphotLigLim_diatoms TCphot_diatoms` |
| `vel_rhs` | `compute_vel_rhs` | `slp_force` (elem2D, 2): -grad(press_air/1000) [m/s2], the l_mslp term (added to `UV_rhs` times `elem_area` on every layer) |
| `transport` | start of `solve_tracers_ale`, before the bolus velocity is added | `uv`, `fer_uv` (elem2D, nl-1, 2); `w`, `w_e`, `w_i`, `fer_w` (nod2D, nl); `hnode`, `hnode_new`, `z_3d_n` (nod2D, nl-1); `helem` (elem2D, nl-1); `zbar_3d_n`, `Kv` (nod2D, nl); `Ki` (nod2D, nl-1); `slope_tapered` (nod2D, nl-1, 3) |
| `tra_in` | tracer loop, before `init_tracers_AB` | `tr<ID>.values`, `.valuesold` |
| `tra_ab` | after `init_tracers_AB` | `.values`, `.valuesold`, `.valuesAB` |
| `tra_adv` | after advection was added to `del_ttf` | `.values`, `.del_ttf` |
| `tra_redi` | `diff_tracers_ale`, after horizontal + explicit vertical Redi | `.values`, `.del_ttf` |
| `tra_bf` | after `diff_ver_recom_expl` (remin tracers 1001 1002 1003 1018 1019 1022) | `.dtr_bf`, `.values` (= values + dtr_bf) |
| `tra_ball` | after `ballast` (sinking tracers 1004-1008, 1013-1017, 1020, 1021, 1025-1028) | `.seawater_visc_3D`, `.rho_particle1`, `.rho_particle2` (nod2D, nl-1); `.scaling_visc_3D`, `.scaling_density1_3D`, `.scaling_density2_3D` (nod2D, nl) |
| `tra_sink` | after `ver_sinking_recom` / `_benthos` and the update (sinking tracers) | `.vert_sink`, `.str_bf` (nod2D, nl-1); `.SinkingVel1`, `.SinkingVel2` (nod2D, nl); `.Benthos_tr` (4, nod2D: this tracer's benthos buffer); `.Benthos`; `.values` (= values + vert_sink + str_bf) |
| `tra_tstar` | after the T* update | `.values`, `.del_ttf` |
| `tra_impl` | after `diff_ver_part_impl_ale` (incl. bc_surface) | `.values` |
| `tra_out` | after `relax_to_clim`, before the halo exchange | `.values` |
| `step_end` | after `oce_timestep_ale` (salinity clip and the Benthos merge done) | `tr<ID>`, `Benthos` |

## Runs (2026-10-25)

All with FESOM2 `fc698a30` / REcoM `b9e98a2` (the instrumented build; `exe_seg1/` in each run directory holds the
binary, md5 `ccd415b9...` for `fesom.x`, `06c690cf...` for `libfesom.so`), one rank on one `compute` node, the
namelist set above, cold start (`REcoM_restart .false.`) unless stated. `dt` 2700 s; ~2.3 s per step on one rank
(the tracer step 65 %, `recom()` 29 %), so a 30-day run takes 37 min and does not fit the 30-min debug QOS: the
30-day jobs ran with `--time=01:00:00`, the 15-day segments with `45:00`.

| run (`recom_ref/<run>`) | what | job | wall | dumps (`<run>/dumps`) |
|---|---|---|---|---|
| `pi_30d_jan` | (a) 30 d from 1958-01-01 | 28417314 | 37 min | mstep 1, 2, 17, 224, 225, 960 (5.2 GB) |
| `pi_30d_jan_nodump` | (a) without `FESOM_RECOM_DUMP_DIR` | 28417315 | 37 min | none |
| `pi_1d_jul` | (b) 1 d from 1958-07-01 (day 182) | 28417316 | 1.7 min | 1, 17 (1.8 GB) |
| `pi_30d_split` | (c) segment 1: 15 d from 1958-01-01 | 28417317 | 19 min | none |
| `pi_30d_split` | (c) segment 2: restart, 15 d (`REcoM_restart .true.`) | 28417318 | 19 min | segment mstep 1, 2 (1.8 GB) |
| `smoke_1d_d`, `_dump`, `_pristine` | 1 d: instrumented without / with dumps (1, 2, 32), uninstrumented | 28417235/6, 28417240 | 1.6 min each | `smoke_1d_d_dump/dumps` |

Earlier attempts (`smoke_1d`, `smoke_1d_b`: job-script bugs, 28417093, 28417128; `smoke_1d_c`, 28417176:
present-day DIC IC, superseded) are left in place. Total cost ~2.0 node-hours.

Checks:

- **No-op:** `pi_30d_jan` and `pi_30d_jan_nodump` are bitwise identical in all 493 variables of the 30-day
  restarts and the daily output (`scripts/recom/fortran/compare_runs.py`); so are the three 1-day smoke runs
  (including the uninstrumented binary).
- **Sanity** (`scripts/recom/fortran/global_diags.py`): 31 BGC tracers in the log ("Actual BGC tracers from
  namelist = 31"), no non-finite value in any output; CO2 uptake 2.1 PgC/yr on day 1 rising to 4.5 PgC/yr on day
  30 (surface dpCO2 -7.5 -> -22 uatm against 278 ppm), NPP grows from the `tiny_chl` cold start (0.003 PgC/yr on
  day 1) to 38 PgC/yr on day 30; `press_air` 0.96-1.04e5 Pa.
- **Dump self-consistency:** from one dumped step, numpy rebuilds `tra_bf.values`, `tra_sink.values` and
  `tra_tstar.values` from `tra_redi` + `dtr_bf` + `vert_sink` + `str_bf` + `hnode`/`hnode_new` bitwise for T, S and
  BGC tracers; `step_end.Benthos` = `recom_out.Benthos` + the per-tracer `Benthos_tr` (to 2e-18, summation order).
- **Carbonate cache refresh** (cells where `recom_out.CO23D` differs from `recom_in.CO23D`): mstep 1 all 101749
  wet cells; 2 and 17 none; 224 (end of day 7) 54986 cells in all 3140 columns (the 7-day lit refresh; at night
  every layer counts as lit); 225 none; 960 (end of day 30) 51799 cells in 1938 columns (the 30-day dark refresh).
  The split run refreshes everything again at its segment mstep 1 (day 16).
- **Split vs unsplit:** days 1-15 bitwise identical; from day 16 on the split run differs, **in the physics as well**
  (day 16: salt 3e-3, SST 2e-4, sea ice 2e-4; day 30: T 0.36 degC, DIC 3 mmol/m3), so FESOM2's restart is not
  bit-continuous in the physics here (REcoM does not feed back on the physics); the split pair is a reference for
  restart semantics as a whole, not only for the BGC `mstep` reset.

## The physics-matched twin (Task 4b, lane P)

`pi_30d_jan_jaxphys`: the same build, REcoM setup, forcing and dumps, with the physics of `configs/pi_recom.yaml`
(= the CORE2 production physics of fesom-jax, `configs/core2_full.yaml`, on pi). Namelists in
`scripts/recom/fortran/namelists_jaxphys/`: the set above with these changes, each taken from the matched CORE2
Fortran run (`docs/CORE2_FORTRAN_SPEC.md`; its run directory is now
`/work/ab0995/a270088/fesom2_core2_SUPERSEDED_experimental_icefct/`):

| file | change from `namelists/` | why |
|---|---|---|
| oce | `mix_scheme 'cvmix_TKE'` (`'KPP'`) | fesom-jax classical TKE (`tke: {}`); fesom-jax has no FESOM-native KPP in production |
| dyn | `opt_visc 7` (5), `use_wsplit .false.` (`.true.`) | fesom-jax has opt_visc 7 only; core2_full runs without w_split |
| ice | `ice_diff 10.0` (0.0); the CORE2 run's explicit `&ice_therm` values (`h0_s 0.5`, `hmin 0.01`, `armin 0.01`, `iclasses 7`, `new_iclasses .false.`, `h_cutoff 3.0`, `snowdist .true.`, `c_melt 0.5`, `open_water_albedo 0`, all equal to the FESOM2 defaults) | fesom-jax `IceConfig` default, as the matched CORE2 run |
| forcing | `chl_data_source 'Sweeney'` with `/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc` (`'None'`, 0.1) | fesom-jax's run driver reads the Sweeney climatology (no constant-chl key), as the matched CORE2 run |

Everything else is unchanged: zstar, GM/Redi (the CORE2 values, already in `namelists/`), mEVP 120 subcycles,
`step_per_day 32`, `l_mslp .true.`, the 33 tracers, the WOA18 5-degree T/S and the BGC IC files, cold start
1958-01-01. Run: `RUN=pi_30d_jan_jaxphys DAYS=30 NML=<repo>/scripts/recom/fortran/namelists_jaxphys
FESOM_RECOM_DUMP_DIR=<run>/dumps FESOM_RECOM_DUMP_STEPS=1,2,3,17,960 sbatch --time=01:00:00 run_pi_recom.sbatch`,
job 28420010, dumps at mstep 1, 2, 3, 17, 960 (the ice in `recom_in(N)` is already the ice after step N, since FESOM2 runs the ice step first; step 3 is a spare). The log
confirms `initialise CVMIX_TKE`. The physics floor of this pair against fesom-jax is in
`scripts/recom/physics_floor.py` and in the plan (Task 4b).

## Things the parent plan should know

- `bin_2p3z2d` itself runs `step_per_day 36`, `whichEVP 0`, `use_wsplit .false.`; the fixed choices override
  these. Its mixing is **FESOM-native KPP** and its viscosity `opt_visc 5`; both are kept (pi runs them).
- The pi test (`setups/test_pi_recom`) initialises DIC from the present-day `tco2_5deg.nc` with 278 ppm (a run
  with it, `smoke_1d_c`, outgasses 11 PgC/yr on day 1); the reference uses `bin_2p3z2d`'s preindustrial DIC, whose
  5-degree version is in `test/input/recom`.
- `__usetp` is always defined with `RECOM_COUPLED` (src/CMakeLists.txt and src/recom/CMakeLists.txt): the
  sinking tracers' benthos goes into per-tracer buffers `Benthos_tr(:, :, tr_num)` and `Benthos` is updated only
  after the whole tracer loop, `Benthos = Benthos + Benthos_tr(:, :, tr)` for tr = 1..33 in order. Inside the
  loop `Benthos` is constant.
- `GloHplus` (restart field `HPlus`) holds the surface **pH** (`GloHplus(n) = ph(1)`), and `flxco2`'s `ph`
  argument is `intent(out)`: the value read back into the local `Hplus` is never used. It is not a solver first
  guess; restarting it changes nothing.
- The SLP term is `p_air = press_air/1000` (not divided by rho0), `Fx = sum(gradient_sca(1:3)*(-p_air))`, added as
  `Fx*elem_area` to every layer of `UV_rhs`; `press_air` is in Pa, REcoM converts it to atm.
- `use_virt_tracers .true.` in `bin_2p3z2d` is inactive under zstar (no virtual tracer fluxes).
- `use_ocean_only_forcing` (land-filled JRA55 fields) is on in FESOM2's JRA template and off here.
