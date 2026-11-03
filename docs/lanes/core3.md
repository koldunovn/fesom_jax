# Lane L-OCN: CORE3 with ice-shelf cavities in fesom_jax

Branch `coupled/core3` (from `coupled/awicm` 5b88770), worktree `/home/a/a270088/port_jax_wt/core3`.
Plan: `~/awicm_jax/docs/plans/20261103-jaxcpl-core3.md` (lane L-OCN). Reference run: Jan's `jaxcpl`
(`/work/bb1469/a270092/runtime/awicm3-develop/jaxcpl/HANDOVER_jaxcpl.md`): FESOM2 main 0aaa2d73, single precision,
CORE3 with cavities, linfs, `cavity_gamma_scale = 0.6`, ocean from the 1855-01-01 spinup.

Outputs live under `/work/ba1583/a270088/awicm/core3/ocn/` (mesh, partitions, jobs, logs) and
`/work/ba1583/a270088/awicm/core3/fesom_oracle/` (the Fortran oracle: source, build, runs, dumps).

## Status (2026-11-03, end of session 2)

| task | state |
|---|---|
| 1. CORE3 mesh in fesom_jax format + cavity metadata + dist_1/2/4/8/16 | **done** |
| 2. FESOM main 0aaa2d73 SP oracle with dumps, run on CORE3 with cavities | **done** |
| 3. cavities behind `use_cavity` | **wired** behind the `cavity:` config slot (default off): EOS/pressure, melt, shelf drag, no ice under the shelves, the shelf-front ice boundary, the virtual-salt integral, the TKE column under the shelves, the SSH matrix and momentum advection at the shelf base; the replay ladder (substeps 1, 3, 4, 5, 7-13, 16) matches the double-precision oracle to round-off / CG tolerance; a whole CORE3 step runs (CPU, GPU) |
| 4. restart import | **done**: jaxcpl's 1855-01-01 ocean + ice restart -> fesom_jax restart, bitwise on every wet lane; the coupled CORE3 config is still to do |

Session 2 in one paragraph: the converter of branch `fortran-restart` is on this branch and reads FESOM2 main
restarts; the replay harness `scripts/core3/replay_ladder.py` checks one kernel per substep against the oracle's
full dumps; three cavity defects in existing kernels were found by it and fixed (SSH matrix depth, momentum
advection at the shelf base, the AB2 bootstrap of a step-0 restart) plus one TKE driver change (the column under a
shelf). Global node order: fesom_jax's CORE3 bundle keeps FESOM's `nod2d.out` order (node i of the bundle = line i+1
of nod2d.out; `geo_coord_nod2D.npy` is that order), which lane L-CPL relies on.

## 1. The mesh

```bash
# raw FESOM files (copies of Jan's, never written in place): /work/ba1583/a270088/awicm/core3/ocn/mesh_core3_raw
python scripts/prepare_mesh.py  mesh_core3_raw  mesh_core3  --cavity        # 32 s on a compute node
python scripts/core3/check_cavity_mesh.py mesh_core3 --mesh-diag /work/ab0246/a270092/input/fesom2/core3/fesom.mesh.diag.nc
```

* `MESH_JAX = /work/ba1583/a270088/awicm/core3/ocn/mesh_core3` (220,509 nodes, 426,403 elements, 647,052 edges,
  nl = 48; 10,047 cavity nodes, 19,001 cavity elements, max `ulevels` 29). Extra arrays: `cavity_depth.npy` (node,
  m, <= 0) and `ocean_areawithcav` in `meta.txt` (3.6688e14 m^2; `ocean_area` 3.6534e14 m^2).
* `DIST_DIR = /work/ba1583/a270088/awicm/core3/ocn/mesh_core3_raw` with `dist_1 dist_2 dist_4 dist_8 dist_16`
  (fesom_jax) and `dist_128` (the oracle), made by 0aaa2d73's own partitioner (`fesom_meshpart` = fvom_init,
  `n_levels = 1, n_part = N`, `use_cavity = .true.`), job `ocn/jobs/partition.sbatch`.
* `prepare_mesh.py --cavity` mirrors `oce_mesh.F90` (`find_levels_cavity`, `find_levels_min_e2n`, `mesh_areas`):
  `ulevels`/`ulevels_nod2D` from the cavity files, `ulevels_nod2D_max`, `area` summed over each cell's levels
  `ulevels..nlevels-1`, `areasvol` = the area of the lower prism face where a cavity cell touches the node's upper
  face, `mesh_resolution` from `areasvol` at the top wet level. Without `--cavity` the output is unchanged.

Verdicts:

| check | job | result |
|---|---|---|
| CORE2 `test_prepare_mesh.py` (no-cavity path byte-faithful to the C export) | 28696089 | 3 passed |
| CORE3 bundle invariants + `load_mesh` | 28696089 (+ rerun of the fixed check, see below) | all OK except my own wrong check "area == areasvol off-cavity" (front nodes legitimately differ); fixed to "where no cavity cell touches the node" |
| CORE3 bundle vs Jan's `fesom.mesh.diag.nc` | 28696089 | `ulevels`, `ulevels_nod2D`, `nlevels`, `nlevels_nod2D` identical; `elem_area` rel 2.2e-12; that file's `nod_area` = our `area` to rel 2.0e-12 (it is `area`, not `areasvol`) |
| CORE3 `area`/`areasvol`/`mesh_resolution`/`elem_area` vs the DOUBLE-precision oracle's own arrays | 28697061 (run_cav_dp) | rel 2.0e-12 / 2.0e-12 / 3.7e-13 / 2.2e-12, zero outside the wet levels in both: the cavity `areasvol` rule is FESOM's |
| partitioner regenerates every derived file from the 4 primitive files (`mesh_core3_regen`) | 28696321 | `nlvls elvls elvls_raw cavity_nlvls cavity_elvls cavity_elvls_raw edges edge_tri edgenum` byte-identical to Jan's |
| our `dist_128` vs Jan's `dist_128` | 28696321 | `rpart.out`, `my_list00000.out`, `com_info00000.out` byte-identical |

## 2. The oracle (FESOM2 main 0aaa2d73, single precision, with dumps)

Source: `fesom_oracle/fesom-0aaa2d73` = `git archive 0aaa2d73` of Jan's tree (no REcoM submodule; RECOM off) as a
local git repo (branch `oracle`): d715474 pristine, 3993842 + ee6b828 the dump instrumentation. The same patch is in
this branch as `docs/lanes/core3_oracle_dumps.patch` (first commit; regenerate with `git format-patch` there).

Build (`fesom_oracle/jobs/build.sbatch`, ~8 min, compute node): `env.sh levante.dkrz.de` (intel 2022 + openmpi 4.1.2,
Jan's toolchain), `-DUSE_SINGLE_PRECISION=ON -DOIFS_COUPLED=OFF -DENABLE_OPENMP=OFF -DBUILD_MESHPARTITIONER=ON
-DCMAKE_BUILD_TYPE=Release` (Jan: `-O3 -fp-model precise`, OpenMP on, OIFS coupled) into `fesom_oracle/build_sp`;
binaries `build_sp/install/bin/{fesom.x, fesom_meshpart}`. A new source file needs a re-run of cmake (GLOB);
`jobs/build_and_run.sbatch` does an incremental `make` + install and then the run.

Run (`fesom_oracle/jobs/run_cav.sbatch`, 1 node, 128 ranks, ~30 s for 6 steps):

* namelists = jaxcpl's as run (`run_18550101-18551231/work/namelist.*`) with these changes only: `meshpath` = our
  `mesh_core3_raw` (identical files, has `dist_128`), `resultpath` = the run dir, `run_length = 6 's'`,
  `restart_length_unit = 'off'`, `use_icebergs = use_icesheet_coupling = .false.`, `use_atm_ice_tskin = .false.`
  (coupled-only), `n_levels = 1, n_part = 128`, clock year 1957/1958 (below).
* **Forcing: standalone JRA55-do v1.4.0, year 1958** — the forcing fesom_jax's CORE2-type runs use. `namelist.forcing`
  = jaxcpl's (already the JRA55-do file set) with `nm_nc_iyear = 1900, nm_nc_tmid = 0` and CORE2 runoff
  (`CORE2_runoff.nc`), i.e. the settings of our earlier Fortran CORE2 runs
  (`/work/ab0995/a270088/fesom_jax_dev/fortran/core2_rst/jan`). No SSS restoring (`surf_relax_s = 0`, as jaxcpl).
* **Initial state: jaxcpl's 1855-01-01 spinup restart, relabelled 1958-01-01**: the run dir holds symlinks
  `fesom.1958-01-01.{oce,ice}.restart -> .../jaxcpl/run_18550101-18551231/work/fesom.1855-01-01.{oce,ice}.restart`
  and `fesom.clock` = `84600 365 1957 / 0 1 1958`, so FESOM reads the 1855 state and the 1958 forcing.
* dumps (env, see `fesom_dump_core3.F90`): `FESOM_DUMP_DIR`, `FESOM_DUMP_MAXSTEPS` (probe records, default 4),
  `FESOM_DUMP_FULL_STEPS` (full owned fields, default 2), `FESOM_DUMP_PROBES[_ELEM]` (id lists from
  `scripts/core3/make_probes.py`: 21 nodes = 9 cavity (deep/mid/shallow), 4 cavity-front, 3 Southern Ocean ice,
  2 Arctic, 3 open; 29 elements; table in `fesom_oracle/probes/probes.txt`).

Dump format. Probe files `probe_nod.<rank>` / `probe_elem.<rank>` are the `io_dump.py` record format (node columns
over levels `1..nlevels_nod2D` INCLUDING the cavity levels as stored, element columns `1..nlevels-1`, vectors as
`<name>_1/_2`, all real64). Full fields `full/s<step>_<substep>_<name>.<rank>` (`int32 n_owned, ncol; int32
gid[n_owned]; float64 [n_owned, ncol]`). Readers: `io_dump.load_probe_dir`, `io_dump.load_full_field`. Substeps
1-16 as before (`io_dump.SUBSTEP_NAMES`), plus 17 `atm_forcing` (shortwave, longwave, Tair, shum, u/v_wind,
prec_rain/snow), 18 `ice_step` (a_ice, m_ice, m_snow, uice, vice), 19 `oce_fluxes` (heat_flux, water_flux,
virtual_salt, relax_salt, real_salt_flux, stress_surf_x/y, stress_node_surf_x/y, a_ice), 20 `pre_fluxes` (the state
`oce_fluxes` reads: T, S, uvnode_1/2, uv, density, density_ref, hnode, Z_3d_n, eta_n, hbar; at step 1 also area,
areasvol, mesh_resolution, elem_area). Per substep: 1 density/pressure/bvfreq, 2 sw_alpha/beta, 3 pgf_x/y (elem),
4 Kv, Av (elem), 5/6/7 uv_rhs after vel_rhs / viscosity filter / implicit viscosity, 8 ssh_rhs, 9 d_eta, 10 uv,
11 hbar, hbar_old, ssh_rhs_old, 12 eta_n, 13 hnode_new, w, 14 fer_uv, fer_w, 15 T, S, Kv_after, 16 hnode, helem.

Runs and verdicts:

| run dir | job | what | result |
|---|---|---|---|
| `run_cav` | 28696443 | binary 3993842, probes steps 1-4, full steps 1-2 | COMPLETED, 6 steps, 11.5 s; SSH CG 52 iterations mean, no non-convergence; no NaN; 2.2 GB of dumps |
| `run_cav2` | 28696550 | + `pre_fluxes` stage | COMPLETED; full fields **bitwise identical** to `run_cav` (512 rank files compared): the oracle is deterministic |
| `run_cav3` | 28696663 | ee6b828 (+ static areas), full step 1 only | COMPLETED; static areas used for the SP geometry row below |
| `run_cav_dp` | build 28696789, run 28697061 | the same source built in **double precision** (`build_dp`, `-r8`), probes 1-4, full 1-2 | COMPLETED, SSH CG 51 iterations mean; the tight rung (see the precision note) |

Sanity of the oracle state (run_cav, step 1): no sea ice under the shelves (a_ice, m_ice, uice all 0 at the
10,047 cavity nodes); T = 0 in the cavity levels above `ulevels` (as stored), top wet T at cavity nodes -2.30 ...
0.89 degC; eta_n = hbar = 0 at cavity nodes (rigid lid), d_eta -1.8 ... -1.5 there; basal melt from
`-water_flux * areasvol(top)` = **878 Gt/yr** at step 1 (jaxcpl's budget: 760-798 Gt/yr annual). Large upward
`heat_flux` (up to 2.9e4 W/m^2, 3,812 nodes above 1,500 W/m^2 at step 1, 1,872 at step 2) at partly ice-covered
open nodes near 77 S: the first-step shock of a coupled restart (OpenIFS sea-ice state) meeting JRA55 forcing; not
a cavity effect, it halves per step.

Rebuild / rerun recipe (also for lane L-SFC, which may use this oracle with its own namelists):

```bash
O=/work/ba1583/a270088/awicm/core3/fesom_oracle
# edit $O/fesom-0aaa2d73/src/...; commit there
mkdir -p $O/run_X/dumps/full && cp $O/run_cav/namelist.* $O/run_cav/fesom.clock $O/run_X/ && cp -P $O/run_cav/fesom.1958-01-01.*.restart $O/run_X/
sbatch --export=ALL,RUNDIR=$O/run_X,MAXSTEPS=4,FULLSTEPS=2 $O/jobs/build_and_run.sbatch     # or jobs/run_cav.sbatch if no rebuild
```

Precision note for the ladder. There are two oracles from one source: `build_sp` (jaxcpl's precision; `run_cav*`)
and `build_dp` (`run_cav_dp`, job `jobs/build_dp.sbatch`, run with `FESOMX=$O/build_dp/install/bin/fesom.x`).
Port kernels against the DOUBLE oracle with fesom_jax in double precision: that is the tight rung (map ~1e-15,
reductions ~1e-12; the cavity mesh and kernels below already meet it). The SP oracle then checks fesom_jax's
single-precision mode at float32 tolerances, with one known, forced deviation: FESOM SP derives its geometry in
float32 (elem_area off the float64 value by up to 8.7e-4 relative on CORE3, median 1.8e-5), fesom_jax SP rounds the
float64 geometry (6e-8). Decision (coordinator 2026-11-03, put to Nikolay): keep fesom_jax's
float64-derived geometry for now. Where that limits the SP comparison: every SP rung that reads areas or gradients
(`elem_area`, `area`/`areasvol`, `gradient_sca`, edge geometry: N^2 smoothing, PGF, momentum, the SSH matrix,
tracer fluxes, the global integrals) carries geometry differences up to ~1e-3 relative; rungs that are column maps
(EOS density/pressure, the melt, the TKE column) do not, and there the SP rung is float32-tight (melt 99.4 % bitwise). Compiler: FESOM's Intel flags are `-r8`/`-r4` (a bare literal is `WP`, not
float32) with `-fp-model precise -no-prec-div -fimf-use-svml -no-prec-sqrt`, so divisions/sqrt/log are not always
IEEE-rounded: expect ulp-level, not bitwise, agreement in a few percent of values.

## 3. Cavities: inventory (FESOM2 main 0aaa2d73)

Size. 112 source lines name `use_cavity*` in 23 files; 26 of them are in seven `icb_*.F90` files (icebergs, out of
scope); `cavity_param.F90` is 588 lines (not 685: the plan's number is from another version) holding 10 routines.
The explicit branches are the small part: every vertical loop in FESOM starts at `ulevels`/`ulevels_nod2D`/
`ulevels_nod2D_max`, and most fesom_jax kernels already carry those bounds from the C port (`nzmin = ulevels-1`),
never exercised with `ulevels > 1`. Under "fesom_jax" below: **ported** = done this session; **mask present** = the
C-port code already has the cavity mask/bound, needs the oracle check; **todo** = missing.

A. Mesh and setup

| file:line | routine | what it does | fesom_jax | state |
|---|---|---|---|---|
| oce_mesh.F90:62, 981-1479 | `mesh_setup`, `find_levels_cavity` | read `cavity_elvls/nlvls.out` -> `ulevels`, `ulevels_nod2D`; `cavity_depth@node.out` | `scripts/prepare_mesh.py --cavity` | ported, verified |
| oce_mesh.F90:1487 | `find_levels_min_e2n` | `ulevels_nod2D_max` | prepare_mesh | ported |
| oce_mesh.F90:2084-2172 | `mesh_areas` | `area` over each cell's wet levels; `areasvol` lower-face rule at cavity contacts; `areasvol_inv`; `mesh_resolution`; `ocean_area`, `ocean_areawithcav` | prepare_mesh; `mesh.py` must load `ocean_areawithcav`/`cavity_depth` when present | ported (mesh.py: todo) |
| oce_mesh.F90:2728 | `check_cavity_mesh_conflict` | refuse cavity files with `use_cavity` off | prepare_mesh warns; `load_mesh` should refuse `ulevels>1` with `use_cavity` off | todo (small) |
| oce_ale.F90:474, 547 | `init_surface_elem/node_depth` | partial cavity cells (`use_cavity_partial_cell`) | — | n/a (off in jaxcpl) |
| oce_ale.F90:91-240 | `init_ale` | `zbar_3d_n`, `Z_3d_n` incl. dummy levels inside the cavity | `mesh.zbar_3d_n`, `Z_3d_n` | same as today without partial cells; check `Z_3d_n` in the cavity |
| oce_setup_step.F90:193, 200 | `ocean_setup` | **`use_cavity` forces `use_density_ref = .true.`**: `init_ref_density` = JM density of (T=2, S=34) on every node's `Z_3d_n`; every `density_m_rho0`, `density_sigma0`, `dbsfc` becomes relative to that profile, globally, not only under the shelves | `eos.py` (today `density_ref = density_0`) | **todo** — the largest global effect of the switch |
| oce_setup_step.F90:198 | `compute_nrst_pnt2cavline` | nearest cavity-line point (IC extrapolation) | `phc_ic.py` | later (cold start only) |
| oce_setup_step.F90:649 | `dynamics_init` | split-explicit SSH refuses cavities | config validation | todo (small) |
| gen_ic3d.F90:261-608 (6) | `nc_ic3d_ini`, `getcoeffld`, `do_ic3d` | PHC under the shelves: extrapolation from the cavity line, zero above `ulevels` | `phc_ic.py` | later (cold start only; jaxcpl starts from a restart) |

B. Forcing and surface fluxes

| file:line | routine | what it does | fesom_jax | state |
|---|---|---|---|---|
| gen_forcing_couple.F90:686 | `update_atm_forcing` | zero wind, humidity, SW, LW, Tair, rain, snow, runoff at cavity nodes; keep `press_air` | `forcing.py` (`open_water`, fesom_bulk.c:248-257), `surface_forcing.py` | mask present; check vs substep 17 |
| ice_oce_coupling.F90:116 / cavity_param.F90:377 | `oce_fluxes_mom` -> `cavity_momentum_fluxes` | quadratic drag of the shelf base on the top wet cell: `stress_surf` (elem), `stress_node_surf` (node) | `cavity.cavity_momentum_fluxes` | **ported**, replayed |
| ice_oce_coupling.F90:354 / cavity_param.F90:143 | `oce_fluxes` -> `cavity_heat_water_fluxes_3eq` | three-equation melt with `cavity_gamma_scale`; overwrites `heat_flux`, `water_flux` at cavity nodes | `cavity.cavity_heat_water_fluxes_3eq`, `potit` | **ported**, replayed |
| cavity_param.F90:329 | `cavity_heat_water_fluxes_2eq` | two-equation variant | — | n/a (not called) |
| ice_oce_coupling.F90:392-397 | `oce_fluxes` | virtual-salt balance (linfs) skips cavity nodes | surface flux / `sss_runoff.py` | todo |
| ice_oce_coupling.F90:410 | `oce_fluxes` | zstar without virtual salt: `virtual_salt = 0` | — | n/a (linfs) |
| ice_oce_coupling.F90:462 | `oce_fluxes` | no SSS relaxation under the shelves | `relax_salt` | todo (zero in jaxcpl anyway: `surf_relax_s = 0`) |
| ice_oce_coupling.F90:569 | `oce_fluxes` | freshwater balance: linfs counts `-water_flux` at cavity nodes (zstar: 0) | water balance | todo |
| ice_oce_coupling.F90:593 | `oce_fluxes` | the balance correction `net/ocean_area` only off the cavities | water balance | todo |
| oce_shortwave_pene.F90:48, 85 | `cal_shortwave_rad` | no shortwave penetration under the shelves | `forcing.py` (C port: no penetration under cavity or ice) | mask present; check |
| fesom_conservation.F90:87 | `conservation_before_step` | diagnostic | — | n/a |

C. Sea ice

| file:line | routine | what it does | fesom_jax | state |
|---|---|---|---|---|
| ice_setup_step.F90:189 / cavity_param.F90:421 | `ice_timestep` -> `cavity_ice_clean_vel` | `uice = vice = 0` under the shelves | `cavity.cavity_ice_clean_vel` | ported, not wired |
| ice_setup_step.F90:265 / cavity_param.F90:445 | `ice_timestep` -> `cavity_ice_clean_ma` | `a_ice = m_ice = m_snow = 0` under the shelves | `cavity.cavity_ice_clean_ma` | ported, not wired |
| ice_maEVP.F90:371 | `EVPdynamics_m` (mEVP, jaxcpl `whichEVP = 1`) | velocity BC `bc_zero` also at nodes on an edge with a cavity cell | `ice_mevp.py` (cavity masks for the tilt/stress; this edge BC: absent) | todo |
| ice_maEVP.F90:868, ice_EVP.F90:621 | aEVP, EVP | same BC | — | n/a (mEVP) |
| ice thermo / advection masks | `ice_thermo.py:321`, `ice_adv.py` | skip cavity nodes/cells | from the C port | mask present; check vs substep 18 |

D. Ocean dynamics and tracers (the implicit part: loops from `ulevels`)

| file | routine | what changes with `ulevels > 1` | fesom_jax | state |
|---|---|---|---|---|
| oce_ale_pressure_bv.F90:198-236 | `pressure_bv` | density in the cavity levels from the top wet T/S; `hpressure(nzmin)` = the weight of the water column the shelf displaces (linfs or cavity) | `eos.py` (nzmin generic; cavity fill and this `hpressure` start: absent) | **todo** |
| oce_ale_pressure_bv.F90:535 | `pressure_force_4_linfs` | no partial cells -> the full-cell PGF from each cell's top level | `pgf.py` (`surf_e = ulevels-1`) | mask present; check vs substep 3 |
| oce_ale_pressure_bv.F90:1095-2461 | easypgf variants | cavity + density_ref corrections | — | n/a (partial cells only) |
| oce_ale.F90:1406 | `init_stiff_mat_ale` | no `areasvol/dt` time-derivative term in cavity rows (rigid lid) | `ssh.py:587` (`nocav`) | mask present; check vs substep 9 |
| oce_ale.F90:1893-1900 | `compute_ssh_rhs_ale` | zstar/zlevel only: `use_cavity_fw2press` (default true) adds `-alpha*water_flux*areasvol` at cavity nodes (melt raises the pressure), no `ssh_rhs_old` term there | `ssh.py` | n/a for linfs (jaxcpl); todo for a zstar cavity run |
| oce_ale.F90:2103-2134 | `compute_hbar_ale` | no water flux / hbar change at cavity nodes; `dhe = 0` on cavity cells | `ssh.py:1208-1224` | mask present; check vs substep 11 |
| oce_ale.F90 `oce_timestep_ale` | eta_n update only where `ulevels_nod2D == 1` | `ssh.py` | mask present; check vs substep 12 |
| oce_ale.F90 `vert_vel_ale` | `w` from `nzmin`; surface freshwater BC only off the shelves | `ale.py:303` (`nocav`) | mask present; check vs substep 13 |
| oce_ale.F90 `impl_vert_visc_ale` | top boundary at `ulevels(elem)` with `stress_surf` = shelf drag | `momentum.py:324-343` | bounds present; check vs substep 7 |
| oce_ale_vel_rhs.F90, oce_dyn.F90 | Coriolis/PGF/advection, viscosity (`opt_visc = 5` in jaxcpl: lane L-SFC) | `momentum.py:215` (edge `nzmin = max(ulevels)`) | bounds present; check vs substeps 5, 6 |
| oce_ale_tracer.F90 | advection (MFCT/QR4C), implicit vertical diffusion with the surface BC at `nzmin` (`impl_bc` from `heat_flux`/`water_flux` = melt under the shelves), ballast | `tracer_adv.py`, `tracer_diff.py` | bounds present; check vs substep 15 |
| oce_fer_gm.F90, oce_redi | GM/Redi between `ulevels_nod2D_max` and `nlevels_nod2D_min` | `gm.py:220,399`, `gm_redi.py` | bounds present; check vs substep 14 |
| cvmix_driver/gen_modules_cvmix_tke.F90:340, 558 | TKE: surface BC at `nun = ulevels_nod2D` (forcing at the shelf base) | `cvmix_tke.py` (asserts `k = 0`, "no cavities") | **todo** |
| oce_ale_mixing / `mo_convect`, `use_instabmix`, MLD | from `nzmin` | `pp.py`, eos MLD | check vs substep 4 |

E. Out of scope here: icebergs (`icb_*`, 26 lines), REcoM cavity NaN scrub (oce_ale.F90:4387, `__recom`), output
(`io_meandata.F90:2431`, `io_mesh_info.F90`), `fvom_init` (Fortran tool, used as is).

What the replay checked (job 28696654, `scripts/core3/replay_cavity_melt.py`): see "Verdicts, task 3" below.

### What exists for task 3 (none of it is called by the step yet; default results unchanged)

* `fesom_jax/cavity.py`: `three_eq_melt` / `cavity_heat_water_fluxes_3eq` (+ `potit`, `pttmpr`, `adlprt`, with
  FESOM's float32 default-REAL literals reproduced), `cavity_momentum_fluxes`, `cavity_ice_clean_vel/_ma`,
  `cavity_boundary_node_mask` (the mEVP velocity BC at the shelf front), `CavityConfig` (the planned `cavity:`
  run-config slot: `gamma_scale`, `density_ref_T/S`).
* `fesom_jax/eos.py`: `reference_density` (`init_ref_density`) and `pressure_bv(..., density_ref=, cavity=)` (the
  `use_density_ref` subtraction, the density fill of the shelf levels, the cavity `hpressure(nzmin)`); both
  arguments default to today's path.
* `fesom_jax/io_dump.py`: substeps 17-20 and the oracle readers.
* `scripts/core3/replay_cavity_melt.py`, `scripts/core3/replay_pressure_bv.py`: the replay rungs (oracle inputs of
  the same step -> fesom_jax kernel -> oracle output), job `ocn/jobs/replay.sbatch` (`DUMPS`, `PREC`, `WHAT`).

### Verdicts, task 3

| check | job | result |
|---|---|---|
| melt + shelf drag replay, single precision, vs SP oracle `run_cav2`, steps 1-2 | 28696654, 28697537 | heat flux bitwise at 99.4 % of the 10,047 cavity nodes, water flux 99.7 %, max diff 1.1e-7 of max |flux| (3.3e-5 W/m^2); stress_surf / stress_node_surf bitwise everywhere; no sea ice under the shelves in the oracle; `CAVITY_MELT_REPLAY_OK`. (The first job printed FAIL on a pointwise relative criterion, which blows up where `tin - tf` cancels; the criterion is now max diff / max |flux| <= 1e-6.) |
| melt + shelf drag replay, double precision, vs DP oracle `run_cav_dp`, steps 1-2 | 28697533 | heat flux max 1.1e-16 of max |flux| (bitwise 99.6 %), water flux 1.1e-16 (99.7 %), drag bitwise. A first DP comparison (28697305) was off by 7e-8: I had emulated bare Fortran literals as float32, but FESOM builds with `-r8`; fixed in `cavity.py` |
| `eos.reference_density` + `pressure_bv(density_ref=, cavity=True)` vs DP oracle, steps 1-2 | 28697498 | `density_ref` bitwise everywhere; density 2.4e-14 of its max (incl. the 152,242 shelf-level fill values, max 4.5e-13 kg/m^3); hpressure 2.5e-14 at cavity nodes (the cavity `hpressure(nzmin)` branch), 3.5e-14 open |
| the same, single precision vs SP oracle | 28697161 | density_ref 2.3e-7, density 2.5e-5, pressure 5.4e-5 of max (float32 + geometry; ~15-20 % bitwise) |
| N^2 (`bvfreq`, substep 1), double | 28697498 -> fix -> 28697820 | first 2.7e-3 of max at every node: my `pressure_bv` change divided N^2 by `density_ref` instead of `density_0` (F90:295 keeps `density_0`); a numpy re-derivation of FESOM's N^2 + `smooth_nod3D` matched the oracle to 1.2e-13 and isolated it. Fixed: N^2 1.2e-13 of max (interior 6e-16 abs), steps 1-2. **Substep 1 with cavities is green in double precision** |
| the same, single | 28697821 | N^2 6.6e-5 of max, density 2.5e-5, pressure 5.5e-5 (float32 + the SP geometry) |
| default path after the fix | 28697822 | 22 passed |
| default path unchanged: `test_eos.py`, `test_dp_lock.py`, `test_prepare_mesh.py` | 28696826 | 22 passed |
| bundle `area`, `areasvol`, `mesh_resolution`, `elem_area` vs the DP oracle's own `mesh%...` | run_cav_dp | rel 2.0e-12, 2.0e-12, 3.7e-13, 2.2e-12; zero outside the wet levels in both |
| SP geometry: oracle `mesh%elem_area`/`area`/`areasvol` vs the bundle | run_cav3 (28696663) | elem_area rel median 1.8e-5, p99 1.8e-4, max 8.7e-4: FESOM SP derives the geometry in float32, fesom_jax SP rounds the float64 geometry (max 6e-8). The cavity `areasvol` rule itself agrees: the same 11,055 (node, level) values differ from `area` in both |

## Session 2 (2026-11-03): restart import, the ladder, cavities wired into the step

### Task 4: the 1855-01-01 restart

The converter of branch `fortran-restart` (9cb9a02..c95e476, cherry-picked; the PORTING_LESSONS append conflict
resolved by keeping both sections) now also reads FESOM2 main restarts: `d_eta` (CG warm start), the mEVP stresses
`sigma11/12/22` and the Newton seed `t_skin` [degC] are taken from the restart when present (a FESOM 2.7 restart
converts as before); optional State fields stay `None`; `convert.py --rest-reference` uses a rest state with
T = S = 0 as the reference (CORE3 has no cached PHC IC; every wet lane comes from the restart, dry and shelf lanes
0 as FESOM stores them). density / hpressure / bvfreq stay 0, as FESOM does not restart `density_m_rho0` either:
the oracle's first step after the restart reads rho = 0 + density_ref in the melt (checked: its `pre_fluxes`
density at step 1 is 0 everywhere).

```bash
python scripts/fortran_restart/convert.py --oce $W/fesom.1855-01-01.oce.restart --ice $W/fesom.1855-01-01.ice.restart \
   --mesh-dir $OCN/mesh_core3 --rest-reference --date 1958-01-01 --start-year 1958 --dt 1800 --out <new dir>
python scripts/core3/check_restart.py <new dir> $OCN/mesh_core3 $W/fesom.1855-01-01.oce.restart $W/fesom.1855-01-01.ice.restart [oracle dumps]
```

Use **`/work/ba1583/a270088/awicm/core3/ocn/restart_jaxcpl_18550101_as_1958_v2`** (standalone JRA55 1958, step 0;
the oracle's setup) and `..._as_1855_v2` (the same state dated 1855-01-01 for the coupled run). Job 28698817:
`RESTART_CHECK_OK`, every restarted field bitwise on its wet lanes (T, S, T_old, S_old, hnode, w, w_e, tke, eta_n,
hbar, d_eta, ssh_rhs_old, a_ice, m_ice, m_snow, u_ice, v_ice, t_skin, sigma11/12/22, uv, uv_rhsAB). Superseded,
not to be used: `..._as_1958`, `..._as_1855` (job 28698495: density recomputed, which FESOM does not do) and
`..._as_1958_v3`, `..._as_1855_v3` (job 28699073: labelled with the previous year to dodge the AB2 bootstrap;
that needs 1957 forcing, which JRA55-do does not have). `run.py` now never re-bootstraps AB2 at the first step
of a restart (`plan_chunks(restarted=True)`; before, a restart at step 0 was taken for a cold start and its first
step ran as the AB2 first step).

### Task 2: the replay ladder (`scripts/core3/replay_ladder.py`, job `ocn/jobs/ladder.sbatch`)

One fesom_jax kernel per substep, its inputs from the double-precision oracle's full dumps of the same step (and
the converted restart for what FESOM carries but does not dump), its output vs the oracle's dump. Step 1, cavity
nodes/elements, the cavity front (open, touching a cavity cell) and the rest; max |diff| / max |field|:

| rung | kernel | job | cavity | front | open |
|---|---|---|---|---|---|
| 1 | `eos.compute_pressure_bv(density_ref, cavity)` (density, pressure, N^2) | 28697820 | 2.4e-14 / 2.5e-14 / 6.7e-14 | 1e-13 | 1.2e-13 |
| 3 | `pgf.pressure_force_linfs` | 28698897 | 3.9e-13 | 3.9e-14 | 3.9e-13 |
| 4 | `tke.mixing_tke(cavity=True)` + `mo_convect`: Kv / Av | 28698897 | 7.8e-15 / bitwise | 1.1e-15 / bitwise | 5.4e-14 / bitwise |
| 5 | `momentum.compute_vel_rhs(cavity=True)` | 28698812 | 1.0e-12 (was 4.4e-4) | 7.6e-15 (was 4.5e-4) | 4.4e-14 |
| 7 | `momentum.impl_vert_visc` with the shelf drag | 28698897 | 3.7e-16 | 3.4e-17 | 8.9e-16 |
| 8 | `ssh.compute_ssh_rhs` | 28698897 | 5.2e-14 | 5.3e-14 | 5.4e-14 |
| 9 | `ssh.solve_ssh` (CG, soltol 1e-5) | 28698897 | 2.5e-5 (was 0.58) | 3.9e-6 | 2.8e-5: both solutions have residual 9.9e-6 / 9.5e-6 under fesom_jax's matrix, i.e. CG tolerance |
| 10 | `momentum.update_vel` | 28698897 | 1.5e-12 | 8e-16 | 5e-15 |
| 11 | `ssh.compute_hbar` (hbar, ssh_rhs_old) | 28698897 | bitwise / 2e-11 | 2e-12 / 1.7e-11 | 4e-12 / 2.8e-10 |
| 12 | `ssh.eta_n_update` | 28698897 | bitwise | bitwise | bitwise |
| 13 | `ale.compute_w` | 28698897 | 6e-13 | 2e-13 | 6e-13 |
| 16 | linfs thickness (hnode, helem) | 28698897 | bitwise | bitwise | bitwise |
| 17-19 | melt, shelf drag, no ice under the shelves (`replay_cavity_melt.py`) | 28697533 | 1.1e-16 / bitwise | | |

What the ladder found in EXISTING kernels (each a cavity case the C port never ran, fixed here):
* **SSH stiffness matrix** (`ssh.build_ssh_operator`): the element depth was `zbar[nlevels-1] - zbar[0]`; FESOM's is
  `zbar_e_bot - zbar_e_srf` with `zbar_e_srf = zbar(ulevels(el))` (oce_ale.F90:1362). Under a shelf the column
  started at the sea surface: the oracle's `d_eta` had a residual of 0.26 under fesom_jax's matrix. Off the shelves
  `ulevels = 1`, so the default matrix is unchanged.
* **Momentum advection** (`momentum.momentum_adv_scalar(cavity=True)`): FESOM adds a cell's top interface
  velocity only for open cells (`if (ule==1)`); fesom_jax took half of it from the half-sum with the dry level
  above (4.4e-4 at cavity and front cells).
* **TKE under a shelf** (`tke.mixing_tke(cavity=True)`): the column core assumes its surface at k = 0; FESOM's CVMix
  driver hands it the column from the top wet level. The driver now shifts every column up by `ulevels_nod2D - 1`
  and back, and caps `dzt` with half the top wet layer (it took `hnode[:, 0]`, which is 0 under a shelf).
* **AB2 at a step-0 restart** (`run.plan_chunks`), above.

Not yet on the ladder: rung 2 (sw_alpha/beta: only GM/KPP read it), 6 (fesom_jax has no `opt_visc = 5`: lane
L-SFC), 14 (GM/Redi with jaxcpl's scalings: L-SFC), 15 (tracers: advection, the implicit diffusion with the melt as
the surface BC at the shelf base, GM/Redi), and the forcing / ice substeps 17-18 as kernels (their cavity masks are
checked in the oracle: no ice and zero atmospheric forcing under the shelves). Rungs at step 1 only: later steps
need the AB2 slot, `w_expl` and the CG warm start from the previous step, which the oracle does not dump.

### Task 3: the wiring (all behind `cavity:`; `None` = today's program)

* `run_config.RunConfig.cavity: Optional[CavityConfig]` (`gamma_scale`, `density_ref_T/S`), in `physics_kwargs`
  only when set (like `recom`); `validate()` refuses cavities with zstar (the zstar branches are not ported) and
  without the prognostic sea ice.
* `run.py`: `cavity.with_cavity_static` (the virtual-salt integral weights each node by `areasvol` at its top wet
  level, as FESOM's `integrate_nod`) and `cavity.cavity_boundary_node_mask` (the GLOBAL ice boundary incl. the shelf
  front). `integrate_sharded.run_steps_sharded_forced(cavity_cfg=)` passes it to `step`; the executable-cache key
  gets it only when set.
* `step.step(cavity_cfg=)`: `eos` with `density_ref` and the cavity branch; `compute_vel_rhs(cavity=True)`;
  `mixing_tke(cavity=True)`; `ice_step.ice_surface_step(cavity_cfg=)`: ice velocity and ice cleared under the shelves
  after the EVP / at the end of the ice step, the three-equation melt overwriting `-flx_h`/`-flx_fw` at cavity nodes
  before the virtual salt (which then reads salt at the top wet level there), the shelf-base drag after the stress
  blend. The melt reads the density the last `pressure_bv` left (`state.density`, relative to `density_ref`), as
  FESOM, where `oce_fluxes` runs before `oce_timestep_ale`.
* Inventory rows now closed: mesh, density_ref, pressure_bv, melt, shelf drag, ice cleaning, mEVP shelf-front BC,
  virtual-salt integral, TKE top boundary, SSH matrix, momentum advection. Still open: `relax_salt` zeroed under the
  shelves before its mean (zero in jaxcpl: `surf_relax_s = 0`, so not done), the linfs freshwater balance (FESOM
  main balances `water_flux` also under linfs, fesom_jax does not; under linfs nothing downstream reads the balanced
  `water_flux`, the virtual salt is formed before it), the coupled ice path (`coupled/flux.ice_surface_step_cpl` has
  no `cavity_cfg`), the sharded mesh (`cavity.py` reads mesh-local arrays only; the shelf-front mask must be built
  globally, which `run.py` does), single precision, and the PHC cold start under the shelves.

### Whole-step checks

* CPU, 4 steps from the restart (job 28698898): finite; T -2.31...32.86, max|uv| 2.35. RSS 31 GB, 1 min 45 s.
* One step vs the double-precision oracle (job 28699335, `scripts/core3/compare_step.py`), max |diff| / max |field|
  (p99): T cavity 5.8e-5 (1.0e-5), front 5.5e-4 (4.8e-5), open 3.6e-3 (7.7e-6); u cavity 1.8e-3, open 1.4e-2;
  eta_n cavity bitwise (rigid lid), open 8.8e-3; a_ice front 8.0e-2 (p99 3.9e-3), open 4.3e-3. This mixes the
  substeps fesom_jax does not have in jaxcpl's form yet (opt_visc 5, GM/Redi scalings, tracer scheme: L-SFC), so
  it is a sanity rung; the cavity nodes are not worse than the open ocean. The front `a_ice` maximum is the one
  number to look at next (the mEVP shelf-front boundary or the front's thermodynamics).
* GPU (task 4 of the session): see below.

### Default path (cavity off) after session 2

`pytest` on the branch, CPU: `test_eos test_dp_lock test_prepare_mesh test_fortran_restart test_run_config
test_ice_step test_tke_replay test_step_pi` 147 passed (job 28699466); `test_core2_step` 6 passed (28700115);
`test_momentum test_ssh test_ssh_ell test_run_entry test_release test_ale` 148 passed (28700740; `test_run_entry`
alone takes 26 min). The branch is ready for the coordinator's coupled-CORE2 bitwise gate vs 5b88770.

### Estimate for the rest (after session 2)

* Rung 15 (tracers with the melt as the surface BC at the shelf base, MFCT/QR4C advection at the shelf base) and
  rung 14 (GM/Redi between `ulevels_nod2D_max` and `nlevels_nod2D_min`): 1-2 days, partly waiting for L-SFC's
  jaxcpl settings (the tracer scheme, GM scalings) so that the rung isolates the cavity part.
* The ice substeps as kernels (rung 18) and the shelf-front `a_ice` difference of the whole-step check: 0.5-1 day.
* The coupled ice path (`coupled/flux.ice_surface_step_cpl(cavity_cfg)`), the coupled CORE3 config, the sharded
  (dist_N) cavity run: 1-2 days.
* Single precision with cavities (with L-SFC's SP linfs): 0.5-1 day once that exists.
Total ~3-6 working days.

## Next session, first steps

1. Rung 15 (tracers) and rung 14 (GM/Redi) in `replay_ladder.py`, against `run_cav_dp`.
2. The shelf-front `a_ice` difference of the whole-step check (rung 18: FCT advection / thermodynamics at front
   nodes).
3. `coupled/flux.ice_surface_step_cpl` with `cavity_cfg`, the coupled CORE3 config (restart `..._as_1855_v2`; the
   OASIS weights are L-CPL's), a dist_4 run.
