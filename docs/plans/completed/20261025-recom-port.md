# REcoM biogeochemistry in fesom-jax

## Overview

Port the REcoM ocean biogeochemistry model (FESOM2's `src/recom` submodule) into fesom-jax so a CORE2
hindcast with biogeochemistry runs on the fast GPU stack, reproduces FESOM2-REcoM, and stays
differentiable for later BGC parameter calibration.

- **Purpose (user, 2026-10-25):** fast forward hindcasts first; the code stays AD-friendly (fixed-iteration
  solvers, no host-side branching, no Fortran-style hidden state) so calibration can follow without a rewrite.
- **Configuration:** the REcoM switch set of upstream `config/bin_2p3z2d` (AWI's CORE2 REcoM setup: 2 phyto,
  3 zoo, 2 detritus = `enable_3zoo2det`, no coccos, 31 BGC tracers = IDs 1001-1030 + 1037 DICremin,
  ballasting with density and viscosity scaling, Albani monthly dust, `constant_CO2` 278 ppm, alkalinity
  restoring, `biostep 1`, `Nmocsy 1`). It is the ecosystem of `setups/test_pi_recom`, which we use on the pi
  mesh for verification with the `bin_2p3z2d` namelists. Every other REcoM switch raises `NotImplementedError`.
  Then CORE2 as fast as the inputs arrive.
- **Sea-level pressure (user, 2026-10-25): port FESOM2's `l_mslp` fully.** REcoM's CO2 and O2 solubility read
  `press_air`, which is zero unless `l_mslp` is on (the upstream pi test has it off; REcoM's own comment,
  `recom_main.F90:~326-336` at 692f804, says a zero pressure outgasses CO2 and O2 without limit).
  `bin_2p3z2d` sets `l_mslp = .true.`, which also adds the SLP gradient to the momentum equation
  (`oce_ale_vel_rhs.F90:202`). fesom-jax gets the JRA55-do `psl` reader and that momentum term; REcoM runs
  require it on. This is a FESOM2 option, not a deviation, but a REcoM run's physics differs from our
  physics-only runs by that term.
- **Precision:** double precision first. Upstream FESOM2 refuses `USE_SINGLE_PRECISION` with `RECOM_COUPLED`
  (FESOM2 `src/CMakeLists.txt:124`: REcoM hardcodes its own `WP = 8`). SP with REcoM is a later, stated,
  opt-in deviation, decided by the user with numbers (STOP 4).
- **Why it fits JAX:** most of the ~180 `#if __recom` lines in FESOM2 are `__recom && __usetp`, the
  tracer-group MPI mode (`num_fesom_groups`), which JAX does not need (the tracer axis is a batch axis).
  What remains: (1) `recom()` once per step before `oce_timestep_ale`, column-local except one global mean;
  (2) the BGC tracers in the same tracer loop as T and S, with the same velocities, Kv, GM/Redi and ALE
  thicknesses, plus REcoM's benthic return flux, sinking and benthos inside that loop; (3) the SLP forcing
  above. No physics reads a BGC tracer (shortwave penetration uses a fixed chlorophyll).

## Context (from discovery)

- **Base:** branch `recom/port` from `perf/core2-adj-fast` @ d44dd08 (contains all of `perf/core2-gpu`, the
  ~900-1000 SYPD CORE2 forward stack, plus the fast sharded adjoint `--fast ice2+se+sp`, `--devices 4`).
  Worktree `/work/ab0995/a270088/fesom_jax_dev/wt_recom` (data symlink in place). The Fram session still
  commits to `perf/core2-adj-fast`; merge its later tips only when needed. Nothing is pushed.
- **REcoM source:** upstream `github.com/RECOM-Regulated-Ecosystem-Model/REcoM` at **692f804**, the commit
  FESOM2 main e96a2b13 pins (`git ls-tree HEAD src/recom`), cloned read-only at
  `/work/ab0995/a270088/REcoM_src` (checked out at 692f804). It is ~20 commits past 28760bf (the pin of the
  older local checkouts), among them d7f2b4e (atmospheric CO2 source for standalone runs: at 28760bf a
  standalone run read `x_co2atm`, zeros outside `__oifs`), d24e583, 192fbf4 (benthic seafloor area), 7d2fcea
  (`tiny_Si`), 70555a0 (surface values at nzmin). Sizes: `recom_sms.F90` ~4.1k lines, `recom_sms_update.F90`
  ~3.6k, `recom_sms_init.F90` ~1k, `recom_modules.F90` ~3.3k, `recom_sinking.F90` ~1.2k, `recom_init.F90`
  ~0.9k, `recom_main.F90` ~0.8k, `recom_forcing.F90` ~0.5k, `recom_carbonate_chemistry.F90`, `mocsy/`.
- **FESOM2 hook points** (upstream main checkout `/work/ab0995/a270088/fesom2_sp_zstar`, e96a2b13):
  - `src/fesom_module.F90:1203` `call recom(...)`: after the forcing update, the ice step and the ocean flux
    coupling, before `oce_timestep_ale`. Reads `a_ice`, `hnode`, `z_3d_n`, `zbar_3d_n`, geo coordinates,
    `areasvol`, all tracers, `dt`, the calendar (`daynew`, `month`, `mstep`, `ndpyr`, `yearold`, `timenew`),
    `rad`, `kappa`, `press_air`, `u_wind`, `v_wind`, `shortwave`. Produces the updated tracers and the
    module-global surface fluxes (`GloCO2flux_seaicemask`, `GloO2flux_seaicemask`, `AtmFeInput`,
    `AtmNInput`, `relax_alk`), `GlodecayBenthos`, `GloHplus`.
  - `src/oce_ale_tracer.F90:729-930`, per BGC tracer in tracer-ID order, AFTER advection and explicit Redi
    and BEFORE the T* update (`:861-866`) and the implicit vertical diffusion (`:914`):
    `diff_ver_recom_expl` (`dtr_bf`, the benthic return flux into the remin tracers 1001/1002/1003/1018/
    1019/1022, from `GlodecayBenthos`), then for the sinking tracers `get_seawater_viscosity`,
    `get_particle_density` (from the CURRENT detritus tracers), `ballast`, `ver_sinking_recom`
    (`vert_sink`) and `ver_sinking_recom_benthos` (`str_bf`, accumulating `Benthos`). All three are added
    to `values`, so they enter the tracer as `Δ·hnode/hnode_new` and are then vertically diffused. The
    salinity clip runs after the whole tracer loop.
  - `src/oce_ale_tracer.F90:2186-2242` `bc_surface`: 1001 DIN (`AtmNInput`), 1002 DIC
    (`GloCO2flux_seaicemask`), 1003 Alk (`relax_alk`), 1018 DSi, 1019 Fe (`AtmFeInput`), 1022 O2
    (`GloO2flux_seaicemask`); the `virtual_*` terms are zero under zstar and no BGC case carries the
    `-sval·water_flux` term; the other IDs have zero surface flux.
  - `src/gen_surface_forcing.F90:1639,1897` `sbc_ini_recom` / `sbc_do_recom`: the dust is a monthly HOLD
    (no time interpolation), switched on the last step of the month (`if (mstep>1) i=i+1`); with
    `constant_CO2` the CO2 file is not read.
  - `src/oce_setup_step.F90:1272`: `Alk_surf` (the restoring target) = the IC surface Alk, set before the
    restart read. `src/io_restart.F90:402`: `HPlus` (`GloHplus` = the surface pH, `recom_main.F90:547`) is a
    restart field, but REcoM never reads it back into the chemistry (`Hplus = GloHplus(n)` at :356 is unused;
    mocsy starts from its own guess): output only.
  - `src/oce_ale_vel_rhs.F90:202` (`l_mslp`): the SLP-gradient momentum term; `gen_forcing_couple.F90:680`
    fills `press_air`.
  - Out of scope: `ice_oce_coupling.F90:444` virtual tracer fluxes (linfs only), `oce_ale.F90:4325` cavity
    NaN scrub (no cavities on CORE2).
- **REcoM internals to mirror exactly:**
  - the interior carbonate system (mocsy `vars_sprac`) is a CACHED state refreshed per LEVEL inside the
    layer loop (`recom_sms.F90:489`): at `mstep == 1`, every 7 days where that layer's
    `PARave > 0.01*SurfSR`, every 30 days where `PARave < 0.01*SurfSR` (periods from `dt_b`; both tests
    strict, so at night, `SurfSR = PARave = 0`, no layer refreshes; corrected by lane M, verified). The surface `flxco2` (`recom_forcing.F90:234`) runs every step.
  - `mstep = n` restarts at 1 in every run segment (`fesom_module.F90:849,983,1049`), so the cache refresh
    and the dust month switch depend on where the run was restarted; the carbonate cache is not in the
    Fortran restart. A Fortran run split at a restart is NOT the unsplit run; we mirror that.
  - `state = max(tiny, state + sms)` plus per-species floors (`recom_forcing.F90:316-341`): not
    conservative; the budget must carry a clip term.
  - the DICremin surface reset (`recom_main.F90:300-310`); the IC Fe ×1e9 conversion and the
    hydrothermal-vent Fe cap (`recom_init.F90:118-123, 706-750`); the `relax_alk` global-mean subtraction
    (`recom_main.F90:81-84`, a global reduction).
  - the pH solver stops per point at `|Δ ln h| < 1e-8` (`mocsy/phsolvers.F90:23,612`).
  - `tra_recom_sms` exists only for `ltra_diag` tracers (`recom_main.F90:490`); gates compare the updated
    tracers instead.
  - `biostep = 1`: the `any(abs(sms) <= tiny)` reset at the sub-step start is a no-op because `sms` is zeroed
    just before (`recom_sms.F90:259`); assert `biostep == 1` instead of porting it.
- **Existing upstream material:** FESOM2 branch `origin/recom4-dumps` (2026-10-16, AWI's REcoM developer,
  "RECOM4 m1") instruments `recom()`, `diff_tracers_ale` and `bc_surface` for single columns; a guide for
  Task 4, which needs full fields. `config/bin_2p3z2d` holds the CORE2 REcoM namelists; its input paths are on
  AWI's Albedo machine (`/albedo/work/projects/p_pool_recom/input/mesh_CORE2_finaltopo_mean/`).
- **fesom-jax patterns this follows:**
  - `fesom_jax/state.py`: the State pytree is FLAT and ENTITY-LEADING (`integrate_sharded._fold`,
    `folded_state`, `zarr_output._all_state_fields` rely on axis 0 = node). Optional groups are None unless
    enabled: `SE_FIELDS` (state.py:231), `ICE_AVG_FIELDS` (state.py:226). REcoM adds a `BGC_FIELDS` group
    registered as optional from the start (`_all_state_fields` exclusion, `read_restart`
    zarr_output.py:~368, `core2_adj.py:~343`), so existing restarts keep loading.
  - `fesom_jax/step.py:563-640`: the T/S stage (`tracer_adv.advect_one_fct`, Redi
    `gm_redi.diff_ver_part_redi_expl`/`diff_part_hor_redi`, K33, `tracer_diff.impl_vert_diff`, `_exch`,
    `S_FLOOR` at :634). Stage 3 goes after it behind `recom_cfg is not None` (the `tke_cfg`/ice precedent:
    a dead branch when off ⇒ byte-identical program), and reads S before the floor (Fortran clips after
    the loop). The increments enter through the content form (`advect_one_fct(content=True)`,
    tracer_adv.py:~695).
  - `fesom_jax/surface_forcing.py:88` `StepForcing` (no `psl`); `_fold_forcing` (integrate_sharded.py:~189)
    treats a None field as a replicated scalar, so the BGC per-step inputs travel in their own optional
    structure.
  - `fesom_jax/run_config.py:161` `RunConfig` optional sub-configs ⇒ `recom: Optional[RecomConfig] = None`.
  - `fesom_jax/zarr_output.py:251` archival restart; `fesom_jax/io_dump.py` dump readers; the Fortran dump
    style of `fortran_ref` 9271ae92 (env-gated, no-op otherwise).
  - fesom-jax implements the tracer advection weights `0., 1.` only (`config.py:144-160`); the upstream pi
    test lists `1., 1.`, `bin_2p3z2d` lists `0., 1.`.
- **Speed baseline** (`docs/PERFORMANCE_CORE2_GPU.md`, job 28303935, 4 GH200, SP, dt 2700): 9.07 ms/step with
  the CG SSH, 7.43 ms with split-explicit; tracer advection of T and S 0.96 ms, GM/Redi 0.40 ms. 31 tracers
  one at a time like T and S would add ~15 ms/step.

## Development Approach

- **Testing approach: gate-first (TDD), user's choice.** Each port task first writes its gate test against
  the Fortran dump (it fails), then ports the code until it passes at the stated tolerance.
- Complete each task before the next; small commits on `recom/port`; English commit messages; stage files
  explicitly (never `git add <dir>`; check `git diff --cached --name-status` for stray `A` lines).
- **Every task includes new or updated tests**, success and error cases; all tests pass before the next task.
- **REcoM off and `l_mslp` off ⇒ byte-identical physics.** `fesom_jax/tests/test_dp_lock.py` and the
  identity gates stay green after every task touching `step.py`, `state.py`, `run_config.py`, the forcing or
  the integrators.
- Tests on a compute node (`scripts/runs/run_suite.sbatch` / `run_suite_parallel.sbatch`, `-p compute`,
  `--hint=nomultithread`); pi-size tests are fine on the login node, CORE2 never.
- One lesson per task in `docs/PORTING_LESSONS.md`.
- **Mirror FESOM2-REcoM exactly**, quirks included; a forced deviation is written down in the task and the
  lessons, never decided silently.
- ➕ new tasks, ⚠️ blockers; keep the plan in sync.
- **Jobs:** submit compute/GPU jobs without asking (report ids and cost). dolpung: at most 2 nodes in total,
  counting fesom3 `fjx_*` jobs; `squeue` + `--dependency` before every dolpung submit. After each submit,
  a background shell watcher (`sacct` every ~10 min, exits on any terminal state). No deletions, no
  `scancel`, no push without an explicit yes.
- **P0 (GPU) and P1 (CPU) run in parallel.**

## Testing Strategy

- **Unit tests** for every new function (`fesom_jax/tests/test_recom_*.py`, `test_mslp_*.py`), CPU, DP.
- **Gates against the Fortran reference** (Tasks 3-4), pytest-marked `recom_gate`, skipping with a clear
  message when the dump directory is absent:

  | gate | what | tolerance (DP, starting value) |
  |---|---|---|
  | GP | `l_mslp`: `press_air` on nodes and the SLP momentum term vs the Fortran dump | rel 1e-12 |
  | G1 | `carbonate.py` vs a standalone Fortran mocsy driver over a (T, S, DIC, Alk, Si, PO4, p) grid, pH solver included | rel 1e-12 |
  | G2 | stage 1 from the dumped `recom()` inputs: updated tracers, surface fluxes, `GlodecayBenthos`, `GloHplus`, carbonate cache, the REcoM 3-D diagnostics (NPP, GPP, grazing, ...) at step 1, a refresh step, lit/dark and ice-covered columns | rel 1e-11 of each field's max |
  | G3 | `dtr_bf`, ballasting densities, `vert_sink`, `str_bf`, `Benthos` from the dumped pre-sinking state, in Fortran tracer order | rel 1e-12 |
  | G4a | stage 3 fed with Fortran's own transport fields | rel 1e-11 |
  | G4b | one full JAX step vs Fortran for the BGC tracers, where the pi physics matches | the physics gate's floor |
  | G5 | 30-day pi run vs Fortran: inventories (DIC, Alk, DIN, DSi, DFe, O2), CO2 flux, NPP; a closed JAX budget (inventory change = surface fluxes + benthos exchange + clip term) | budget rel 1e-12; inventories: set after measuring the physics-only T/S divergence of the same JAX/Fortran pair |
  | G6 | `jax.grad` of a 1-day NPP cost w.r.t. 2-3 REcoM parameters vs central FD | finite; rel 1e-4 |

  A gate needing a looser tolerance gets the reason (reassociation, iterative solver) in the test.
- No UI, no e2e tests.

## Progress Tracking

- `[x]` immediately when done; ➕ discovered tasks; ⚠️ blockers; job ids and numbers next to the item.

## What Goes Where

- **Implementation Steps** (`[ ]`): code, tests, benchmarks, Fortran instrumentation in our own clone, docs.
- **Post-Completion** (no checkboxes): things that need the user or colleagues.

## Implementation Steps

### Phase P0: speed ceiling (cost ~1 GPU node-hour; in parallel with P1)

### Task 1: Optional BGC fields + batched passive-tracer transport

**Files:**
- Create: `fesom_jax/recom/__init__.py`, `fesom_jax/recom/transport.py`
- Modify: `fesom_jax/state.py` (optional `BGC_FIELDS`), `fesom_jax/zarr_output.py`, `fesom_jax/core2_adj.py`
  (optional-field handling only)
- Modify: `fesom_jax/step.py` (stage-3 call after the T/S stage, behind `recom_cfg is not None`)
- Modify: `fesom_jax/run_config.py` (`recom: Optional[RecomConfig]`; a `passive_tracers: int` debug key)
- Create: `configs/pi_recom.yaml` (forced pi, the physics of the Fortran reference)
- Create: `fesom_jax/tests/test_recom_transport.py`

- [x] tests first: (i) a stack of copies of S, given S's surface BC, reproduces `S_new` after one forced pi
      step (DP, CPU; bitwise for `loop`, a stated ulp bound for `vmap` if XLA reassociates); (ii) pi
      `dist_4` sharded equals serial; (iii) REcoM off ⇒ `test_dp_lock.py` unchanged; (iv) a restart written
      without BGC fields still loads (REcoM off) and one with them loads (REcoM on)
- [x] `BGC_FIELDS` registered optional (None unless REcoM is on) in State, the restart writer/reader and
      `core2_adj.load_state`
- [x] `transport.py`: batched FCT advection + explicit Redi for a `(nod2D, nl, ntr)` stack (reusing
      `advect_one_fct`, `gm_redi.*`), then T* and the single-tracer core of `impl_vert_diff` (no sw_3d);
      two layouts behind `gpu.bgc_layout`: `loop` (the reference) and `vmap` over axis 2 (halo collectives
      batch into one per FCT stage)
- [x] check `_exch(x, "nod")` and the padded/coloured transports with a trailing axis (fix + test if not)
      ➕ result (`test_recom_transport.py`, 10 tests): forced pi, 3 steps dt 2700, zstar + w_split + mEVP + TKE +
      GM: `loop` bitwise S dense and on FESOM2's pi `dist_4` (default all_gather halo); `vmap` within 4 ulp dense,
      10/158400 values off by ≤ 1.2e-15 rel sharded (`VMAP_RTOL` 1e-14); physics bitwise with the stage on;
      restart round trip with and without BGC fields; sharded with the default all_gather AND the padded halo.
- [x] run tests (CPU suite on a compute node) - must pass before Task 2 (job 28417154: DP lock + identity gates +
      REcoM 19 passed; the 4 sharded variants on 4 host devices passed separately)

### Task 2: Measure the transport ceiling on the fast stack

**Files:**
- Create: `configs/core2_gpu_fast_bgc31.yaml`, `scripts/recom/bench_bgc_transport.sbatch`
- Modify: `docs/PERFORMANCE_CORE2_GPU.md` (section: BGC transport)

- [x] 31 passive tracers on CORE2 from the v3 1990 restart, 4 GH200, dt 2700: steady ms/step for `loop`
      and `vmap`, in DP; plus an SP COST estimate with the stack cast to f32 (benchmark only, not the
      value-faithful SP path of Task 16)
- [x] one nsys profile of the faster layout: transport ms/step, NCCL count, launches
- [x] ledger rows with job ids; the SYPD range with REcoM = measured transport + an estimate for the
      sources/sinks (stated as an estimate; Task 15 measures it)
- [x] pick the layout (into the plan with the numbers); a tracer-minor layout only if Task 15 shows a need
      ➕ `vmap` (default) with ordered sums: SP + 31 47.3 ms/step = 156 SYPD (estimate), DP + 31 67.2 = 110 SYPD,
      against 7.6 / 16.7 without BGC; `loop` 3.9 ms/tracer vs `vmap` 2.2 in DP; cost linear in the tracer count;
      nsys: scatters 34 + advection 28 + Redi 9 + NCCL 8 ms of 88 (DP, ordered sums off). Jobs 28417245,
      28417424 (profile), 28417425 (A100), 28417702; ledger section "REcoM: the BGC tracer stage"
- [x] ⛔ **STOP 1: report the ceiling to the user** (continue? target?) ➕ done 2026-10-25: user "continue faithful port"

### Task 2b: The BGC tracers every k-th physics step (user 2026-10-25: "make it possible early")

A stated deviation, opt-in `recom.bgc_every: k` (default 1 = FESOM2, the every-step program unchanged). User's
choice: the k-step MEAN transport (the `ice_ave_steps` idea for the BGC tracers), not a snapshot and not sources/sinks
only. Expected from Task 2: SP + 31 at k = 2 ~270 SYPD, k = 4 ~420 (transport only).

**Files:**
- Modify: `fesom_jax/state.py` (`BGC_MR_FIELDS`: `bgc_steps`, `bgc_uvh`, `bgc_we`, `bgc_wi`, `bgc_kv`, `bgc_bc`,
  `bgc_h0`), `fesom_jax/recom/config.py` (`bgc_every`), `fesom_jax/recom/transport.py` (`init_every_k`,
  `bgc_every_k`, `helem_flux`), `fesom_jax/step.py`, `fesom_jax/run.py`
- Modify: `fesom_jax/tests/test_recom_transport.py`

- [x] each physics step adds `uv_adv·helem` (the FCT sees the velocity only through the edge volume flux, so the
      summed product is exactly the mean mass transport), `w_e_adv`, `w_i`, the K33-augmented `Kv` and the surface BC
      to interval sums; the k-th step runs the BGC stage once with `k·dt` on the means, from `bgc_h0` (its zstar
      geometry, its vertex-mean helem) to `hnode_new`; Redi uses the k-th step's slopes/Ki (stated); `lax.cond` on the
      shared counter (the `ice_ave_steps` pattern; a static trace-time schedule like `gpu.ice_static` if the GPU host
      sync shows)
- [x] tests: k = 1 strips the sums and is the old program; hold-then-step pattern; salinity copies hold S's inventory
      exactly at the interval ends; a uniform tracer stays uniform (k = 1 and 2); mid-interval restart round trip
- [x] GPU: SP and DP + 31 at k = 2 and 4 (ms/step), and the k-step scheme's error over a month (copies of S vs S)
- [x] sources/sinks: stage 1 runs on the BGC steps with `dt_bgc = k·dt` (lane S takes `dt_bgc`; wiring in Task 10)
      ➕ lane W (`bgc_every_k(full=...)`): stage 1 + stage 3 on the k-th step on the mean transport from the interval-start
      thickness; stage 1 reads T/S of that step's start, `mstep` counts BGC steps; budget closed at k = 2 (4.6e-15)
- ➕ results (742d5c7, test fix 35c1608): CPU pi — salinity copies keep S's inventory to 1e-12 at k = 2 (max |X−S|
  0.031 psu after two intervals, rms 3e-4); a uniform tracer under linfs stays exactly 1 at k = 1 and 2 (under zstar
  it dilutes with the freshwater input, its content is constant); sharded k = 2 vs dense k = 2 differs by 2.9e-2 =
  the physics' own sharded-vs-dense S difference (3.0e-2 after 4 steps, ⚠️ below). GH200: SP 156 / 276 / 430 SYPD
  at k = 1 / 2 / 4, DP 110 / 168 / 227 (ledger). Open: the k-step scheme's error over a month on CORE2.
- ⚠️ finding outside REcoM: on pi `dist_4` with zstar + w_split + mEVP + TKE + GM, 4 forced steps, the SHARDED physics
  differs from the dense one by max |ΔS| = 0.030 psu (`test_every_k_sharded_equals_dense` prints it), not bitwise
  as the post-af5429a CORE2 note suggests; which module is untested (w_split a suspect: the fast CORE2 stack runs
  without it). Reported to the user; not chased here.

### Phase P1: Fortran reference (cost: a few CPU node-hours on `compute`; in parallel with P0)

### Task 3: Build and run FESOM2 + REcoM (DP) on pi

**Files:**
- Create (outside the repo): `/work/ab0995/a270088/fesom2_recom/` (fresh clone of FESOM2 main +
  `git submodule update --init src/recom` ⇒ 692f804)
- Create: `scripts/recom/fortran/build_fesom2_recom.sh`, `scripts/recom/fortran/run_pi_recom.sbatch`
- Create: `docs/recom/FORTRAN_REFERENCE.md`

- [x] build DP with `RECOM_COUPLED=ON` (Intel, FESOM2's CMake flags); record both commits
- [x] pi run with the `bin_2p3z2d` REcoM namelists (31 tracers incl. 1037, advection weights `0., 1.`), the
      pi REcoM inputs (`test/input/recom`: Albani dust on the pi mesh, 5° IC files), JRA55-do forcing
      with `l_mslp = .true.` (JRA55-do `psl`), physics as close to `configs/pi_recom.yaml` as the namelists
      allow (zstar, mEVP, `use_wsplit`, `step_per_day 32`)
- [x] runs: 1 day, 30 days (one job), and the same 30 days split into 2 restart segments (the `mstep`
      semantics reference); timings and the REcoM global diagnostics from the log
- [x] `docs/recom/FORTRAN_REFERENCE.md` (rebuild recipe, runs, output locations)
- [x] test `fesom_jax/tests/test_recom_reference.py`: reference layout and tracer list (skips if absent)

### Task 4: Env-gated dumps + the standalone mocsy driver

**Files:**
- Modify (our clone only): `src/recom/recom_main.F90`, `src/recom/recom_sinking.F90`,
  `src/oce_ale_tracer.F90`, `src/oce_ale_vel_rhs.F90`, `src/fesom_module.F90` (behind
  `FESOM_RECOM_DUMP_DIR`, steps from `FESOM_RECOM_DUMP_STEPS`; `origin/recom4-dumps` as a guide)
- Create: `scripts/recom/fortran/mocsy_driver.F90` + build script
- Modify: `fesom_jax/io_dump.py`
- Create: `fesom_jax/tests/test_recom_dump_io.py`

- [x] full-field dumps in global node numbering: every `recom()` input and output (incl. `press_air`,
      `GlodecayBenthos`, `GloHplus`, `Alk_surf`), the carbonate cache, the surface fluxes, the REcoM 3-D
      diagnostics; per BGC tracer in loop order: values before/after `dtr_bf`, the ballasting densities,
      `vert_sink`, `str_bf`, `Benthos`, values after T* and after the implicit diffusion; the step's
      transport fields (uv, w_e/w_i, hnode/hnode_new, Kv, GM bolus, Redi) for G4a; the SLP momentum term
- [x] dump steps: 1, the first carbonate refresh, lit/dark and ice-covered steps (Jan and Jul), the first step
      of the second restart segment
- [x] mocsy driver: a grid of inputs → `vars_sprac`/`flxco2` outputs (G1)
- [x] JAX reader + test: shapes, IDs, a round trip
- [x] the dumps are a no-op without the env variable (a 1-day run is bitwise the Task 3 run)
- ➕ lane F done (recom/fortran-ref 0623dc1/96d9147/0e69077, merged 32b619e; ~2 node-h): FESOM2 e96a2b13 + hooks,
  REcoM 692f804 + dump module, Intel 2022 DP, one rank on a `dist_1` made with `fesom_meshpart`
  (`/work/ab0995/a270088/recom_ref/`). Runs: `pi_30d_jan` (dumps steps 1, 2, 17, 224, 225, 960), `pi_1d_jul`
  (1, 17), `pi_30d_split` (15 d + restart + 15 d), the no-op pair bitwise. Reader `io_dump.load_recom_dump`.
  Catalogue and namelists: `docs/recom/FORTRAN_REFERENCE.md`. Findings to mirror: (1) `__usetp` is always
  compiled with REcoM, so the sinking tracers' benthos goes to per-tracer buffers and `Benthos` is updated after
  the whole tracer loop; (2) `GloHplus` = surface pH, output only; (3) the SLP momentum term uses
  `press_air/1000` (not /rho0); (4) FESOM2's own restart is not bit-continuous in the PHYSICS (split vs unsplit:
  S 3e-3 on day 16, T 0.36 °C by day 30; days 1-15 bitwise). Choices taken from `bin_2p3z2d`: the preindustrial
  GLODAPv2 PI DIC IC (with 278 ppm; the CI test's present-day DIC outgasses 11 PgC/yr), `use_ocean_only_forcing`
  off (as every earlier Fortran JRA reference of ours). Sanity: 31 BGC tracers, CO2 uptake 2.1 -> 4.5 PgC/yr,
  NPP 0.003 -> 38 PgC/yr over the 30-day cold start, cache refreshes at steps 1, 224, 960.
- ⚠️ the reference runs `bin_2p3z2d`'s PHYSICS (FESOM-native KPP, `opt_visc 5`), which fesom-jax does not have
  (opt_visc 7 only) — my brief said "mixing as in bin_2p3z2d". The dump-fed gates (G2, G3, G4a) do not care;
  G4b and G5 need ➕ Task 4b.
- ✅ Task 4b done (lane P): `recom_ref/pi_30d_jan_jaxphys`, job 28420010, namelists
  `scripts/recom/fortran/namelists_jaxphys/`. Physics floor vs JAX `configs/pi_recom.yaml` from the same step-0 state
  (`scripts/recom/physics_floor.py`, job 28420384): step 1 forcing 1e-14, a_ice 2.7e-7, uv 1.1e-6, Kv bitwise,
  T 3.4e-3 degC / S 1.4e-3 psu (top 4 layers, ice-covered high latitudes; enters in the tracer step); step 2 uv
  2.4e-4, Kv 0.1 (a convective switch), T 0.10 degC. G4b is judged against this floor.
- ➕ **Task 4b: a physics-matched Fortran rerun** — the same REcoM setup with our JAX physics (cvmix TKE,
  `opt_visc 7`, mEVP, zstar, GM/Redi, w_split as `configs/pi_recom.yaml`), 30 days with dumps at steps 1 and 2,
  for G4b and G5 (~1 node-h). Before Task 12.


### Phase P2: port + gates

### Task 5: `l_mslp`: JRA55-do sea-level pressure + the momentum term — gate GP

**Files:**
- Modify: `fesom_jax/jra55.py`, `fesom_jax/surface_forcing.py`, `fesom_jax/forcing_local.py` (on-device
  tables), `fesom_jax/momentum.py`, `fesom_jax/run_config.py` (`forcing.mslp`)
- Create: `fesom_jax/tests/test_mslp.py`

- [x] GP test first from the Task 4 dumps (`press_air`, the momentum term)
- [x] read `psl.YYYY.nc` exactly like the other JRA55-do fields (time interpolation, mesh interpolation,
      units as `gen_forcing_couple.F90:680`); carried in the forcing stack only when `forcing.mslp` is on
- [x] the SLP-gradient term of `oce_ale_vel_rhs.F90:202` in `momentum.py`, behind the flag (`press_air/1000`,
      not /rho0: lane F)
- [x] `RecomConfig` requires `forcing.mslp: true` (error otherwise, with the reason) — in `RunConfig.validate`, for
      REcoM proper only: the passive-tracer transport mode (`recom.passive_tracers > 0`) runs no chemistry and keeps
      its physics, so `core2_gpu_fast_bgc31.yaml` and the Task 1 tests stay as they are
- [x] `mslp` off ⇒ `test_dp_lock.py` unchanged; serial/sharded/on-device forcing paths agree
- [x] run tests - must pass before Task 6
      ➕ lane P (`recom/physics`, plan `20261025-recom-lane-physics.md`): GP passes with one stated split: the SLP
      term from Fortran's own `press_air` at 3.4e-13 rel, the JAX `press_air` at 4e-16 on steps 1-2 and
      1.5e-11..4.3e-11 later. The residual is the evaluation form: FESOM2 e96a2b13 evaluates every forcing field as
      `coef_b + (rdate - t0)·coef_a` in DP too (#940) at `timenew - dt/2`; fesom-jax's DP reader keeps
      `rdate·coef_a + coef_b` for all nine fields (the same bracket in FESOM2's form matches to 8e-16, asserted).
      Porting the point-slope form to the DP reader would move every forcing field and the DP lock: a decision for
      the user, not taken here. The split-explicit SSH passes the term through `du` like the rest of the slow RHS
      (untested against Fortran, whose SE scheme differs anyway).

### Task 6: REcoM parameters, tracer map and config

**Files:**
- Create: `fesom_jax/recom/params.py`, `fesom_jax/recom/tracers.py`
- Modify: `fesom_jax/run_config.py` (`RecomConfig`, yaml block `recom:`)
- Create: `fesom_jax/tests/test_recom_params.py`

- [x] tests first: the `bin_2p3z2d` namelist values load; every out-of-scope switch (ciso, atbox, coccos,
      rivers, erosion, MEDUSA, transit/age, `use_virt_tracers`, cavities, `biostep != 1`, `NitrogenSS`,
      `useAeolianN`, `constant_CO2 = .false.` until the CO2 file path is ported) raises
      `NotImplementedError`; unknown keys raise
- [x] `params.py`: a frozen dataclass of the REcoM parameters this configuration uses (defaults from
      `recom_modules.F90` + the namelist); differentiable leaves separate from static switches
- [x] `tracers.py`: ID → slot map (1001-1030, 1037), Fortran loop order, sinking set, remin set, IC-file
      map, bc_surface IDs
- [x] run tests - must pass before Task 7

### Task 7: Calendar and BGC per-step inputs through the integrators

**Files:**
- Create: `fesom_jax/recom/inputs.py` (optional per-step structure: `daynew`, `ndpyr`, `month`, `mstep`,
  `timenew`, dust index, CO2)
- Modify: `fesom_jax/step.py` (an optional `bgc_in` argument), `fesom_jax/integrate.py`,
  `fesom_jax/integrate_sharded.py` (`run_steps_sharded_forced`, folding), `fesom_jax/run.py` (chunk
  driver: segment-relative `mstep`, dates)
- Create: `fesom_jax/tests/test_recom_inputs.py`

- [x] tests first: the per-step calendar of a chunked run equals FESOM2's (`daynew`, `ndpyr`, `month`,
      `mstep` restarting at 1 per run segment), across a month end and a year end; the dust month switch
      on the last step of the month (`gen_surface_forcing.F90:~2065`)
- [x] the structure travels as its own optional `xs` (not None fields in `StepForcing`) through the serial,
      sharded, on-device-tables and chunk paths
- [x] REcoM off ⇒ no new leaves, `test_dp_lock.py` unchanged
- [x] run tests - must pass before Task 8
      ➕ lane P part 2 (`recom/inputs`, plan `20261025-recom-lane-inputs.md`): the calendar (`daynew`, `ndpyr`) and
      the held Albani dust (`fe_dust`) as optional `StepForcing` fields (the calendar as replicated scalars,
      `surface_forcing.STEP_SCALARS`; on the tables path in `ForcingSched` + a dust month table), built by
      `surface_forcing.BgcForcing`; equal to the Fortran `recom_in` records bitwise (see the lane plan's Results).
      `mstep` stays with the main session (a State counter), as the lane plan says; the "own xs" design above was
      replaced by the lane plan's optional fields.

⛔ **STOP 2 before Task 8: the user picks sequential or 3 parallel lanes.** Lanes (own sub-plan, high
effort, own branch off `recom/port`, time box, cost stated first): M = Task 8; S = Tasks 9-10; T = Tasks
11-12. Tasks 5-7 and 13-14 stay in the main session. Sequential = no agents.
➕ **2026-10-25, user: up to three agents in parallel.** Launched at 54ea945: lane F (Tasks 3-4,
`docs/plans/20261025-recom-lane-fortran-ref.md`, branch `recom/fortran-ref`, 8 h box) and lane M (Task 8,
`docs/plans/20261025-recom-lane-mocsy.md`, branch `recom/mocsy`, 6 h box); the main session does Task 1.

### Task 8: Carbonate chemistry (mocsy) — gate G1

**Files:**
- Create: `fesom_jax/recom/carbonate.py`
- Create: `fesom_jax/tests/test_recom_carbonate.py`

- [x] G1 test first from the Task 4 driver output
- [x] port `vars_sprac` (constants, K's, pH solver) and `flxco2`; the pH solver runs a fixed iteration
      count ≥ Fortran's worst case and FREEZES each point once `|Δ ln h| < 1e-8` (same values as Fortran's
      per-point stop); the convergence check lives in the test
- [x] the per-level refresh rule (`mstep == 1`, 7-day lit / 30-day dark per layer) as masks plus one
      `lax.cond` on the step schedule ⚠️ moved to Task 10 (stage 1 owns the cache; lane M ported the chemistry only)
      ➕ done in Task 10: the refresh mask equals Fortran's cell for cell
- [x] edge cases: tiny/negative DIC or Alk in shallow bottoms, ice-covered surface, night (`SurfSR = 0`)
- [x] run tests - must pass before Task 9 ➕ lane M 7f0a778, merged: G1 on 4324 points, pH ≤ 8e-14 abs, species
      ≤ 1.9e-13 rel, K's ≤ 4.7e-13; iterations ≤ 11 (ocean 2-4); suite 75 passed (job 28417248). The caller-side
      rules (input clamps/floors, depth = top interface, unit conversions, run-stops, cavity nodes) are in the
      `carbonate.py` docstring and lane M's report; they belong to Task 10

### Task 9: Sources and sinks (`REcoM_sms`, init, update) — gate G2 (tracers)

**Files:**
- Create: `fesom_jax/recom/sms.py`
- Create: `fesom_jax/tests/test_recom_sms.py`

- [x] G2 test first: from each dumped `recom()` input, the updated tracers and the 3-D diagnostics match
      (the diagnostics localise a failing process)
- [x] port `sms_initialize_variables`, `REcoM_sms` (PAR down the column as a cumulative sum,
      photosynthesis, limitation, grazing with variable preference, mucus, O2-dependent remineralisation,
      iron chemistry, calcite, photodamage, ...) and `sms_update_state`, vectorised over `(nod2D, nl)`, in
      REcoM's f64; `biostep == 1` asserted
- [x] `state = max(tiny, state + sms)` and the per-species floors (`recom_forcing.F90:316-341`), with the
      clipped amount returned for the budget
- [x] a test per sub-process where the diagnostics allow; error-path tests (one-layer columns, nzmin > 1
      rejected)
- [x] run tests - must pass before Task 10

### Task 10: REcoM forcing + stage 1 assembly — gate G2 (fluxes)

**Files:**
- Create: `fesom_jax/recom/forcing.py`, `fesom_jax/recom/stage1.py`
- Modify: `fesom_jax/step.py` (stage 1 at the Fortran position: after the ice step and the ocean flux
  coupling, before EOS/momentum)
- Create: `fesom_jax/tests/test_recom_stage1.py`

- [x] G2 flux tests first: `GloCO2flux_seaicemask`, `GloO2flux_seaicemask`, `AtmFeInput`, `AtmNInput`,
      `relax_alk`, `GlodecayBenthos`, `GloHplus`, pCO2surf
- [x] `forcing.py`: Albani dust as the monthly hold of Task 7, constant CO2, `press_air` (Task 5), winds,
      shortwave, `a_ice`
- [x] `stage1.py`: DICremin surface reset, `bio_fluxes` with the global-mean subtraction as a global
      reduction (psum; the ordered-sums path when `gpu.ordered_sums`), the sources and sinks, the fluxes into
      `BGC_FIELDS`
- [x] REcoM off ⇒ `test_dp_lock.py` unchanged; sharded `dist_4` equals serial (lane W: full step 4.0e-13 worst, median
      6.9e-15, physics T 2.5e-14)
- [x] run tests - must pass before Task 11
- ➕ lane S done (recom/sources ca81b71/8ceedab/9cf8162/860c41f, merged bc4ca55; 116 tests, job 28420498): G2 at
  round-off on all 8 dump steps — tracers ≤ 1.2e-14, 3-D diags ≤ 1.9e-14, fluxes/GloHplus/pCO2surf ≤ 1.0e-13,
  cache ≤ 1.3e-13, benthos bitwise; refresh mask = Fortran cell for cell (101749 / 54986 / 51799 cells at steps
  1 / 224 / 960). Signature `recom_sources_sinks(bgc, carb, benthos, alk_surf, temp, salt, geom, inputs, dt_bgc,
  params, *, reduce_sum, n_iter) -> Stage1Result(bgc, carb, sfc, benthos, sms)`, inputs `BgcStepInputs(daynew,
  ndpyr, mstep, fe_dust, n_dust, co2_atm, press_air, u_wind, v_wind, shortwave, a_ice)`. Correction (dump-verified,
  lane S): at night PAR is floored at a tiny positive value, so a layer can still count as lit — the code mirrors
  Fortran's mask exactly whatever the light. Not mirrored: Fortran's run-stops (NaN/DIC/flux guards) and print-only
  diagnostics; REcoM's overwrite of FESOM's `kappa` (nothing reads it).
- [x] wiring into `step()` (with Task 7's inputs): stage 1 at the Fortran position on the BGC steps; the shapes
      differ — stage 1 works on `(nod, nl-1, 31)`, `State.bgc` is `(nod, nl, ntr)`; `relax_alk`'s global mean via
      the sharded psum with `area_top` zero on halo copies; under `bgc_every = k`, `inputs.mstep` counts BGC steps
      and the refresh periods follow `dt_bgc` (stated deviation); RunConfig requires zstar with REcoM


### Task 11: Benthic return flux, ballasting, sinking, benthos — gate G3

**Files:**
- Create: `fesom_jax/recom/sinking.py`
- Create: `fesom_jax/tests/test_recom_sinking.py`

- [x] G3 test first from the per-tracer dumps, in Fortran tracer order ➕ lane T: `test_g3_per_tracer`, every field within 1.7e-15 of its max on all 8 dumped steps
- [x] port `diff_ver_recom_expl` (remin tracers), `get_seawater_viscosity` (T and S after their diffusion,
      S before the clip), `get_particle_density` (from the CURRENT detritus tracers), `ballast`,
      `ver_sinking_recom`, `ver_sinking_recom_benthos` (per-tracer `Benthos_tr` buffers, merged into `Benthos`
      after the whole tracer loop, in tracer order: `__usetp` is always compiled with REcoM)
- [x] sequential semantics: a tracer's density uses the detritus tracers already finished this step and the ➕ one batch of 23 + four rounds of two, bitwise equal to the sequential loop (`test_g3_loop_sequential_equals_batched`)
      post-stage-1 values of the rest; an optional value-preserving optimisation batches the tracers between
      two detritus completions (constant density), tested against the sequential reference
- [x] edge cases: one-layer columns, the bottom layer, zero sinking speed ➕ `test_one_layer_and_bottom_conservation`, `test_zero_sinking_speed`, `test_cfl_cap_long_step`
- [x] run tests - must pass before Task 12

### Task 12: Stage 3 complete, in Fortran order — gates G4a/G4b

**Files:**
- Modify: `fesom_jax/recom/transport.py`, `fesom_jax/step.py`
- Create: `fesom_jax/tests/test_recom_step.py`

- [x] G4a test first: Fortran's transport fields → stage 3 → BGC tracers after the tracer step
- [x] order per tracer: batched advection + explicit Redi → `dtr_bf`, `vert_sink`, `str_bf` added in content
      form (enter as `Δ·hnode/hnode_new`) → T* → implicit vertical diffusion with the bc_surface cases;
      the per-tracer part follows Task 11's sequence
- [x] G4b: one full JAX step vs the Fortran step ➕ lane W, job 28421905: all 31 BGC tracers <= 1.8e-8 rel, below
      the pair's T floor 4.1e-8 (1.2e-6 degC; S 1.5e-8); Benthos <= 3e-11; CO2 flux 7.5e-9, pCO2 3.6e-14, NPP 2.2e-10
- [x] sharded `dist_4` equals serial for the BGC tracers (2 full steps: worst 4.0e-13, Benthos 1.8e-16)
- [x] run tests - must pass before Task 13
- ➕ c859bb5: `transport.recom_stage3` (batched `bgc_advect_redi`, then lane T's `recom_tracer_loop_terms` with a
  `finish` = T* with the modified values (`X_adv + (values_mod − X1)·hnode/hnode_new`) + `bgc_diffuse` with
  `surface_bc`); the passive path now runs on the same two halves (its tests still bitwise / 1e-14). G4a (7 dump
  steps) FAILS by ~3e-4 rel — localized stage by stage: advection, Redi, T* at round-off, the implicit diffusion off;
  cause = FESOM2 #944 (Redi K33 weights) missing in fesom-jax (investigation G) ⇒ G4a xfail until
  `physics/upstream-944-1116` is merged.
- ➕ with #944 merged (8707dce) G4a PASSES at round-off: worst 1.2e-14 rel over the 31 tracers and 7 dump steps
  (median ~1.5e-15), Benthos included (job 28421676)
- [x] step wiring of full mode (stage 1 + stage 3 + `bgc_every`): lane W (`docs/plans/20261025-recom-lane-wiring.md`)


### Task 13: Initial conditions, restart, output, `run_from_config`

**Files:**
- Create: `fesom_jax/recom/ic.py`, `fesom_jax/recom/output.py`
- Modify: `fesom_jax/run.py`, `fesom_jax/zarr_output.py`
- Create: `fesom_jax/tests/test_recom_ic.py`, `fesom_jax/tests/test_recom_restart.py`

- [x] IC test first: the interpolated fields (Fe with ×1e9 and the hydrothermal-vent cap, O2, Si, Alk, DIC,
      DIN) and the constant initial values of the other tracers match Fortran's step-0 dump; `Alk_surf` =
      the IC surface Alk
- [x] `ic.py`: the 5° climatologies → nodes with `gen_ic3d`'s algorithm
- [x] restart: `BGC_FIELDS` incl. `HPlus` through the archival restart; `Alk_surf` from the IC, never from
      the restart (as Fortran); the carbonate cache NOT restarted (as Fortran)
- [x] test: a JAX run split at a restart reproduces Fortran's REcoM split semantics (`mstep` back to 1, cache
      refresh at the segment start, the dust month switch), checked on the REcoM pieces only: FESOM2's own restart
      is not bit-continuous in the physics (lane F finding 4), so the split pair is not a whole-state gate
- [x] output: monthly means, default 3-D DIC, Alk, DIN, DSi, DFe, O2, total Chl; 2-D CO2 flux, pCO2surf,
      integrated NPP, export at 100 m; configurable; the ushow schema
- [x] run tests - must pass before Task 14
- ➕ lane W: restart = `bgc`, `bgc_old`, `bgc_benthos`, `bgc_alk_surf` (Alk_surf KEPT in the restart: Fortran rebuilds
  it from the IC at every run start, the same values); never written: `bgc_carb`, `bgc_mstep`, `bgc_diag` (GloHplus =
  surface pH, output only, so `HPlus` is not restarted either), `bgc_budget`. Every run start resets mstep and the
  cache (`wiring.start_segment`); a physics-only restart into a REcoM run starts the BGC from the IC. Split test
  (2 + restart + 1 vs 3 steps): physics bitwise, mstep 1 vs 3, the cache refilled in 100 % of the wet cells. Output:
  `recom.output` (default DIC, Alk, DIN, DSi, DFe, O2, Chl, co2flux, pco2surf, npp, export100) joins the daily and
  monthly mean streams (ushow schema); export100 = POC (PhyC, DetC, DiaC, DetZ2C) sinking flux through the interface
  nearest 100 m, a fesom-jax diagnostic (REcoM has none)

### Task 14: 30-day pi run vs Fortran, budget, AD smoke — gates G5, G6

**Files:**
- Create: `fesom_jax/tests/test_recom_budget.py`, `fesom_jax/tests/test_recom_grad.py`
- Create: `scripts/recom/compare_pi_30d.py`, `docs/recom/VALIDATION.md`

- [x] budget test first: 1- and 10-step closed budgets for N, C, Si, Fe, Alk, O2 (surface fluxes + benthos
      exchange + clip term)
- [x] measure the physics-only T/S divergence of the JAX/Fortran pair over 30 days, then set and run G5
- [x] G6: `jax.grad` of 1-day NPP w.r.t. 2-3 parameters (max photosynthesis rate, a grazing rate, a
      remineralisation rate) vs central FD; no NaN/Inf in the cotangents
- [x] numbers into `docs/recom/VALIDATION.md`
- ➕ lane W: budget closed per tracer to <= 4.6e-15 (4 steps, k = 1 and 2) and <= 6.9e-15 over 30 days; N, Si, C
  element budgets (+ benthos) close to the inventory-sum rounding. G5 (job 28422570): daily inventories of DIC, Alk, DIN,
  DSi, DFe, O2 within 1.1e-9 of the twin on all 30 days, CO2 flux 1.9e-5, NPP 5.0e-6; day-30 pointwise T 7.1e-3 rel,
  BGC 3e-4..1.5e-2 (DetZ2* up to 0.31). G6 (job 28424311, from the day-30 state): P_cm 8.8e-6, rho_N 4.9e-5, alfa_d
  6.9e-7; graz_max numerically zero (no grazers yet). Two derivative-only fixes (sinking areasvol, photodamage exp)
- [x] ⛔ **STOP 3: report P2 to the user**; the CORE2 inputs needed (see Post-Completion) ➕ user 2026-10-26: "Profile + 1 year, k=1"; the CORE2 inputs found on Levante (Task 15)
- ➕ first CORE2 + full REcoM speed (job 28424908, 4 GH200, dt 2700, DP, v3 1990 restart, BGC from the IC,
  `configs/core2_gpu_recom.yaml`): k = 1 120.1 ms/step = 61.6 SYPD; `bgc_every` 4 47.1 ms/step = 157.1 SYPD; the
  31-tracer passive DP reference in the same job 67.1 ms/step = 110.3 SYPD (STOP 1: 110). So stage 1 (sources/sinks,
  mocsy, sinking/benthos terms) adds ~53 ms/step to the ~59 ms/step of the BGC transport; peak 14.8 GB a GPU

### Phase P3: CORE2

### Task 15: CORE2 + REcoM, 1 year

**Files:**
- Create: `configs/core2_recom.yaml` (DP), `configs/core2_gpu_recom.yaml`
- Modify: `docs/PERFORMANCE_CORE2_GPU.md`, `docs/recom/VALIDATION.md`

- [x] CORE2 inputs FOUND (a colleague's tip, 2026-10-25): `/work/ab0246/a270092/input/fesom2/recom/CORE2/
      DustClimMonthlyAlbani.nc` (`DustClim(months=12, nod2d=126858)`, umol/m2/s, Albani 2014 — node count = our CORE2
      mesh; verify the node ORDER against `data/mesh_core2` with a land/ocean or coastline check), plus the 1° IC
      set one level up (GLODAPv2 PI TCO2 + TAlk, WOA13 i00/n00, WOA18 o00, PISCES Fe) and `MonthlyAtmCO2_gcb2023.nc`.
      Reference run with the same REcoM switches: AWI-ESM3-cc on CORE3 (cavities, 48 steps/day, coupled,
      `restore_alkalinity = .false.`), `/work/bb1469/a270092/runtime/awiesm3-v3.5.1-cc-blazerh/blazerh_fix`
      (namelists in `config/fesom/`, bio restarts, output) — read-only, for namelist and order-of-magnitude checks
- ➕ node order VERIFIED (lane I, lesson 2026-10-25): annual log-dust differs by 0.027 across mesh edges against
  0.92 for random pairs and 0.13 when shifted by one node; the maximum is off Senegal. GPU config:
  `configs/core2_gpu_recom.yaml` (the fast stack in DP + full REcoM + mslp + dust)
- [x] 1 year CORE2 + REcoM in DP on 4 GH200 from the v3 1990 restart (BGC from the climatology IC):
      SYPD, inventories, CO2 flux, NPP maps (nereus, never scatter)
- [x] nsys profile: sources/sinks, carbonate refresh steps, transport; hot spots fixed value-neutrally
- ➕ the year (job 28431153, 91b8210; STOP 3 user choice 2026-10-26 'profile + 1 year, k=1'): 1990 from the v3
  restart with `dt: 1800` + `dt_ramp: {after_step: 561024, dt: 2700}` (the se10 runs' way: the calendar is n*dt from
  1958; a first try with plain dt 2700, job 28430160, ran on 2006 forcing, and `BgcForcing` needed the ramp, fixed in
  91b8210); 1708 s for 11680 steps = 50.6 SYPD in the production loop (monthly output, CMDBUF on), finite. Annual
  means: CO2 uptake 2.03 Pg C/yr (atmosphere 278 uatm constant, surface pCO2 263-274 uatm), NPP 47.3 Pg C/yr, POC
  export at 100 m 5.63 Pg C/yr; inventory change Dec vs Jan mean: DIC -4.0e-5, Alk +6.2e-6, DIN -1.2e-3, DSi -4.6e-4,
  DFe -3.0e-3, O2 -5.3e-4; maps `runs/recom/core2_y1990_k1_ramp2/summary/maps.png` (scripts/recom/core2_year_summary.py)
- ➕ nsys (job 28430158): kernels 122 ms/step; REcoM's own kernels ~17 ms (sinking 9.2, sms 4.2, carbonate 0.4); the
  rest is transport of the 33 tracers: no value-neutral hot spot in the biology (ledger 'full REcoM on CORE2')
- ➕ found: with a dt ramp the JRA mid-step offset stays 0.5 * base dt (`surface_forcing._midstep_days`): after the
  switch the physics forcing is read 450 s early against FESOM2's timenew - dt/2 (the se10/sp10 runs too); FIXED on
  `physics/ramp-midstep` (74e8698, user 2026-10-26 'fix on its own branch'), merged a8b0066
- ➕ k = 4 error month (user 2026-10-26; jobs 28432702/3/5 + 28433011, comparison 28433021,
  `scripts/recom/kerr_compare.py`, `runs/recom/kerr_1990_01`): January 1990 from the v3 restart, ramp config.
  Floor (k = 1 twice, same code): monthly global integrals <= 1e-6 (Chl), T rms 2.5e-5. k = 4 against k = 1 (both
  with the mid-step fix): the physics equals the floor (T rms 2.8e-5); the conserved inventories hold (DIC 1.5e-6,
  Alk -1.2e-7, DIN 1.2e-5, DFe 1.8e-5, O2 -1.6e-5) but the biology does not: monthly NPP -8.1 %, CO2 flux -6.0 %,
  export100 -14.4 %, Chl -5.5 % (rms 10 %); end state PhyC -9 %, DetC -13.5 % (rms 26 %). So k = 4 (dt_bgc 3 h, stage
  1 with the k-th step's light) is NOT a faithful shortcut for the biology; the cause (the 3 h source/sink step or
  the light sampled once per interval) is not yet separated. The mid-step fix itself over the month (k1new - k1):
  NPP +0.21 %, CO2 flux +0.13 %, export +0.34 %, T rms 7.3e-5 (3x the floor)
- [ ] optional (user decides): a Fortran CORE2 + REcoM year, or colleagues' runs, for comparison
- [x] a CORE2 smoke test (few steps, compute node) in the suite
- ➕ `test_recom_core2.py` (job 28425120, 4 min): 3 steps from the v3 1990 restart, BGC from the 1-degree IC, on ONE
  CPU device: 4 in-process CPU devices abort in XLA:CPU's all_gather rendezvous on CORE2 (also without REcoM, job
  28425055; lesson 2026-10-26)

### Task 16: SP opt-in measurement

**Files:**
- Modify: `fesom_jax/recom/transport.py` (the WP/WP_full pattern of `_tracers_advect_sp`)
- Modify: `docs/PERFORMANCE_CORE2_GPU.md`, `docs/recom/VALIDATION.md`

- [x] SP transport for the BGC stack with WP_full accumulators like T/S; stage 1 and mocsy stay f64 ➕ lane SP
      (`docs/plans/20261026-recom-lane-sp.md`, branch `recom/sp`, 73e2171): the #1207 scheme of T/S with
      `State.bgc_carry`, REcoM's code in float64 with its WP inputs widened, every REcoM increment as float64 content
      into the one carried rounding (deviations R0-R6); opt-in, `bgc_every` 1 only; DP programs unchanged
- [x] 1-year DP vs SP twin on CORE2: inventory drift (DIC, Alk, DIN), CO2 flux, NPP, SYPD ➕ 4 GH200, v3 1990
      restart, jobs 28439582 (DP) / 28439583 (SP), same commit: SP - DP of the December inventories DIC 1.1e-8,
      Alk -1.3e-9, DIN 4.5e-8, DFe -4.4e-7, O2 -4.0e-7; December CO2 flux +8.5e-5, NPP +4.3e-5, export -2.9e-4;
      annual means equal to three digits; A/B 76.6 vs 120.1 ms/step = 96.6 vs 61.6 SYPD (SP/DP 0.64, job 28439576),
      production loop 72.3 vs 51.5 SYPD; the floor (DP twice) and the reading in the lane plan's STOP 4 report
- [x] tests: SP vs DP one-step difference at the f32 floor; budgets in SP ➕ `test_recom_sp.py`: the scheme alone
      (stage 3, same transport) <= 2.5e-7; the model step pair is dominated by the physics' own SP - DP (T 7.7e-3);
      budget with the carry median 8e-14, worst 1.7e-11 (open item), without the carry 1e-7 to 7e-7
- [x] ⛔ **STOP 4: the user decides on SP with REcoM** (upstream refuses SP + REcoM) ➕ report in the lane plan;
      user 2026-10-26: accepted as an opt-in (spin-ups, ensembles, sensitivity runs), DP stays the default and the
      production mode; merged into `recom/port` (e65413e); the open items (the 1e-11 budget residual of five
      tracers, DOC's sharded amplification, one year only) stay documented in the lane plan

### Task 17: Verify acceptance criteria

- [x] REcoM off and `mslp` off: `test_dp_lock.py` + identity gates green ➕ the DP programs match the 7e4dd4c
      baseline once dp_hlo_hash imports fesom_jax from the checkout it hashes (696b5f3)
- [x] GP, G1-G6 green; numbers in `docs/recom/VALIDATION.md` ➕ suite array 28445934 (unit/core2 groups)
- [x] CORE2 1-year run done; SYPD in the ledger ➕ job 28431153 (DP, 50.6 SYPD), the SP twin 28439582/3/750
- [x] full CPU suite (`scripts/runs/run_suite_parallel.sbatch`) and the sharded REcoM tests ➕ array 28445934,
      then the fixes f7db13c/c5f11e3/b3fc402/696b5f3 rerun (28519728, 28522536, 28534589, 28560735; sharding
      28519727_4 + 28561842 + 28560839). All green except test_the_cost_split_is_still_worth_having (ci 39 files
      against core2 48: core2 grew by real CORE2 tests; left red on purpose, the regrouping needs a timing run).
      The failures were stale tests (the optional BGC/carry State fields), a restartcheck that compared None as
      NaN, a one-sample fdlin noise floor and a subprocess that imported the main checkout; none in the model.
      ⚠️ The sharding group no longer fits one 8-h job (~8.3 h of tests) and test_gradient_sharded alone needs
      more than 240 GB (it passed on a 512 GB node, 1 h 18 min)

### Task 18: [Final] Documentation

- [x] `docs/PORTING_LESSONS.md` entries complete ➕ one per lane and finding, the last the every-k rates lesson
- [x] `README.md`: switching REcoM on, the supported configuration, the inputs, `forcing.mslp` ➕ section
      "Biogeochemistry (REcoM)": the YAML block, configs, commands, speed, checks, the two opt-in deviations, not ported
- [x] `docs/recom/FORTRAN_REFERENCE.md`, `docs/recom/VALIDATION.md` final ➕ VALIDATION gains a summary table and
      the STOP 4 decision; FORTRAN_REFERENCE unchanged (complete since lane F/P)
- [x] move this plan to `docs/plans/completed/`

## Technical Details

### State (`BGC_FIELDS`, all None when REcoM is off)

| field | shape | meaning |
|---|---|---|
| `bgc` | `(nod2D, nl, ntr)` | BGC tracers (slot order of `tracers.py`) |
| `bgc_old` | `(nod2D, nl, ntr)` | AB2 old values (`valuesold`) |
| `bgc_benthos` | `(nod2D, 4)` | benthic N, C, Si, Calc |
| `bgc_carb` | `(nod2D, nl, 8)` | cached CO2, pH, pCO2, HCO3, CO3, Ω_calc, K_sp, ρ_sw (not restarted, as Fortran) |
| `bgc_hplus` | `(nod2D,)` | `GloHplus` = surface pH (restart `HPlus`; output only, never read back) |
| `bgc_alk_surf` | `(nod2D,)` | the alkalinity restoring target (IC surface Alk, never from the restart) |
| `bgc_sfc` | `(nod2D, k)` | stage-1 outputs for stage 3: CO2, O2, Fe, N fluxes, `relax_alk`, `GlodecayBenthos` terms, pCO2surf |

The tracer axis is trailing because every State leaf is entity-leading. The brainstorm's separate `BgcState`
carry became this optional field group (the `SE_FIELDS` precedent); the BGC per-step calendar and dust inputs
travel separately (Task 7), so the integrator signatures gain one optional argument.

### Step order (mirrors `fesom_module.F90` + `oce_timestep_ale`)

1. forcing update (with `psl` when `forcing.mslp`), ice step, ocean flux coupling
2. **stage 1**: DICremin surface reset, `bio_fluxes` (global mean), per-level carbonate refresh on schedule,
   surface `flxco2`, dust, sources and sinks + clip → `bgc`, `bgc_sfc`, `bgc_hplus`
3. ocean dynamics (with the SLP momentum term), SSH, ALE, the T/S tracer stage (unchanged otherwise)
4. **stage 3**: batched FCT advection + explicit Redi for the BGC stack; then in Fortran tracer order:
   `dtr_bf` (remin tracers), ballasting density from the current detritus, `vert_sink`, `str_bf` +
   benthos, T*, implicit vertical diffusion with the bc_surface cases
5. salinity clip (after the whole tracer loop, as Fortran), thickness commit

### Config

```yaml
forcing:
  mslp: true              # required with recom
recom:
  namelist: <namelist.recom of bin_2p3z2d>
  inputs:
    dust: <DustClimMonthlyAlbani on this mesh>
    ic_dir: <dir with the 5-degree IC files>
  output: [DIC, Alk, DIN, DSi, DFe, O2, Chl, co2flux, pco2surf, npp_int, export100]
gpu:
  bgc_layout: vmap        # loop | vmap (Task 2 decides the default)
```

### Cost (stated up front)

- P0: ~1 GPU node-hour (dolpung within the 2-node limit, or A100).
- P1: build + pi runs (1 d, 30 d, 30 d split) + dumps: a few CPU node-hours.
- P2: CPU tests, a few node-hours; most cost is session tokens (the ~9k-line sources/sinks port). With 3
  lanes: 3 high-effort agents in parallel for roughly a day each, a large share of a weekly limit;
  sequential: slower, cheaper to steer.
- P3: 1 CORE2 year ≈ minutes of GPU time; the DP/SP twin ~2 GPU node-hours; an optional Fortran CORE2 year a
  few CPU node-hours.

### Out of scope (each raises `NotImplementedError`)

ciso, atbox, coccolithophores/Phaeocystis, rivers and erosion, aeolian N, MEDUSA sediments, transit and age
tracers, the CO2 time series (`constant_CO2 = .false.`), linfs virtual tracer fluxes, cavities, the
tracer-group MPI mode (`__usetp`/`num_fesom_groups`), multi-rate or offline BGC, a tracer-minor layout.

## Post-Completion

*Items requiring the user or other people — no checkboxes.*

- **CORE2 inputs:** found on Levante (Task 15); open: whether `bin_2p3z2d` is the namelist the AWI group runs on
  CORE2 standalone (the AWI-ESM3-cc run turns alkalinity restoring off).
- Whether `origin/recom4-dumps` ("RECOM4 m1", AWI's REcoM developer) is a parallel port effort to coordinate
  with.
- The STOP decisions: STOP 1 (continue after the ceiling), STOP 2 (sequential or lanes), STOP 3 (CORE2
  inputs), STOP 4 (SP with REcoM).
- A science comparison against colleagues' FESOM2-REcoM CORE2 runs, if they share them.
- Upstream reports for anything found while mirroring (e.g. the pi test running with zero SLP and advection
  weights `1., 1.`, the 30-vs-31 tracer list of `namelist.tra.working`): the user decides; no post without the
  exact text approved.
