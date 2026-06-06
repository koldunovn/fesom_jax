# FESOM2 → JAX Port — Phase 5: CORE2 Single-Device (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 5 outline).
**Created:** 2026-06-06. **Status:** DRAFT for review (no tasks started).

---

## 0. Scope correction (READ FIRST — supersedes the parent outline)

The parent plan's Phase-5 outline lists **"zlevel ALE (surface-layer thickness change;
local-zstar fallback)"**, **"partial cells"**, and **"bring back the `w_i` advective terms
in `impl_vert_visc` if `use_wsplit` is on."** **All three are dropped** — they are NOT in
the C port, which is our algorithmic source of truth (the golden rule: match the C port
kernel-by-kernel). Verified by reading the source this session:

- **`fesom_ale.c` is linfs-only.** The zlevel / local-zstar algorithm exists *only* in the
  Fortran `oce_ale.F90:2132-2557` (`vert_vel_ale`), which we do **not** mirror.
  FRESH_START §14.7 states it directly: `which_ALE = 'zlevel'`, **but we will use linfs**.
- **The C port is full-cell, no cavities, no partial cells.** `fesom_mesh.c:617-634`
  (`compute_zbar_3d_n`) sets `zbar_3d_n[n,nz] = zbar[nz]` (global column, truncated by
  `nlevels`); there is **no `Z_3d_n` array** (`fesom_eos.c:319`: `Z_3d_n[nz,n] = Z[nz]`).
  Bottom depths are z-level-snapped. So the global `zbar`/`Z` assumption the pi kernels
  rest on stays valid; EOS / SSH operator / PP need **no change**.
- **`use_wsplit = 0` on CORE2** (`fesom_constants.h:48-57`, same as pi) and `do_wimpl` is
  always false because `tra_adv_lim == 'FCT'` (`fesom_tracer_diff.c:102`,
  `oce_ale_tracer.F90:616`). So **`w_i ≡ 0`** — the `impl_vert_visc` w_i tridiagonal terms
  stay inert; nothing to re-enable.

> The parent outline described *real FESOM / the Fortran*. The C port is a **deliberately
> simplified** FESOM (linfs, full-cell, no cavities). Per the golden rule + the locked
> decision "match the C port," **Phase 5 ports the existing pi physics on the CORE2 mesh.**
> **zstar** (the user's future intent) and partial cells require C-side work first and are
> **out of scope** for Phase 5.

**Therefore Phase 5 =** the assembled single-step pi model (PP mixing + **linfs** ALE +
FCT advection + opt_visc=7, **no** GM/KPP/ice) run on the **CORE2 mesh** with **PHC initial
conditions** and **real JRA55 + SSS-restoring + runoff** forcing, verified per-substep
against a CORE2 C-port dump, stable for 1–10 days, and AD-re-checked on a CORE2 slice.

**Design seam for future ALE (user note 2026-06-06):** all ALE capabilities (**zstar**
first) will be ported later. Phase 5 implements only the `'linfs'` branch (the existing
`ale.thickness_linfs`), but keep the seam so zstar slots in without restructuring: route the
thickness step through a `which_ale` dispatch, keep `hnode_new` a first-class State field
(it already is), and keep the ALE mass-correction `del_ttf += T·(hnode − hnode_new)` term
in tracer reconstruction (already present, ≡0 under linfs, auto-activates when
`hnode_new ≠ hnode`). Likewise leave the dynamic-depth path reachable (a `which_ale`-gated
`Z_3d_n`/`zbar_3d_n` rebuild) rather than hardcoding the static global `zbar`/`Z`. **Do not
port zstar now** — just don't bake in linfs-only assumptions that would block it. zstar +
partial cells also need the C port extended first (it is linfs/full-cell today).

## 1. Reference path — Path A (user-confirmed 2026-06-06)

Per-substep C-port dump on CORE2 at the **JAX-matched config**, exactly mirroring Phase 0
(Path A). The C port already runs this config; the toggles are env-driven on the existing
binary:

- `FESOM_MIX_SCHEME=PP`, `FESOM_NO_GMREDI=1`, ice off (`FESOM_NO_ICE_DYN/ADV/THERMO=1`).
- linfs / opt_visc=7 / FCT / full-cell are the C port's hardcoded defaults
  (`fesom_step.c:149-150`, `fesom_momentum.c:632-650`).
- `dt = 500` (FRESH_START §15; safe for a short dump from smooth PHC IC; same dt JAX↔C).
- Mesh dir: **`/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`** (confirmed in
  `jobs/production_pp.sh` `MESH=` and `work_linfs_pp/namelist.config:49`).

JAX↔C diffs are then pure FP reassociation (tightest gate), config auto-matched — same as
pi. A climate-level cross-check vs an existing CORE2 run is **not** used (those runs are a
different config: GM/Redi + ice on, dt=1800 — not comparable).

## 2. Verification ladder (unchanged classes from the parent plan)

Per-substep probe-column dump at pinned CORE2 probes, truncate to `nlevels`,
`verify.assert_close(col, rec, kind=…)`: **map/gather 1e-15, scatter/reduction 1e-12**
(calibrate `atol`). **Re-run `tests/test_gradient.py` (CORE2 slice) at GATE 5.** AD rule
stays: any divide/sqrt whose denominator/arg can vanish in a masked lane must compute a
FINITE value (`where(d==0,1,d)` / double-`where` safe-sqrt) — the forward `where` does not
stop a `0·inf` NaN in the backward pass.

**Probes:** the C dump's `PROBE_GIDS` are hardcoded `{1001,1500,2000,2500,3000}`
(`fesom_dump.c:15-17`) — valid on CORE2 but clustered in the Southern Ocean. Re-pin to
useful CORE2 coverage (1-line edit + rebuild), **including a node by the Aleutian Trench
hotspot** (global elem 194724; its vertex node gids are 94122/100637/21532 — use 94122).

## 3. Config (the CORE2 reference run)

PP mixing, **linfs** ALE (`hnode_new = hnode`), FCT tracers, opt_visc=7
(γ0/γ1/γ2=0.003/0.1/0.285), **`use_wsplit=0`** (`w_e=w`, `w_i=0`), CG SSH (α=1), full-cell
(global `zbar`/`Z`), no GM/KPP/ice. **dt=500.** IC = PHC `phc3.0_winter.nc`. Forcing =
JRA55-do v1.4.0 + L&Y09 bulk + PHC2 SSS restoring + CORE2 runoff. `state_equation=1`
(JM-EOS), `C_d=0.0025`, `K_ver=1e-5`, `A_ver=1e-4`, `surf_relax_S=1.929e-6 s⁻¹`,
`ref_sss_local=1`, `vcpw=4.2e6`.

---

## Implementation Steps

### Task 5.1: CORE2 mesh export + load

**Files:** Modify (C, `port2`): SLURM job only (clone `jobs/jax_mesh_export_pi.sh`). JAX:
none expected (verify). Create: `data/mesh_core2/*.npy`; `tests/test_mesh_core2.py`.

> **✅ DONE 2026-06-06.** Export job 25386129 (17 s, peak 5.6 GB) wrote
> `data/mesh_core2/` (31 arrays + meta: nod2D=126858, elem2D=244659, edge2D=371644,
> nl=48, no cavity). Log confirms **`orient_cw: swapped 244654/244659`** (CORE2 raw mesh
> ~all CCW → normalized to CW; export captured post-swap). `load_mesh('data/mesh_core2')`
> works with **zero `mesh.py`/`state.py` change** (the design claim held). New
> `jobs/jax_mesh_export_core2.sh` (port2). `tests/test_mesh_core2.py` = **12 passed**
> (counts/bit-for-bit/indices/CSR/masks/geometry/no-cavity/all-CW). `tests/test_step_core2.py`
> rest-state = **PASS** (max|uv|=1.8e-14, |eta|=2.4e-15, T/S bit-exact, no NaN; ~32 s/step
> eager on CPU). Full suite re-run green (329).

- [x] **Export (Path A, npes==1, no C code change):** the existing `fesom_mesh_export.c`
  already writes everything CORE2 needs — `nlevels_nod2D_min` (`:99`), `ulevels_nod2D_max`
  (`:101`), `zbar_3d_n` (`:107`), `area`/`areasvol` (`:105-106`); rotation + CW orientation
  are baked into the exported `coord_nod2D`/`elem_nodes`. Clone the pi export job, set
  `MESH=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`, `EXPORT=.../data/mesh_core2`,
  `phc="" jra=0 nsteps=1`; bump SLURM `--mem` 8G→~16G (helem `nod2D*nl` ~94MB). Counts fit
  in int32.
- [x] **JAX load:** `load_mesh('data/mesh_core2')` already adapts — it reads `nl` from
  `meta.txt`, and the four ragged masks already encode per-node variable depth (the only
  real pi→CORE2 mesh difference). **Confirm no `mesh.py`/`state.py` change is needed**
  (the research says none; treat any required change as a finding to log).
- [x] ✅ **Orientation gate (the pi↔CORE2 trap — guard already added this session):**
  `mesh.check_cw_orientation` (mirrors the C `orient_cw`, `fesom_mesh.c:430`) runs inside
  `load_mesh` on **every** load and raises if any triangle is CCW/degenerate. The C
  normalizes pi **and** CORE2 to CW *before* deriving `gradient_sca` (verified:
  `orient_cw`@1193 → `elem_area`@1219 → `gradient_sca`; `elem_area` is `abs`; edge geometry
  is centroid-based ⇒ orientation-free). pi confirmed 5839/5839 CW. CORE2's raw mesh is
  ~all CCW, so this is the load-time catch if a future export ever skips the swap (→ wrong
  stiffness sign → Aleutian blow-up). `test_mesh_core2.py` re-asserts it on CORE2.
- [x] **Gate:** `tests/test_mesh_core2.py` mirroring `test_mesh.py` — counts
  (nod2D=126858, elem2D=244659, nl=48), index ranges (`elem_nodes∈[0,nod2D)`,
  `edge_tri∈{−1}∪[0,elem2D)`), CSR consistency, mask/level consistency, pytree round-trip,
  all-CW orientation; `State.rest(mesh, T0, S0)` builds and `hnode` derives from `zbar_3d_n`.
- [x] **Rest-state sanity:** constant T/S + zero wind on CORE2 stays at rest to machine
  precision (a few steps; the no-spurious-flow gate, like pi).
- [x] run — must pass before Task 5.2. **Lesson:** append (esp. whether the mesh port was
  truly zero-code as predicted). — **DONE: confirmed zero-code; orientation now guarded.**

### Task 5.2: PHC initial conditions

**Files:** Create `fesom_jax/phc_ic.py` (numpy, one-time) + `data/ic_core2/{T,S}_ic.npy`;
`tests/test_phc_ic.py`. C (`port2`): add a few full-column probe dumps if vertical-fill
verification needs it (cheap).

- [ ] **numpy reader** (one-time, offline; NOT in the AD path — IC is setup, though the
  produced field IS a valid grad target). Source: `phc3.0_winter.nc` under
  `…/INITIAL/phc3.0/` (**INITIAL, not FORCING**; `t_insitu=1`). Dims depth=33/lat=180/
  lon=360, vars `temp`(°C in-situ)/`salt`(psu), land=NaN, regular 1° geographic grid.
  Mirror `fesom_phc.c`: cyclic-lon +2 halo (`:428-443`); per-node bilinear bracket via
  `binarysearch_d` on **geographic** node coords (`:451-452`); per-depth bilinear (`:210-223`)
  with the surface-corner / per-depth dummy gate; linear vertical interp onto `mesh.Z`
  (`:226-250`); then **`extrap_nod3D`** + vertical fill (`:264-369`); below-5500 m
  vertical-fill from above; final dummy→0 + below-`nlevels` zero + K→C; **`ptheta`**
  insitu→pot (`:66-83`). Cache `T_ic/S_ic [nod2D, nl]`; set `T,S,T_old,S_old` via
  `dataclasses.replace`.
- [ ] ⚠️ **#1 fidelity risk — `extrap_nod3D` is sequential Gauss-Seidel, node-index ordered,
  in-place** (`fesom_phc.c:318-342`): each dummy wet node filled **once**, value = mean of
  neighbors valid *at fill time* (multiplicity-weighted via `nod_in_elem2D`, no dedup). A
  vectorized Jacobi gives **different** values (not rounding) past the data frontier.
  Replicate the in-place sequential loop (numba/loop ok — one-time). Verify it *binds*.
- [ ] **Gate (two-stage vs C dump):** the C emits `phc_dump_preextrap` (gid,T,S,bilin_i/j,
  lon,lat) + `phc_dump_postload` (surface) under `FESOM_EVP_DUMP_DIR`; diff harness
  `scripts/phc_dump_diff.py` exists. (a) pre-extrap surface T/S + bilin indices → bracket+
  bilinear @ **1e-15**; (b) post-load surface (post extrap+fill+ptheta) → **1e-12** (Jacobi
  fails this). Add a small full-column dump for a deep (>5500 m) + a coastal-extrap probe to
  verify vertical fill. Validate on the **pi** mesh first (near-global coverage, available
  now) before CORE2.
- [ ] **Fallback if extrap parity stalls:** load the C-dumped T/S `.npy` directly (IC is
  non-differentiable setup) and keep the numpy reimpl only for verification. Decide
  explicitly; log.
- [ ] run — must pass before Task 5.6. **Lesson:** append.

### Task 5.3: JRA55 forcing reader (host numpy)

**Files:** Create `fesom_jax/jra55.py` (numpy reader + bilinear stencil + time/cache
driver) + `tests/test_jra55.py`. C: probe-dump the 8 jra fields at a fixed (year,day,sec).

- [ ] **Reader** mirroring `fesom_jra55.c`: files
  `…/FORCING/JRA55-do-v1.4.0/{var}.{YEAR}.nc`, field order **uas,vas,huss,rsds,rlds,tas,
  prra,prsn** (`fesom_jra55.h:50-59` — note `tas` is 6th); source grid read from file dims
  (320×640), cyclic-lon **+2 halo** (`:154`), lat-flip if stored −90→90 (`:271-279`). Build
  the bilinear stencil **once** (4 src indices + 4 weights per node — shared by all 8
  fields) on **geographic** coords (`:295,458`); time-grid via `julday`+rebase+**mid-interval
  shift** (`nm_nc_tmid=0`, `:262-268`); 2-slice raw cache + linear time-interp (`:683-700`).
- [ ] **Wind g2r rotation** (`:692-694`, `fesom_vector_g2r` `fesom_mesh.c:169-192`): rotate
  (uas,vas) geographic→model-rotated per node (Euler 50/15/−90), magnitude-preserving;
  scalars NOT rotated. Unit conv: Tair K→°C, prra/prsn /1000 → m/s (`:698-700`).
- [ ] **Cache strategy:** keep 2 raw 3-hourly slices (×8×nod2D×8B ≈ 13 MB/slice on CORE2),
  refresh every ~3 model-hours; output `[nod2D, 8]` device-constant per step. Do **not**
  precompute all ~2920 records to nodes (~24 GB).
- [ ] **Gate:** dump `jra->{u_wind,v_wind,Tair,shum,shortwave,longwave,prec_rain,prec_snow}`
  at all nodes for a fixed (year,day,sec) vs C `fesom_jra55_step` — map/gather ~1e-13 (the
  bilinear gather; literal index/weight parity required).
- [ ] run — must pass before Task 5.4. **Lesson:** append (geographic-vs-rotated coord trap,
  field-order trap, mid-interval shift).

### Task 5.4: L&Y09 bulk formulae (AD-safe JAX)

**Files:** Modify `fesom_jax/forcing.py` (add bulk) + `tests/test_forcing.py`.

- [ ] **`ncar_ocean_fluxes_mode`** (`fesom_bulk.c:49`): neutral 10 m coeffs LY2009 11a/b,
  then the Monin-Obukhov stability loop — **run a FIXED 5 iterations, unrolled / `lax.scan`,
  drop the data-dependent early break** (`:171`; post-convergence iters are no-ops →
  identical result, AD-safe). Relative wind `|u_atm−u_ocn|` floored at 0.3 (`:70-73`).
- [ ] **`obudget`** (`:187`): sat-humid `b=3.8e-3·exp(17.27·t/(t+237.3))`; LWout=−ε·σ·(t+273.15)⁴;
  sensible/latent/evap; `qns=−(LWin+LWout+sens+lat)`; constants (`:21-39`) ρ_air=1.3,
  cp=1005, L=2.501e6, σ=5.67e-8, ε=0.97, **albw=0.1** (CORE2, not 0.066). ⚠️ stress/coeffs
  use **relative** wind but `obudget`'s `ug` uses **absolute** wind (`:283`) — deliberate
  Fortran mismatch, preserve it.
- [ ] **Outputs** (`:291-342`): `heat_flux[nod]=qns−qsr`, `water_flux[nod]=evap−prra−prsn`,
  `stress_node_surf=Cd·ρ_air·|Δu|·Δu` (relative wind), then node→elem **simple mean-of-3**
  (`:332-342`; NOT the pi area-weighted double-average). Cavity nodes zeroed (none on CORE2).
- [ ] ⚠️ **AD-safe guards:** `sqrt(Δu²+Δv²)` (floor inside, or safe-sqrt — kink at 0);
  `x2=sqrt(|1−16ζ|)` singular at ζ=1/16 (`:99,115,132`); replace `copysign` step-switches
  with `jnp.where`; divides by `ustar²`/`tv`/`cd+1e-8` are structurally nonzero. The
  **SST→heat_flux and current→stress feedback is differentiable** and lives here.
- [ ] **Gate:** dump `Cd/Ce/Ch` (`fesom_bulk.c`) + `heat_flux/water_flux/stress_node_surf/
  stress_surf` after `fesom_bulk_compute` vs C (~1e-12). + a finite-iter-vs-C-early-break
  equivalence check.
- [ ] **Defer** `cal_shortwave_rad` / `sw_3d` penetration (`fesom_bulk.c:355`) — needs chl
  climatology + 3-D plumbing; gate whether the CORE2 reference run has it on
  (`USE_SW_PENE` is gated on `use_jra` → likely ON; if so it's a Task-5.6 sub-item, else
  skip). Confirm against the C config before porting.
- [ ] run — must pass before Task 5.6. **Lesson:** append.

### Task 5.5: SSS restoring + runoff + oce_fluxes balance

**Files:** Create `fesom_jax/sss_runoff.py` (numpy readers + AD-safe JAX flux math) +
`tests/test_sss_runoff.py`.

- [ ] **Port `fesom_sss_runoff.c` exactly — it already mirrors the Fortran sbc, no
  invented modeling choice.** (SSS/runoff is validated in the C port — no SSS problems —
  so the discipline is a faithful 1:1 port, gated by the dump; do NOT substitute
  FRESH_START §9's `water_flux += (S−S_clim)·v` / `water_flux −= runoff` *shorthand*, which
  is just a simplified description.) What the C does: SSS restoring = a **virtual salt flux
  + relax_salt fed into the S surface BC** (`fesom_tracer_diff.c:58-69`), and in the
  **no-ice** path (Phase 5) runoff enters only via the global-mean water balance (the local
  runoff term lives in ice thermo, which is off here). The per-substep dump gate confirms
  JAX == the C-port-no-ice run; faithfulness is *verified*, not assumed.
- [ ] **numpy readers** (one-time): SSS `PHC2_salx.nc` (`SALT` time=12/lat=180/lon=360,
  psu, missing=−99 → 30-cell expanding-neighborhood fill) → precompute **all 12 months** to
  nodes `Ssurf_clim[12, nod2D]`; runoff `CORE2_runoff.nc` (`Foxx_o_roff`, single record,
  (kg/s)/m² → /1000 m/s) → `runoff_node[nod2D]` once. Port `interp_2d_field` + missing-fill
  **literally** (bilinear cyclic-lon; scipy/xarray will miss tolerance).
- [ ] **AD-safe JAX flux math** (`fesom_sss_runoff.c:384-440`), pure fn of
  `(S_top, water_flux, Ssurf_month, runoff_node, areasvol_surf, ocean_area, masks)`:
  `rsss = S_top` (**`ref_sss_local=1`**, not 34.7); `virtual_salt = rsss·water_flux`;
  `relax_salt = surf_relax_S·(Ssurf−S_top)`; subtract the **area-weighted global mean** of
  each (non-cavity nodes only) — `integrate_nod_2D(x)/ocean_area = Σ(mask·x·areasvol_surf)/
  ocean_area`; then `water_flux += mean(water_flux + runoff)` (**all** nodes, no cavity
  skip — the asymmetry is real, `:422-440`).
- [ ] ⚠️ **Traps:** signs (`virtual_salt=rsss·water_flux` with water_flux>0=ocean loses FW;
  `relax_salt` removes salt when model too salty); the month index — fire on first step of
  new month, **no legacy `+1`** (`:351-365`); `S_top` masked-finite before the reduction.
- [ ] **Gate:** dump `Ssurf/runoff/virtual_salt/relax_salt/water_flux` (post-balance) vs C
  at step 1 + after a month crossing (map/gather 1e-15 for the multiplies; reduction 1e-12
  for the global-mean).
- [ ] run — must pass before Task 5.6. **Lesson:** append.

### Task 5.6: Wire surface BCs into the step + assemble CORE2 forcing

**Files:** Modify `fesom_jax/tracer_diff.py`, `fesom_jax/step.py`, `fesom_jax/params.py`
(forcing seam if useful) + tests.

- [ ] **Surface BCs** (currently `bc_surface=0`, Phase 2): in `tracer_diff` set
  `bc_T = −dt·heat_flux/vcpw` (`fesom_tracer_diff.c:56`) and
  `bc_S = dt·(virtual_salt + relax_salt)` (`:58-69`; linfs ⇒ `real_salt_flux=0`,
  `is_nonlinfs=0`, so `water_flux` has no *direct* T/S BC — only via `virtual_salt`). Wire
  the bulk `stress_surf` into momentum (replace pi's analytical wind path).
- [ ] **Thread forcing through `step`/`integrate`:** SST tap `T[:,0]`, surface-current tap
  `uvnode[:,0,:]` feed the bulk; the per-step jra atmo array + the month index are
  loop-carried/closed-over. Keep `params=None ⇒ defaults` transparency intact for the
  existing 313-test pi suite (CORE2 forcing must not perturb the pi gates).
- [ ] **Gate:** per-substep CORE2 dump at step 1 (tight, all kernels via `step()`); SST/SSS
  evolution vs C over a few steps. **Lesson:** append.

### Task 5.7: Matched C dump run + CORE2 stability

**Files:** Modify (C, `port2`): re-pin `PROBE_GIDS` for CORE2 (`fesom_dump.c:15-17`) +
SLURM dump job. Create `tests/test_step_core2.py`; `docs/REFERENCE_RUNS.md` (CORE2 section).

- [ ] **Generate the CORE2 per-substep dump (Path A):** existing binary, env
  `FESOM_MIX_SCHEME=PP FESOM_NO_GMREDI=1 FESOM_NO_ICE_*=1`, PHC IC, JRA55 on, dt=500,
  ~10 steps; re-pinned CORE2 probes (incl. Aleutian-Trench node 94122). Capture
  `fixtures/core2_cdump.00000`. (npes small / dist_16; ~10 steps cheap.)
- [ ] **Assemble + run CORE2:** `step`/`integrate` on CORE2 mesh + PHC IC + full forcing.
  CORE2 1-day (~172 steps, dt=500) + a multi-day run; assert stable per FRESH_START §15:
  no NaN, SST∈[−2,35], |SSH|<5 m, max|vel|<3 m/s. Watch the Aleutian Trench.
- [ ] ⚠️ **Stability risk to surface empirically:** PHC IC + JRA55, **no ice** at dt=500 →
  high-lat super-cooling / the historic Aleutian blowup. The C reference at the matched
  config is the arbiter — **if the C port itself blows up without ice at this config, that
  is a finding** (ice would have to move from Phase 6 into Phase 5; flag immediately, do not
  paper over it).
- [ ] **Gate:** step-1 per-substep dump-tight; 1-day + multi-day stable. Snapshot
  climate-close stays indirect unless a matched C snapshot is produced. **Lesson:** append.

### Task 5.8: GATE 5 — gradient check on a CORE2 slice

**Files:** Modify `fesom_jax/tests/test_gradient.py` (add a CORE2-slice variant) + a GPU
memory sbatch.

- [ ] **Re-run the permanent AD gate on a CORE2 slice (small N):** `d(mean SST)/d(k_ver)`
  AD↔FD plateau (signal-lifted `k_ver`), flowing through the CG `custom_linear_solve`;
  `d(loss)/d(T₀)` finite everywhere incl. masked lanes (the strong masked-NaN probe).
- [ ] **NEW differentiable feedbacks** (the Phase-5 additions): `d(heat_flux)/d(SST)` and
  `d(stress)/d(surface_current)` AD vs FD (validates the bulk's differentiable seam — the
  whole point of real forcing for hybrid ML). Stay clear of the bulk's `where`/safe-sqrt
  kinks (modest perturbations).
- [ ] **Memory:** CORE2 is ~40× pi nodes → confirm the checkpointed N-step backward fits the
  A100-40 (the pi N=200 backward was 4.23 GB; expect to drop N or use O(√N) nested
  checkpointing). GPU sbatch.
- [ ] run — full suite green (pi 313 + CORE2 additions). **Lesson:** append.

**GATE 5 (acceptance):** CORE2 (PP/linfs/FCT/opt_visc7, PHC IC, JRA55+SSS+runoff, no
GM/KPP/ice) reproduces the CORE2 C per-substep dump at step 1 within tolerance; runs 1-day
+ multi-day stable (physical SST/SSH/vel); the gradient gate passes on a CORE2 slice incl.
the new SST→flux / current→stress feedbacks; full suite green.

---

## Risks / watch-list

- **CORE2 stability without ice** (dt=500, PHC+JRA55) — the biggest scope risk; the matched
  C run is the arbiter (Task 5.7). May force ice earlier than Phase 6.
- **PHC `extrap_nod3D` sequential-GS parity** (Task 5.2) — Jacobi gives wrong values;
  the #1 fidelity risk. Fallback: load the C-dumped IC.
- **JRA55 reader literal parity** (bilinear index/weight, field order, mid-interval shift,
  geographic-not-rotated interp) — Task 5.3.
- **Aleutian Trench (global elem 194724)** — every historical CORE2 blowup; pin a probe.
- **Backward-pass memory at CORE2 scale** — Task 5.8.
- **`sw_3d` shortwave penetration** — confirm whether the matched C config has it on before
  deciding to port (Task 5.4).

## Out of scope (deferred — NOT in the C reference)

zlevel / zstar ALE, local-zstar fallback, partial cells, `Z_3d_n`, the `w_i` advective
terms, GM/Redi, KPP, sea ice. zstar is the user's future intent and needs C-side changes
first. GM/KPP/ice are Phase 6.

## Revision Log

- **2026-06-06 — created** (Phase-5 sub-plan). **Scope-corrected from the parent outline:**
  dropped zlevel ALE / partial cells / w_i re-enable (NOT in the linfs-only, full-cell C
  port — confirmed by reading `fesom_ale.c`, `fesom_mesh.c:617-634`, FRESH_START §14.7).
  Phase 5 = pi physics on the CORE2 mesh + PHC IC + JRA55/SSS/runoff forcing. **Path A**
  reference confirmed by user (per-substep CORE2 C dump at the matched config). Task ladder
  5.1–5.8 from this session's source research (5 module briefs).
- **2026-06-06 — user review revisions.** (1) Scope confirmed ("scope is fine"); PP-first
  confirmed. (2) **Triangle orientation made a checked invariant** (user: "remember the
  problems we had between pi and core on orientation — check and fix"): added
  `mesh.check_cw_orientation` + `load_mesh` guard + 3 tests (pi 5839/5839 CW verified;
  CCW/degenerate raise). Confirmed in C that `orient_cw`@1193 normalizes pi+CORE2 to CW
  before `gradient_sca`; `elem_area` is abs; edge geom is centroid-based. Task 5.1 gate
  updated. (3) **Task 5.5 SSS/runoff de-scoped to a faithful 1:1 C port** — removed the
  wrongly-introduced "C-literal vs §9-shorthand" modeling choice (user: "I hope the C does
  exactly what Fortran is doing; we don't have SSS problems anymore"); the dump gate
  verifies JAX == C. (4) **zstar design seam** added to §0 (user: port all ALE later, leave
  a placeholder) — `which_ale` dispatch + keep `hnode_new`/mass-correction/dynamic-depth
  paths reachable; do not port zstar now.
- **2026-06-06 — Task 5.1 DONE (CORE2 mesh export + load).** Export job 25386129 (17 s,
  5.6 GB) → `data/mesh_core2/` (31 arrays; nod2D=126858/elem2D=244659/edge2D=371644/nl=48,
  no cavity); log: `orient_cw swapped 244654/244659`. `load_mesh` works **zero-code**;
  `test_mesh_core2.py` (12) + `test_step_core2.py` rest-state (max|uv|=1.8e-14, T/S exact)
  green. New `port2/jobs/jax_mesh_export_core2.sh`. Lessons logged (zero-code port, the
  244654 swap confirmation, eager ~32 s/step ⇒ use jit/GPU for 5.7). Next: **Task 5.2
  (PHC initial conditions)**.
