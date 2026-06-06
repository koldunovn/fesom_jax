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

> **✅ DONE 2026-06-06.** `fesom_jax/phc_ic.py` (faithful numpy port of `fesom_phc.c`:
> bilinear interp + **sequential GS** `extrap_nod3D` + vertical fill + `ptheta`) +
> `build_and_cache_ic` + `core2_initial_state`. Verified vs the C dump (job 25386555,
> `data/phc_dump_core2/`): **bracket indices EXACT**, pre-extrap surface **~1e-14**,
> **post-load surface ~1e-14** (the order-dependent GS replicated exactly), 0 nodes off
> by >1e-12. Cache `data/ic_core2/{T,S}_ic.npy` (T∈[−2.06,30.05]°C, S∈[5.63,41.12] wet).
> `tests/test_phc_ic.py` = **5 passed**. **netCDF4 installed** into the env (user-approved;
> numpy/jax unchanged). New `port2/jobs/jax_phc_dump_core2.sh`. ⚠️ The C dump is
> surface-only → vertical interp/deep-ptheta verified indirectly (physical-range here +
> the Task-5.7 density gate); add a full-column C dump if 5.7 shows a depth mismatch.
> `T_old`/`S_old` step-1 AB2 history set = PHC fields for now; finalized in 5.7.

- [x] **numpy reader** (one-time, offline; NOT in the AD path — IC is setup, though the
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
- [x] ✅ **#1 fidelity risk CLEARED — `extrap_nod3D` is sequential Gauss-Seidel, node-index ordered,
  in-place** (`fesom_phc.c:318-342`): each dummy wet node filled **once**, value = mean of
  neighbors valid *at fill time* (multiplicity-weighted via `nod_in_elem2D`, no dedup). A
  vectorized Jacobi gives **different** values (not rounding) past the data frontier.
  Replicate the in-place sequential loop (numba/loop ok — one-time). Verify it *binds*.
- [x] **Gate (two-stage vs C dump):** the C emits `phc_dump_preextrap` (gid,T,S,bilin_i/j,
  lon,lat) + `phc_dump_postload` (surface) under `FESOM_EVP_DUMP_DIR`; diff harness
  `scripts/phc_dump_diff.py` exists. (a) pre-extrap surface T/S + bilin indices → bracket+
  bilinear @ **1e-15**; (b) post-load surface (post extrap+fill+ptheta) → **1e-12** (Jacobi
  fails this). Add a small full-column dump for a deep (>5500 m) + a coastal-extrap probe to
  verify vertical fill. Validate on the **pi** mesh first (near-global coverage, available
  now) before CORE2.
- [~] **Fallback if extrap parity stalls:** NOT NEEDED — the numpy reimpl matched the C
  to ~1e-14 (incl. the GS extrap), so we keep the in-repo numpy reader (no C-dump fallback).
- [x] run — must pass before Task 5.6. **Lesson:** append. — **DONE: test_phc_ic 5 passed.**

### Task 5.3: JRA55 forcing reader (host numpy)

**Files:** Create `fesom_jax/jra55.py` (numpy reader + bilinear stencil + time/cache
driver) + `tests/test_jra55.py`. C: probe-dump the 8 jra fields at a fixed (year,day,sec).

> **✅ DONE 2026-06-06.** `fesom_jax/jra55.py` — faithful numpy port of `fesom_jra55.c`
> (julday/binarysearch/time-grid transform + per-field mid-interval shift + shared bilinear
> stencil + per-field `getcoeffld` cache + `fesom_jra55_step` + **wind g2r rotation**).
> Verified vs the C dump (job 25388630, year 1958, `data/jra_dump_core2/`) at **two** dates —
> (day1,sec0) boundary + (day100,12:00) interior: **6 scalar fields BIT-EXACT** (max|diff|=0
> over all 126858 nodes, both dates), wind **~3.5e-15** (g2r `sin`/`cos` libm). `test_jra55.py`
> = **5 passed**. **#1 trap cleared:** the C time-interp `field=rdate·coef_a+coef_b` cancels two
> ~2.4e6 Julian-day numbers → a folded-weight gather's ~1e-13 reassociation blew up to ~6e-8;
> fixed by a **bit-identical** `(s·dx)·dy`-order + divide-at-end gather. C-side: `dump_jra_fields`
> in `fesom_main.c` (gated `FESOM_JRA_DUMP_DIR`) + `jobs/jax_jra_dump_core2.sh` (untracked,
> port2). `flip_lat=0` (lat ascending); field order uas,vas,huss,rsds,rlds,**tas**,prra,prsn.

- [x] **Reader** mirroring `fesom_jra55.c`: files
  `…/FORCING/JRA55-do-v1.4.0/{var}.{YEAR}.nc`, field order **uas,vas,huss,rsds,rlds,tas,
  prra,prsn** (`fesom_jra55.h:50-59` — note `tas` is 6th); source grid read from file dims
  (320×640), cyclic-lon **+2 halo** (`:154`), lat-flip if stored −90→90 (`:271-279`). Build
  the bilinear stencil **once** (4 src indices + 4 weights per node — shared by all 8
  fields) on **geographic** coords (`:295,458`); time-grid via `julday`+rebase+**mid-interval
  shift** (`nm_nc_tmid=0`, `:262-268`); 2-slice raw cache + linear time-interp (`:683-700`).
- [x] **Wind g2r rotation** (`:692-694`, `fesom_vector_g2r` `fesom_mesh.c:169-192`): rotate
  (uas,vas) geographic→model-rotated per node (Euler 50/15/−90), magnitude-preserving;
  scalars NOT rotated. Unit conv: Tair K→°C, prra/prsn /1000 → m/s (`:698-700`).
- [x] **Cache strategy:** per-field `getcoeffld` cache refreshing only when `rdate` leaves the
  current `[t_indx, t_indx_p1]` bracket (a pure optimization — the result depends only on the
  bracket, not call history). Output a `JRAFields` of 8 `[nod2D]` numpy arrays per step (→ jnp
  device constants in the 5.6 driver). Do **not** precompute all ~2920 records.
- [x] **Gate:** dump `jra->{u_wind,v_wind,Tair,shum,shortwave,longwave,prec_rain,prec_snow}`
  at all nodes for a fixed (year,day,sec) vs C `fesom_jra55_step` — achieved **bit-exact**
  scalars + ~3.5e-15 wind (tighter than the ~1e-13 target). Two dates (boundary + interior).
- [x] run — must pass before Task 5.4. **Lesson:** appended (the cancellation/bit-exact-gather
  trap, the interior-vs-boundary gate, field-order/geographic-vs-rotated/mid-shift traps).

### Task 5.4: L&Y09 bulk formulae (AD-safe JAX)

**Files:** Modify `fesom_jax/forcing.py` (add bulk) + `tests/test_forcing.py`.

> **✅ DONE 2026-06-06.** `fesom_jax/forcing.py` — AD-safe port of `fesom_bulk.c`
> (`ncar_ocean_fluxes_mode` fixed-5 unrolled + `obudget` + `bulk_surface_fluxes` with the
> node→elem mean-of-3). Verified vs a new C `bulk_dump_*` all-node dump (job 25389451,
> `data/bulk_dump_core2/`) at 3 configs — d1z (day1, zero curr), inz (day100/noon, zero curr),
> ins (day100/noon, synthetic curr): **cd/ce/ch ~1e-17, heat_flux ~6e-13, stress ~5e-16** over
> all 126858 nodes (essentially bit-exact, MAP-class). `test_forcing.py` = **10 passed**
> (forward gate ×3 + elem stress ×3 + synthetic-current active + early-break-bound + AD-finiteness
> + ordering). ⚠️ **FINDING — the "drop the break ⇒ identical" assumption was WRONG:** the M-O
> loop doesn't robustly converge at calm nodes, so fixed-5 vs early-break diverges (`ch` up to
> **~88%** at the calmest tropical nodes); but the **physical** impact is bounded — heat_flux
> ≤7.2 W/m² at ~4 nodes (mean 2e-4; <0.1 W/m² for 126848/126858), stress ≤4e-3 N/m². JAX runs
> fixed-5 (AD-safe) verified vs a **fixed-5** C dump (`FESOM_BULK_FIXED_ITERS`); Task 5.7 must
> set that flag on the per-substep reference. New C: `fixed_iters` param + `fesom_bulk_dump` +
> `jax_bulk_dump_core2.sh`. `USE_SW_PENE=1` in the C ⇒ shortwave penetration is ON → it's a
> Task-5.6 sub-item (heat_flux here = `qns−qsr`, pre-penetration).

- [x] **`ncar_ocean_fluxes_mode`** (`fesom_bulk.c:49`): neutral 10 m coeffs LY2009 11a/b,
  then the Monin-Obukhov stability loop — **run a FIXED 5 iterations, unrolled, drop the
  data-dependent early break** (`:171`, AD-safe: a `while`-break is not reverse-mode diff'able).
  Relative wind `|u_atm−u_ocn|` floored at 0.3 (`:70-73`). ⚠️ **CORRECTION to the original claim
  "post-convergence iters are no-ops → identical":** they are NOT — the loop is non-convergent at
  calm nodes (see the DONE note; `ch` up to ~88% fixed-5-vs-early-break). The fix is to verify vs
  a fixed-5 C dump, and bound the (small) physical residual vs the early-break production.
- [x] **`obudget`** (`:187`): sat-humid `b=3.8e-3·exp(17.27·t/(t+237.3))`; LWout=−ε·σ·(t+273.15)⁴;
  sensible/latent/evap; `qns=−(LWin+LWout+sens+lat)`; constants (`:21-39`) ρ_air=1.3,
  cp=1005, L=2.501e6, σ=5.67e-8, ε=0.97, **albw=0.1** (CORE2, not 0.066). ⚠️ stress/coeffs
  use **relative** wind but `obudget`'s `ug` uses **absolute** wind (`:283`) — deliberate
  Fortran mismatch, **preserved** (validated by the synthetic-current dump mode).
- [x] **Outputs** (`:291-342`): `heat_flux[nod]=qns−qsr`, `water_flux[nod]=evap−prra−prsn`,
  `stress_node_surf=Cd·ρ_air·|Δu|·Δu` (relative wind), then node→elem **simple mean-of-3**
  (`:332-342`; NOT the pi area-weighted double-average). Cavity nodes zeroed (none on CORE2).
- [x] ⚠️ **AD-safe guards:** `u`/`mag` use a double-`where` safe-sqrt (the `current→stress`
  gradient at Δu=0 is otherwise `0·inf` NaN); `x2=sqrt(|1−16ζ|)` (singular at ζ=1/16) is
  `sqrt(max(|1−16ζ|,1))` — bit-identical to the C floor AND smooth; `copysign` step-switches
  ported **literally** via `jnp.copysign` (exact at ±0, gradient 0). The **SST→heat_flux and
  current→stress feedback is differentiable** and lives here (AD-finiteness gated).
- [x] **Gate:** dump `Cd/Ce/Ch` + `heat_flux/water_flux/stress_node_surf/stress_surf` after
  the bulk vs C — achieved ~1e-17 (coeffs) / ~6e-13 (heat) / ~5e-16 (stress), well inside 1e-12.
  + the finite-5-vs-C-early-break check (recast as a **bounded-divergence** gate, per the finding).
- [~] **Defer** `cal_shortwave_rad` / `sw_3d` penetration (`fesom_bulk.c:355`) — **confirmed ON**
  (`FESOM_PHASE1_USE_SW_PENE=1`). heat_flux here is the pre-penetration `qns−qsr`; the 0.54-visible
  removal + `sw_3d` column build is a **Task-5.6 sub-item** (needs chl + 3-D plumbing).
- [x] run — must pass before Task 5.6. **Lesson:** appended (5 bulk lessons + 2 workflow). — **DONE: test_forcing 10 passed.**

### Task 5.5: SSS restoring + runoff + oce_fluxes balance

**Files:** Create `fesom_jax/sss_runoff.py` (numpy readers + AD-safe JAX flux math) +
`tests/test_sss_runoff.py`.

> **✅ DONE 2026-06-06.** `fesom_jax/sss_runoff.py` — faithful numpy readers
> (`interp_2d_field` bilinear with **lat-clamp / lon-cyclic-wrap** + the 30-cell
> expanding missing-fill via `scipy.ndimage.uniform_filter`) producing
> `Ssurf_clim[12, nod2D]` + `runoff_node[nod2D]` (/1000), plus the **AD-safe** JAX flux
> math `sss_runoff_fluxes` (`virtual_salt = S_top·water_flux`, `relax_salt =
> surf_relax_S·(Ssurf−S_top)`, each minus its area-weighted global mean; then
> `water_flux += ⟨water_flux + runoff⟩` over all nodes). Verified vs a new C `sss_dump_*`
> all-node dump (job 25390216, year 1958, `data/sss_dump_core2/`) at 2 months — m1 (Jan,
> day1) + m4 (Apr, day100 **month crossing**): **runoff bit-exact** (max|Δ|=0 all 126858);
> **Ssurf bit-exact at 105148/126858 ocean-bracket nodes** (95p ~3.6e-14, max 2.8e-12 at
> ~35 coastal/fill-bracket nodes — the 30-cell-mean reduction); **virtual_salt/relax_salt/
> water_flux ~1e-20..1e-22** (the global-mean reductions barely reassociate — the
> ÷`ocean_area`=3.6e14 crushes them; the flux math is fed the C's own
> `(S_top, water_flux_in, Ssurf, runoff)`, so it's apples-to-apples MAP-class). The two
> differentiable seams flow (`d/d(water_flux)`=SST→flux via the bulk, `d/d(S_top)`=restoring).
> `test_sss_runoff.py` = **9 passed**. New C `fesom_sss_runoff_dump` (gated
> `FESOM_SSS_DUMP_DIR`/`_DAY`/`_SEC`/`_MONTH`) + `jobs/jax_sss_dump_core2.sh` (untracked,
> port2). `ref_sss_local=1`, `surf_relax_S=1.929e-6`; **no legacy month +1**.

- [x] **Port `fesom_sss_runoff.c` exactly — it already mirrors the Fortran sbc, no
  invented modeling choice.** (SSS/runoff is validated in the C port — no SSS problems —
  so the discipline is a faithful 1:1 port, gated by the dump; do NOT substitute
  FRESH_START §9's `water_flux += (S−S_clim)·v` / `water_flux −= runoff` *shorthand*, which
  is just a simplified description.) What the C does: SSS restoring = a **virtual salt flux
  + relax_salt fed into the S surface BC** (`fesom_tracer_diff.c:58-69`), and in the
  **no-ice** path (Phase 5) runoff enters only via the global-mean water balance (the local
  runoff term lives in ice thermo, which is off here). The per-substep dump gate confirms
  JAX == the C-port-no-ice run; faithfulness is *verified*, not assumed.
- [x] **numpy readers** (one-time): SSS `PHC2_salx.nc` (`SALT` time=12/lat=180/lon=360,
  psu, missing=−99 → 30-cell expanding-neighborhood fill) → precompute **all 12 months** to
  nodes `Ssurf_clim[12, nod2D]`; runoff `CORE2_runoff.nc` (`Foxx_o_roff`, single record,
  (kg/s)/m² → /1000 m/s) → `runoff_node[nod2D]` once. Port `interp_2d_field` + missing-fill
  **literally** (bilinear cyclic-lon; scipy/xarray will miss tolerance).
- [x] **AD-safe JAX flux math** (`fesom_sss_runoff.c:384-440`), pure fn of
  `(S_top, water_flux, Ssurf_month, runoff_node, areasvol_surf, ocean_area, masks)`:
  `rsss = S_top` (**`ref_sss_local=1`**, not 34.7); `virtual_salt = rsss·water_flux`;
  `relax_salt = surf_relax_S·(Ssurf−S_top)`; subtract the **area-weighted global mean** of
  each (non-cavity nodes only) — `integrate_nod_2D(x)/ocean_area = Σ(mask·x·areasvol_surf)/
  ocean_area`; then `water_flux += mean(water_flux + runoff)` (**all** nodes, no cavity
  skip — the asymmetry is real, `:422-440`).
- [x] ⚠️ **Traps:** signs (`virtual_salt=rsss·water_flux` with water_flux>0=ocean loses FW;
  `relax_salt` removes salt when model too salty); the month index — fire on first step of
  new month, **no legacy `+1`** (`:351-365`); `S_top` masked-finite before the reduction.
- [x] **Gate:** dump `Ssurf/runoff/virtual_salt/relax_salt/water_flux` (post-balance) vs C
  at step 1 + after a month crossing (map/gather 1e-15 for the multiplies; reduction 1e-12
  for the global-mean). — **DONE: 2 months (m1 Jan, m4 Apr-crossing); runoff bit-exact,
  Ssurf bit-exact at 105148/126858, flux math ~1e-20.**
- [x] run — must pass before Task 5.6. **Lesson:** appended. — **DONE: test_sss_runoff 9 passed.**

### Task 5.6: Wire surface BCs into the step + assemble CORE2 forcing

**Files:** Modify `fesom_jax/tracer_diff.py`, `fesom_jax/step.py`, `fesom_jax/integrate.py`,
`fesom_jax/forcing.py`, `fesom_jax/sss_runoff.py`, `fesom_jax/phc_ic.py`; create
`fesom_jax/core2_forcing.py`, `tests/test_surface_bc.py`, `tests/test_core2_step.py`.
C (`port2`): `fesom_dump.c` (env probes), `fesom_step.c` (surf dump), `fesom_bulk.c`
(`FESOM_BULK_FIXED_ITERS` in `fesom_bulk_compute`); `jobs/jax_step_dump_core2.sh`.

> **✅ DONE 2026-06-06.** Surface BCs wired (`bc_T=−dt·heat_flux/vcpw`,
> `bc_S=dt·(virtual_salt+relax_salt)`, shortwave penetration `sw_3d` divergence into T) +
> the bulk/SSS/runoff/shortwave forcing assembled per step (`core2_forcing.py`:
> bulk → sss_runoff → **ice stress blend** → cal_shortwave_rad). Verified vs a new CORE2
> per-substep C dump (job 25391647, `data/step_dump_core2/`, 7 re-pinned probes incl.
> Aleutian 94122, `FESOM_BULK_FIXED_ITERS=1`): **step-1 post-step T 7.1e-15 / S 2.1e-14**
> (bit-exact — the comprehensive gate), surface forcing heat_flux 1.1e-13 / water_flux
> 9e-22 / virtual_salt 2.7e-20 / relax_salt 9.5e-20 / sw_3d 8.5e-22; dynamics density
> 2.3e-13 / uv 1e-10 / d_eta 2e-9 / w 4e-12; **steps 2-3 T/S ~1e-9** (threading). pi path
> bit-identical (313 tests). `test_surface_bc.py` (7) + `test_core2_step.py` (5).
> **chl = Sweeney monthly climatology** (the C default; constant-0.1 seam kept). **THREE
> bugs found+fixed** (see the lessons): (1) the C "no-ice" run keeps a **static `a_ice=0.9`
> mask** (IC SST<0) gating cal_shortwave penetration + the momentum stress blend — **user:
> match it**; (2) step-1 `T_old`=constant base 10/35, NOT PHC (pi blob analog); (3)
> `fesom_bulk_compute` didn't honor `FESOM_BULK_FIXED_ITERS` (early-break vs fixed-5 at calm
> nodes). **Finding:** in linfs the balanced `water_flux` is inert ⇒ runoff doesn't affect
> the Phase-5 trajectory.

- [x] **Surface BCs** (was `bc_surface=0`, Phase 2): `tracer_diff.impl_vert_diff(±bc_T/bc_S/
  sw_3d)` — `bc_T = −dt·heat_flux/vcpw`, `bc_S = dt·(virtual_salt + relax_salt)` (linfs ⇒
  `real_salt_flux=0`, `is_nonlinfs=0`); `sw_3d` divergence into T (`fesom_tracer_diff.c:
  298-308`). `cal_shortwave_rad` added to `forcing.py` (AD-safe, cumulative-OR break).
  Bulk `stress_surf` (with the ice blend) replaces the analytical wind. Defaults `None` ⇒
  the pi path bit-identical.
- [x] **Thread forcing through `step`/`integrate`:** `core2_forcing.compute_surface_fluxes`
  taps `T[:,0]`/`uvnode[:,0]`; the per-step jra atmosphere + month SSS/chl are the scanned
  `xs` (`StepForcing`), the runoff/areasvol/ocean_area/`a_ice` closed over (`ForcingStatic`).
  `step(..., step_forcing, forcing_static)` (lazy-imports `core2_forcing`); `params=None`
  transparency intact.
- [x] ✅ **Static ice mask (user decision: match the C).** `ice_ic_aice` replicates
  `fesom_ice_initial_state` (`a_ice=0.9` where IC SST<0); gates shortwave penetration +
  blends the stress (`ice_drag·a + atm·(1−a)`, `u_ice=0`, `ρ·Cd=1030·5.5e-3`).
- [x] **Gate:** per-substep CORE2 dump at step 1 (bit-exact T/S 7e-15, all kernels) + SST/SSS
  evolution steps 2-3 (~1e-9). **Lesson:** appended (the 3 bugs + the linfs-runoff-inert +
  the AD-seam findings).

### Task 5.7: Matched C dump run + CORE2 stability

**Files:** Modify (C, `port2`): re-pin `PROBE_GIDS` for CORE2 (`fesom_dump.c:15-17`) +
SLURM dump job. Create `tests/test_step_core2.py`; `docs/REFERENCE_RUNS.md` (CORE2 section).

> **✅ DONE 2026-06-06.** Per-substep dynamics gates added + the assembled CORE2 model
> run 1-day + multi-day, jitted, with the matched C arbiter. **(1) Per-substep dynamics
> gate** (`test_core2_step.py::test_step1_dynamics_per_substep`, +1 test → suite 371):
> step-1 pressure/PGF/Av/uv_rhs/ssh_rhs/d_eta/uv/hbar/eta_n/w/hnode all **bit-exact-class**
> (pre-solve ~0..1e-17; CG-derived ~1e-16..8e-15; the big intermediates ssh_rhs ~1e5 /
> pressure ~5e5 match ~1e-11 *relative*); element fields gated at the dump's incident-element
> gids; `test_evolution_steps23` extended with uv/d_eta (steps 2-3 ~1e-6, the discrete CG
> iter-count + FCT amplifying the step-1 ~1e-15). **(2) Stability run**
> (`scripts/core2_stability_run.py` + `core2_stability_gpu.sh`, A100, jitted ~0.06 s/step):
> **numerically stable days 1–7** (no NaN; max|vel| ≤ 1.9 < 3; |SSH| ≤ 2.8 < 5; Aleutian
> 94122 calm). **(3) C arbiter** (`jobs/jax_core2_stability.sh`, matched config + per-step
> monitor): the C is stable + JAX **tracks it to 3 sig figs** on SST_min/max|uv|/max|eta|
> (step 216: SST_min −6.60=−6.60, uv 1.389≈1.39, eta 2.715≈2.71). **FINDING (anticipated
> risk #1):** with no sea ice the SST **supercools without bound** (−1.9 IC → −5.8 d1 →
> −16.5 d5 → −22.8 d8); past ~−20 °C the JM-EOS is out of range, and at **model day ~8.1
> max|vel| crosses 3 m/s**. This is a *physical* no-ice limitation (the C does the same —
> NOT a JAX bug, and NOT the "C blows up ⇒ move ice to Phase 5" finding) — sea ice (Phase 6)
> caps it. Gate met: step-1 dump-tight (per-substep) + 1-day & multi-day numerically stable.

- [x] **Generate the CORE2 per-substep dump (Path A):** done in 5.6
  (`data/step_dump_core2/core2_cdump.00000`, 3 steps, 7 node + 7 incident-element probes
  incl. Aleutian 94122, `FESOM_BULK_FIXED_ITERS=1`). 5.7 added the per-substep **dynamics**
  gates on it (5.6 had only the comprehensive T/S + density/bvfreq/Kv).
- [x] **Assemble + run CORE2:** jitted `step_jit` loop with stacked `step_forcings`
  (`CoreForcing.stack(dates_for_steps(1958, 500, N))`) + `forcing_static`, monitored per
  step. CORE2 1-day (172 steps) + 10-day (1728) on an A100. Numerically stable through
  day 7; the Aleutian Trench stayed calm.
- [x] ⚠️ **Stability risk resolved empirically:** the matched C arbiter is **stable and
  tracks JAX** — so the no-ice run does **not** numerically blow up (ice stays Phase 6). The
  real limitation is **unbounded high-lat supercooling** (the C supercools identically): no
  NaN/dynamical blowup for ~7 days, then the sub-−20 °C EOS-invalid SST drives max|vel|>3 at
  day ~8. Flagged, not papered over (PORTING_LESSONS Task 5.7).
- [x] **Gate:** step-1 per-substep dump-tight (dynamics added); 1-day + multi-day
  numerically stable, JAX↔C 3-sig-fig trajectory match. Snapshot climate-close stays
  indirect (no matched C snapshot). **Lesson:** appended.

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

- **CORE2 stability without ice** (dt=500, PHC+JRA55) — ✅ **RESOLVED (Task 5.7): does NOT
  force ice earlier.** Numerically stable days 1–7 (vel/SSH bounded, no NaN); the matched C
  arbiter is stable too and JAX tracks it to 3 sig figs. The only no-ice limitation is
  **unbounded high-lat supercooling** (SST → −22 by day 8, the C identically) which past the
  EOS-valid range (~−20 °C) destabilizes the dynamics at day ~8 — a physical limitation capped
  by sea ice in Phase 6, not a numerical failure.
- **PHC `extrap_nod3D` sequential-GS parity** (Task 5.2) — Jacobi gives wrong values;
  the #1 fidelity risk. Fallback: load the C-dumped IC.
- **JRA55 reader literal parity** (bilinear index/weight, field order, mid-interval shift,
  geographic-not-rotated interp) — Task 5.3.
- **Aleutian Trench (global elem 194724)** — every historical CORE2 blowup; pin a probe.
- **Backward-pass memory at CORE2 scale** — Task 5.8.
- **`sw_3d` shortwave penetration** — confirm whether the matched C config has it on before
  deciding to port (Task 5.4).

## Runoff handoff to Phase 6 (the deferred-runoff plan — user-required understanding)

**Decision (2026-06-06, user):** Phase 5 keeps the C's no-ice config in which **runoff is
inert** (Option 1), CONDITIONAL on a locked, complete plan for activating it later. Here it
is. *Runoff is NOT broken — it works fully in the C's ice-on (production) run; the Phase-5
no-ice config just doesn't exercise it, by the C's own design.*

**Why it's inert in Phase 5 (the mechanism, C-verified):** the C deliberately routes runoff
through **sea-ice thermodynamics**, not the standalone sbc (`fesom_sss_runoff.c:376-380`
"Phase C3b: removed runoff subtraction… runoff is now folded into `ice->flx_fw` inside
`fesom_therm_ice`… subtracting again here would double-count"). The full path (ice ON):

```
runoff_node ─▶ fesom_ice_thermo.c:318  prec = rain + runo + snow·(1−A)
            ─▶ fesom_ice_thermo.c:509  flx_fw = prec + evap + fwice + fwsnw   (incl. runoff)
            ─▶ fesom_ice_coupling.c:139 water_flux = −flx_fw                  (OVERWRITES the bulk evap−prec)
            ─▶ fesom_sss_runoff.c:391   virtual_salt = rsss·water_flux        (now incl. runoff)
            ─▶ bc_S = dt·(virtual_salt + relax_salt)  → linfs salinity BC → river-mouth freshening (advects)
```

With `FESOM_NO_ICE_THERMO=1` (Phase 5) that whole block is gated off, so `water_flux` stays
the **bulk** `evap−prec` (no runoff) and `virtual_salt = rsss·(evap−prec)`; the standalone
balance `water_flux += ⟨water_flux+runoff⟩` still runs but is **inert in linfs** (the only
`water_flux` consumers are the non-linfs `ssh_rhs`/ALE paths). Net: runoff has **zero** effect
on the Phase-5 linfs trajectory. (Consequence to accept for Phase 5: no river freshwater ⇒ a
salty coastal/Arctic bias vs. a complete run.)

**What is ALREADY done (needs no change for Phase 6):**
- The runoff **reader** — `sss_runoff.runoff_node` (bit-exact, Task 5.5), carried in
  `core2_forcing.ForcingStatic.runoff_node`.
- The **salt/water balance** — `sss_runoff.sss_runoff_fluxes(water_flux, …, runoff_node, …)`
  is **pure in `water_flux`** (Task 5.5, dump-verified). The global-mean balance already
  consumes `runoff_node`.

**What Phase 6 adds to activate runoff (the ONLY missing links):**
1. Port `fesom_ice_thermodynamics` (the ice obudget) so it folds runoff into `flx_fw`
   (`prec = rain + runo + …`) — this is part of "port sea ice" anyway.
2. Port `fesom_ice_oce_fluxes` so it sets `water_flux = −flx_fw` (incl. runoff) + the
   heat-flux ice blend — also part of porting sea ice.
3. **The JAX seam is already clean:** in `core2_forcing.compute_surface_fluxes`, the ice-on
   branch computes `flx_fw` and passes `water_flux = −flx_fw` into the EXISTING
   `sss_runoff_fluxes` (instead of the bulk's `evap−prec`). No restructuring of the reader or
   the balance — just feed a different `water_flux`. `runoff_node` is already plumbed.
4. **No double-count:** follow the C3b design exactly (runoff lives in `flx_fw`; the sbc's
   local `−runoff` term stays removed; the balance's `+runoff` is the global-mean term only).
   Verify with the proven dump recipe (dump `flx_fw`/`water_flux`/`virtual_salt` at
   river-mouth nodes, gate JAX vs the **ice-on** C run).

**So:** runoff "comes online" automatically the moment Phase 6 ports the ice freshwater
budget; nothing in the Phase-5 runoff code needs to be revisited or undone. If a future
decision wants runoff in a *no-ice* run, that is a C-side change to the no-ice sbc branch
(add the local runoff to `virtual_salt`) — match whatever the C is then made to do.

## Out of scope (deferred — NOT in the C reference)

zlevel / zstar ALE, local-zstar fallback, partial cells, `Z_3d_n`, the `w_i` advective
terms, GM/Redi, KPP, sea ice. zstar is the user's future intent and needs C-side changes
first. GM/KPP/ice are Phase 6. **Runoff activates with the ice freshwater budget in Phase 6 —
see "Runoff handoff to Phase 6" above (reader + balance already done; pure-in-`water_flux`
seam).**

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
- **2026-06-06 — Task 5.2 DONE (PHC initial conditions).** `fesom_jax/phc_ic.py` — faithful
  numpy port of `fesom_phc.c` (bilinear interp + sequential-GS `extrap_nod3D` + vertical
  fill + Bryden-1973 `ptheta`); `build_and_cache_ic` + `core2_initial_state`. Verified vs the
  C surface dumps (job 25386555): bracket indices EXACT, pre/post-load surface **~1e-14**
  (GS replicated exactly — the #1 risk cleared). Cache `data/ic_core2/`; `test_phc_ic.py`
  5 passed. **Env: netCDF4 installed** (user-approved; numpy 2.4.6 / jax 0.10.1 unchanged;
  benign `ndarray size changed` ABI warning). New `port2/jobs/jax_phc_dump_core2.sh`.
  Next: **Task 5.3 (JRA55 forcing reader)**.
- **2026-06-06 — Task 5.4 DONE (L&Y09 bulk formulae, AD-safe).** `fesom_jax/forcing.py` —
  `ncar_ocean_fluxes_mode` (fixed-5 unrolled) + `obudget` + `bulk_surface_fluxes` (node→elem
  mean-of-3). Verified vs a new C `bulk_dump_*` all-node dump (job 25389451, year 1958) at 3
  configs (zero + synthetic current, day1 + day100/noon): **cd/ce/ch ~1e-17, heat_flux ~6e-13,
  stress ~5e-16** over all 126858 nodes. `test_forcing.py` 10 passed. ⚠️ **The sub-plan's
  "drop the break ⇒ identical result" assumption was WRONG** — the M-O loop is non-convergent
  at calm nodes (`ch` diverges up to **88%** fixed-5-vs-early-break), but the **physical** impact
  is bounded (heat_flux ≤7.2 W/m² at ~4 nodes; <0.1 W/m² for 126848/126858). JAX runs fixed-5
  (AD-safe; a data-dependent `while`-break is not reverse-mode diff'able) verified against a
  **fixed-5** C dump via a new `FESOM_BULK_FIXED_ITERS` env gate; **Task 5.7's per-substep
  reference must set that flag.** AD-safe `x2=sqrt(max(|1−16ζ|,1))` (bit-identical + smooth through
  ζ=1/16), double-`where` safe-sqrt for `u`/`mag`, literal `jnp.copysign` switches; the deliberate
  relative-vs-absolute wind mismatch preserved + validated by a synthetic-current dump mode.
  `USE_SW_PENE=1` confirmed ⇒ sw penetration deferred to Task 5.6 (heat_flux = pre-pene `qns−qsr`).
  C: `fixed_iters` param + `fesom_bulk_dump` (+`T_oc`/early-break columns) + `jax_bulk_dump_core2.sh`.
  **Artifacts moved to `/work`** (`data → /work/.../port_jax/data` symlink; user rule). Next:
  **Task 5.5 (SSS restoring + runoff)**.
- **2026-06-06 — Task 5.3 DONE (JRA55 forcing reader).** `fesom_jax/jra55.py` — faithful numpy
  port of `fesom_jra55.c` (julday/binarysearch + per-field time-grid transform with the
  mid-interval shift + shared bilinear stencil + per-field `getcoeffld` cache + `step` + the
  g2r wind rotation). Verified vs a new C all-node dump (job 25388630, year 1958,
  `data/jra_dump_core2/`) at two dates (day1/sec0 boundary + day100/12:00 interior): **6 scalar
  fields bit-exact** (max|diff|=0, all 126858 nodes), wind ~3.5e-15. `test_jra55.py` 5 passed.
  **#1 fidelity trap cleared:** the C time-interp `field=rdate·coef_a+coef_b` cancels two ~2.4e6
  Julian-day numbers, so a folded-`1/denom` bilinear gather's ~1e-13 reassociation amplified to
  ~6e-8 — fixed by a **bit-identical** `(s·dx)·dy`-order + divide-at-end gather; the interior
  dump (not just the t=0 boundary, which degenerates to `field=d1`) is what exposed it. C-side:
  `dump_jra_fields` in `fesom_main.c` (gated `FESOM_JRA_DUMP_DIR` + `_DAY`/`_SEC`) +
  `port2/jobs/jax_jra_dump_core2.sh` (untracked). `flip_lat=0` (lat ascending); field order
  uas,vas,huss,rsds,rlds,**tas**,prra,prsn. Next: **Task 5.4 (L&Y09 bulk formulae, AD-safe)**.
- **2026-06-06 — Task 5.5 DONE (SSS restoring + runoff).** `fesom_jax/sss_runoff.py` —
  faithful numpy readers (`interp_2d_field` lat-clamp/lon-cyclic-wrap bilinear +
  `read_other_NetCDF` 30-cell expanding missing-fill via `scipy.ndimage.uniform_filter`)
  → `Ssurf_clim[12, nod2D]` + `runoff_node[nod2D]` (/1000); + the AD-safe JAX flux math
  `sss_runoff_fluxes` (virtual_salt/relax_salt with area-weighted global-mean subtraction +
  the water balance). Verified vs a new C all-node dump (job 25390216, year 1958,
  `data/sss_dump_core2/`) at 2 months (m1 Jan/day1 + m4 Apr/day100 month-crossing): **runoff
  bit-exact** (max|Δ|=0); **Ssurf bit-exact at 105148/126858 ocean-bracket nodes** (95p
  ~3.6e-14, max 2.8e-12 at ~35 coastal/fill-bracket nodes); **virtual_salt/relax_salt/
  water_flux ~1e-20..1e-22** (the global-mean reductions barely reassociate — fed the C's own
  inputs, the ÷`ocean_area`=3.6e14 crushes the integral). Both differentiable seams flow
  (`d/d(water_flux)`, `d/d(S_top)`). `test_sss_runoff.py` 9 passed. New C
  `fesom_sss_runoff_dump` (gated `FESOM_SSS_DUMP_DIR`/`_DAY`/`_SEC`/`_MONTH`) +
  `jobs/jax_sss_dump_core2.sh`. `ref_sss_local=1`, `surf_relax_S=1.929e-6`; no legacy month
  +1. Next: **Task 5.6 (wire surface BCs into the step + assemble CORE2 forcing)**.
- **2026-06-06 — Task 5.6 DONE (wire surface BCs + assemble CORE2 forcing).** New
  `fesom_jax/core2_forcing.py` (the per-step driver: bulk → sss_runoff → ice stress blend →
  cal_shortwave_rad → bc_T/bc_S/sw_3d/stress_surf; host readers + device AD-safe math).
  `tracer_diff` gained `bc_T`/`bc_S`/`sw_3d`; `forcing` gained `cal_shortwave_rad`;
  `sss_runoff` gained `build_chl_clim`; `step`/`integrate` thread `step_forcing`/
  `forcing_static` (pi path `None` ⇒ bit-identical). Verified vs a new CORE2 per-substep C
  dump (job 25391647, 7 probes incl. Aleutian 94122): **step-1 T 7.1e-15 / S 2.1e-14**
  (bit-exact), surface forcing 1e-13..1e-22, steps 2-3 ~1e-9. `test_surface_bc.py` (7) +
  `test_core2_step.py` (5). **chl = Sweeney monthly** (C default). **Three bugs found+fixed:**
  (1) ⚠️ the C "no-ice" run keeps a **static `a_ice=0.9` mask** (`fesom_ice_initial_state`,
  IC SST<0) that gates cal_shortwave penetration + blends the momentum stress — **user review
  decision: match the C** (replicate the mask, not truly-ice-free); (2) step-1 `T_old` is the
  **constant base 10/35**, not PHC (pi-blob analog); (3) `fesom_bulk_compute` ignored
  `FESOM_BULK_FIXED_ITERS` (only the dump fn honored it) → fixed-5-vs-early-break divergence
  at calm/cold nodes (now wired). **Finding:** the balanced `water_flux` is inert in linfs ⇒
  **runoff has no effect on the Phase-5 trajectory** (it feeds only the non-linfs ssh_rhs/ALE
  paths). C edits on the `jax-mesh-export` branch; `jobs/jax_step_dump_core2.sh` untracked.
  Next: **Task 5.7 (matched C dump run + CORE2 stability, 1-day + multi-day)**.
- **2026-06-06 — Task 5.7 DONE (matched C dump run + CORE2 stability).** Two deliverables.
  **(A) Per-substep dynamics gates** (`test_core2_step.py`): `test_step1_dynamics_per_substep`
  gates pressure/PGF/Av/uv_rhs/ssh_rhs/d_eta/uv/hbar/eta_n/w/hnode at step 1 — **bit-exact
  class** (pre-solve ~0..1e-17; CG-derived ~1e-16..8e-15; big intermediates ssh_rhs ~1e5 /
  pressure ~5e5 match ~1e-11 relative); element fields compared at the dump's incident-element
  gids (`_emaxabs`); `test_evolution_steps23` extended with uv/d_eta (steps 2-3 ~1e-6 — the
  discrete CG iter-count + FCT amplify the step-1 ~1e-15). Suite **371** (was 370). **(B) CORE2
  stability run** (`scripts/core2_stability_run.py` + `core2_stability_gpu.sh`, A100 jitted
  ~0.06 s/step; eager ~32, CPU ~3): **numerically stable days 1–7** (no NaN; max|vel| ≤ 1.9
  < 3; |SSH| ≤ 2.8 < 5; Aleutian 94122 calm/warm). **(C) Matched C arbiter**
  (`jobs/jax_core2_stability.sh`, same config + `FESOM_PRINT_EVERY` monitor): stable, and JAX
  **tracks it to 3 sig figs** on SST_min/max|uv|/max|eta| (step 216: −6.60=−6.60, 1.389≈1.39,
  2.715≈2.71) despite per-element chaotic divergence — robust min/max reductions track the
  shared forced response. **FINDING (anticipated risk #1, NOT a bug):** no sea ice ⇒ SST
  supercools without bound (−1.9 IC → −16.5 d5 → −22.8 d8); past ~−20 °C the JM-EOS is invalid
  → spurious convection → max|vel|>3 at model day ~8.1. The C does the same ⇒ ice stays Phase 6
  (the "C blows up ⇒ ice into Phase 5" trigger did NOT fire); a physical SST simply needs the
  ice cap. C job untracked on `port2` `jax-mesh-export`; JAX driver + GPU job committed on
  `main`. Next: **Task 5.8 (GATE 5 — gradient on a CORE2 slice)**.
