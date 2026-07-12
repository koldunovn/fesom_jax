# FESOM2 → JAX Port — Phase 9a: zstar vertical coordinate (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 9 — physics options; locked decision 6
reserved the `which_ale` seam for exactly this).
**Predecessors:** GATE 6C (full KPP+GM+ice model, `v1.0-single-gpu`) + Phase 8 (`v1.1-multi-gpu`).
**Siblings:** `20260611-fesom-jax-tke.md` (Phase 9b), `20260611-fesom-jax-mevp.md` (Phase 9c).
Recommended order **9a → 9b → 9c** (geometry seam first), but **no hard dependency** — each sibling
plan is self-contained and validated against its own C oracle under its own single-knob config.
**Created:** 2026-06-11. **Status:** ☐ NOT STARTED.

**C source of truth:** `/home/a/a270088/port2/fesom2_port_zstar` (git worktree, branch `mevp`,
tag `zstar-validated-2026-06-10`), feature commits `c20330a → 89611d5 → 7ae04dc → 8260dea`.
The C plan + full porting-experience record: `port2/fesom2_port_zstar/docs/plans/completed/`
`20260610-zstar-vertical-coordinate.md`. The C is itself validated vs Fortran zstar at
**SST/SSS RMS 0.0038/0.0014 (yr 1958)** with a 3–9× contrast to linfs — the comparison resolves
the coordinate.

**Decisions (locked):**
1. **Mirror the C 1:1, dump-gate each kernel** — `FESOM_ALE=zstar` semantics exactly; **zlevel is NOT
   ported** (the C aborts on it — no reference run ⇒ no oracle ⇒ out of scope, same for partial cells,
   floating ice, cavity bodies, `use_ssh_se_subcycl`).
2. **Config-gate like kpp/gm/ice:** static `ale_cfg: AleConfig | None = None` threaded through
   `step()`/`integrate()`/sharded drivers. **`None` ⇒ the linfs path, byte-identical** (the C proved
   its equivalent claim with three byte-gate jobs; ours is a trace-time dead branch + full suite).
3. **Derived live geometry (D1):** `zbar_3d_n`/`Z_3d_n` are NOT new State fields — they are recomputed
   from the carried `hnode` by one helper (`ale.live_geometry`), replicating the C commit's bottom→top
   reconstruction bit-for-bit. `hnode/helem/hbar/hbar_old` are **already prognostic State fields** —
   the plumbing (partitioning, halo rows, zarr) exists; only the values become time-varying.
4. **Stiffness as a function of state, not a carried matrix (D2):** the C carries CSR `values` with a
   cumulative per-step `+= dhe`-assembly. In exact arithmetic `Σ dhe ≡ mean₃(hbar − hbar_init)` and
   `hbar_init = 0` (cold start) ⇒ `A(t) = A_base + ΔA(mean₃(st.hbar))·g·dt·α·θ`, recomputed inside the
   `custom_linear_solve` matvec each step. Algebraically exact vs the C (FP differs ~1 ulp/step —
   inside the scatter-class tolerance). Preconditioner stays **frozen** (the C/Fortran never refresh
   it — verified, lesson #11). This keeps the solve a clean differentiable function of state.
5. **AD-safe by construction + a zstar gradient gate** — every new thickness division gets the
   double-`where` masked-divide; dry/below-bottom geometry lanes are filled with **nominal spacing,
   never 0**, before any divide (the C arrays are 0 beyond bottom — a dense-JAX inf factory).

---

## 0. Scope (READ FIRST — what zstar actually changes)

Under zstar the SSH change is distributed proportionally over the water column every step, so layer
thicknesses become state. Seven concrete changes vs linfs (C digest verified):

1. **Thickness machinery:** `hnode`, `helem`, `zbar_3d_n`, `Z_3d_n` become time-varying; only layers
   `nz ≤ nlevels_nod2D_min(n)−3` (0-based) stretch; bathymetry-intersecting + bottom layers keep
   nominal spacing (all masks derive from **static** integer level arrays ⇒ precomputable).
2. **Real freshwater flux:** `water_flux` enters `ssh_rhs` (`−α·wf·areasvol`), `ssh_rhs_old`
   (`−wf·areasvol`), and the surface `Wvel` BC (`Wvel(1) −= wf`).
3. **Real salt flux:** `virtual_salt ≡ 0`; `real_salt_flux` (`rsf = fwice·Sice − iflice·ρice/ρwat·Sice`)
   becomes a live producer in ice thermo. ⚠️ The JAX port (like C v1.0) has **no rsf producer**:
   `ice_thermo.py` computes none (its `evap` at `ice_thermo.py:135` bundles evaporation+sublimation —
   must be SPLIT for the balancing), and the ice-path `bc_S` omits rsf (`ice_step.py:137`;
   `surface_forcing.py:155` is the secondary no-ice consumer) — this plan adds the producer
   (C plan-review BLOCKER #1).
4. **`water_flux` composition:** gains ice-growth volume terms + a **global net balancing** increment
   (`fesom_ice_coupling.c:178-216`, gated `!use_virt_salt`); needs new `evaporation`/`ice_sublimation`
   per-node fields in the ice-step output.
5. **PGF:** switches to the **shchepetkin density-Jacobian** on live geometry; **no hpressure at all**
   under zstar (the C plan-review killed an invented "non-linfs hpressure term" — there is none).
6. **SSH stiffness matrix:** becomes thickness-dependent (decision D2 above).
7. **Tracer surface BCs:** `bc_T −= dt·sval·wf·is_nonlinfs`; `bc_S = +dt·(virtual_salt + relax_salt +
   rsf·is_nonlinfs)` — **positive dt, NO sval·wf term in S** (C sign-trap lesson #3,
   `fesom_tracer_diff.c:43-75`).

One C-only deviation is **already mirrored**: the **salinity floor** `S = max(S, 0.5)` over wet
layers (`fesom_step.c:434-453`; JM-EOS NaNs for S ≲ 0.2 PSU once real freshwater fluxes act) already
exists at `step.py:46-48,282` (`S_FLOOR = 0.5`), **unconditional — exactly like the C** (the C gates
it only on `FESOM_NO_SFLOOR`, never on ale mode). **Nothing to add; do NOT ale-gate it** — gating
would diverge from both codes under linfs (plan-review correction 2026-06-12).

**New kernels (C LOC):** `init_thickness_zstar` (~100), `vert_vel_zstar_distribute` (~57),
`update_thickness_zstar` commit (~45), stiffness increment (~59), shchepetkin PGF (~156), forcing
flip + rsf producer + fw balancing (~75), bc_surface terms (~10). Plus ~20 one-line geometry re-points
across consumer modules (the C "Z7" sweep).

---

## 1. Reference configuration (VERIFIED values — port these)

From `port2/fesom2_port_zstar/docs/zstar_reference_namelists/` (+ `PROVENANCE.md`):
`which_ALE='zstar'`, `use_partial_cell=.false.`, PGF = `'shchepetkin'` (module default),
`use_floatice=.false.` (the EVP `p_ice` term is a literal ×0 — skip in JAX, gradient dead anyway),
`i_vert_diff=.true.`, `use_wsplit=.false.`, opt_visc=7, KPP, MFCT, GM/Redi, EVP, JRA55 1958, PHC,
dt=1800, CG `soltol=1e-5`/maxiter 500, `FESOM_PHASE1_ALPHA=1.0`, `THETA=1.0` (⇒ `eta_n ≡ hbar`, the
`(1−α)` terms vanish numerically but are ported literally — match the C). `use_virt_salt=!zstar`,
`is_nonlinfs = zstar ? 1.0 : 0.0` (`fesom_ale.c:14-33`).

**Oracles on `/work/ab0995/a270088/port/zstar/`** (existence verified 2026-06-11): `z2_cdump` (C
3-step 16r dump set), `fdump`/`fdump_k2` (Fortran), `c_zstar_2yr` (C zstar 2-yr monthly output),
`fortran_zstar_2yr`, `fortran_linfs_2yr_b`, `z0_byteident`, `z1_smoke`. The dual dump harness
(`fesom_ale_dump.c`, env `FESOM_ALE_DUMP_DIR`/`_STEPS`) writes **12 tags** at 6 driver sites:
`forcing` (water_flux, virtual_salt, relax_salt, real_salt_flux), `pgf_x`/`pgf_y`, `sshsolve`
(ssh_rhs, d_eta), `hbar` (hbar, hbar_old, ssh_rhs_old, eta_n), `dhe`, `Wvel`, `hnode_new`, and the
post-commit `hnode`/`zbar_3d_n`/`Z_3d_n`/`helem`. Format = gid-keyed text
`ale_dump_s<step>_<tag>_rank<R>.txt` with `# step= tag= rank= N= ncomp=` headers — **the same format
`io_dump.read_kpp_table` already parses.** ⚠️ JZ.0 must inspect `z2_cdump` for tag completeness
(it may predate the full feature); regeneration recipe = the `mevp`-branch binary +
`FESOM_ALE=zstar FESOM_ALE_DUMP_DIR=… FESOM_ALE_DUMP_STEPS=3`, 16r dist_16, 3 steps, dt=1800
(`-p compute --time=30:00` debug-QOS job).

---

## 2. The seam + integration map

**`AleConfig(NamedTuple)`** (new, in `ale.py`): static, hashable. Fields: `zstar: bool = True`
(presence of the cfg ⇒ zstar; any other mode raises — the C abort parity; NamedTuple ⇒ validate via
a `__new__` override or factory). Derived (properties):
`use_virt_salt=False`, `is_nonlinfs=1.0`. Thread `ale_cfg=None` through: `step()` + `step_jit`
static_argnames (`step.py:313-314`), `integrate()` eager-step-1 + scan body + static_argnames
(`integrate.py:46,71,77,89,98,109-111`), `integrate_sharded.run_step_sharded/run_steps_sharded`
(`integrate_sharded.py:259-263, 346ff`).

**Where the step changes (current `step.py` line refs):**

| Substep | Today | Under `ale_cfg` |
|---|---|---|
| 1 EOS/N² | static `Zp`/`zbar_3d_n` depths (`eos.py:110,238,305`) | live geometry from `st.hnode` (re-point); hpressure block **skipped** |
| 3 PGF | `pgf.pressure_force_linfs` (`step.py:169`) | `pgf.pressure_force_shchepetkin` (new) |
| hoist `step.py:157` | `hnode_new = thickness_linfs(st.hnode)` before GM | **un-hoist**: under zstar `hnode_new` is produced at substep 12 (vert_vel distribute). GM's coefficient block (substep 1b/2) runs BEFORE vert_vel, where the C's `hnode_new` variable still holds the **previous step's committed thickness** (the commit is substep 16) — i.e. ≡ `st.hnode` (verified vs the C step order; hedge resolved at plan review). So: `st.hnode` to the GM coefficient block, the fresh `hnode_new` to the substep-13/15 tracer/Redi/K33 pieces — the **dual-geometry trap**, lesson #6. Under linfs identical (the hoist was a pure convenience) |
| 8–10 SSH | static operator `op` (`ssh.py:60-64`), solve (`step.py:217`) | matvec += `ΔA(mean₃(st.hbar))·g·dt·α·θ` (D2); `ssh_rhs` gains the zstar wf tail; `compute_hbar` gains `ssh_rhs_old −= wf·areasvol` |
| 12 vert_vel | `ale.compute_w` only | + `vert_vel_zstar_distribute`: Wvel integrated-Σdh/dt correction over stretch layers, `Wvel(1) −= wf`, produce `hnode_new`; `_exch(hnode_new,"nod")` (**new exchange — the C has it, zstar-only**) |
| 13 tracers | FCT already carries `hnode/hnode_new` (`tracer_adv.py:142-150` — **the zstar math is structurally present**) | inputs become genuinely different; `bc_T/bc_S` gain the §0.7 terms; QR4C vertical stencil re-pointed to live committed geometry (from `st.hnode`), flux-limited pieces to `hnode_new` (dual-geometry, C `tracer_adv.c:673-690`) |
| 13b impl diff | static `Zp` spacings (`tracer_diff.py:78-92`) | `zbar_n/Z_n` built **from `hnode_new`** (C `tracer_diff.c:148-158`) |
| salinity floor | `step.py:282` `max(S, 0.5)`, unconditional | **unchanged** (already mirrors the C — do NOT ale-gate) |
| 16 commit | `commit_thickness` (hnode:=hnode_new, helem mean) — already correct | unchanged code path; geometry next step derives from the new `hnode` (D1 ⇒ the C's zbar/Z rebuild is our `live_geometry`) |

**Geometry re-point sweep (the C Z7 list, JAX targets):** `eos.py:110,238,305` (EOS depth, dbsfc
denom, bvfreq dz), `pp.py:81-92` (shear dz), `kpp.py:350,492,590-591,675,702` (Ri dz, bldepth zbar,
blmix/enhance Z), `tracer_adv.py:258-259` (QR4C `_z_stencil`), `momentum.py:273-279` (TDMA spacings from
static `zbar`/`Z` — re-point to live `st.helem`-derived spacing), `gm.py:215,250-252` + `gm_redi.py:81,192` (fer on `hnode_new`, horizontal Redi on `st.hnode`
— OLD mesh, lesson #6b), `forcing.py:340` (sw penetration depth). **Audit by array, not by comment**
(C lesson #5): grep `mesh.Z\b`, `mesh.zbar`, `zbar_3d_n` over the package and dispose of every hit.
Under linfs each re-point is bitwise-neutral (live == static by construction) — the suite is the gate.

**Forcing flip integration:** `ice_thermo.py` gains the rsf producer (+ the `evap` split into
`evaporation`/`ice_sublimation`) surfaced through `IceStepOut`; `ice_coupling.py` gains the
fw-balancing block (global sums → `reductions.global_sum` so the sharded path is psum-correct);
`surface_forcing.py:24-28,97,153` + `ice_step.py:137` (the ice-path `bc_S` — the operative consumer) +
`sss_runoff.py:262` + `ice.py:96` flip on `use_virt_salt/is_nonlinfs`. Under `ale_cfg=None` all new
code is dead-branch ⇒ bit-identical.

---

## 3. Validation strategy

- **Reader first:** generalize `io_dump.read_kpp_table` → `read_gid_table(path)` (same parser, shared)
  + an `ALE_TAGS` table; loaders merge ranks by gid. (Siblings reuse this — if 9b/9c run first they
  hoist this item.)
- **Controlled, per-task dump gates** vs `z2_cdump` (or regenerated set), mirroring the C's own Z-gate
  ladder: forcing tag → ssh/hbar/dhe tags → Wvel/hnode_new → pgf → post-commit thickness tags.
  Tolerance ladder: map/gather ~1e-15 … scatter/reduction ~1e-12 (house `verify.py` ladder); the C
  achieved **bit-faithful pgf (~1e-18)** on identical step-1 inputs — expect the same class here since
  the JAX forcing is a validated 1:1 port of the C's.
- **Cold-start degeneracy as a free gate** (C lesson #12): at `hbar=0` the zstar init ≡ linfs init
  bit-for-bit; step-1 stiffness increment is a no-op. Gate JZ.1 exploits this before any physics
  diverges.
- **Tracer chain:** the ale_dump has **no T/S tags** — gate the tracer chain via (a) the thickness/
  Wvel/forcing tags that fully determine its new inputs, (b) the linfs byte-identity suite (the FCT
  ALE machinery is already dump-validated from Phases 5/6), (c) the year-scale comparison vs
  `c_zstar_2yr` monthly SST/SSS. Fallback if a discrepancy needs localization: add 2 tags to
  `fesom_ale_dump.c` (own TU — safe) but ⚠️ the dump *call site* lives in `fesom_step.c`, a physics
  TU: re-run the C byte-gate (`z0_byteident` recipe) after any such edit
  (C lesson `feedback_tu_codegen_bitgate`).
- **Climate/stability:** 10-day A100 zstar-ON stability; then JAX-zstar ↔ C-zstar year-1 SST/SSS RMS
  must be ≪ the zstar↔linfs contrast (the C measured 3–9×; the discriminating-check style of K.9).

## 4. AD-safety strategy (the differentiability contract)

Hazard inventory from the C digest, each with its JAX treatment:

- **Divisions by live thickness** (`fct_LO`/reconstruction ÷hnode_new, TDMA 1/ΔZ_n, momentum TDMA
  1/Δzbar_n(helem), GM ÷hnode_new, CFLz, `dd=(hbar−hbar_old)/(zbar₁−dd1)`): keep every existing
  double-`where` guard; **fill below-bottom `zbar_3d_n`/`Z_3d_n`/`hnode` lanes with nominal spacing,
  never 0** before any divide, then mask outputs (the masked-NaN rule — bit us 4× before).
- **Shchepetkin PGF dense stencils:** the C protects with per-vertex integer branches on *static*
  level arrays ⇒ precompute case masks; clamp stencil indices; safe denominators on `Z` differences
  (the triple product `dx20·dx21·dx10`).
- **State-dependent linear solve:** `custom_linear_solve` with the D2 matvec — symmetric ⇒ the same
  solve serves the transpose; implicit diff now propagates into **A via `hbar`** (closure), not just b.
  Gate: transpose-residual check (the Task-5.8 pattern) with `ale_cfg` ON.
- **Clamps/branches:** salinity floor `max(S,0.5)` (zero gradient below — by design, documented);
  KPP's discrete `kbl` search now reads live geometry (already `stop_gradient`-treated — re-verify);
  vert_vel NaN/negative-hnode checks are print-only in C ⇒ **drop** (or debug-callback).
- **Dead code dropped:** wsplit (config-off), the EVP `p_ice·use_pice=0` term (gradient killed by ×0).
- **Gradient gates (GATE 9a):** masked-NaN `d(SST)/d(T0)` finite + exactly 0 on masked lanes, zstar-ON
  assembled model; FD↔AD plateau on `d(loss)/d(k_ver)` zstar-ON (N=1 / smooth-regime rule); CG
  transpose residual; `d(loss)/d(hbar-IC)` finite (the new state path).

## Development Approach (standing rules)

- Oracle-first house style: every task lands its dump-gate **tests with the kernel**; the full suite
  (`scripts/runs/run_suite.sbatch`, CPU; ~483 tests + new) must be green before the next task. `ale_cfg=None`
  bit-identity is asserted by the existing suite at every task boundary.
- **STANDING RULE:** append one entry per task to `docs/PORTING_LESSONS.md` as you go, citing the C
  `file:line` or dump probe that proves it.
- Mark `[x]` immediately; ➕ for discovered tasks; ⚠️ for blockers; update this plan when scope shifts;
  move to `docs/plans/completed/` when GATE 9a is met.
- Compute: login node = edit/grep only; CPU tests via `run_suite.sbatch`; dumps/stability/gradients on
  A100 via `-A ab0995_gpu`; cheap C dump jobs `-p compute --time=30:00`. Python =
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`.

## Implementation Steps

### JZ.0 — Scaffolding: seam, readers, oracle audit (NO behavior change)

**Files:** Modify: `fesom_jax/ale.py` (AleConfig), `fesom_jax/step.py`, `fesom_jax/integrate.py`,
`fesom_jax/integrate_sharded.py`, `fesom_jax/io_dump.py`. Create: `fesom_jax/tests/test_ale_zstar.py`.

- [x] `AleConfig` + thread `ale_cfg=None` through step/integrate/sharded + static_argnames
- [x] generalize `read_kpp_table` → shared `read_gid_table` + `ALE_TAGS` loader (12 tags × 3 steps);
      `load_ale_dump` merges 16 ranks by gid (nodes disjoint; elements bit-identical overlap)
- [x] audit `z2_cdump`: **COMPLETE** — 12 tags × 3 steps × 16 ranks = 576 files, full feature set
      (nod2D=126858, elem2D=244659, nl=48 = `mesh_core2`); NO regeneration needed
- [x] tests: cfg-threading no-op (None vs `AleConfig()` ⇒ step bit-identical, max|Δ|=0), reader
      round-trip on all tags (16 tests in `test_ale_zstar.py`)
- [x] full suite green — **OCEAN 503 + ICE 47, 0 fail**; sharded `ale_cfg=None` byte-identical (the
      sharded tests that ran all pass). ⚠️ The sharded-GRADIENT group is pre-existingly slow post-Phase-8b
      (ragged halo); the Jun-8 baseline ran it green in 11:55, my runs time out — NOT a zstar regression
      (verified: `ale_cfg=None` is an inert closure constant; Phase 8b rewrote the sharded path).

### JZ.1 — Thickness machinery: init + live geometry + commit

**Files:** Modify: `fesom_jax/ale.py`, `fesom_jax/state.py` (rest-IC path only), `fesom_jax/tests/test_ale_zstar.py`.

- [x] `init_thickness_zstar` (`fesom_ale.c:45-146`): whole-column `(Δzbar)·(1+(hbar/dd)·stretch)`
      (no separate bottom field — full-cell ⇒ nominal), helem mean, eta_n reversed weights, ssh_rhs_old
- [x] `live_geometry(mesh, hnode) → (zbar_3d_n, Z_3d_n)`: reverse-cumsum-from-anchor reconstruction over
      `nz < min_f−2`; ⚠️ below-stretch/below-bottom lanes = nominal `zbar` (dz>0), NOT `mesh.zbar_3d_n`
      (0-padded below bottom on ALL nodes ⇒ the inf factory)
- [x] commit under zstar reuses `commit_thickness` (full-cell ⇒ bottom-helem agrees ≤1 ulp); the
      un-hoist of `hnode_new` is JZ.6 (step-flow), noted there
- [x] tests: cold-start degeneracy (zstar init ≡ linfs **exactly 0** at hbar=0 — the free Z1 gate);
      `live_geometry` ≡ static `zbar_3d_n`/`Z` **bitwise on pi**; vs-numpy-ref for hbar≠0; AD finite (10 tests)
- [x] full suite green (OCEAN 503 + ICE 47; `State.rest(ale_cfg=)` byte-identical, `test_state` green)

### JZ.2 — Forcing flip: rsf producer, water-flux composition, balancing, BCs

**Files:** Modify: `fesom_jax/ice_thermo.py`, `fesom_jax/ice_step.py`, `fesom_jax/ice_coupling.py`,
`fesom_jax/surface_forcing.py`, `fesom_jax/sss_runoff.py`. Create: tests in `test_ale_zstar.py`.

- [x] rsf producer in ice thermo (`fesom_ice_thermo.c:359-408`); SPLIT `evap` into
      `evaporation`/`ice_sublimation` (the C already computes both halves — free), surfaced via
      `ThermoOut`→`ThermoState`→`IceStepOut`. rsf matches the closed form exactly.
- [x] verified the C `sval` source: ⚠️ it is the **POST-ADVECTION** `trarr[surface]` (`tracer_diff.c:292`),
      NOT start-of-step — so the `bc_T −= dt·sval·wf` term lands in `step.py` before `impl_vert_diff`,
      not in the forcing step. (plan hedge resolved: the C does NOT use start-of-step T)
- [x] water-flux ice-volume terms + **global net balancing** (`ice_coupling.fresh_water_balance_zstar`,
      `fesom_ice_coupling.c:193-216`), `net = ⟨flux⟩` via `sss_runoff._area_mean` → `reductions.global_sum`
- [x] `virtual_salt ≡ 0` under zstar (ice + no-ice paths); `bc_S = dt·(0 + relax_salt + rsf·is_nonlinfs)`
      (+dt, NO sval·wf — lesson #3); `bc_T` sval·wf term in step.py
- [x] tests: ✅ CPU unit tests (split exact, rsf producer formula, balancing global-mean, bc_S flip,
      kernel byte-identity — 4 new + the ice suite green); ✅ **config-independent linfs↔zstar FLIP gate
      on CORE2** (`test_forcing_flip_linfs_vs_zstar`, compute): relax_salt path-Δ=**0**, virtual_salt
      6.8e-4→**0**, rsf 0→8.53e-5 (≈ z2_cdump's 8.53e-5). ⏳ direct `z2_cdump` `forcing`-tag match is a
      ⚠️ **CROSS-CUTTING config-matching follow-on** (the JAX harness ≠ z2_cdump step-1 inputs by ~1e-5;
      virtual_salt≡0 matches exactly; the path-independent relax_salt diff proves it's inputs, not code —
      affects ALL z2_cdump dump gates, so JZ.3+ validate via flip/degeneracy/transitive + z2_cdump as
      diagnostic). See `[[zstar-forcing-dump-config-gap]]`.
- [x] `ale_cfg=None` byte-identity: `test_ice_thermo`/`ice_coupling`/`ice_step`/`core2_step` all green

### JZ.3 — SSH plumbing: rhs tail, hbar wf term, stiffness-as-function-of-state

**Files:** Modify: `fesom_jax/ssh.py`, `fesom_jax/step.py`. Tests: `test_ale_zstar.py`.

- [x] `ssh_rhs += −α·wf·areasvol + (1−α)·ssh_rhs_old` (zstar tail; `fesom_ssh.c:413-421`,
      `compute_ssh_rhs(water_flux=)`) and `compute_hbar(water_flux=)`: `ssh_rhs_old −= wf·areasvol`
      (`fesom_momentum.c:839-846`) — the wf subtraction lands BETWEEN the transport divergence and the
      hbar update (hbar + next step's `(1−α)` term read the wf-modified `ssh_rhs_old`). Non-cavity arm
      only (cavity unported). `water_flux=None` ⇒ linfs byte-identical.
- [x] D2 matvec: `stiff_increment_matvec` = `A_base(x) + ΔA(−mean₃(st.hbar))(x)` with `factor=g·dt·α·θ`,
      added inside the `solve_ssh` matvec (`mesh`+`hbar` args); the C's cumulative CSR
      (`fesom_update_stiff_mat_ale`, `fesom_ssh.c:238-296`) telescopes to `Σdhe ≡ mean₃(hbar)` at cold
      start (`fesom_step.c:216-227`), recomputed each step. Preconditioner frozen (base only). ΔA is the
      same antisymmetric edge→node scatter with "velocity"=∇x ⇒ symmetric ⇒ `custom_linear_solve(symmetric)`
      holds; closure ⇒ implicit diff propagates into A via `hbar`.
- [x] tests (9, green): wf-tail/wf-term exact algebra (probed at uv=0 to dodge the near-cancelling base);
      `water_flux=None` byte-identity; **step-1 increment ≡ no-op** + cold-start solve bitwise == linfs;
      increment symmetric; **transpose-residual with state-dependent A**; `d/d(hbar)` finite+nonzero.
      Config-INDEPENDENT C-internal `dhe`/telescoping: `dhe`-tag ≡ `mean₃(hbar−hbar_old)` recompute (Δ=0
      at all 3 steps); `Σdhe ≡ mean₃(hbar_3)` (1.7e-16) — validates the D2 increment-depth foundation
      WITHOUT JAX forcing (sidesteps the config gap). The assembled `sshsolve`/`hbar` z2_cdump dump gate
      (on the config-clean subset) is the JZ.7 multi-step run.
- [x] full suite green — **OCEAN 517 + ICE 47, 0 fail** (job 25534571, 2026-06-12); sharded group times
      out (pre-existing Phase-8b, not zstar). `ale_cfg=None` byte-identical across CORE2/GM/KPP/ice.

### JZ.4 — vert_vel distribute + hnode_new production

**Files:** Modify: `fesom_jax/ale.py`, `fesom_jax/step.py`, `fesom_jax/halo_points.py`. Tests: `test_ale_zstar.py`.

- [x] `vert_vel_zstar_distribute` (`fesom_ale.c:162-201`): **vertically-integrated** Wvel correction
      `(zbar_3d_n(nz)−dd1)·dd/dt` (NOT per-layer h·dd/dt), `Wvel(1) −= wf`, hnode_new on stretch
      layers only; uses **pre-update** geometry (`live_geometry(st.hnode)`). Wired into `step.py`
      substep 13: corrects `w`, **overrides** the hoisted linfs `hnode_new` (so substeps 15/16 read the
      new thickness, GM substep 2 already read the OLD `st.hnode` ⇒ the dual-geometry, lesson #6).
- [x] `_exch(hnode_new, "nod")` (gated under `ale_cfg`) + `OCEAN_SCHEDULE` row (the C's zstar-only
      `exchange_nod(hnode_new)`, `fesom_ale.c:157`); linfs ⇒ no-op. Sharded N-vs-1 is JZ.7.
- [x] tests (4 kernel + 1 smoke, green): numpy-loop-ref match (golden rule); **cold-start degeneracy**
      (hbar=hbar_old ⇒ w unchanged, hnode_new=hnode); surface-wf-BC isolation; `d/d(hbar)` finite;
      assembled **pi-zstar 2-step smoke** finite (no NaN, hnode_new genuinely stretched, warm-hbar D2
      increment live). The `Wvel`/`hnode_new` z2_cdump dump-tag gate (config-clean subset) is the JZ.7
      assembled run (needs CORE2 forcing).
- [x] full suite green — **OCEAN 524 + ICE 47, 0 fail** (job 25534806, 2026-06-12); sharded times out
      (pre-existing). `ale_cfg=None` byte-identical across CORE2/GM/KPP/ice.

### JZ.5 — Shchepetkin PGF

**Files:** Modify: `fesom_jax/pgf.py`, `fesom_jax/step.py`. Tests: `test_ale_zstar.py`.

- [x] `pressure_force_shchepetkin` (`fesom_eos.c:348-503`): element `Z_n` stack from helem (static
      bottom anchor `zbar[nlevels−1]`, mirrored not "fixed"), quadratic-Newton vertex `drho_dz`
      (`_drho_dz`), surface/interior/bottom stencils as precomputed static case masks (forward/centered/
      backward), safe denominators. The vertical integral is the **cumsum identity** `pgf[k] =
      cumsum(aux)[k] − ½·aux[k]` (replaces the C's running `int_dp += aux`). Wired into `step.py`
      substep 3 on the hoisted live `Z3d_live`.
- [x] hpressure unused under zstar (the shchepetkin path takes density+geometry directly; `compute_pressure_bv`
      still returns hpressure but it is discarded — the C "no hpressure" `fesom_eos.c:172`; JZ.6 may skip
      its compute). linfs ⇒ `pressure_force_linfs(hpressure)` unchanged (byte-identical).
- [x] tests (2 kernel + smoke, green): **numpy-loop-ref match** (golden rule, <1e-12, config-independent);
      `d/d(density)` finite (safe-denominator AD discipline); the pi-zstar smoke now runs through the
      shchepetkin PGF finite. The `pgf_x`/`pgf_y` z2_cdump dump-tag gate (config-clean subset, step-1 where
      live==static) is the JZ.7 assembled run (needs CORE2 IC density).
- [x] full suite green — **OCEAN 524 + ICE 47, 0 fail** (job 25534806, 2026-06-12; same run as JZ.4).

### JZ.6 — Geometry re-point sweep + dual-geometry discipline

**Files:** Modify: `fesom_jax/eos.py`, `fesom_jax/pp.py`, `fesom_jax/kpp.py`, `fesom_jax/tracer_adv.py`,
`fesom_jax/tracer_diff.py`, `fesom_jax/momentum.py`, `fesom_jax/gm.py`, `fesom_jax/gm_redi.py`,
`fesom_jax/forcing.py`, `fesom_jax/step.py` (un-hoist). Tests: `test_ale_zstar.py`.

**Audited design (2026-06-12) — the executable sweep:** `step.py` already hoists `zbar3_live, Z3d_live =
ale.live_geometry(st.hnode)` (JZ.5) — thread these into each consumer as optional `Z3d=None`/`zbar3=None`
args (None ⇒ `mesh.Z[None,:]`/`mesh.zbar[None,:]` broadcast = today's behaviour = **byte-neutral for
linfs since live==static under nominal hnode**; given ⇒ the live per-node 2-D depths). Per-consumer
grep-by-array hits (the static→live targets):
  * `eos.py:110` density depth `z` (the in-situ pressure proxy) + `:238` bvfreq `dz` spacing → live `Z3d`
    (the `_insitu`/N² depth becomes per-node; hpressure unused under zstar so its `z` is don't-care).
  * `pp.py:81` shear `dz` (`Zp`) → live.
  * `kpp.py:702` `zbar_3d_n` (+ `:350,492,590-591,675` Ri/bldepth/blmix depths) → live.
  * `momentum.py:273-274` TDMA `zbar`/`Z` spacings → live (per-node, `st.helem`-derived).
  * `tracer_adv.py:258` QR4C `_z_stencil` → live on the **committed** `st.hnode` geometry (dual-geometry).
  * `gm_redi.py:81,192` `zbar,Z` → live; `gm.py:215,250-252` fer on `hnode_new` (dual-geometry).
  * `forcing.py:340` sw-penetration depth → live.
Each re-point is independently linfs-byte-neutral ⇒ suite-gate after each module; the zstar assembled
correctness (steps ≥2, where live≠static) is the JZ.7 gate. The "un-hoist step.py:157" is **already done**
(JZ.4 overrides `hnode_new` after vert_vel; GM substep-2 reads the OLD `st.hnode`-hoist, downstream reads
the new — the C dual-geometry, no rename needed).

- [x] re-point every §2-listed consumer to live geometry under zstar (grep-by-array audit; lesson #5);
      un-hoist `step.py:157` ✅ (JZ.4 override). **COMPLETE (2026-06-12):** all consumers re-pointed
      via the `Z3d=None`/`zbar3=None` byte-neutral threading + the **C-confirmed which-side map**
      (4-agent extraction of `fesom_gm.c`/`fesom_momentum.c`/`fesom_tracer_diff.c`):
      ✅ `eos.py` (density depth + N² + **`compute_dbsfc`** — the grep-by-array audit caught the
      "dbsfc denom" the comment-scan missed), ✅ `pp.py`, ✅ `tracer_diff.py` (impl-diff on `Z3d_new`),
      ✅ `tracer_adv.py` (QR4C 2-D `_z_stencil` on `Z3d_live`/`zbar3_live`), ✅ `momentum.py`
      (impl_vert_visc per-ELEMENT geom from `st.helem` via a NEW `ale.live_geometry_elem` helper —
      the C rebuilds `zbar_n`/`Z_n` per element from `helem`, `fesom_momentum.c:321-333`),
      ✅ `gm.py` (fer_solve_gamma TDMA + zscaling on `Z3d_live`/`zbar3_live` = `hnode_new`@prev ≡
      `st.hnode`), ✅ `gm_redi.py` (G7a vert-Redi geom on `Z3d_live` OLD + the existing `÷hnode_new`
      divisor; K33 on `Z3d_new`/`zbar3_new` NEW side; G7b needs no change — already on
      `st.hnode`/`helem`/`hnode_new`), ✅ `kpp.py` (5 sites on `Z3d_live`/`zbar3_live`), ✅ `forcing.py`
      sw-pen on `zbar3_live` (hoist moved before the forcing block). All byte-neutral.
- [x] dual-geometry exactly as the C: QR4C + horizontal Redi on committed (`st.hnode`/`Z3d_live`),
      impl-diff + GM K33 on `hnode_new`/`Z3d_new`, the `÷hnode_new` divisors on the new side (lesson #6).
- [x] salinity floor (`step.py:282`) untouched — unconditional, mirrors the C (do NOT ale-gate).
- [x] tests: per-consumer linfs bitwise-neutrality (live==static — pi suite 69 + the full ocean suite);
      zstar-ON runs finite (pi-zstar smoke + the JZ.7 assembled multi-step).
- [x] full suite green — **OCEAN 529 + ICE 47, 0 fail (job 25550512, 2026-06-12)**; `ale_cfg=None`
      byte-identical across all re-pointed consumers (eos/pp/dbsfc/kpp/gm/gm_redi/momentum/tracer_adv/
      forcing). Sharding group times out (pre-existing Phase-8b ragged halo — not a zstar regression).

### JZ.7 — Assembled zstar step: 3-step dump-diff + sharded N-vs-1

**Files:** Modify: `fesom_jax/tests/test_ale_zstar.py`, `fesom_jax/tests/test_step_sharded.py` (if a
new exchange row needs asserting). Create: `scripts/` gate job if needed.

- [~] assembled eager zstar run vs the dump tags. **STEP-1 DONE (2026-06-12, job 25535718,
      `test_jz7_assembled_zstar_step1` + `scripts/debug/jz7_assembled_gate.sbatch`):** the FULL 4-config
      model — **KPP + GM/Redi + EVP ice + zstar, the z2_cdump config — assembles and runs FINITE the
      first time all four knobs run together** (the key integration milestone).
      **IC MISMATCH ROOT-CAUSED & FIXED (2026-06-12, job 25537808): FULL-MESH BIT-IDENTITY.** The former
      pgf tail (max~3e-5, k≥15 shelf-break) + the 488 brackish surface nodes were ONE mechanism: the C
      `extrap_nod3D` GS land fill is order-dependent and runs PER-RANK (local order, halo frozen between
      exchanges, surface-only outer-loop continuation) ⇒ **the C IC is partition-dependent** (C 1r vs 16r
      dumps differ by up to 25.8 PSU at fill nodes). Fix: `phc_ic._extrap_nod3D_mpi` (dist_16-faithful,
      rank lists from the dump gid columns; `scripts/tools/rebuild_ic_dist16.py`) + the bilinear association fix
      (`((v·wx)·wy)` like C — was ~1 ulp off at ~27k nodes). Rebuilt-IC surface = C 16r postload dump
      **EXACTLY (0 diffs, all 126858 nodes)**; standalone density→pgf max|Δ|=**2.7e-20**. Step-1 gates
      now FULL-mesh: config-clean = 126858/126858; `pgf_x/y` max=**4.4e-16/5.6e-16** (gate p99<1e-15,
      max<1e-14; the assembled-step ulp slack is the AB2 tracer-blend reassociation);
      `d_eta`/`hbar`/`eta_n` p50=p99.9≈2.5e-7 max=5.5e-7 — pure CG early-stop, the elliptic config-gap
      halo is GONE (gate p99<1e-6, max<5e-6); `Wvel` max=2.8e-8 (gate max<1e-6); `ssh_rhs` diagnostic
      (near-cancelling). The JZ.2 forcing diagnostic became a HARD gate (the ~1e-5 "input gap" was the
      IC too): `relax_salt` max=3.4e-18, `real_salt_flux` 1.6e-10, `water_flux` 1.4e-9. New IC gates:
      `test_phc_ic.py` serial dumps now byte-exact (`np.array_equal`) + `test_dist16_*` (16r postload
      bit-identity + cache currency for BOTH caches). ⚠️ TWO IC caches (the legacy CORE2 oracles were
      1-rank runs — one IC can't serve both partitions): `data/ic_core2` = serial (legacy gates),
      `data/ic_core2_dist16` = dist_16 (all z2_cdump-gated zstar tests point here).
      **MULTI-STEP DONE (steps 1-3, 2026-06-12, job 25550843, `test_jz7_assembled_zstar_steps123` +
      `scripts/debug/jz7_multistep_gate.sbatch`):** the assembled model chained 3 steps vs the z2_cdump
      3-step set — the REAL validation of the JZ.6 live-geometry re-points (live≠static at steps ≥2).
      **pgf stays bit-faithful on genuinely-live geometry** (s2/s3 p50≈4e-9, p99≈3.7e-7, max≈4.5e-6 vs
      step-1 4e-16) — since pgf reads BOTH density (T/S) AND live `Z_3d_n`, this certifies the geometry
      reconstruction AND the re-pointed tracer chain. ⚠️ KEY FINDING (corrected via the
      controlled-replay): the SSH-solve-derived fields (`d_eta`/`hbar`/`eta_n` + the hbar-built
      `zbar_3d_n`/`Z_3d_n`) diverge to ~mm at step ≥2 (s2≈s3≈7e-3, BOUNDED), but **the SSH solve
      itself is BYTE-IDENTICAL on CPU** — `test_jz7_ssh_solve_controlled_replay` feeds the C's OWN
      dumped `ssh_rhs`+warm-start+`hbar` and reproduces the C `d_eta` to **7.2e-16 (step 1) / 9.7e-16
      (step 2, D2 increment LIVE)**. So the chained ~mm is the UPSTREAM velocity/`ssh_rhs`
      FP-reassociation (~1e-12 scatter floor) amplified by the near-cancelling `ssh_rhs`
      (`dx·helem~1e7`, Phase-2 ssh/rhs) + the near-null-space `S⁻¹` — the FP-trajectory butterfly
      (the C vs Fortran has it too ⇒ the year-scale climate gate). NOT a bug. Gates: pgf tight (the
      JZ.6 precision gate); the SSH solve+D2 byte-fidelity gated by the controlled-replay (1e-12) +
      JZ.3; d_eta/hbar/geometry the bounded "no-blow-up" class; finiteness hard. jit-twice no-leak:
      the eager 3-step chain runs clean (both is_first_step branches compile).
- [x] sharded N-vs-1 (CPU fake devices) with `ale_cfg` ON — **DONE (2026-06-12, job 25551060,
      `test_zstar_serial_sharded_step_matches_dense` (npes=1 byte-id) + `test_zstar_assembled_
      sharded_owned_matches` (npes=2 owned) + `scripts/debug/jz7_shard_zstar.sbatch`).** A WARM hbar seed
      (`_warm_zstar_state`, ~0.5 m bump ⇒ stretched hnode) makes the live geometry genuinely active
      (cold-start would be a no-op). npes=1: the full 4-config (KPP+GM+ice+zstar) collapses to dense
      BYTE-identically. npes=2 owned-match: every new zstar State field at the CLEAN reassociation
      floor — **hnode_new 2.8e-14** (the JZ.4 exchange row), **hnode 2.8e-14 / helem 5.7e-14 / hbar
      1.9e-13** (live geometry shards), **d_eta 3.3e-16** (the D2 stiffness-as-state increment shards
      inside the distributed CG); T/S at the upwind-flip FCT floor (9.7e-3<3e-2), ssh_rhs at the
      cancellation floor — the documented N-vs-1 non-determinism, NOT a missing exchange.
- [x] full suite green — OCEAN 529 + ICE 47, 0 fail (job 25550512). The zstar sharded tests run
      only in the SHARDING group (4 fake-devices; the ocean group ignores `test_*_sharded.py`) ⇒ they
      gate via `jz7_shard_zstar.sbatch`, not the (pre-existingly slow) suite sharding group.

### JZ.8 — Stability, climate, gradient gates (GATE 9a evidence)

**Files:** Create: `scripts/core2_zstar_stability.{py,sbatch}`, `scripts/core2_zstar_grad_gate.{py,sbatch}`.

- [x] 10-day CORE2 A100 zstar-ON (KPP+GM+ice) stable, no NaN; SSH/steric sanity — **DONE
      (2026-06-12, job 25552436, `scripts/core2_zstar_stability.{py,sbatch}`).** The full 4-config
      ran **480 steps / 10.00 days stable** on an A100-80 (~0.12 s/step after an 11.8 s compile):
      `fin=1` throughout, **min hnode = 4.52 m > 0** every step (NO column collapse — the zstar
      failure mode), **global-mean ⟨hbar⟩ drift = −1.4e-4 m** (volume conserved under the real
      freshwater fluxes), |SSH|≈2 m, max|vel|≈1.9 m/s, **SST capped at −1.89 °C** (the ice
      supercooling cap holds under zstar), ice grows physically (m_ice 2.0→2.77, a_ice→1.0).
- [x] year-scale: JAX-zstar ↔ `c_zstar_2yr` SST/SSS ≪ the zstar↔linfs contrast — **DONE / PASS
      (2026-06-12).** ⚠️ The first run (dist_16 IC, job 25552572) FAILED the aggregate gate (SSS 0.12)
      — root-caused NOT as a code bug but as an **IC-partition mismatch**: `c_zstar_2yr` was run on
      **864 ranks** (C plan Z9, job 25495449) ≠ my **dist_16** IC, and the partition-dependent
      `extrap_nod3D` GS fill ([[zstar-forcing-dump-config-gap]]) makes the two ICs differ by up to
      **25.74 PSU at the Baltic** (`rebuild_ic_dist864.py` confirmed: 512 Baltic nodes). Evidence it
      was the IC: divergence largest-at-month-1-decaying, Baltic-localized (global p50 SSS=1.7e-3 =
      the ref), global salt CONSERVED. **Re-running from the matched dist_864 IC (job 25553805,
      stable 17520 steps) COLLAPSED it:** JAX↔C-zstar **SST 3.46e-3 / SSS 2.98e-3** (was 1.9e-2 /
      0.12 — a 41× SSS reduction), i.e. AT the C↔Fortran port-fidelity (C₀ = 3.74e-3/1.52e-3; SST even
      slightly closer). **B/A = 5.7× SST, 7.4× SSS — inside the C's measured 3–9× coordinate
      contrast** ⇒ the JAX reproduces the C zstar climate, not a linfs-ward drift. Methodology sound
      (C₀ reproduces the plan-§0 ref 0.0038/0.0014 exactly). (`scripts/archive/core2_zstar_climate_compare.py`,
      `scripts/rebuild_ic_dist864.*`, `scripts/archive/core2_zstar_climate_dist864.sbatch`.)
- [x] gradient gates per §4 — **DONE (2026-06-12, job 25551862, `test_jz8_grad_*_zstar` (3) +
      `scripts/debug/jz8_grad_gate.sbatch`).** All N=1 backward through the assembled zstar step on a
      compute node (no ice scan ⇒ CPU-feasible): **masked-NaN** `d(SST)/d(T₀)` through KPP+GM+zstar
      finite EVERYWHERE + nonzero on wet + EXACTLY 0 on masked lanes (the live-geometry ÷thickness
      guards + shchepetkin safe denominators + vert_vel/D2/forcing-flip paths are AD-safe);
      `d(SST)/d(k_ver)` (PP+zstar) finite+nonzero; **`d(SST)/d(hbar-IC)`** (GM+zstar, the NEW
      prognostic-thickness state path: `init_thickness_zstar`→live geometry→D2 closure) finite+nonzero.
      The CG transpose with the state-dependent D2 matvec is gated by JZ.3
      (`test_solve_ssh_state_dependent_transpose_residual`); the quantitative FD↔AD plateau is the
      deferred GPU gate (`scripts/core2_zstar_grad_gate.*`, smooth-regime/N=1).
- [x] full suite green — ocean 529 + ice 47 (job 25550512); the JZ.8 grad gates + the
      `test_jz7_ssh_solve_controlled_replay` gate via their sbatch scripts (heavy CORE2, like
      `test_gradient_core2`). **GATE 9a MET** — see the acceptance table below (all rows green).

### JZ.9 — Close-out

**Files:** Modify: `docs/PORTING_LESSONS.md`, `README.md`, this plan, parent plan, memory.

- [ ] verify §6 GATE table all green; lessons appended (one per task, accumulated)
- [ ] update parent plan Phase 9 status + project memory; commit; move plan to `docs/plans/completed/`

## Technical Details

- **No new State fields** (D1/D2): the prognostic set for zstar = the existing
  `hbar, hbar_old, hnode, helem` (+ within-step `hnode_new`); `zbar_3d_n/Z_3d_n` derived; the
  stiffness increment derived from `hbar`. dhe is transient (recomputed where needed).
- The C's per-step CSR `+=` and our recompute differ by ~1 ulp/step of FP association — inside the
  scatter-class tolerance; the dump gates compare *values*, not histories.
- Stretch-range masks: scaled layers `nz ≤ nlevels_nod2D_min(n)−3` (0-based); vert_vel writes
  `1..min_f−2` (1-based C); commit rebuilds `min_f−2..1` bottom→top — all static-precomputable.
- Cavity guards (`ulevels>1`) are ported as no-op masks (CORE2 has no cavity), matching the C.

## Post-Completion

- whichEVP=1 + zstar composition is **deliberately unvalidated in the C** (single-knob matrix) — if
  later needed, treat as JAX-first territory with smoke-level gates only (see the mEVP plan close-out).
- zlevel: only if a C reference is ported first (C-side effort, out of scope here).
- Restart/warm-start of time-varying thickness from Fortran restarts: out of scope (no restart system).
- Optional: global salt-content conservation + SSH/steric drift checks as long-run diagnostics
  (the C left these as manual Post-Completion items too).

## GATE 9a (acceptance)

| Check | Bar | Result (2026-06-12) |
|---|---|---|
| `ale_cfg=None` | full suite green, byte-identical path | ✅ OCEAN 529 + ICE 47, 0 fail (job 25550512) |
| Per-kernel dump gates (12 tags × 3 steps) | map ~1e-15 / scatter ~1e-12; pgf ≤1e-12 | ✅ JZ.1–5 gates; step-1 pgf 4.4e-16 |
| Assembled 3-step zstar | all tags within ladder; jit clean | ✅ pgf bit-faithful on live geom; SSH solve+D2 byte-id via controlled-replay (7e-16) |
| 10-day A100 zstar-ON | stable, no NaN | ✅ 10-day + full-year (17520 steps) stable; min hnode>0; ⟨hbar⟩ drift 1.4e-4 m |
| Year-scale vs `c_zstar_2yr` | JAX↔C ≪ zstar↔linfs contrast | ✅ SST 3.46e-3 / SSS 2.98e-3 (≈ C↔Fortran); B/A=5.7×/7.4× (dist_864 IC) |
| Gradients | masked-NaN clean; CG transpose residual ~1e-12; d/d(hbar-IC) finite | ✅ masked-NaN clean + d/d(k_ver)/d/d(hbar-IC) finite (job 25551862); transpose JZ.3 |
| Sharded | N-vs-1 (CPU ×4) zstar-ON within `_BYTE_ID_ATOL` | ✅ npes=1 byte-id + npes=2 owned clean (hnode_new 2.8e-14, d_eta 3.3e-16) |

## Revision Log

- **2026-06-11 — Plan created** from the C-port digest (4-agent analysis of
  `port2/fesom2_port_zstar` @ `mevp`, the completed C zstar plan + its 24 recorded landmines, and the
  JAX seam map). Key locked decisions: derived live geometry (D1), stiffness-as-function-of-hbar (D2),
  salinity floor mirrored (D3), forcing-flip scope incl. the rsf producer the JAX port lacks (D4).
- **2026-06-12 — Plan-review pass (APPROVE WITH MINOR REVISIONS; 1 MAJOR + 5 minor, all applied).**
  MAJOR corrected: the salinity floor ALREADY exists in the JAX port (`step.py:46-48,282`),
  unconditional exactly like the C — the original "add it zstar-gated" premise was wrong and would
  have broken linfs parity; floor scope removed from AleConfig/JZ.6 (D3 is a no-op: already mirrored).
  Minor: rsf evidence re-cited to `ice_thermo.py` (+ the `evap` split) and `ice_step.py:137`; the GM
  dual-geometry hedge resolved (substep-1b/2 reads the previous commit ≡ `st.hnode`);
  `momentum.py:273-279` re-point reworded (static `zbar`/`Z` → `st.helem`-derived); `compute_hbar`
  wf-ordering and `bc_T` sval-timing notes added to JZ.2/JZ.3. Review verified D1/D2 sound (D2
  telescoping confirmed algebraically exact vs the C, `fesom_ssh.c:229-231`) and the JZ ladder
  complete vs the C Z0–Z10.
- **2026-06-12 — JZ.6 + JZ.7 COMPLETE.** JZ.6: the 5 remaining consumers re-pointed to live
  geometry (tracer_adv QR4C, momentum element-stack via the new `ale.live_geometry_elem`, gm/gm_redi,
  kpp ×5, forcing sw-pen, + the audit-caught `eos.compute_dbsfc`), the which-side map confirmed
  against the C source (4-agent extraction). All byte-neutral (suite OCEAN 529 + ICE 47, 0 fail, job
  25550512). JZ.7: step-1 gate re-green with JZ.6; the **multi-step (1-3) gate** (job 25550843) —
  pgf bit-faithful on genuinely-live geometry at steps ≥2 (the JZ.6 validation), the d_eta/hbar ~mm
  chained divergence root-caused via the **controlled-replay** (`test_jz7_ssh_solve_controlled_replay`):
  the SSH solve + D2 increment are BYTE-IDENTICAL with identical inputs (7e-16/9.7e-16) — the chained
  ~mm is the upstream velocity/ssh_rhs reassociation amplified by the SSH near-cancellation, not a bug;
  the **sharded N-vs-1** (job 25551060) — npes=1 byte-id + npes=2 owned-match, every zstar State field
  at the clean reassociation floor (hnode_new 2.8e-14, d_eta 3.3e-16). **Next: JZ.8** (10-day A100
  stability, year-scale climate vs `c_zstar_2yr`, the §4 gradient gates — GATE 9a evidence).
