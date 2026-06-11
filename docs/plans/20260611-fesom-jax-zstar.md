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
   `core2_forcing.py:155` is the secondary no-ice consumer) — this plan adds the producer
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
`core2_forcing.py:24-28,97,153` + `ice_step.py:137` (the ice-path `bc_S` — the operative consumer) +
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
  (`scripts/run_suite.sbatch`, CPU; ~483 tests + new) must be green before the next task. `ale_cfg=None`
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

- [ ] `AleConfig` + thread `ale_cfg=None` through step/integrate/sharded + static_argnames
- [ ] generalize `read_kpp_table` → shared `read_gid_table` + `ALE_TAGS` loader (12 tags × 3 steps)
- [ ] audit `/work/ab0995/a270088/port/zstar/z2_cdump` tag completeness; regenerate via the documented
      16r/3-step job if stale/incomplete (archive job script in `scripts/`)
- [ ] tests: cfg-threading no-op (None ⇒ step output bit-identical), reader round-trip on all tags
- [ ] full suite green

### JZ.1 — Thickness machinery: init + live geometry + commit

**Files:** Modify: `fesom_jax/ale.py`, `fesom_jax/state.py` (rest-IC path only), `fesom_jax/tests/test_ale_zstar.py`.

- [ ] `init_thickness_zstar` (`fesom_ale.c:45-146`): `hnode=(Δzbar)·(1+hbar/dd)`, helem mean, hnode_new=hnode;
      ⚠️ init `eta_n = α·hbar_old + (1−α)·hbar` — **reversed weights vs per-step** (C lesson #7)
- [ ] `live_geometry(mesh, hnode) → (zbar_3d_n, Z_3d_n)`: bottom→top reconstruction over
      `nz ≤ min_f−2`, deeper interfaces keep nominal; below-bottom lanes = nominal spacing (AD rule)
- [ ] commit under zstar: NO per-step `hnode_new=hnode` copy on the linfs pattern (C lesson #8 — the
      C *skips* the linfs memcpy; our un-hoisted flow must match)
- [ ] tests: cold-start degeneracy (zstar init ≡ linfs bit-for-bit at hbar=0 — the free Z1 gate);
      `live_geometry(st.hnode)` ≡ static `zbar_3d_n`/`Z` under linfs (bitwise); ≤1-ulp helem note
      (C lesson #13)
- [ ] full suite green

### JZ.2 — Forcing flip: rsf producer, water-flux composition, balancing, BCs

**Files:** Modify: `fesom_jax/ice_thermo.py`, `fesom_jax/ice_step.py`, `fesom_jax/ice_coupling.py`,
`fesom_jax/core2_forcing.py`, `fesom_jax/sss_runoff.py`. Create: tests in `test_ale_zstar.py`.

- [ ] rsf producer in ice thermo (`fesom_ice_thermo.c:360-405,516-525`); SPLIT the existing combined
      `evap` (`ice_thermo.py:135` bundles evaporation+sublimation) into `evaporation`/`ice_sublimation`,
      surfaced via `IceStepOut`
- [ ] verify the C `bc_surface` `sval` source (surface-T timing) and match it in the JAX `bc_T` build
      (the JAX builds `bc_T` in the forcing step from start-of-step `T[:,0]` — confirm the C agrees)
- [ ] water-flux ice-volume terms + **global net balancing** (`fesom_ice_coupling.c:178-216`), sums via
      `reductions.global_sum` (sharded-correct)
- [ ] `virtual_salt ≡ 0` under zstar; `bc_T`/`bc_S` per §0.7 — ⚠️ S has **+dt and NO sval·wf** (lesson #3)
- [ ] tests: replay vs the `forcing` ale_dump tag (wf, virtual_salt≡0, relax_salt, rsf) at the C's
      "explained-by-input" bar; `ale_cfg=None` byte-identity
- [ ] full suite green

### JZ.3 — SSH plumbing: rhs tail, hbar wf term, stiffness-as-function-of-state

**Files:** Modify: `fesom_jax/ssh.py`, `fesom_jax/step.py`. Tests: `test_ale_zstar.py`.

- [ ] `ssh_rhs += −α·wf·areasvol + (1−α)·ssh_rhs_old` (zstar tail; `fesom_ssh.c:405-426`) and
      `compute_hbar`: `ssh_rhs_old −= wf·areasvol` (`fesom_momentum.c:843-860`) — ⚠️ the wf subtraction
      lands BETWEEN the transport divergence and the hbar update (hbar consumes the wf-modified
      `ssh_rhs_old`; the wf-modified value is also what's stored for next step's `(1−α)` term)
- [ ] D2 matvec: `A(x) = base(x) + ΔA(mean₃(st.hbar))(x)·g·dt·α·θ` (edge-assembly increment,
      `fesom_ssh.c:238-296` semantics); preconditioner untouched (frozen — lesson #11)
- [ ] tests: `sshsolve` + `hbar`/`dhe` dump tags (note dhe is recomputable as `mean₃(hbar−hbar_old)` —
      compare derived value); step-1 increment ≡ no-op; transpose-residual with state-dependent A
- [ ] full suite green

### JZ.4 — vert_vel distribute + hnode_new production

**Files:** Modify: `fesom_jax/ale.py`, `fesom_jax/step.py`, `fesom_jax/halo_points.py`. Tests: `test_ale_zstar.py`.

- [ ] `vert_vel_zstar_distribute` (`fesom_ale.c:162-218`): **vertically-integrated** Wvel correction
      `(zbar_3d_n(nz)−dd1)·dd/dt` (NOT per-layer h·dd/dt), `Wvel(1) −= wf`, hnode_new on stretch
      layers only; uses **pre-update** geometry (from `st.hnode`)
- [ ] `_exch(hnode_new, "nod")` + `OCEAN_SCHEDULE` row (the C's zstar-only exchange, lesson #9)
- [ ] tests: `Wvel` + `hnode_new` dump tags; surface-interface Wvel budget closes to machine zero
      (the C's exact-budget check)
- [ ] full suite green

### JZ.5 — Shchepetkin PGF

**Files:** Modify: `fesom_jax/pgf.py`, `fesom_jax/step.py`. Tests: `test_ale_zstar.py`.

- [ ] `pressure_force_shchepetkin` (`fesom_eos.c:348-503`, 156 LOC): 1-based local stacks from helem,
      static bottom anchor `zbar_n[nle+1]=zbar[nlevels−1]` (the C reads *static* depth there —
      mirror, don't "fix"), quadratic-Newton vertex `drho_dz`, surface/bottom one-sided stencils as
      precomputed static case masks, safe denominators
- [ ] hpressure gated OFF under zstar (`fesom_eos.c:172`)
- [ ] tests: `pgf_x`/`pgf_y` dump tags — the C hit ~1e-18 on step-1; gate at ≤1e-12 all-elems
- [ ] full suite green

### JZ.6 — Geometry re-point sweep + dual-geometry discipline

**Files:** Modify: `fesom_jax/eos.py`, `fesom_jax/pp.py`, `fesom_jax/kpp.py`, `fesom_jax/tracer_adv.py`,
`fesom_jax/tracer_diff.py`, `fesom_jax/momentum.py`, `fesom_jax/gm.py`, `fesom_jax/gm_redi.py`,
`fesom_jax/forcing.py`, `fesom_jax/step.py` (un-hoist). Tests: `test_ale_zstar.py`.

- [ ] re-point every §2-listed consumer to live geometry under zstar (grep-by-array audit; lesson #5);
      un-hoist `step.py:157`
- [ ] dual-geometry exactly as the C: QR4C on committed (st.hnode) geometry, flux-limited + impl-diff
      + GM fer on `hnode_new`, horizontal Redi on `st.hnode` (lesson #6)
- [ ] confirm the existing unconditional salinity floor (`step.py:282`) stays untouched (mirrors the
      C; do NOT ale-gate — plan-review 2026-06-12)
- [ ] tests: per-consumer linfs bitwise-neutrality (live==static); zstar-ON 1-step runs finite
- [ ] full suite green

### JZ.7 — Assembled zstar step: 3-step dump-diff + sharded N-vs-1

**Files:** Modify: `fesom_jax/tests/test_ale_zstar.py`, `fesom_jax/tests/test_step_sharded.py` (if a
new exchange row needs asserting). Create: `scripts/` gate job if needed.

- [ ] assembled 3-step eager+jit zstar run vs ALL 12 dump tags × 3 steps (post-commit thickness tags
      included); jit-twice no-leak regression
- [ ] sharded N-vs-1 (CPU fake devices ×4) with `ale_cfg` ON — generic State-field loops cover
      hnode/helem/hbar; hnode_new exchange row asserted
- [ ] full suite green

### JZ.8 — Stability, climate, gradient gates (GATE 9a evidence)

**Files:** Create: `scripts/core2_zstar_stability.{py,sbatch}`, `scripts/core2_zstar_grad_gate.{py,sbatch}`.

- [ ] 10-day CORE2 A100 zstar-ON (KPP+GM+ice) stable, no NaN; SSH/steric sanity (global mean drift)
- [ ] year-scale: JAX-zstar ↔ `c_zstar_2yr` SST/SSS — must be ≪ the zstar↔linfs contrast (3–9×)
- [ ] gradient gates per §4 (masked-NaN probe, k_ver plateau zstar-ON, CG transpose residual,
      d/d(hbar-IC) finite)
- [ ] full suite green

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

| Check | Bar |
|---|---|
| `ale_cfg=None` | full suite green, byte-identical path |
| Per-kernel dump gates (12 tags × 3 steps) | map ~1e-15 / scatter ~1e-12; pgf ≤1e-12 |
| Assembled 3-step zstar | all tags within ladder; jit clean |
| 10-day A100 zstar-ON | stable, no NaN |
| Year-scale vs `c_zstar_2yr` | JAX↔C ≪ zstar↔linfs contrast |
| Gradients | masked-NaN clean; k_ver plateau ≤1e-4; CG transpose residual ~1e-12; d/d(hbar-IC) finite |
| Sharded | N-vs-1 (CPU ×4) zstar-ON within `_BYTE_ID_ATOL` |

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
