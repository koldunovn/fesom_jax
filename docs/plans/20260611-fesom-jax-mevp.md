# FESOM2 → JAX Port — Phase 9c: mEVP sea-ice rheology (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 9 — physics options).
**Predecessors:** GATE 6 (sea ice / standard EVP, `ice_evp.py`) + Phase 8.
**Siblings:** `20260611-fesom-jax-zstar.md` (9a), `20260611-fesom-jax-tke.md` (9b). Recommended order
9a → 9b → **9c**, but mEVP is **fully independent** of both (zero TKE interaction; reads the same
`hbar` tilt either coordinate) — it can float anywhere in the order.
**Created:** 2026-06-11. **Status:** ☐ NOT STARTED.

**C source of truth:** `port2/fesom2_port_zstar` (branch `mevp`, tag `mevp-validated-2026-06-11`),
commits `a74aacf → 8f325f0 → 0ca1baf → 7ec565d → 17214db → 398448a → d4ac9c5 → 6da0b0e`; C plan:
`docs/plans/completed/20260611-mevp-sea-ice-rheology.md` (incl. the **14-item fidelity-trap
checklist** with line citations — reproduced below; it IS the hard part of this port). The C is
validated vs Fortran mEVP at SST/SSS RMS 0.0049/0.0024 (yr 1), ice extent ≤0.3%/volume ≤0.7%, a
**vector** drift gate (±0.014° median angles), and a diff-of-diffs liveness proof
(pattern corr +0.96/+0.90/+0.88).

**Decisions (locked):**
1. **whichEVP=1 ONLY** — the literal port of Fortran `EVPdynamics_m` (`ice_maEVP.F90:429-882`, the
   NR-optimized version with ssh2rhs/stress_tensor_m/stress2rhs_m inlined). **aEVP (whichEVP=2) is
   NOT ported** (the C aborts on it — no reference ⇒ no oracle; `c_aevp` doesn't exist in the C).
   Also not ported: `use_floatice` ssh branch (levitating branch only), cavity bodies, standalone
   ssh2rhs/stress_tensor_m (dead in Fortran too).
2. **Config-gate inside `IceConfig`:** add `whichEVP: int = 0`, `alpha_evp: float = 250.0`,
   `beta_evp: float = 250.0` (static NamedTuple ⇒ a trace-time Python branch — no `lax.cond`).
   **`whichEVP=0 ⇒ byte-identical`** (the existing EVP path untouched; the `None ⇒ PP` analogue).
   Constructor raises on `whichEVP=2` (C abort parity).
3. **New module `fesom_jax/ice_mevp.py`** mirroring the C monolith `fesom_ice_maevp.c` (363 LOC) for
   1:1 traceability — but **importing** the genuinely shared pieces from `ice_evp.py`:
   `boundary_node_mask`, `_safe_sqrt`/safe-speed, the strain-rate block and the stress-divergence
   scatter stack (identical formulas — extract as shared helpers; pure refactor, traced graph for
   EVP unchanged), and the checkpointed fixed-length `lax.scan` + per-substep `exch` scaffold.
   The setup/relaxation/solve pieces are **rewritten** against the C (the formulas genuinely differ —
   see the trap list; "copy the EVP template" is the #1 failure mode the C documented).
4. **No new prognostic state:** the State carry stays `(u_ice, v_ice, σ11, σ12, σ22)` — σ is NOT
   zeroed on entry (already the JAX convention). Inside one call the **scan carry is
   `(u_aux, v_aux, σ11, σ12, σ22)`**, while the **frozen entry `(u_ice, v_ice)` are closed-over
   constants** anchoring the backward-Euler rhs (`rhsu = u_ice + … + β·u_aux`) — ⚠️ UNLIKE std-EVP,
   whose velocity update bases the rhs on the **current iterate** (`ice_evp.py:185`); copying that
   template solves the wrong fixed point (the cold-start it2 dump catches it: correct rhs uses
   entry=0, the buggy form uses `u_aux(it1)≠0` — plan-review 2026-06-12). The one new static input is
   the `bc_index` mask (derivable from the existing `boundary_node_mask` — no new files).
5. **AD-safe by construction + a gradient gate.** Conditioning, stated precisely (plan-review
   2026-06-12): the additive `1/(Δ+δmin)` is **C¹-continuous** where std-EVP's `max(Δ,δmin)` clamp
   is kinked (locally zero gradient) — continuous but **LARGE** near rigid pack (`p ~ P/δmin ~
   1e12–13`; `∂p/∂Δ = −P/(Δ+δmin)²`, far larger still — both finite). The **diagonal** carry-Jacobian
   blocks are damped (`det1 ≈ 0.996`, `β/(1+β+drag) ≤ 0.996`, solve denominator ≥ 63 001) and non-ice
   nodes carry **identity** (vs std-EVP's gradient-killing hard zero); full-map conditioning rests on
   the empirical forward stability (the C's 120-iter convergence; σ memory decay `det1¹²⁰ ≈ 0.62`).
   The **binding metric is the assembled `d(loss)/d(ice IC)` measured in JM.4** — expected in
   std-EVP's stiff-but-finite class (~1e16); a larger RAW `∂p/∂Δ` is not a bug. Acceptable either way
   (trainable seams don't route through rheology).

---

## 0. Scope (READ FIRST — what mEVP actually is)

mEVP replaces the standard-EVP elastic subcycling (120 substeps of pseudo-elastic waves over
`dte = ice_dt/120` with `Tevp_inv` damping) by a **pseudo-time fixed-point iteration for the
backward-Euler VP problem over the FULL ice step `rdt = ice_dt = 1800 s`**, stabilized by two
relaxation constants (Bouillon et al. 2013):

- stress: `σ_{k+1} = det1·σ_k + det2·(VP target)`, `det2 = 1/(1+α)`, `det1 = α·det2` (α=250)
- momentum: `+β·u_aux` on the rhs and `(1+β+drag)` in the implicit 2×2 determinant (β=250)
- **120 fixed iterations** (the same `evp_rheol_steps` namelist variable, reused as iteration count);
  no CFL logic, no adaptivity (that's aEVP), no `theta_io` rotation, no `Tevp_inv`/`dte`/`zeta`.

Per call (`fesom_ice_maevp.c:66-363`): aux init from `u/v_ice` → inlined ssh-tilt
(`g·(area/3)·∇hbar` scatter, **UNMASKED** — all non-cavity elements) → node precompute
(`inv_thickness`, the **verbatim regularizer** `mass = M/((1+M²)·area)` with
`M = ρᵢ·m_ice + ρₛ·m_snow` — the per-area mass, NOT divided by `a_ice`, contrast `inv_thickness`
which does divide — `ice_nod = a_ice ≥ 0.01`) →
element precompute (`pressure_fac = det2·pstar·msum·exp(−20(1−asum))`, `ice_el = mean₃(m_ice) > 0.01`,
**NO 0.5**) → 120× [strain rates (same formula as EVP) → `Δ = √(…)`, `p = pressure_fac/(Δ + δmin)`
(**ADDITIVE** δmin) → α-relaxed σ update (0.5 inside σ11/σ22 only) → guarded scatter (same as EVP) →
β-relaxed node solve with **rdt-carrying drag** and `det = bc_index/((1+β+drag)² + (rdt·f)²)`;
non-ice nodes **keep** their velocity (no else-zero) → edge-BC zeroing → halo exchange] → final
`u_ice = u_aux`.

**What to copy from `ice_evp.py` vs rewrite** (from the C digest): copy strain-rate block,
divergence scatter, safe helpers, boundary mask, scan+checkpoint+exch scaffold. Rewrite: setup
(masks/mass/pressure_fac/unmasked tilt), Δ + additive δmin, the σ relaxation, the β-relaxed
bc_index-masked solve, constants.

---

## 1. Reference configuration (VERIFIED values — port these)

From `docs/mevp_reference_namelists/` (+ PROVENANCE): `whichEVP=1`, **`alpha_evp=250`,
`beta_evp=250`** (namelist == module default — verified explicitly, the `namelist_over_codedefault`
trap class), `evp_rheol_steps=120` (= mEVP iteration count), `Pstar=30000`, `ellipse=2.0`
(`vale=0.25`), `c_pressure=20`, `delta_min=1e-11` (**additive**), `Cd_oce_ice=0.0055`,
`ice_ave_steps=1`, ρ₀=1030, g=9.81, **`rdt = ice_dt` (FULL step — not dte!)**. No `c_aevp`, no
`theta_io`, no `Tevp_inv`.

**Oracles on `/work/ab0995/a270088/port/mevp/`** (verified 2026-06-11): `cdump_16r` + `fdump_16r`
(C + Fortran per-substep dump sets — **points Q/U0/F (entry inputs), P (precomputes),
it1/it2/it60/it120 (per-iterate `u_aux`+σ after edge-BC, before exchange), UF (final)**; 16r dist_16,
2 steps, dt=1800; format = gid-keyed text `evp_dump_s<step>_<point>_<kind>_rank<R>.txt`),
`baseline_2714071` (the C byte-gate baseline), `c_mevp_2yr`, `c_evp_2yr` (the diff-of-diffs legs),
`c_mevp_5yr`, `fortran_mevp_2yr`. C-side diff tool precedent: `scripts/maevp_dump_diff.py`.
Regeneration: `mevp`-branch binary, `FESOM_WHICH_EVP=1 FESOM_EVP_DUMP_DIR=…`, 16r, 2 steps, dt=1800
(⚠️ the *Fortran* dump patch is OMP-unsafe — `OMP_NUM_THREADS=1`; the C side has no such constraint).

---

## 2. The seam + integration map

- **`IceConfig`** (`ice.py:35-127`): add `whichEVP=0`, `alpha_evp=250.0`, `beta_evp=250.0`; derived
  properties `mevp_det2 = 1/(1+alpha_evp)`, `mevp_det1 = alpha_evp·det2`. Raise on `whichEVP not in
  (0, 1)`. ⚠️ `ice_dt` is force-derived at `ice_step.py:73` (`ice_ave_steps·dt`) — mEVP's `rdt` must
  read THAT (the historic `ice_dt` config-desync lesson).
- **Dispatch:** in `ice_step.py` (or `evp_dynamics`'s caller): static Python branch
  `cfg.whichEVP == 1 ⇒ ice_mevp.mevp_dynamics(...)` with the **same signature/outputs** as
  `evp_dynamics` (`ice_evp.py:203-241`): inputs incl. `elevation = hbar`, `stress_atmice` (built with
  the previous-step `u_ice`, `ice_step.py:83-84`), `boundary_node`, `exch`; outputs
  `(u_ice, v_ice, σ11, σ12, σ22)`.
- **`bc_index`** = `1.0 − boundary_node_mask(mesh)` as float (the C rebuilt its array to the
  global-edge-id convention — which is exactly what `ice_evp.boundary_node_mask` already implements:
  edges ≥ `edge2D_in`). Sharded: derive from the **GLOBAL** mask partitioned in (the existing
  `boundary_node_p` precedent, `integrate_sharded.py:263,286-291`) — a local recompute mis-flags
  partition seams (the C's multi-rank trap #1, `MOD_ICE.F90:889-895`).
- **Scratch-array note:** the C reuses `rhs_a/rhs_m` (advection arrays) as ssh-tilt scratch — legal
  because ice advection rewrites them after dynamics. In JAX these are just local values; **no
  ordering hazard exists** (pure functions) — document, don't replicate the aliasing.
- Sharded N-vs-1: σ stays in the `_DIAG_FIELDS` exclusion (the EVP precedent — VP-kink noise
  amplification; the C's own M3 saw e-13→e-8 growth over 120 iterations at near-rigid elements while
  velocity stayed 5e-11); `u_ice/v_ice` gate strictly.

## 3. Validation strategy

- **Per-iterate dump gates** vs `cdump_16r` (JAX-vs-C — the same algebra, so tighter than the C's own
  C-vs-Fortran floor): entry/precompute points (Q/U0/F/P) at ~1e-13; `u_aux` at it1/it2/it60/it120 at
  ≤1e-12; σ tracked with the documented noise-amplification context (velocity is the binding check).
  Step s1 = controlled cold start (inputs bit-identical); s2 inherits the known step-1 envelope —
  the C's std-EVP-control finding says: **diff s2 only against the C, never against Fortran directly**.
- **Liveness (diff-of-diffs):** `(JAX-mEVP − JAX-EVP)` 10-day fields must pattern-correlate with
  `(c_mevp_2yr − c_evp_2yr)` early-period fields — proves the option is live and faithful, not inert
  (absolute agreement alone cannot catch a silently-dead knob).
- **Ice metrics with a VECTOR gate:** extent/volume/drift-magnitude AND per-node speed-ratio + angle
  medians (rotation-class errors — e.g. wrongly porting `theta_io` — are invisible to all magnitude
  metrics; C lesson). ⚠️ Frame: the C/Fortran NetCDF `uice/vice` output is now **geographic** by
  default (`FESOM_IO_VECTOR_FRAME`, commit 75406d3); JAX internals are rotated/native — rotate before
  comparing against NetCDF (dumps/snapshots stay native). `nan_to_num→0` before any time-averaging
  of ice metrics.
- **Stability:** 10-day A100 mEVP-ON; note the C found mEVP **damps** the autumn velocity transient
  (5-yr peak |uv| 2.56 vs std-EVP 4.65) — a sanity direction, not a gate.

## 4. AD-safety strategy (the differentiability contract)

- `(ρᵢm+ρₛms)/a_ice` — division by zero at ice-free nodes, branch-guarded in C ⇒ the masked-divide
  double-`where` (`where(a≥0.01, a, 1)`; the `ice_evp.py:81` idiom).
- `Δ = √(sum of squares)` — exactly 0 at zero strain (true at cold start) ⇒ `_safe_sqrt` (copy).
- `umod = √(du²+dv²)` — 0 when `u_aux == u_w` (cold start) ⇒ safe-speed (copy).
- `p = pressure_fac/(Δ + 1e-11)` — never divides by zero; **C¹-continuous** (vs the EVP clamp kink)
  but large-magnitude near rigid pack (`∂p/∂Δ = −P/(Δ+δmin)²`, huge-but-finite) ⇒ stiff backward,
  the known acceptable class; the assembled JM.4 gradient is the binding measurement.
- `mass = M/((1+M²)·area)` — smooth everywhere incl. M=0 (do NOT "simplify"); `1/max(·,9)` kink fine.
- Masks: `ice_nod`/`ice_el` computed **once per call** (constant across the 120 iterations); element
  skip freezes σ (carry old — the `ice_evp.py:139-140` idiom); node skip = **identity carry of
  `u_aux`** (NOT zero — gradient-friendlier and the faithful semantics).
- Fixed 120-length `lax.scan` + `jax.checkpoint` (copy the EVP scaffold; backward memory already
  proven at 120 substeps).
- **Gradient gates:** finite (non-NaN) `d(loss)/d(ice IC)` through the assembled mEVP step;
  masked-NaN probe over `a_ice=0` lanes; document the **assembled** stiffness scale (the binding
  metric; expect std-EVP's ~1e16 class — raw per-term `∂p/∂Δ` may be far larger, not a bug);
  confirm trainable-seam gradients (`d(SST)/d(k_ver)` mEVP-ON) keep their plateaus.

## Development Approach (standing rules)

- Oracle-first; suite green per task; `whichEVP=0` byte-identity asserted throughout (the EVP-path
  refactor in JM.1 is the one step that must prove graph-identity: same jit HLO / suite bitwise).
- **STANDING RULE:** one lesson per task to `docs/PORTING_LESSONS.md`.
- `[x]` immediately; ➕/⚠️ conventions; keep plan in sync; move to `completed/` at GATE 9c.
- Compute: as siblings (suite sbatch CPU; A100 for stability/gradients; debug-QOS C jobs).

## Implementation Steps

### JM.0 — Scaffolding: cfg, dispatch stub, readers, oracle audit (NO behavior change)

**Files:** Modify: `fesom_jax/ice.py`, `fesom_jax/ice_step.py`, `fesom_jax/io_dump.py`.
Create: `fesom_jax/ice_mevp.py` (stub), `fesom_jax/tests/test_mevp.py`.

- [ ] `IceConfig` fields + raises (NamedTuple has no `__init__` hook ⇒ a `__new__` override or a
      validating factory); dispatch stub (whichEVP=1 ⇒ NotImplemented for now)
- [ ] `EVP_DUMP_TAGS` readers on the shared `read_gid_table` (base = the existing gid-keyed
      `read_kpp_table`, `io_dump.py:236`; hoist from JZ.0/JT.0, or JM.0 owns the generalization if 9c
      runs first); audit `/work/ab0995/a270088/port/mevp/{cdump_16r,fdump_16r}`; regeneration job
      documented (C filename format `evp_dump_s<step>_<point>_<kind>_rank<R>.txt`,
      `fesom_ice_maevp.c:52-54`)
- [ ] tests: whichEVP=0 byte-identical (suite); whichEVP=2 raises; readers round-trip all points
- [ ] full suite green

### JM.1 — Shared-helper extraction + bc_index (EVP-path graph-identity)

**Files:** Modify: `fesom_jax/ice_evp.py`. Create: tests in `test_mevp.py`.

- [ ] extract the strain-rate block + the **raw** elem→node σ-divergence scatter (the
      `·inv_areamass + tilt` / `·mass + rhs_a` tails STAY in their respective modules — do not pull
      EVP-specific masking into the shared helper) + reuse `_safe_sqrt`/safe-speed/
      `boundary_node_mask` — pure refactor. ⚠️ keep the **EVP association** (`mfac·(ΣV)/3.0`,
      `ice_evp.py:118`); mEVP then differs from the C's `meancos = val3·metric_factor` form by ~1e-16
      association — do NOT chase that residual in the it-dumps. (Fallback if the refactor fights:
      duplicate the ~15 lines — the C's own monolith choice)
- [ ] `bc_index = 1.0 − boundary_node_mask` (float; sharded from the global mask)
- [ ] tests: EVP path bitwise-unchanged after refactor — `max|Δ|=0` on the existing EVP dump gates +
      sharded EVP gates (the BINDING gate; HLO comparison advisory only); bc_index complement
      property on the CORE2 mesh; dist_16 partition spot-check (no seam nodes flagged — the C trap #1)
- [ ] full suite green

### JM.2 — The mEVP kernel (`ice_mevp.py`), per-iterate dump-gated

**Files:** Modify: `fesom_jax/ice_mevp.py`, `fesom_jax/tests/test_mevp.py`.

- [ ] setup: unmasked ssh tilt (**no has_ice mask** — trap 6/“do not copy EVP's mask”), node
      precompute (`ice_nod`, `inv_thickness`, verbatim `mass`), element precompute (`ice_el` by
      **mean₃(m_ice)>0.01**, `pressure_fac` with det2 folded, **no 0.5**)
- [ ] iteration body: strain (shared) → Δ + **additive** δmin → α-relaxed σ (0.5 in σ11/σ22 only;
      no eps12 half) → scatter (shared) → β-relaxed solve (**drag carries rdt; `drag·u_w` outside the
      rdt group; rhs uses entry `u_ice` + `β·u_aux`**; det = bc_index/((1+β+drag)²+(rdt·f)²); **no
      theta_io**; non-ice = identity carry) → edge-BC zero → exch — 120× scan + checkpoint
- [ ] final copy; σ enters/leaves as State carry untouched on entry (trap 11); no `uice_old` (trap 14)
- [ ] **walk the 14-trap checklist** (below) as explicit review items, each ticked with a JAX
      line citation
- [ ] **trap-13 unit test** (the cold-start dumps CANNOT distinguish identity-vs-zero — all
      velocities are 0 at s1): seed a NON-ice interior node (`a_ice<0.01`) with `u_ice=5`, one mEVP
      iteration ⇒ velocity **retained** (identity carry, ≈5); paired boundary-node assert
      (bc_index + edge-BC ⇒ 0) [plan-review MAJOR]
- [ ] tests: Q/U0/F/P precompute gates ~1e-13; it1/it2/it60/it120 `u_aux` ≤1e-12 (s1); σ tracked with
      context; cold-start σ≡0 at it1 reproduced; **it2 entry-anchor check** (rhs uses the frozen
      entry, not the it1 iterate — Decisions #4)
- [ ] full suite green

### JM.3 — Step wiring + stability + liveness + ice metrics

**Files:** Modify: `fesom_jax/ice_step.py`. Create: `scripts/core2_mevp_stability.{py,sbatch}`,
tests in `test_mevp.py`.

- [ ] live dispatch end-to-end (eager + jit); s2 dump compare (vs C only) — establish the
      **whichEVP=0 JAX-vs-C s2 diff as the control floor** and judge the mEVP s2 diff against that
      rheology-independent envelope (the C's std-EVP-control methodology)
- [ ] 10-day A100 mEVP-ON stable; ice extent/volume/drift vs `c_mevp_2yr` early period; **vector
      gate** (speed ratio + angle medians; geographic-frame rotation for NetCDF comparisons)
- [ ] diff-of-diffs liveness: (JAX-mEVP − JAX-EVP) vs (c_mevp − c_evp) pattern correlation
- [ ] full suite green

### JM.4 — Gradient gates

**Files:** Modify: `fesom_jax/tests/test_gradient.py` (or `test_mevp.py`). Create:
`scripts/core2_mevp_grad_gate.{py,sbatch}` if GPU-scale needed.

- [ ] finite `d(loss)/d(ice IC)` through the assembled mEVP step (stiffness scale documented)
- [ ] masked-NaN probe over `a_ice=0` lanes
- [ ] trainable-seam plateaus unchanged mEVP-ON (`d(SST)/d(k_ver)`)
- [ ] full suite green

### JM.5 — Sharded N-vs-1 + close-out matrix

**Files:** Modify: `fesom_jax/tests/test_step_sharded.py` (only if exclusions need touching),
`docs/PORTING_LESSONS.md`, `README.md`, this plan, parent plan, memory.

- [ ] N-vs-1 (CPU ×4) mEVP-ON: u_ice/v_ice strict, σ in `_DIAG_FIELDS`
- [ ] ➕ optional track close-out: the all-ON triple (zstar+TKE+mEVP) 10-day smoke — ⚠️ **JAX-first
      territory** (the C deliberately validated single knobs only; zstar+TKE is the only C-validated
      combination) — smoke-level gate only (stable, no NaN, fields physical), explicitly NOT a
      fidelity gate
- [ ] GATE 9c table green; lessons; parent plan + memory; commit; move to `completed/`

## The 14 fidelity traps (from the C plan — review checklist for JM.2)

1. **`rdt = FULL ice_dt`** and **drag carries rdt** (`drag = rdt·cd·umod·ρ₀·inv_thickness`);
   `drag·u_w` sits OUTSIDE the `rdt·(…)` group. The #1 copy-from-EVP error (×120/×1800 wrong).
2. `pressure_fac` has **NO 0.5**; the 0.5 lives inside the σ11/σ22 updates only (σ12 has none).
3. **No `theta_io` rotation** — drop EVP's ax/ay entirely.
4. Mask semantics: element `mean₃(m_ice) > 0.01` (mean, m-only); node `a_ice ≥ 0.01` — NOT EVP's
   all-vertex `m>0 && a>0`.
5. `mass = M/((1+M²)·area)` **verbatim** (smooth small-mass regularizer — never "simplify").
6. ssh-tilt scatter is **unmasked** (all non-cavity elements); stress scatter is owned-guarded —
   an asymmetry to preserve (in JAX: just don't apply `has_ice` to the tilt).
7. rhs zeroing/scaling subtleties are C-MPI artifacts — in JAX pure values; document only.
8. aux init AND final copy over the full node extent; no exchange after the final copy (halo current
   from the last substep).
9. `bc_index` from the **global-edge-id** convention (= `boundary_node_mask`); never local-recompute
   on a partition.
10. `delta_min` is **ADDITIVE** (`/(Δ+δmin)`), not EVP's `max(Δ, δmin)`.
11. σ NOT zeroed on entry (persists across steps and no-ice intervals; decays by `det1¹²⁰ ≈ 0.62`).
12. All Fortran literals are doubles (`-r8`); `meancos = metric_factor/3`.
13. Non-ice nodes are **SKIPPED, not zeroed** (identity velocity carry — porting EVP's else-zero
    would zero ice-edge velocities every iteration).
14. No `uice_old/vice_old` saves (std-EVP's are dead weight anyway).

## Technical Details

- Iteration count fixed at `evp_rheol_steps=120` — same checkpointed scan budget as EVP; no
  adaptivity anywhere (adaptive α/β IS aEVP — out of scope).
- New constants live in `IceConfig` (static); nothing mEVP goes in `Params` (rheology is not a
  training seam; gradients flow but are stiff by nature).
- mEVP has **no zstar-conditional code** (the levitating ssh branch covers linfs AND zstar); the
  composition is untested in C — hence the smoke-only triple gate.

## Post-Completion

- **aEVP (whichEVP=2):** would require a C-side port first (EVPdynamics_a + `find_alpha/beta_field_a`
  + `c_aevp=0.15`; note the Fortran oddity `beta_evp_array = ice%alpha_evp` at `MOD_ICE.F90:753`).
  Out of scope until a validated C reference exists.
- mEVP×zstar fidelity validation (beyond smoke) if that combination ever becomes the production
  config — needs a C-side combined reference first.

## GATE 9c (acceptance)

| Check | Bar |
|---|---|
| `whichEVP=0` | full suite green, bitwise (incl. post-refactor JM.1) |
| Precompute dumps (Q/U0/F/P) | ~1e-13 |
| Per-iterate `u_aux` (it1/2/60/120, s1) | ≤1e-12 (σ tracked, velocity binding) |
| 10-day A100 mEVP-ON | stable; ice metrics + vector gate sane vs `c_mevp_2yr` |
| Liveness | (JAX-mEVP − JAX-EVP) ≁ 0 and pattern-matches (C-mEVP − C-EVP) |
| Gradients | finite d/d(ice IC); masked-NaN clean; k_ver plateau unchanged mEVP-ON |
| Sharded | N-vs-1 (CPU ×4) mEVP-ON; u/v strict, σ excluded |

## Revision Log

- **2026-06-11 — Plan created** from the C-port digest (completed C mEVP plan, its 14-trap fidelity
  checklist + 24 lessons, the M3 s1/s2 dump-diff methodology, the std-EVP-control finding, and the
  JAX `ice_evp.py` anatomy). Locked: separate `ice_mevp.py` with shared-helper extraction
  (graph-identity proven in JM.1), `bc_index` from the existing global boundary mask, additive-δmin
  smooth rheology, identity-carry masking, σ excluded from N-vs-1.
- **2026-06-12 — Plan-review pass (APPROVE WITH MINOR REVISIONS; 3 MAJOR-class clarity/coverage
  items + minors, all applied).** (1) Decisions #4 restated: scan carry = `(u_aux, v_aux, σ)` with
  the **frozen entry `(u_ice, v_ice)` closed over as the backward-Euler rhs anchor** — std-EVP bases
  its rhs on the current iterate, so template-copying solves the wrong fixed point; an it2
  entry-anchor check added to JM.2. (2) A **trap-13 unit test** added (seeded non-ice node retained,
  not zeroed) — the cold-start dumps provably cannot distinguish identity-vs-zero. (3) Conditioning
  prose corrected: `~1e13` is p's VALUE not its derivative (`∂p/∂Δ = −P/(Δ+δmin)²`, up to ~1e22);
  "C¹-continuous but large" replaces "smoother ⇒ better"; the assembled JM.4 gradient is the binding
  metric. Minors: JM.1 extraction boundary pinned (raw scatter only; EVP association kept, ~1e-16 vs
  the C's meancos form — don't chase; duplication fallback noted), suite-bitwise is the binding JM.1
  gate (HLO advisory), the whichEVP=0 s2 control-floor made explicit in JM.3, `M` defined
  (ρᵢ·m_ice+ρₛ·m_snow, not /a_ice), reader base named (`read_kpp_table`, `io_dump.py:236`),
  NamedTuple-validation mechanism noted. Review verified all spot-checked traps (1, 10, 13, bc_index
  complement, det1/β numbers) against the C and the M0–M6 completeness.
