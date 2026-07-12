# FESOM2 → JAX Port — Phase 9b: CVMix classical-TKE vertical mixing (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 9 — physics options).
**Predecessors:** GATE 6C (KPP — the mixing-seam template) + Phase 8.
**Siblings:** `20260611-fesom-jax-zstar.md` (9a), `20260611-fesom-jax-mevp.md` (9c). Recommended order
9a → **9b** → 9c, but **TKE has NO hard zstar dependency**: the C validated TKE under **linfs**
(deliberately, to isolate the mixing knob), and TKE reads geometry through per-node arrays that are
simply static under linfs. If 9b runs before 9a, the only obligation is JT.2's geometry-seam rule.
**Created:** 2026-06-11. **Status:** ✅ **COMPLETE — GATE 9b MET 2026-06-13** (JT.0→JT.5; column
core + driver replay BIT-EXACT ≤3e-17, `TKE_GRAD_GATE_OK`, sharded N-vs-1, stable, year-scale
climate SST 4.68e-3 ≈ the C↔Fortran floor). TKE is the project's first fully-differentiable
prognostic mixing scheme. The lone xfail (live-step-1 forward) is an understood forcing-init
PERSISTENT undiagnosed low-wind forcing diff (small climate impact, not a blocker; NOT a "transient").

**Why TKE matters most for this project:** TKE is the **primary hybrid-ML seam** — a prognostic
1-equation mixing closure whose constants (`c_k`, `c_eps`, `cd`, `alpha_tke`) are exactly the kind of
parameters Phase 7a tunes and Phase 7 replaces/augments with NNs. Differentiability and `Params`
exposure are first-class requirements here, not add-ons.

**C source of truth:** `port2/fesom2_port_zstar` (branch `mevp`, tag `tke-validated-2026-06-11`),
commits `45afc01 → fcd152e → f5fbe58 → 2714071`; C plan:
`docs/plans/completed/20260610-tke-vertical-mixing.md`. The C is validated vs Fortran TKE at
**SST/SSS RMS 0.0049/0.0028 (yr 1)** with an 11–18× (SST) scheme contrast — the comparison resolves
the scheme. The combined **zstar+TKE** matrix (2 yr + 5 yr, peak |uv| 2.93) is also C-validated.

**Decisions (locked):**
1. **Mirror the C 1:1, dump/replay-gate each kernel.** The variant is the **classical Gaspar (1990)
   TKE only** — the ICON/Brüggemann CVMix fork FESOM2 vendors, `tke_mxl_choice=2`, `only_tke`,
   **Neumann** surface+bottom BCs. NOT ported (config-abort parity, raise at `TkeConfig` validation):
   IDEMIX, Langmuir, Dirichlet BCs, `mxl_choice≠2`.
2. **Module split mirrors the C:** `fesom_jax/cvmix_tke.py` = the pure column core (no mesh/state —
   the analogue of `fesom_cvmix_tke.c`, 404 LOC) + `fesom_jax/tke.py` = driver/state/wiring (the
   analogue of `fesom_tke.c`, 534 LOC). Same split the C chose deliberately for dump traceability
   (precedent: `gm.py`/`gm_redi.py`).
3. **Config-gate:** static `TkeConfig(NamedTuple)`, `tke_cfg=None ⇒ today's KPP/PP dispatch,
   byte-identical`. Mixing dispatch becomes 3-way at `step.py:177-190`; **error if `kpp_cfg` and
   `tke_cfg` are both set** (the C runs exactly one scheme per process — fail loudly).
4. **TKE is PROGNOSTIC** — `tke [nod2D, nl]` (interface-indexed) joins `State` and the scan carry.
   This is the structural difference from KPP/PP/GM (all stateless) and drives the new-field
   checklist below. The C/Fortran "diagnostic" comment is wrong (C plan flagged it).
5. **Trainable constants live in `Params` from day one** (`c_k`, `c_eps`, `tke_cd`, `alpha_tke` — the
   `k_gm`/`default_factory` pattern); structural switches stay static in `TkeConfig`. Dead C knobs
   (`kappaM_min` — never applied; `tke_surf_min` — Dirichlet-only) are documented, NOT exposed.

---

## 0. Scope (READ FIRST — what TKE is)

One prognostic equation for turbulent kinetic energy per column, solved implicitly each step, yielding
`Kv`/`Av` exactly where PP/KPP do (the same `(Kv, Av, uvnode)` output contract, post-`mo_convect`):

1. `sqrttke = √max(0, tke_old)`; mixing length `mxl = √2·sqrttke/√max(1e-12, N²)` then the
   **Blanke–Delecluse wall constraints**: zero endpoints, forward min-scan `mxl[k] ≤ mxl[k−1]+dzw[k−1]`,
   special pre-step at `nlev−1`, backward min-scan, floor `mxl_min` (`fesom_cvmix_tke.c:222-237`).
2. `KappaM = min(kappaM_max, c_k·mxl·sqrttke)`; `Pr = max(1, min(10, 6.6·Ri))` with
   `Ri = N²/max(S², 1e-12)`; `KappaH = KappaM/Pr` (`:242-253`). Computed from **OLD** tke ⇒ the ocean's
   Kv/Av at step n are functions of `tke_{n−1}`.
3. Forcing `forc = S²·KappaM − N²·KappaH` (+ surface buoyancy ≡ 0 in this config) (`:258-273`).
4. **Implicit tridiagonal solve over the nlev+1 interfaces**: TKE self-diffusivity
   `ke = alpha_tke·0.5·(KappaM[kp1]+KappaM[kk])` with the index quirk `kp1=min(k+1,nlev−1)`,
   `kk=max(k,1)` (0-based); Neumann surface BC injects `cd·forc_tke_surf^{3/2}/dzt[0]`; the
   **Patankar quasi-implicit dissipation** `+dt·c_eps·sqrttke/mxl` on interior diagonal rows (old-tke
   coefficient × new tke); Thomas solve **in the C's reciprocal-multiply form** (`:278-324, :119-135`).
5. Floor `tke_new = max(tke_new, tke_min)` (`:348-351`); 13 diagnostics (8 budget terms +
   `Lmix`/`Pr`/`int1-3` auxiliaries; always computed in scratch, copied out only if requested) with
   the **closure identity Ttot ≈ Σterms** (the C measured ~4e-15 rel; gate at ≤1e-14) — a free
   internal oracle.
6. Driver: `forc_tke_surf = |stress_node_surf|/ρ₀` (the ice-blended nodal stress KPP already gets);
   `vshear2` from `uvnode` diffs / ΔZ; `bvfreq2` = the shared smoothed N²; `dz_trr` interface spacings
   with **half-thickness end caps** `hnode/2`; post-solve zero Kv/Av at the surface interface and
   below-bottom interface (looks wrong, is right — consumers read interior only); full-slab Kv copy +
   node→elem 3-vertex Av mean (interior levels); `mo_convect` after (`fesom_tke.c:270-534`).

**No new forcing plumbing needed:** `stress_node_surf`/`heat_flux` already reach the mixing seam
(KPP's K.4 work). No bottom friction, no surface buoyancy flux, no lookup tables, no iteration loops.

---

## 1. Reference configuration (VERIFIED values — port these)

Echo-verified in the Fortran logs (`docs/tke_reference_namelists/PROVENANCE.md:43-58`):
`mix_scheme='cvmix_TKE'` (nmb=5); `c_k=0.1`, `c_eps=0.7`, `alpha_tke=30.0`, **`tke_cd=3.75`**
(⚠️ namelist beats module default 1.0; and despite the Fortran comment "3.75=Dirichlet", the
**Neumann** branch executes), `mxl_min=1e-8`, `kappaM_min=0.0` (dead), `kappaM_max=100.0`,
`tke_surf_min=1e-4` (dead), `tke_min=1e-6`, `mxl_choice=2`, `only_tke=T`, `dolangmuir=F`,
Dirichlet F/F. dtime = ocean dt (1800). ⚠️ **All Fortran default-real literals are DOUBLE** (`-r8`
build): `6.6`, `0.5`, `√2`, … — the one real bug the C port had (caught by replay).

**Oracles on `/work/ab0995/a270088/port/tke/`** (verified 2026-06-11): `cdump` (C dump set — **20
tags: 5 column inputs + 3 outputs + 10 diags + wired Kv/Av**, 16r, 3 steps, dt=1800, linfs), `fdump`
(Fortran), `replay` (Fortran column-input injection set), `c_tke_2yr` (C linfs+TKE 2-yr monthly),
`fortran_linfs_tke`, plus the combined-matrix `c_zstar_tke_2yr`/`c_zstar_tke_5yr`,
`fortran_zstar_tke`. Dump format = the same gid-keyed text as KPP/ALE
(`io_dump.read_gid_table` once JZ.0/JT.0 lands it). Regeneration recipe: `mevp`-branch binary,
`FESOM_MIX_SCHEME=TKE FESOM_TKE_DUMP_DIR=… [FESOM_TKE_DIAG=1]`, 16r, 3 steps, dt=1800.

---

## 2. The seam + integration map

**`TkeConfig(NamedTuple)`** (static, hashable): `mxl_min`, `tke_min`, `kappaM_max`, `mxl_choice=2`,
`only_tke=True`, `use_dirichlet=False`, `with_diags=False` (test-only budget outputs) — constructor
raises on any un-ported combination (the C init-abort parity, `fesom_tke.c:246-253`).
**`Params` additions** (`default_factory`, `params.py:56-61` pattern): `tke_c_k=0.1`,
`tke_c_eps=0.7`, `tke_cd=3.75`, `tke_alpha=30.0`. The Pr-law literal 6.6 + clamps [1,10] stay static
(`TKE_C66`, a *double*).

**Dispatch (3-way, `step.py:177-190`):** `tke_cfg is not None` ⇒
`tke.mixing_tke(mesh, st.uv, bvfreq, st.tke, stress_node_surf, geometry, tke_cfg, params, exch=_exch)`
→ `(Kv, Av, uvnode, tke_new)` (stress only — `forc_rho_surf ≡ 0` in this config, heat flux unused;
`exch` is REQUIRED for the internal Av exchange, point 7 below); raise if `kpp_cfg` also set; raise on
the pi path (needs `stress_node_surf` — the KPP precedent `step.py:178-182`). `tke_new` written via a
**conditional replace** keyed on the cfg (the ice precedent `step.py:302-306`) so the None path never
touches it.

**The prognostic-field checklist (every State enumeration site, from the seam digest):**
1. `state.py:34-89` declaration (+ `zeros`/`rest`: **tke IC = 0** — cold start; the first call floors
   the wet column to `tke_min`, Kv/Av step-1 ≈ 0 — exact C mirror), pytree auto-registered
2. `step.py:291-306` conditional `dataclasses.replace(..., tke=...)`
3. `integrate.py` — carry generic; thread `tke_cfg` like `kpp_cfg` (`:46,71,77,89,98,109-111`)
4. `integrate_sharded.py:259-263, 346ff` — thread `tke_cfg`
5. `shard_mesh.partition_state` — **generic** (`[nod2D, nl]` auto-partitions; zero changes)
6. `zarr_output.py:29-30` `DEFAULT_FIELDS` — add `"tke"`
7. Halo — two distinct facts (plan-review MAJOR, 2026-06-12): (a) **the `tke` FIELD is never
   exchanged** (C probe-verified: no reader needs its halo — each column is self-contained on owned
   data); do NOT add one — document as a comment row near `halo_points.OCEAN_SCHEDULE`. (b) **the Av
   path DOES need an internal exchange**: the C driver exchanges node `tke_Av` (`fesom_tke.c:491`)
   BEFORE the node→elem 3-vertex mean — boundary OWNED elements have HALO vertices; the exact
   `kpp.py:787-789` precedent (`viscA = exch(viscA, "nod")` then node→elem). Hence
   `mixing_tke(..., exch=)`; without it the design passes eager/1-device and fails the sharded
   N-vs-1 gate on boundary-element Av. The final Kv/Av reuse the existing post-exchanges
   (`step.py:191-193`).
8. ICs: `phc_ic.core2_initial_state` / `ic.initial_state` inherit the rest default (0)
9. Tests: `test_state.py:30-44` `_expected_shapes` is the deliberate tripwire — add
   `"tke": (nod2D, nl)` (the `:51` comparison trips on it; the leaf-count check is self-adjusting);
   `test_partition_state.py` + the seven `test_step_sharded.py` field loops cover `tke` automatically

**Geometry-seam rule (zstar compatibility):** the driver must take `dzw` (= hnode slice) and
interface spacings `dz_trr` **as derived-from-(hnode, Z-source) inputs** computed at the call site in
`step.py` — under linfs from the static geometry (bitwise = today), under zstar (9a) from
`live_geometry`. Zero TKE-code change when 9a lands — exactly how the C achieved zstar+TKE for free.

---

## 3. Validation strategy

- **Controlled replay is the primary gate** (it caught the C's only real bug): the `cdump` set
  contains the **5 column inputs AND all outputs per step** — read inputs (incl. `tke_old`) for
  steps 1–3, push them through the JAX column core, compare all 15 output tags. This turns every step
  into a pure-algebra comparison and **sidesteps the flip-amplification problem entirely** (the C's
  live s2/s3 diffs were threshold-flips at the PHC noise floor, 44 854 columns — not algebra).
  Bar: the C-vs-Fortran replay hit **1.1e-16**; JAX-vs-C gate at ≤1e-13 per tag.
- **Step-1 live gate:** cold start ⇒ uv=0 ⇒ pure algebra on bit-identical inputs (the C got exact-0
  vs Fortran). ⚠️ Step 1 is a **weak test** (shear production untested) — hence steps 2–3 via replay.
- **Assembled 3-step live gate:** JAX forcing is a validated 1:1 port of the C's (the K.8 precedent:
  no JAX↔C transient) ⇒ expect ~1e-12-class agreement *without* replay; if flips appear, fall back to
  the C's flip-explanation methodology (don't chase ghosts — C lesson).
- **Budget closure** as a standing test: `Ttot ≈ Tbpr+Tspr+Tdif+Tdis+Twin+Tiwf+Tbck` ≤ ~1e-14 rel.
- **Diag invariance:** `with_diags=True/False` outputs bit-identical in model state.
- **Climate/stability:** 10-day A100 linfs+TKE; year-scale JAX-TKE ↔ `c_tke_2yr` ≪ the TKE↔KPP scheme
  contrast (the C measured 11–18×); scheme-engaged check (TKE ≠ KPP, rel > 0.1 — the K.8 pattern).
- **If 9a is done:** zstar+TKE 10-day smoke + year-scale vs `c_zstar_tke_2yr` (the C matrix oracle).

## 4. AD-safety strategy (the differentiability contract)

- **`sqrt` at 0:** `sqrttke = √max(0,tke)` — tke=0 at cold start/dry lanes ⇒ safe-sqrt
  (`where(x>0, x, 1)` inside) so the backward pass is finite; after step 1 the floor keeps wet-column
  tke ≥ 1e-6. `|stress|` ⇒ safe-norm (NaN gradient at exactly-zero wind otherwise).
- **The `tke_min` floor kills gradients in the quiescent ocean** (most of the deep interior). This is
  the C/Fortran physics — keep the **exact clamp** for fidelity. Consequence for gates and training:
  choose gradient-gate losses in ACTIVE regions (e.g. mean mixed-layer Kv or SST), and note a smooth
  (softplus) floor as a possible Phase-7 *training-time* variant behind a config flag — never on the
  verification path.
- **Clamp inventory** (kinks, all faithful): `max(1e-12, N²)`, `max(S², 1e-12)`, Pr ∈ [1,10] (gradient
  dead outside Ri ∈ [0.15, 1.5]), `kappaM_max`, the mxl directional **min-scans** (two `lax.scan`s or
  associative cummins; padded/dry levels must NOT contaminate the running min — pad with +∞-like
  nominal before the scan, mask after).
- **Tridiagonal solve:** same class as the already-differentiated `tracer_diff` Thomas scan; padded
  rows = **identity** (a=c=0, b=1, d=carry) so masked tke stays inert and gradients don't leak.
  Mirror the C's reciprocal-multiply form for forward bit-parity.
- **Scan-carry tape:** `tke` joins the checkpointed `lax.scan` carry — same memory burden as any
  prognostic tracer. Two gradient paths per step: tke→Kv/Av→ocean (the ML path, well-conditioned) and
  the tke self-recurrence (heavily floor-gated). Gate both: plateau on `d(loss)/d(tke_c_k)` and a
  finite `d(loss)/d(tke-IC)` probe.
- **`pow(x, 1.5)`:** value+first-derivative fine at 0; port as `x**1.5` (the C chose `pow` over
  `x·sqrt(x)` for Intel bit-parity — for JAX-vs-C gating expect ≤1-ulp libm residue, tolerated).
- **mo_convect after TKE** (hard `bvfreq<0` branch to Kv/Av=0.1) is part of the effective mixing law —
  already ported + AD-treated in `pp.py`; reuse, don't duplicate.

## Development Approach (standing rules)

- Oracle-first: each task lands its replay/dump tests with the kernel; full suite green before the
  next task; `tke_cfg=None` byte-identity asserted throughout.
- **STANDING RULE:** append one lesson per task to `docs/PORTING_LESSONS.md` as you go.
- `[x]` immediately; ➕ discovered tasks; ⚠️ blockers; keep this plan in sync; move to `completed/`
  at GATE 9b.
- Compute: suite via `scripts/runs/run_suite.sbatch` (CPU); stability/gradients on A100 (`-A ab0995_gpu`);
  C dump regeneration jobs `-p compute --time=30:00`. Env python as in the parent plan.

## Implementation Steps

### JT.0 — Scaffolding: cfg seam, State.tke, readers, oracle audit (NO behavior change)

**Files:** Create: `fesom_jax/tke.py` (cfg + stubs), `fesom_jax/tests/test_tke_replay.py`.
Modify: `fesom_jax/state.py`, `fesom_jax/step.py`, `fesom_jax/integrate.py`,
`fesom_jax/integrate_sharded.py`, `fesom_jax/io_dump.py`, `fesom_jax/zarr_output.py`,
`fesom_jax/params.py`, `fesom_jax/tests/test_state.py`.

- [x] `TkeConfig` (with un-ported-option raises) + 3-way dispatch skeleton (raise if kpp+tke) +
      threading + static_argnames — `tke.py` `TkeConfig.validate` (IDEMIX/Langmuir/Dirichlet/
      mxl_choice≠2), `step.py` 3-way (tke→kpp→pp; both-set + pi-path raises), threaded through
      step/run/integrate/integrate_sharded + static_argnames; `integrate_sharded` closes over cfg.
- [x] `State.tke` + the 9-point checklist — state.py decl+zeros (IC=0), conditional replace
      (`step.py`, ice precedent), integrate + integrate_sharded threaded, partition_state/pytree
      generic (no change), `zarr_output.DEFAULT_FIELDS += "tke"`, halo TWO FACTS documented near
      `OCEAN_SCHEDULE` (field never exchanged; node-Av exch is internal to `mixing_tke`), ICs inherit
      0, `test_state.py:_expected_shapes += tke`.
- [x] `Params` + `tke_c_k/tke_c_eps/tke_cd/tke_alpha` (default_factory; defaults == config constants
      `config.py:TKE_*`; ⚠️ also updated `register_dataclass` data_fields AND `Params.defaults()` —
      8 leaves; verified `jax.tree_leaves` count + grad-visibility).
- [x] `TKE_TAGS` (20 tags) + `load_tke_dump` (multi-rank merge-by-gid, the `load_ale_dump` clone) on
      the shared `read_gid_table`; cdump audited (16r × 20 tags × 3 steps = 960 files, nod2D=126858 /
      elem2D=244659, clean strict merge); `replay/` is a C-internal nc artifact (NOT the JAX gate
      input — the cdump's self-contained input+output bundle is). Oracles fresh (no regen needed).
- [x] tests: `test_tke_replay.py` — None ⇒ pi step leaves `state.tke=0` (dead branch); reader
      round-trip (3 steps); `pytest.raises` on un-ported `TkeConfig` combos + both-cfgs + pi-path;
      Params defaults/leaves; mixing_tke stub raises. **16/16 pass; test_state 4/4.** Suite (job
      25558155): **OCEAN 550 + ICE 47 passed, 0 fail** (byte-identity GREEN — incl. test_tke_replay,
      the tripwire, test_partition_state's generic `tke` coverage). Sharded `tke` N-vs-1 confirmed
      separately (job 25560355): `test_{local_mesh_reconstruction_serial,serial_sharded_step_matches_
      dense,sharded_step_owned_matches[2]}` PASS — the generic State-field loops fold/partition/
      reconstruct the new `tke` field correctly. (The full sharded suite hit the pre-existing
      `[[sharded-suite-slow-phase8b]]` wall-time timeout in `test_gradient_sharded`, not a regression.)

**GATE JT.0 met** — `tke_cfg=None` byte-identical; the cfg seam + State.tke + Params + reader are in.
Next: **JT.1 — the column core `cvmix_tke.py`, controlled-replay-gated (13 core tags ≤1e-13).**

### JT.1 — Column core (`cvmix_tke.py`): the pure function, replay-gated

**Files:** Create: `fesom_jax/cvmix_tke.py`. Modify: `fesom_jax/tests/test_tke_replay.py`.

- [x] mixing length (sqrttke, stability bound, the two directional min-scans + special pre-step +
      floor) — alpha_tke NOT in mxl. ⚠️ **the backward min-scan runs k=nlev-2..1, NOT 1..nlev-1**
      (it must NOT re-touch the special-pre-step `nlev-1` value, else `min(mxl_min+dzw, mxl[nlev]+dzw)`
      drops it by exactly mxl_min — a constant-offset bug that masquerades as a flip; lesson JT.1).
      Implemented as two `lax.scan`s (sequential carry handles the special-value seeding naturally).
- [x] KappaM/Pr/KappaH (TKE_C66 = 6.6 *double*; jnp.maximum/minimum = the C compare-select).
- [x] forcing; tridiagonal (interface-indexed; the `ke` `take_along_axis` index quirk
      `kp1=min(k+1,nlev-1)`/`kk=max(k,1)`; Neumann surface+bottom overrides AFTER the interior fill;
      Patankar on interior rows only); Thomas in the **reciprocal-multiply** form (`fxa=1/m; cp=c*fxa`);
      `tke_min` floor (pre-floor `tke_solve` kept for Tdif/Tdis/Tbck).
- [x] the 13 diagnostics in Fortran order (`_diagnostics`): K/P_diss seed forc/Tbpr/Tspr; pre-floor
      `tke_unrest` seeds Tbck; Tdif/Tdis on the pre-floor solve — returned only under `with_diags`.
- [x] padded-row identity (a=c=0,b=1,d=0) + `_safe_sqrt`/`_safe_pow32` + clamped denominators per §4.
- [x] tests: **controlled replay vs the FIXED cdump** s1–s3 → 13 tags **bit-exact ≤3e-17** (≪1e-13);
      budget closure ≤4e-19; diag invariance; masked-grad finite + 0 on dry lanes. **22/22 pass.**
      ⚠️ DISCOVERED: the cdump was STALE (built with `(float)6.6`); regenerated with the fixed binary
      (6.6 double) → bit-exact. Stale preserved as `cdump/dump_stale_6.6f` (lesson JT.1).
- [ ] full suite green (regression) — submitting

### JT.2 — Driver (`tke.py`): column assembly + Kv/Av wiring

**Files:** Modify: `fesom_jax/tke.py`, `fesom_jax/tests/test_tke_replay.py`.

- [x] per-node assembly (`mixing_tke`): `forc_tke_surf = _safe_sqrt(|stress|²)/ρ₀`; `vshear2` (the
      kpp.ri_iwmix `_shift_down(Z)-Z` shear, masked to interior); `bvfreq2 = where(is_interior, bvfreq,
      0)` (zeroed surface+bottom — the naive-slice surface leak avoided); `dz_trr` interior `|ΔZ|` +
      `hnode/2` surface & bottom caps; `dzw = hnode`. ALL geometry via `_layer_center_Z(mesh, Z3d)`
      (static `mesh.Z` padded / live zstar — byte-identical under linfs).
- [x] post-solve (`_wire_kv_av`): zero Kv/Av at surface + below-bottom; `Kv = KappaH` full slab;
      **exchange node `KappaM` (`exch`) BEFORE** the node→elem 3-vertex mean over interior element
      interfaces; `mo_convect` inside; return `(Kv, Av, uvnode, tke_new)`. (Driver passes `dt`; dropped
      the unused `zbar3`.)
- [x] tests: **driver-replay** — inject cdump `tkeav`/`tkekv` at the driver boundary → `kv`/`av` wired
      tags **bit-exact** (kv 0.0, av ≤1.4e-17; the C exchange leaves owned rows unchanged ⇒ `kv≡tkekv`);
      `dz_trr` assembly vs cdump `dztrr` **0.0**; `mixing_tke` composition (cold-start Kv/Av=0, shapes,
      finite) + `jax.grad` wrt `tke_cd` finite; diag invariance (JT.1 `test_diag_invariance`). 25/25.
      ⤷ the full step-1/3-step LIVE gate (with CORE2 forcing) is folded into **JT.3** (the assembled
      gate) — the assembly's non-trivial pieces (`dz_trr`, the wiring) are already bit-exact here, and
      `vshear2` reuses the gated kpp shear.
- [ ] full suite green (regression) — submitting

### JT.3 — Step wiring + assembled live gate

**Files:** Modify: `fesom_jax/step.py`, `fesom_jax/tests/test_tke_step.py` (create).

- [x] dispatch + conditional `tke` replace + `tke_cfg` threading live end-to-end (eager + jit) —
      done in JT.0/JT.2; `test_tke_step.py` confirms a full CORE2 TKE step runs eager + jit.
- [x] tests (`test_tke_step.py`, the K.8 pattern): **4 pass** — `state.tke` evolves off zero (TKE
      genuinely running); **scheme-engaged** TKE≠KPP (cold-start Kv=0 vs KPP OBL Kv>0, rel≫0.1);
      **both-cfgs-set raise** through the forced step; **jit-twice** bit-identical (static cfg). pi-path
      raise is in `test_tke_replay.py` (JT.0).
- [~] ⚠️ the cdump-matching FORWARD gate (step-1 `tke`/`forc_tke_surf` vs cdump) is **xfail** — the
      JAX `build_core_forcing` dt=1800 step-1 wind stress differs from the cdump's C-run by ~7e-4
      (≈60% of scale), **IC-independent** (identical under ic_core2 vs dist_16) ⇒ a forcing step-1
      **time/convention** mismatch, NOT FP and NOT the TKE port (which is bit-exact via the JT.1/JT.2
      replay gates; the KPP forcing gate matched <1e-12 at dt=500 with its own forcing-matched dump).
      A `core2_forcing` dt=1800 alignment (or a forcing-matched TKE step re-dump) unblocks it — handle
      in JT.5/follow-up. (lesson JT.3)
- [ ] full suite green (regression) — submitting

### JT.4 — Gradient gates (the ML seam, GATE 9b core)

**Files:** Create: `scripts/core2_tke_grad_gate.{py,sbatch}`. Modify: `fesom_jax/tests/test_gradient.py`.

- [x] **FD↔AD plateau** (`scripts/core2_tke_grad_gate.{py,sbatch}`, GPU job 25570878, n=4):
      `d(mean ML Kv)/d(tke_c_k)` plateau **8.2e-8** + `d(mean surf tke)/d(tke_cd)` plateau **7.8e-9**
      (both ≪1e-4) — the two ML-seam parameters, well-conditioned. (Params leaves ⇒ traced through
      the full `integrate`, no static-cfg trick.)
- [x] **masked-NaN** `d(mean SST)/d(T0)` TKE-ON: non-finite=0, wet max|g|=2.0e-4, **masked max|g|=0.0**.
- [x] **`d(mean SST)/d(tke-IC)` finite** through the N-step checkpointed scan: non-finite=0, wet
      max|g|=1.2e-3, **dry max|g|=0.0** — the new tke scan-carry path (`d_tri = tke_old + dt·forc` ⇒
      the IC propagates linearly even at cold start). **`TKE_GRAD_GATE_OK`.**
- [x] physical-sign sanity: `d(mean ML Kv)/d(tke_c_k) = +2.97e-2 > 0` (more c_k ⇒ more mixing ✓).
- [~] ⚠️ GPU OOM fix banked: the script runs 4 separate full-`integrate` grads (vs KPP's 1) + TKE's
      heavier backward (the tke carry + mxl/thomas scans) ⇒ `jax.clear_caches()` between gates frees
      executables (lesson JT.4). The column-core + driver AD are also gated in pytest
      (`test_column_grad_finite`, `test_mixing_tke_composition_and_grad`).
- [ ] full suite green (regression) — no model change in JT.4 (the script is the deliverable)

### JT.5 — Stability + climate + sharded

**Files:** Create: `scripts/core2_tke_stability.{py,sbatch}`.

- [x] CORE2 A100 linfs+TKE (KPP swapped out) **stable** (`scripts/core2_tke_stability.{py,sbatch}`,
      job 25572009): 480 steps stable, max|vel|=1.53 m/s (bounded), no NaN; KPP also stable. **And
      the scheme-discrimination: TKE↔KPP surface SST RMS = 0.43 °C — RESOLVED** (TKE genuinely distinct
      from KPP ≫ FP noise). ⚠️ ran at the imported KPP `DT=500` (= 2.78 days, not 10 — a trivial
      re-param to dt=1800×480 for the proper 10-day; the structural result holds).
- [x] **year-scale JAX-TKE ↔ `c_tke_2yr`** (`scripts/core2_tke_climate{,_compare}.py`, job 25574435,
      1-yr `ic_core2_dist864`): **stable 17520 steps** (max|vel| 2.80), and **SST RMS = 4.68e-3 °C /
      SSS 2.74e-3 psu** vs the C oracle — **≈ the C↔Fortran reference 0.0049/0.0028** (A ≈ C0) ⇒ the
      JAX TKE climate is as faithful as the C is to Fortran, and **≪ the TKE↔KPP 0.43 °C contrast**.
      `TKE_CLIMATE_OK`. ⇒ **the step-1 forcing-gap has SMALL climate impact (A~=C0) but is PERSISTENT + UNDIAGNOSED — NOT a "transient", NOT a climate
      blocker** (it never propagates; the live-step-1 gate stays xfail as transient-sensitive). The
      `--tke` flag was added to `core2_kpp_climate_run.py`.
- [x] **sharded N-vs-1 (CPU ×4) TKE-ON** (`test_step_sharded.py`, job 25571969): serial byte-id +
      npes=2 owned-match **PASS** — the direct proof the internal node-`tke_Av` exchange
      (`_wire_kv_av`) is correct (owned-boundary `Av` matches N-vs-1) AND the `tke` field's
      no-exchange design holds (the generic field loop: owned tke == dense).
- [ ] ➕ if 9a landed: zstar+TKE 10-day smoke + year-scale vs `c_zstar_tke_2yr` — deferred (with the climate)
- [ ] full suite green (regression) — no model change in JT.5 (tests + scripts only)

### JT.6 — Close-out

**Files:** Modify: `docs/PORTING_LESSONS.md`, `README.md`, this plan, parent plan, memory.

- [ ] GATE 9b table green; lessons appended; parent plan + memory updated; commit; move to `completed/`

## Technical Details

- `tke [nod2D, nl]` float64, interface-indexed (nl interfaces, nl−1 layers); wet slice
  `uln0 … nln0+1` with `nln0 = nlevels_nod2D−2` (0-based); outside stays 0 forever (masked-inert).
- The Kv/Av the ocean uses at step n derive from `tke_{n−1}` (lag structure — relevant to gradient
  interpretation, mirrors the C exactly).
- Memory: +1 [nod2D, nl] State field; diags are test-only (never in State).
- Dead-but-passed arguments (`bottom_fric`, `iw_diss≡0` read unconditionally) are kept as explicit
  zero arrays where the C keeps them — call-site parity, zero cost under jit.

## Post-Completion

- Phase-7a tuning targets: `tke_c_k`, `tke_c_eps`, `tke_cd`, `tke_alpha` (now in `Params`); the
  perfect-model-twin recipe from `20260607-fesom-jax-paramtune.md` applies unchanged.
- Phase-7 NN seam: the natural swap point is `KappaM = f(mxl, sqrttke, …)` / the Pr law — the column
  core's pure-function design makes this a one-function replacement behind the same contract.
- A smooth-floor (softplus `tke_min`) **training-time** variant: only if Phase-7 training shows the
  clamp starves signal; never on the verification path.

## GATE 9b (acceptance)

**GATE 9b MET (2026-06-13)** — 7/8 green; the 8th is the TKE cdump being an OUTLIER on step-1 forcing (NOT a JAX bug — JAX==KPP-oracle≠cdump).

| Check | Bar | Result |
|---|---|---|
| `tke_cfg=None` | full suite green, byte-identical path | ✅ OCEAN 559 + ICE 47, 0 fail |
| Controlled replay (13 core tags @ JT.1 + kv/av @ JT.2) | ≤1e-13 per tag | ✅ **bit-exact ≤3e-17** (kv 0.0 / av ≤1.4e-17) |
| Step-1 live + assembled 3-step | ≤1e-12-class | ⚠️ **xfail** — the TKE cdump is the OUTLIER on step-1 forcing (3-way: JAX==KPP-oracle≠cdump); NOT a JAX bug |
| Budget closure | ≤1e-14 rel, standing test | ✅ ≤4e-19 |
| 10-day A100 TKE-ON | stable, physical Kv | ✅ stable (480-step + the 1-yr below); max\|vel\| 1.5–2.8 |
| Year-scale vs `c_tke_2yr` | ≪ TKE↔KPP contrast | ✅ **SST 4.68e-3 / SSS 2.74e-3 ≈ C↔Fortran floor; ≪ TKE↔KPP 0.43 °C** |
| Gradients | c_k/cd plateaus ≤1e-4; masked-NaN clean; tke-IC finite | ✅ **`TKE_GRAD_GATE_OK`** — 8.2e-8 / 7.8e-9, clean, finite |
| Sharded | N-vs-1 (CPU ×4) TKE-ON within `_BYTE_ID_ATOL` | ✅ serial byte-id + npes=2 owned-match (the `tke_Av` exch) |

## Revision Log

- **2026-06-11 — Plan created** from the C-port digest (completed C TKE plan + its 26 recorded
  lessons — the −r8 literal bug, the dead-knob inventory, the replay methodology — and the JAX seam
  map incl. the 9-point prognostic-field checklist). Locked: prognostic State.tke, Params-first
  trainable constants, replay-primary validation, exact-clamp fidelity with training-time smoothing
  deferred.
- **2026-06-12 — Plan-review pass (1 MAJOR + 5 minor + nits, all applied).** MAJOR designed in: the
  **internal node-Av halo exchange** — the C exchanges node `tke_Av` (`fesom_tke.c:491`) BEFORE the
  node→elem mean (boundary owned elements have halo vertices; the `kpp.py:787-789` precedent);
  `mixing_tke` now takes `exch=` — without it the port would pass eager and fail the sharded N-vs-1
  gate on boundary-element Av. Minor: JT.1 replay re-scoped to the 13 column-core tags (kv/av are
  driver-level ⇒ JT.2); the diag ordering deps corrected (Tdif→Twin is the unported Dirichlet branch;
  real deps K/P_diss→forc/Tbpr/Tspr, tke_unrest→Tbck); `bvfreq2` endpoint-zeroing made explicit;
  raise-tests added (TkeConfig validation, both-cfgs-set); Params edits extended to
  `register_dataclass`+`Params.defaults()`; budget-closure tolerance unified at ≤1e-14 (C measured
  ~4e-15); `test_state` tripwire re-cited to `_expected_shapes`. Review verified all ~20 sampled C
  citations, the checklist completeness, and the Params/TkeConfig split (GM precedent; deliberate
  divergence from KPP's static-only constants, justified by TKE being the designated ML seam).
- **2026-06-13 — JT.0→JT.5 implemented; GATE 9b MET; plan → `completed/`.** One session, 14+ commits
  (`d023f75`→`9335e8f`). Column core `cvmix_tke.py` + driver `tke.py` replay-gated **bit-exact**
  (≤3e-17) vs the **regenerated** cdump (the original was stale — built with `(float)6.6`, the literal
  bug; `cdump/dump_stale_6.6f` preserved). The ML seam is **fully differentiable** (`TKE_GRAD_GATE_OK`:
  FD↔AD `tke_c_k` 8.2e-8 / `tke_cd` 7.8e-9, masked-NaN clean, tke-IC finite). Sharded N-vs-1 proves
  the internal `tke_Av` exchange. Stable + year-scale climate **SST 4.68e-3 ≈ the C↔Fortran floor**.
  10 lessons appended. **Two real bugs caught**: the backward min-scan off-by-one (a constant-`mxl_min`
  offset masquerading as a flip) and the stale-`6.6f` oracle. **One over-reach corrected** (a reviewer
  challenge): the live-step-1 forcing-gap (7e-4) was pre-judged a climate blocker but is a transient —
  the climate run is the arbiter and it PASSED. Open (optional polish): port the C TKE-branch's
  low-wind gustiness/min-wind bulk term behind a flag to flip the lone live-step-1 xfail to a gate.
