# FESOM2 → JAX Port — Phase 9b: CVMix classical-TKE vertical mixing (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 9 — physics options).
**Predecessors:** GATE 6C (KPP — the mixing-seam template) + Phase 8.
**Siblings:** `20260611-fesom-jax-zstar.md` (9a), `20260611-fesom-jax-mevp.md` (9c). Recommended order
9a → **9b** → 9c, but **TKE has NO hard zstar dependency**: the C validated TKE under **linfs**
(deliberately, to isolate the mixing knob), and TKE reads geometry through per-node arrays that are
simply static under linfs. If 9b runs before 9a, the only obligation is JT.2's geometry-seam rule.
**Created:** 2026-06-11. **Status:** 🚧 IN PROGRESS — **JT.0 GATE met 2026-06-13** (`tke_cfg=None`
byte-identical: OCEAN 550 + ICE 47 pass; sharded `tke` N-vs-1 confirmed). Next: JT.1 column core.

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
- Compute: suite via `scripts/run_suite.sbatch` (CPU); stability/gradients on A100 (`-A ab0995_gpu`);
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

- [ ] mixing length (sqrttke, stability bound, the two directional min-scans + special pre-step +
      floor) — ⚠️ alpha_tke does NOT appear in mxl (it is only the TKE-diffusivity multiplier)
- [ ] KappaM/Pr/KappaH (TKE_C66 as a *double*; clamp order = C compare-select semantics)
- [ ] forcing terms; tridiagonal assembly (interface-indexed; the `ke` index quirk; Neumann BCs
      overwrite boundary rows AFTER the interior fill; Patankar dissipation on interior rows only);
      Thomas solve in reciprocal-multiply form; `tke_min` floor (keep pre-floor value for Tbck)
- [ ] the 13 diagnostics computed in column scratch in Fortran order — the real ordering deps:
      `K_diss_v`/`P_diss_v` seed `forc`/`Tbpr`/`Tspr` (`:260-266`), pre-floor `tke_unrest` seeds
      `Tbck` (the Tdif→Twin coupling is the UNPORTED Dirichlet branch — no such dep here; plan-review
      correction) — returned only under `with_diags`
- [ ] padded-row identity + safe-sqrt/safe-divide guards per §4
- [ ] tests: **controlled replay** — cdump inputs s1–s3 → the **13 column-core output tags** (tke,
      KappaM, KappaH + 10 diags) ≤1e-13 (the `kv`/`av` wired tags are driver-level ⇒ JT.2); budget
      closure ≤1e-14 rel; masked-lane finiteness of a `jax.grad` through one column
- [ ] full suite green

### JT.2 — Driver (`tke.py`): column assembly + Kv/Av wiring

**Files:** Modify: `fesom_jax/tke.py`, `fesom_jax/tests/test_tke_replay.py`.

- [ ] per-node assembly: `forc_tke_surf = safe|stress_node_surf|/ρ₀`; `vshear2` (zero at surface +
      bottom interfaces); `bvfreq2` (the shared smoothed N², **zeroed at surface + bottom interfaces**
      — nonzero only on `[nzmin+1, nzmax−1]`, `fesom_tke.c:362-364`; a naive slice leaks a nonzero
      surface value); `dz_trr` with **hnode/2 end caps**; `dzw = hnode` slice — ALL geometry via the
      §2 seam inputs
- [ ] post-solve: zero Kv/Av at surface + below-bottom interfaces (faithful); full Kv adoption;
      **exchange node `tke_Av` (`exch`) BEFORE** the node→elem 3-vertex Av mean over interior levels
      (`fesom_tke.c:491`; the `kpp.py:787-789` idiom — plan-review MAJOR); `mo_convect` inside;
      return `(Kv, Av, uvnode, tke_new)`
- [ ] tests: step-1 live gate (cold-start pure algebra vs cdump, target ≤1e-12); driver-level replay
      (inject inputs at the driver boundary) incl. the **`kv`/`av` wired tags**; diag invariance
      (with_diags on/off bit-identical model outputs)
- [ ] full suite green

### JT.3 — Step wiring + assembled live gate

**Files:** Modify: `fesom_jax/step.py`, `fesom_jax/tests/test_tke_step.py` (create).

- [ ] dispatch + conditional `tke` replace + `tke_cfg` threading live end-to-end (eager + jit)
- [ ] tests (the K.8 pattern): assembled 3-step vs cdump live tags (~1e-12 expected; flip-methodology
      fallback documented); scheme-engaged (TKE ≠ KPP rel > 0.1); pi-path raise; **both-cfgs-set
      raise** (`kpp_cfg` + `tke_cfg` ⇒ error); jit-twice no-leak
- [ ] full suite green

### JT.4 — Gradient gates (the ML seam, GATE 9b core)

**Files:** Create: `scripts/core2_tke_grad_gate.{py,sbatch}`. Modify: `fesom_jax/tests/test_gradient.py`.

- [ ] FD↔AD plateau on `d(loss)/d(tke_c_k)` and `d(loss)/d(tke_cd)` (active-region loss; plateau
      ≤1e-4; the `test_gradient.py:89-121` pattern)
- [ ] masked-NaN probe: `d(SST)/d(T0)` finite everywhere, 0 on masked lanes, TKE-ON
- [ ] `d(loss)/d(tke-IC)` finite through the N-step checkpointed scan (the carry path)
- [ ] physical-sign sanity at defaults (e.g. d(mixed-layer Kv)/d(c_k) > 0)
- [ ] full suite green

### JT.5 — Stability + climate + sharded

**Files:** Create: `scripts/core2_tke_stability.{py,sbatch}`.

- [ ] 10-day CORE2 A100 linfs+TKE (KPP swapped out) stable; Kv/hbl-analogue fields physical
- [ ] year-scale JAX-TKE ↔ `c_tke_2yr` ≪ TKE↔KPP contrast (discriminating-check style)
- [ ] sharded N-vs-1 (CPU ×4) TKE-ON — `tke` auto-covered by the generic field loops; confirm
      no-halo-exchange design holds (the halo-probe analogue: N-vs-1 equality IS the proof)
- [ ] ➕ if 9a landed: zstar+TKE 10-day smoke + year-scale vs `c_zstar_tke_2yr`
- [ ] full suite green

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

| Check | Bar |
|---|---|
| `tke_cfg=None` | full suite green, byte-identical path |
| Controlled replay (s1–s3; 13 core tags @ JT.1 + kv/av @ JT.2) | ≤1e-13 per tag |
| Step-1 live + assembled 3-step | ≤1e-12-class (flip-fallback documented) |
| Budget closure | ≤1e-14 rel, standing test |
| 10-day A100 TKE-ON | stable, physical Kv |
| Year-scale vs `c_tke_2yr` | ≪ TKE↔KPP contrast |
| Gradients | c_k/cd plateaus ≤1e-4; masked-NaN clean; tke-IC path finite |
| Sharded | N-vs-1 (CPU ×4) TKE-ON within `_BYTE_ID_ATOL` |

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
