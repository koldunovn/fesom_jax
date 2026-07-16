# ============================================================================
# >>> 2026-07-16: ►►► CURRENT BRIEF = `docs/HANDOFF-20260716-m7levers-session3.md` ◄◄◄
# >>> (Sessions 1+2 DONE: branch gated GREEN + main ff'd to 293293a (LOCAL, push pending);
# >>> fusions −8.6 % @ CORE2-8 / WASH @ dars-32 (small-shard lever); on-device forcing
# >>> −12 % CORE2 all-on, 1-yr climate-certified, default OFF (user), flag documented in
# >>> the configs. NEXT = harvest dt120 dars A/B 26299153, finish ladder (NG5 local-forcing
# >>> increment, CGPOLY judged at NG5-64), then the paper RE-MEASURE with on_device OFF
# >>> (user-confirmed). NEVER submit port_kokkos jobs — their session owns them.)
# >>> Everything below is OLDER.
# ============================================================================
# >>> 2026-07-13: ►►► PREVIOUS BRIEF = `docs/HANDOFF-20260713-paper-review-halo-bench.md` ◄◄◄
# >>> (GMD-paper review pass DONE + three-transport halo benchmarks; paper_jax is now a LOCAL
# >>> git repo, commits c070ace→af78607; FIRST ACTION next session = read the NG5-64 result,
# >>> job 26228433, `scripts/logs/bench_ng5_halo64.26228433.out`.) Everything below is OLDER.
# ============================================================================
# >>> NOTE: a SECOND, separate effort is underway — the GMD **model-description paper**
# >>> ("a JAX FESOM2 exists / on-par with Fortran FESOM / scales").
# >>> ►►► START-OF-SESSION BRIEF = `docs/HANDOFF-20260623-model-paper.md` ◄◄◄ (read it FIRST).
# >>> Plan = `docs/plans/20260619-fesom-jax-model-paper.md`. (20260619/20/21 handoffs = older briefs.)
# >>> STATUS 2026-06-23: MODEL DONE. Part A (usability) COMPLETE; B0 + NG5 R0 + dars-unblocked + scaling
# >>> analysis all GREEN. FORCA20 3-yr JAX headline run IN FLIGHT (~18.5%); Fortran FORCA20 reference DONE.
# >>> The FIGURE LAYER is entirely UNBUILT (every scripts/fig_*.py missing). NEXT SESSION (per user) =
# >>> MAKE THE PAPER PLAN + WRITE THE MANUSCRIPT. Two open decisions: NG5 scope/budget + CORE2 matched config.
# >>> Everything BELOW in this file is the OTHER (JAMES *capabilities*) paper — don't conflate them.
# ============================================================================
# STATUS — session 2026-06-18 (F1/F2 paper wrap-up: verification + docs DONE; 2 open forks)
# ============================================================================

**TL;DR: the F1/F2 wrap-up is done for everything that doesn't need a decision. All §1/§2/§3 + infra
`*_OK` tokens verified green; the 3 headline figures (Fig 2/3/4) + the window-SNR supplement are on
disk; the `params=None`/NN→0 bit-identical invariant holds (suite-guarded). README gained a
"Differentiable capabilities" section (committed `7c5cdd3`, NOT pushed). PORTING_LESSONS + the
memories are complete. Suite RE-CONFIRMED GREEN (job 25736458: ocean 656 + ice 47 passed, matching
the prior green 25717958; sharding = the pre-existing `shard_map`-compile timeout). TWO items left,
both the user's call (see forks).**

- **F1 tokens — GREEN:** Infra (OBS_OPERATOR/OBS_ICE/CALIBRATE_SEAM/EKI/TKE_NN/FORTRAN_TRANSFER/
  WINDOW_DERISK), §1 SENSITIVITY_MAP, §2 TWIN_RECOVER/TKE_CALIB/GM_EKI_TWIN/GM_EKI_BUDGET/D2C_HELDOUT,
  §3 NN_TWIN_SHARDED/NN_OBS. (D2C_HELDOUT_OK is emitted by `fig_calibration.py`, not a `core2_*` driver
  — a `[A-Z_]+_OK` grep misses it because the `2` in `D2C` is a digit; not a real gap.)
- **F1 figures:** `fig_sensitivity.png` (Fig 2), `fig_calibration.png` (Fig 3), `fig_hybridml.png`
  (Fig 4), `fig_window_snr.png` (supplement). **Fig 1 (foundation) absent = §0 B1 blocked.**
- **F2 docs:** README "Differentiable capabilities" section (3 pillars + honest caveats + figure
  pointers, committed `7c5cdd3`); PORTING_LESSONS complete (45 paper-era markers, incl. the E2
  offline/online lesson); memory `fesom-jax-port.md` capstone added (points at the 5 dedicated paper
  memories: sensitivity-c1-complete, d2-calibration-complete, e1-nn-twin-memory-fix,
  multigpu-sharded-adjoint-horizon, e2-nn-obs-offline-online).
- **EXTERNAL-only remaining (NOT a regression):** §0 B1/Fig 1 (`FOUNDATION_BASELINE_OK` — needs the
  parallel-session all-on climate run) + D3/`FORTRAN_IMPROVES_OK` (the operational Fortran run,
  Post-Completion). A8 Option B (free-drift ice adjoint) also still planned, not blocking.

## Two open forks (the user's call — deferred both rather than guess)
1. **Move the plan to `docs/plans/completed/`?** I did NOT — §0 B1/Fig 1 is a checkboxed step still
   blocked on the parallel climate run, so the plan stays active until §0 lands. (The 3 JAX pillars +
   infra ARE complete; D3 is external.) Alternative: move it now, treating §0/D3 as Post-Completion.
2. **Pursue `NN_DRIFT_OK` (the persisted-benefit closer)?** Explicitly OPTIONAL + "not overnight" (a
   big compute job). Paths: (a) a longer differentiable rollout / O(√N) checkpointing past N≈20 against
   a multi-day-aggregated MLD target; (b) a physically-informed prior (trust-region toward the D2a 2×);
   (c) EKI on the deployed-MLD metric (forward-only ⇒ immune to the fast/slow misalignment). §3 is
   already DONE with its honest finding (the offline/online gap) ⇒ the paper does NOT need this.

## Commit (code only — plan + this prompt + .claude + memory uncommitted, NOT pushed)
- **`7c5cdd3`** `paper/F2: README — add a 'Differentiable capabilities' section (§1/§2/§3 + honest caveats)`

## Recommended next step
F1/F2 is done. If the goal is "paper experiments closed on the JAX side": nothing more is needed —
wait for §0's parallel climate run, then do B1/Fig 1 + D3 (both external). For a stronger §3: fork #2
(the SLOW-target NN-MLD closer). Otherwise the suite re-confirm (25736458) is the only thing in flight.

# ============================================================================
# STATUS — autonomous session 2026-06-18 (§3 E2 NN obs-training: DONE — NN_OBS_OK + the offline/online finding)
# ============================================================================

**TL;DR: §3 E2 is COMPLETE with a DUAL result. The NN closure trains end-to-end through the global adjoint
to reduce REAL held-out obs misfit (`NN_OBS_OK`) — but the short-window optimum does NOT deploy: a 90-d
forward shows the trained NN is STABLE (a trust-region reg ⇒ drift ≈ default, no blow-up) yet its obs
benefit does NOT persist (the offline-trained/online-deployed gap, now PROVEN to be a fast/slow wrong-sign
misalignment). Committed `1966e7e` (code + lesson only; plan + this prompt + .claude uncommitted, NOT pushed).**

- **`NN_OBS_OK`** (job 25722727, all3 frozen-ice, N=12 batched SEASONAL windows over the 12 monthly
  `nn_twin_snaps`): held-out MLD **−2.1%** = train −2.1% (the D2c clean no-overfit bar), SST −0.1%; bounded
  multiplier (mean 0.97, ∈[0.5,2.0]) ⇒ PD diffusivities + a **spatially-structured** correction a global
  scalar can't make. **The adjoint-trains-NN-on-real-obs capability is PROVEN.**
- **`NN_DRIFT` = STABLE but benefit DOESN'T persist** (90-d forward, default NN→0 vs trained): drift T 1.01× /
  S 1.01× default, finite+physical throughout — but the trained NN's long-forward MLD is WORSE than default at
  EVERY horizon (+11% seasonal-mean). Naive (un-reg) training is far worse: a **bang-bang over-mixing** NN
  (mean 2.8×, saturating both [⅓,3] caps) that minimizes the 5-h misfit but **blows up the 90-d MLD ×2–4**;
  the data loss itself OVERSHOOTS (non-monotone — best at it≈15). **Fix (stability only):** keep-best iterate +
  a **trust-region reg** (area-wt penalty on `(log m)²` toward default; =0 at NN→0 ⇒ bit-identical invariant
  holds) + tighter `m_max`=2 ⇒ stable deploy, but the misalignment remains.
- **The finding (PROVEN, not inferred):** the short-window adjoint optimizes the **FAST** (6-h) MLD response,
  misaligned with the **SLOW** deployed equilibrium — the NN's net multiplier flips **below 1** (less mixing
  helps 6 h), opposite D2a's validated "more mixing deepens MLD". A **uniform-multiplier diagnostic**
  (`--const-mult`, job 25725349): global **2× IMPROVES** the 90-d MLD (+0.5%, stable), **0.7× WORSENS** it
  (−0.4%) — monotone in D2a's direction — yet the NN sits at net **0.97×** (wrong side). ⇒ the deployed MLD is
  a **SLOW target** (a longer differentiable rollout / EKI / a prior toward the D2a 2×) — the SAME adjoint↔EKI
  boundary as GM→T/S (D2b). **The long-forward drift+persisted-benefit gate is ESSENTIAL** — a held-out
  short-window obs reduction is necessary but NOT sufficient for a deployable closure (offline ML-closure
  papers that stop at the short-window number hide the online failure).
- **Suite GREEN** (ocean **656** + ice **47**, job 25717958; sharding = the pre-existing timeout). E2 is
  scripts-only ⇒ zero library surface (the `params=None`/NN-off bit-identical invariant holds).
- **Fig 4** (`scripts/paper/fig_hybridml.py` → `fig_hybridml.png`): (A) E1 twin recovery; (B) per-season held-out
  MLD reduction (train≈held-out); (C) stable deploy (drift ≈ default); (D) the offline/online gap.
- Code: `scripts/core2_paper_nn_obs.{py,sbatch}` (`--reg`/`--m-max`/`--const-mult`/keep-best/`--mode
  train|validate`), `scripts/paper/core2_paper_nn_obs_diag.sbatch`, `scripts/paper/fig_hybridml.py`. Memory:
  `e2-nn-obs-offline-online`. PORTING_LESSONS entry added.

## Commit (code only — plan + this prompt + .claude deliberately uncommitted, NOT pushed)
- **`1966e7e`** `paper/§3 E2: NN obs-training — held-out obs reduced (NN_OBS_OK) + the offline/online deploy gap`

## Recommended next step (§3 substantially done: E1 twin + E2 obs, each with its honest caveat)
1. **F1/F2 wrap-up** — verify all `*_OK` tokens + the four headline figures; update `README.md`; move the plan
   to `docs/plans/completed/`. (§3's two pillars are E1 `NN_TWIN_SHARDED_OK` + E2 `NN_OBS_OK`; `NN_DRIFT` is
   reported as the rigorous stable-but-not-persisted finding, like E1's PARTIAL field + D2b's window-limited GM.)
2. **(Optional, the persisted-benefit closer — a SLOW-target run, not overnight):** the diagnostic shows more
   mixing helps the deployed MLD, so a path to `NN_DRIFT_OK` is (a) train against a multi-day-aggregated MLD
   target (not the 6-h endpoint) via O(√N) checkpointing past N≈20, or (b) a physically-informed prior
   (trust-region toward the D2a 2× instead of toward 1) so the NN refines spatial structure on the right side,
   or (c) EKI on the deployed-MLD metric (forward-only ⇒ immune to the fast/slow misalignment).
⚠️ Reminders: single-GPU adjoint; all3 obs-calib uses the frozen-ice adjoint + remat_blocks (fits N=12, ~41 GB);
strong-type optimizer/normalizer scalars (weak-type recompile); reg=0 at NN→0 keeps the bit-identical invariant;
train↔validate handoff via the small NN pkl on /work (forward-only validate gets a fresh allocator pool).

# ============================================================================
# STATUS — session 2026-06-17 (§3 multi-GPU sharded NN-twin adjoint: FIXED + VALIDATED; chaotic horizon found)
# ============================================================================

**TL;DR: the multi-GPU sharded NN-of-TKE twin WORKS and is VALIDATED, and it RESOLVES the §3 field-recovery
question — but with a NEGATIVE finding that redirects the plan. Committed `d16509d` (code only; plan + this
prompt + .claude uncommitted, NOT pushed).**

- **Fixed the sharded-grad NaN** (the blocker): a *masked-NaN-in-reverse-mode* bug — `cvmix_tke`'s surface-flux
  denom `dzt_surf` divided by `dzt[0]=0` on all-dry SHARDED PADDING columns (`_default_pad` int→0 ⇒ nlev=−1 ⇒
  driver never sets `dz_trr[0]`; core's `is_surf=(k==0)` still divides). Forward masked-finite, reverse
  `0·inf=NaN` leaked via `cd=tke_cd·m_NN` into the NN weights. Fix = `dzt>0` guard (mirrors Part 4's `dzt_s`);
  **bit-identical, 38/38 TKE tests pass.** Also `integrate_sharded.return_grad_fn` (params a replicated
  `shard_map` input; const args placed once outside the grad trace) fixed the TracerArrayConversionError.
- **VALIDATED**: sharded multi-step NN-grad == dense (`|g|` 9.62 vs 9.63 @N=4); the sharded twin reproduces the
  dense recovery (evolution misfit ratio ~0.14, corr_all ~0.80 @N=4). all_gather (`use_ragged=False`) adjoint
  is autodiff-correct end-to-end for the full TKE+NN+GM+ice(frozen)+zstar config; P=4 N=48 @ 24.7 GB/dev.
- **FINDING (redirects the plan)**: the stiff all3 adjoint amplifies EXPONENTIALLY with window — it=0 `|g|` =
  9.6(N4) → 12(N6) → 2.5e4(N8) → 4.6e15(N12) → 2.4e124(N48). Recovery holds only **N≤6**; N≥8 diverges (raw
  AND clipped Adam). Field corr is BEST at the SHORTEST window and DEGRADES with N ⇒ the "long continuous
  window imprints the field" hypothesis is **FALSIFIED**. **Multi-node N≫48 is MOOT for field recovery** (the
  chaotic-adjoint horizon, not memory, binds). See `docs/PORTING_LESSONS.md` + memory
  `multigpu-sharded-adjoint-horizon`.
- **RESOLVED (field gate, user-approved reframe)**: amp-sweep at N=4 ⇒ `--truth-amp 2.0` is the Goldilocks
  perturbation (corr_active 0.32→0.56, **corr_all 0.83→0.88**, evolution misfit 0.052); weaker (1.5) under-
  signals, driving the loss lower destabilizes the stiff optimization and does NOT move corr_active ⇒ the
  strong-anomaly quartile is equifinality/saturation-limited. Twin gate REFRAMED: PRIMARY = `corr_all` (bulk
  field, `--corr-tol` default 0.8) + `corr_pw`; `corr_active` reported as a DIAGNOSTIC, not gated. **Canonical
  §3 sharded twin = amp=2.0 N=4 ⇒ evolution recovered + corr_all ~0.88 = `NN_TWIN_SHARDED_OK`** (job 25716978).
- **NEXT: §3 E2 obs-training** (`core2_paper_nn_twin_batched.py` / `core2_paper_nn_obs` — reduce MLD/upper-T
  vs obs over batched SHORT windows; does NOT need field identifiability; long-forward drift + persisted-benefit
  gates). Then F1/F2 wrap-up.
- Code: `scripts/core2_paper_nn_twin_sharded.{py,sbatch}` (now has `--clip-norm`),
  `scripts/repro_sharded_grad_nan*.{py,sbatch}`, `scripts/verify_sharded_tke_grad.{py,sbatch}`.

# ============================================================================
# STATUS — session 2026-06-15 (§2 D2c — held-out validation + Fig 3: DONE, GREEN)
# ============================================================================

**TL;DR: §2 D2c is COMPLETE — `D2C_HELDOUT_OK`. The TKE `c_k`→WOA-MLD calibration (D2a) was held-out
cross-validated TWO ways on one A100-80 (reusing the D2a spin-up S0): a `random` 50/50 cell split proves
it's NOT overfitting (held-out reduction ≈ train), and a blocked 60° `lon` split exposes the honest
structural limit (a single global scalar doesn't transfer uniformly across regions). The recovered `c_k`
is robust across all 5 splits. Fig 3 drawn. Code committed (`8ea6d19`), suite green. The obs result from
D2b (job 25603998) was already recorded — recommended-step #1 needed nothing further.**

- **`D2C_HELDOUT_OK`** (jobs 25610800 = full+lon, 25611718 = random):
  | split | recovered c_k | train MLD | **held-out MLD** | verdict |
  |---|---|---|---|---|
  | full-domain | 0.235 | +2.0% | — | (D2a headline) |
  | random fold0 / fold1 | 0.235 / 0.238 | +1.6% / +2.5% | **+2.44% / +1.65%** | **OK — not overfitting** |
  | lon fold0 / fold1 | 0.239 / 0.222 | +3.0% / +0.3% | **−0.15% / +2.75%** | spatial transfer (asymmetric) |
  - **random ≈ train** (statistically-identical halves) ⇒ the calibration fits real signal, not per-cell
    noise — the clean overfitting gate.
  - **blocked `lon` is asymmetric**: a global `c_k` *helps* the held-out deep-convection sectors (+2.75%)
    but slightly *over-mixes* the held-out low-bias sectors (−0.15%) ⇒ a single global scalar can't fix a
    spatially-structured bias (matches the §1 C1 map; **motivates the §1 field-leaf / §3 NN params**).
  - **`c_k` robust across ALL 5 splits**: {0.222 … 0.239}, spread **7.3%**, random cross-fold scatter
    **1.2%**, all ∈ [0.05, 0.30] ⇒ the VALUE is well-determined regardless of the split.
  - **SST** ΔRMSE **+0.0022…+0.0037 °C — at/UNDER the C↔Fortran 0.0049 °C floor** ⇒ honestly, SST is NOT
    meaningfully improved over the fast window; **MLD is the constrained channel**.
- **Fig 3** (`scripts/paper/fig_calibration.py` → `scripts/fig_calibration.png`): (A) D1 twin bowl + 800→1499
  recovery; (B) train-vs-held-out MLD% per fold (the random≈train / lon-asymmetry contrast); (C) `c_k`
  across every split in the plausibility band. Emits `D2C_HELDOUT_OK`.
- **Two design choices worth keeping** (now a `PORTING_LESSONS` entry): (1) random vs blocked CV answer
  DIFFERENT questions for a global scalar (overfitting vs spatial transfer) — report both; (2) score the
  held-out cells via `has_aux` so no gradient leaks (and `--holdout none` stays bit-identical to D2a).

## Commit (code only — plan + this prompt deliberately uncommitted, NOT pushed)
- **`8ea6d19`** `paper/D2c: held-out cross-validation of the TKE->MLD calibration (random + blocked folds)`
  — `core2_paper_calib_tke_obs.py` (`--holdout`/`--fold`/`build_holdout`), `core2_paper_calib_tke_xval.sbatch`
  + `_random.sbatch`, `fig_calibration.py`, the PORTING_LESSONS entry.

## Suite (regression guard) — GREEN
job 25610867: **ocean 652** + **ice 47** (unchanged baseline — D2c is scripts-only, zero library surface);
sharding = the pre-existing timeout (memory `sharded-suite-slow-phase8b`).

## Recommended next step (§2 fully proven on the JAX side; pick the direction)
1. **§3 E1 — NN-of-TKE perfect-model twin** (the natural next pillar; reuse `tke_nn` + `calibrate.optimize`
   + the frozen-ice adjoint): truth = a seeded `tke_nn` instance, train NN→0 back to it through the global
   adjoint. `NN_TWIN_OK`. D2c's "a global scalar can't fix the spatial bias" is the direct motivation.
2. **D3 Fortran transfer** — `write_namelist.py` patches the recovered `c_k`≈0.235 (+ the D1/D2b `k_gm`)
   into `namelist.oce`; the operational Fortran run is Post-Completion (external).
3. **GM production EKI** (~170 GPU-h, budgeted in D2b) for the real equilibrium `GM_CALIB` number — a
   separate long job; D2c's held-out methodology applies (basin held-out for the GM T/S target).
⚠️ Reminders: single-GPU adjoint; the obs calib uses the FULL all-on model + frozen-ice adjoint; strong-type
optimizer inits; reuse the D2a S0 (`/work/.../calib_tke_obs_S0.pkl`) for any further TKE-obs calibration.

# ============================================================================
# STATUS — autonomous session 2026-06-15 (§2 D2b — GM→T/S via EKI: twin PROVEN + budget)
# ============================================================================

**TL;DR: D2b's hard half is DONE — the perfect-model EKI GM twin is PROVEN on the FULL all-on model with
LIVE mEVP ice (forward-only ⇒ immune to the A8 ice-rheology adjoint instability that forced D1/D2a to
freeze the ice — the adjoint↔EKI split paying off). Budget measured. Code committed (`e268617`), suite
green. The scoped obs demonstration was running (overnight job 25603998 stage 3) at handoff — see "⏳ obs".**

- **`GM_EKI_TWIN_OK`** (headline job 25603998, n=240/10 members; reproduced by de-risk job 25603626 at n=96/6):
  inject `k_gm=1500`, freeze its basin-mean T/S, recover from a prior ensemble via EKI on `log k_gm`. Bowl argmin
  EXACTLY at 1500; recovered **k_gm=1500.51 (rel 0.034 %)** in **5 EKI iters** (ensemble 1500.5±0.5), misfit
  1.4e-5→4.2e-11, peak 23.8 GB. (The 2-day de-risk: 1500.97, 0.065 %, 3 iters.)
- **`GM_EKI_BUDGET_OK`**: all-on forward **~4.3 steps/s** on A100-80 (forward-only, peak 16 GB, memory ~flat
  in N) ⇒ a 16-member × 8-iter × **3-yr production EKI ≈ ~170 GPU-h** (scoped 14/28/57 GPU-h for 0.25/0.5/1 yr).
  ⇒ the full equilibrium `GM_CALIB_OK` is a **production run, NOT overnight**.
- **`GM_EKI_SEAM_OK`** (`tests/test_gm_eki.py`, 6/6 CPU): basin reduction vs a loop ref + masked/empty-finite
  + the log-`k_gm` EKI recovery wiring.
- **obs DONE** (`--mode obs`, job 25603998 n=480/10 members): the machinery runs end-to-end (`GM_CALIB_OK`
  gate fires: `k_gm=1356`, plausible, misfit reduced) **but the recovered VALUE shows it weakly constrained** —
  ensemble **1406±347** (26 %, no collapse), misfit **−0.2 %** (2.3635→2.3594), bowl **argmin at the grid edge
  (1800), flat ≈2.35–2.40 across [600,1800]**. The WOA misfit is IC/spin-up-dominated over a 10-day window ⇒
  `k_gm` barely moves it. **The twin proved the OPTIMIZER; the obs proves the WINDOW is the limit** — empirically
  confirming the slow-target thesis. The recovered-value ±347 is the rigor catch (the D2a `c_eps`→0 analogue). A
  meaningful obs GM number needs the multi-year **production** ensemble (~170 GPU-h). Results in
  `scripts/calib_gm_eki_results.jsonl` + `scripts/calib_gm_eki_obs.npz`.

## The observable (reused for the obs application + any future adjoint↔EKI cross-check)
Basin-mean **upper-ocean/thermocline T/S profiles**: model T/S → `obs_compare.to_obs` (live zstar z-interp)
→ `obs_compare.basin_mean_profiles` over **5 lat-band basins × 8 WOA levels (0–1500 m) × {T,S}** = an 80-vec.
The SAME fixed reduction (fixed basin weights + a fixed common-validity mask = model-valid ∧ WOA-valid) hits
model AND obs ⇒ a clean *linear* observable; members differ only through the physics. WOA target =
`scripts/tools/make_woa_ts_targets.py` → `woa_ts_targets.npz` (annual; physically textbook basin profiles verified).

## Two findings (now in `docs/PORTING_LESSONS.md`)
1. **The GM→T/S signal over a short window is TINY (~3e-4 °C at 2 d) but CLEAN** ⇒ the twin needs
   **auto-Γ = (ε·ensemble-signal)²** (ε=0.05); a physical absolute σ_T=0.5 °C swamps it (Γ ≫ C_gg ⇒ no
   update) — the EKI analogue of D1's "normalize the loss by J0". (And exactly why short-window *obs*, which
   must use physical σ, is weakly constrained.)
2. **A jit closing over the pre-stacked forcing makes XLA constant-fold the `to_obs` scatter over mesh
   constants** — a slow ONE-TIME compile (~20 s for one scatter-add), amortized over the EKI forwards; for
   the production multi-year run pass forcing as a *traced arg*. Plus: `cf.stack(N)` caps a pre-stack at
   ~weeks (10 MB/step) ⇒ multi-year needs **chunked re-stacking** (the `core2_kpp_climate_run.py` pattern).

## Commit (code only — plan + this prompt deliberately uncommitted, NOT pushed)
- **`e268617`** `paper/D2b: GM→T/S calibration via forward-only EKI (twin proof + budget)` — the 8 files:
  `obs_compare.basin_mean_profiles`, `eki.{sequential_eval, map_fn}`, the driver (+2 sbatch), the WOA T/S
  target builder, `test_gm_eki.py`, the 2 PORTING_LESSONS entries.

## Suite (regression guard) — GREEN
job 25603676: **ocean 652** (= 646 + 6 new `test_gm_eki`) + **ice 47**; sharding = pre-existing timeout.

## Recommended next step (D2b twin+budget done; obs demonstration in flight)
1. **Finish reading the ⏳ obs result** (job 25603998 stage 3) — record the recovered `k_gm` + the honest
   weakly-constrained caveat in the plan's D2b `[~]` line + Fig-3 notes.
2. **§2 D2c** — held-out validation + the falsifiable recovered-value bars (Fig 3): combine D1 twin (k_gm
   800→1500) + D2a TKE obs + the D2b EKI twin; the production GM_CALIB run (~170 GPU-h, budgeted) gives the
   real held-out GM number — schedule it as a separate long job if wanted.
3. **§3 E1** — NN-of-TKE perfect-model twin (reuse `tke_nn` + `calibrate.optimize` + the frozen-ice adjoint).
4. **D3** Fortran transfer (`write_namelist.py` ready). ⚠️ Reminders: single-GPU; forward-only EKI uses the
   FULL all-on model with LIVE ice (no frozen-ice needed); strong-type optimizer inits; the obs window is
   pre-stack-limited to ~weeks (chunk for multi-year).

# ============================================================================
# STATUS — session 2026-06-14 (§2 CALIBRATION — twins + obs application DONE, all-on)
# ============================================================================

**TL;DR: §2 is substantially complete — both perfect-model twins AND the TKE obs calibration run on the
FULL paper model (zstar+TKE+mEVP+GM) with the new frozen-ice adjoint.**
- **D1 twin** `TWIN_RECOVER_OK`: recovered **k_gm=1498.88 (rel 0.075%)** from 800 (job 25592258).
- **D2a twin** `TKE_TWIN_OK`: recovered **tke_c_k=0.14974 (rel 0.176%)** in 9 it (job 25592771).
- **D2a obs calib** `TKE_CALIB_OK`: TKE→**WOA MLD+SST** through the all-on adjoint + the `obs_compare`
  operator (first real-data use; baseline gaps 33 m / 0.81 °C). c_k-only: **c_k=0.243 (plausible), MLD
  −3.0%** (job 25602108); 2-param overfits (`c_eps`→0, caught by plausibility — the rigor working).

5 code commits (`3f475e6`, `111b06e`, `f13fc7d`, `0c7086f`, `b8806f8`); plan + this prompt uncommitted
per instruction; results gitignored. Three big findings this session: the **frozen-ice adjoint** (A8;
the mEVP adjoint explodes, `stop_gradient` it in the backward), the **monthly-WOA MLD** (per-month then
average — annual-profile MLD is Jensen-biased), and the **weak-type recompile** (`jnp.asarray(1.0)` →
the it=2 "Failed to load CUBIN" OOM, NOT memory — strong-type the optimizer init).

## Two user steers this session (HONORED — keep following)
1. **Use the plan in `docs/plans/`** (`20260614-fesom-jax-paper-experiments.md`). 2. **For real runs use
ALL options (zstar+TKE+mEVP) — proper ocean model, not single-physics toys.** D1 (and D2a/D2b/E1/E2) now
default to the all-on config.

## The headline finding + fix (mEVP sea-ice adjoint) — Task A8
The all-on **FORWARD is clean** (twin bowl argmin exactly at the truth) but the naive **ADJOINT explodes
through the mEVP sea-ice rheology** (`|g|≈9e51`, mis-directed, at N=12; ocean-only adjoint clean per C1) —
the classic VP/EVP sea-ice adjoint instability MITgcm/ECCO hit. **Fix shipped = Option A frozen-ice
adjoint** (`IceConfig.adjoint_mode="frozen"`): full mEVP in the forward, `stop_gradient` the ice in the
backward. `stop_gradient` is identity in the forward ⇒ **forward bit-identical** (frozen J0 == exact J0);
`|g|` 9e51→**2.31**. **Option B (free-drift adjoint, `custom_vjp`, MITgcm-style) is PLANNED** (plan A8;
restores ice-momentum sensitivity + enables adjoint sea-ice calibration). Ice-thermo/rheology-mediated
targets → **EKI** (forward-only, immune).

## What I built (D1 + D2a twins + D2a obs)
| File | Role |
|---|---|
| `scripts/paper/core2_paper_calib_twin.py` (+`.sbatch`) | D1 all-on `k_gm` twin: bowl scan + cosine-Adam recovery; `--config all3/tkegm/gm`; `--grad-check` |
| `scripts/paper/core2_paper_calib_tke.py` (+`.sbatch`) | D2a all-on `tke_c_k`→MLD twin (frozen-ice adjoint) — `TKE_TWIN_OK` |
| `scripts/paper/core2_paper_calib_tke_obs.py` (+`.sbatch`) | D2a obs calib: spin-up split + WOA MLD+SST via `obs_compare` + frozen-ice adjoint — `TKE_CALIB_OK` |
| `scripts/tools/make_woa_targets.py` | WOA MLD (per-month→avg, fixes Jensen bias) + SST obs target → `woa_targets.npz` |
| `scripts/paper/core2_paper_calib_twin_diag.sbatch` | adjoint grad-check diagnostic (config×N) |
| `fesom_jax/ice.py`, `fesom_jax/step.py` | `IceConfig.adjoint_mode` + frozen-ice backward (`stop_gradient`) |
| `fesom_jax/tests/test_calib_twin.py` | CPU pi recipe guard — `TWIN_RECIPE_OK` (4/4) |

## Two conditioning lessons (now in `docs/PORTING_LESSONS.md`)
1. **Frozen-ice adjoint** (above) — run full physics forward, skip the unstable block in the backward.
2. **Normalize the loss by J0** — the GM short-window misfit is ~1e-6 ⇒ raw grad ~1e-11 is swamped by
   Adam's `eps=1e-8`; optimize `loss/J0` on a scale-free leaf `u=k_gm/1000`. `lr=0.1/iters=100` from a pi
   probe (lr=0.05/80 missed the 2% bar at 2.2%; the normalized-bowl dynamics are mesh-independent).

## Suite (regression guard) — GREEN
job 25591721: **ocean 646** (= 642 + 4 new `test_calib_twin`) + **ice 47** (post-edit ⇒ confirms the
frozen-ice `step.py`/`ice.py` edits don't regress the `"exact"` path); sharding = pre-existing timeout.

## Commits (code only — plan + this prompt uncommitted, NOT pushed)
- **`3f475e6`** D1 twin driver + CPU recipe test (`TWIN_RECIPE_OK`).
- **`111b06e`** frozen-ice adjoint (A8) + all-on twin `TWIN_RECOVER_OK` + the two lessons.
- **`f13fc7d`** D2a TKE→MLD twin (`TKE_TWIN_OK`) + the MLD-fast-target lesson.
- **`0c7086f`** D2a-obs target builder + obs-calib driver (monthly-WOA MLD; gradient validated).
- **`b8806f8`** D2a-obs FIX: the it=2 "CUBIN OOM" was a weak-type recompile → strong-type the init ⇒
  the obs calib runs all-on (`TKE_CALIB_OK`).

## Recommended next step (§2 twins + TKE obs calib done; pick the direction)
1. **§2 D2b — GM→T/S via EKI** (`fesom_jax.eki`, forward-only ensemble on the full all-on forward —
   immune to the ice adjoint; the slow-target half of §2). The natural next §2 piece.
2. **§2 D2c** — held-out validation + the falsifiable recovered-value bars (Fig 3): calibrate on one
   period/region, validate on an independent one; the TKE c_eps overfit (2-param → c_eps→0) is the
   case-in-point for the plausibility-bound rigor. **D2a obs uses `--params ck`** (clean) for the headline.
3. **Option B — free-drift sea-ice adjoint** (plan A8; `custom_vjp`: forward = full mEVP, backward drops
   ∇·σ) — restores ice-momentum sensitivity + unlocks adjoint sea-ice calibration.
4. **§3 E1** — NN-of-TKE perfect-model twin (reuse `tke_nn` + `calibrate.optimize` + the frozen-ice adjoint).
Then D3 (Fortran transfer). ⚠️ Reminder: strong-type optimizer inits (weak-type ⇒ it=2 recompile/CUBIN OOM);
calibration uses the FULL all-on model + frozen-ice adjoint; single-GPU; descend N on OOM.

# ============================================================================
# STATUS — session 2026-06-14 (§1 C1 SENSITIVITY MAPS — DONE, GREEN)
# ============================================================================

**TL;DR: §1 Task C1 is COMPLETE and GREEN — `SENSITIVITY_MAP_OK`.** One backward pass promotes a
`Params` scalar to a `[nod2D]` field leaf (`calibrate.build_params`, ZERO kernel change) → the
instantaneous adjoint sensitivity map, for two targets. Both proven adjoint==FD, FD-spot-checked, and
(for k_gm) cross-checked against EKI. CPU unit test green (`SENSITIVITY_SEAM_OK`, 6/6); full suite
green (ocean 642 + ice 47). 3 code commits (`36911d0`, `62c2ed9`, `2326184`); plan + this prompt left
uncommitted per instruction; results (`.npz`/`.png`/`.jsonl`) gitignored.

## What I built (Task C1)

| File | Role |
|---|---|
| `scripts/paper/core2_paper_sensitivity.py` (+`.sbatch`) | field-leaf backward → `[nod2D]` map; adjoint==FD proof; h-sweep FD spot-check; adjoint↔EKI cross-check |
| `scripts/paper/fig_sensitivity.py` | Fig 2 — two maps + adjoint↔EKI inset → `scripts/fig_sensitivity.png` |
| `fesom_jax/tests/test_sensitivity.py` | CPU seam unit test (6/6, `SENSITIVITY_SEAM_OK`) |

## GPU results (job 25589761, A100-80GB, single-GPU)

| Target | Config | N | ∂J/∂θ | adjoint==FD plateau | FD spot (h-sweep) | adjoint↔EKI |
|---|---|---|---|---|---|---|
| **mld_ck** `∂(mean MLD)/∂c_k` | zstar+TKE | 12 | +2.91 | **6.08e-7** | 3.39e-7 (Weddell Sea) | — |
| **ts_kgm** `∂(upper-ocean T)/∂k_gm` | zstar+GM | 12 | +1.49e-7 | **6.50e-7** | 2.97e-5 (Fram Strait) | **rel 6.6%**, sign+descent ✓ |

- **mld_ck map peaks in the Weddell/Labrador/Southern-Ocean deep-convection regions** (physically
  correct — where TKE mixing controls MLD). ts_kgm peaks in the polar/eddy-active high latitudes.
- **Honest labelling:** these are the *fast/instantaneous* (~6-h N=12 window) sensitivities, NOT the
  equilibrium. The tiny GM signal (max|g|≈6.5e-10 vs MLD/c_k's 2e-2) reconfirms GM→T/S is beyond the
  adjoint window ⇒ §2 GM calib uses EKI; the adjoint↔EKI cross-check (rel 6.6%) ties the two tools.

## Two findings (now in `docs/PORTING_LESSONS.md`)

1. **The FIELD-leaf backward is heavier than A7's SCALAR ⇒ N_max=12, not 20.** The scary "63.27 GiB
   Failed to allocate" is a NON-FATAL XLA probe (it falls back; N=12 ran at peak 44.5 GB *with that
   message printed*). Trust the exit code, not the alloc line. The `.sbatch` descends N (12→6) as a
   safety net. Also disable CUDA-graph command buffers (`XLA_FLAGS=--xla_gpu_enable_command_buffer=`)
   + reuse one jitted forward (the first run OOMed on "38 alive graphs", not the working set).
2. **A single-h single-node FD spot-check is fragile** (rel 0.79 at the |g|-max node from a too-large
   step); an h-sweep over the top-3 |g| nodes (best plateau) fixes it — both targets then clean.

## Recommended next step

C1 done. Next cheapest pillars (no new dependency):
1. **§2 D1** perfect-model `k_gm` twin (reuse `calibrate.optimize` + the D1 recipe; grid-scan bowl
   first, then recover 800→1500 over a short window). Then **D2a** TKE→MLD calib (N=12 batched windows)
   and **D2b** GM→T/S via **EKI** (`fesom_jax.eki`).
2. **§0 B1 foundation** once the parallel-session all-on climate run lands (consume with `obs_compare`).
3. Re-run `stage_obs.sh` for dBM when IFREMER is back (still the only obs gap).

## Commits (code only — plan + this prompt deliberately uncommitted, NOT pushed)
- **`36911d0`** C1 machinery (field-leaf promotion + adjoint↔EKI + unit test).
- **`62c2ed9`** field-leaf OOM fix (disable command buffers + descending-N).
- **`2326184`** rigorous FD spot-check (h-sweep over top-|g| nodes) + N_max=12 + GPU-memory lesson.
- **`3be7759`** PORTING_LESSONS: FD spot-check needs an h-sweep (single-h is fragile).

# ============================================================================
# NEXT SESSION — §2 CALIBRATION (the "useful" pillar): D1 → D2a/D2b → D2c → D3
# ============================================================================

**Plan:** `docs/plans/20260614-fesom-jax-paper-experiments.md` (PART D). ⚠️ **DO NOT COMMIT the plan or
this prompt**; commit *code* only (modules/tests/scripts) with clear messages; **DO NOT push**. Mark plan
checkboxes `[x]` as you finish; append a `docs/PORTING_LESSONS.md` line per task; keep the suite green.

**State (all green, all reusable):** the full gradient + obs + optimizer seam is BUILT and validated by
§1 C1 — `calibrate.{optimize,grid_scan,build_params}`, `obs_compare.{to_obs,mld_density_threshold,misfit,
aggregate_windows}`, `eki.{eki_run,eki_update,eki_step}`, `scripts/write_namelist.py`. The scalar↔field
leaf seam differentiates cleanly; **field N_max=12** (TKE+MLD; the FIELD backward is heavier than A7's
scalar — scalar is cheaper, use it for scalar calibration). Adjoint↔EKI agree on `k_gm` (C1).

**Build order (cheapest first):**
1. **D1 — perfect-model `k_gm` twin** (`scripts/paper/core2_paper_calib_twin.py`+`.sbatch`; model on
   `core2_gm_grad_gate.py`). Inject truth = run with `k_gm=1500`; **grid-scan the misfit bowl FIRST**
   (forward-only) to confirm argmin sits at 1500; then recover `k_gm` 800→1500 via `calibrate.optimize`
   (Adam + cosine decay) over a short window. **`TWIN_RECOVER_OK`** (within ~2% of 1500, misfit ≪ initial).
   This is the adjoint-as-optimizer proof (short-window twin target IS adjoint-reachable — unlike the slow
   *obs* GM target, which is D2b/EKI).
2. **D2a — TKE→MLD/SST calib via short batched-window adjoint** (FAST). Window = N_max (12 from C1),
   **batched over start-dates/seasons** (target the dBM climatology, not one realization — `aggregate_windows`).
   Tune `{tke_c_k, tke_c_eps}` to MLD-vs-dBM (+ SST-vs-WOA) via `obs_compare.misfit`. ⚠️ **dBM still blocked**
   (IFREMER 503) → use a WOA-derived density-threshold MLD or a perfect-model MLD target as fallback; document.
3. **D2b — GM→T/S via EKI** (SLOW). `eki_run` on `k_gm`(+`redi_kmax`) vs upper-ocean T/S-vs-WOA over a
   warm-started **16–32-member, few-year** ensemble (the A4 budget); record GPU-h. NOT a short adjoint.
4. **D2c — held-out validation + recovered-VALUE reporting** (catch structural-bias compensation); falsifiable
   bars (held-out misfit reduced **≥ the C↔Fortran SST/SSS floor ~0.0049/0.0028 + the EN4 spread**). Fig 3.
5. **D3 — Fortran transfer** (`write_namelist.py` ready): patch calibrated scalars into `namelist.oce`
   (honor `Redi_Kmax=K_GM_max` sync); the operational Fortran run is Post-Completion (external).

**Guardrails (unchanged):** single-GPU for ALL adjoint experiments (sharded ragged-halo AD bug); GPU sbatch
= `-A ab0995_gpu -p gpu --gres=gpu:a100_80:1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MEM_FRACTION=0.95`,
and **`XLA_FLAGS=--xla_gpu_enable_command_buffer=`** + reuse one jitted forward (the C1 OOM lesson). Cheap
CPU jobs `-p compute --time=30:00`; large files on `/work`. IC per-oracle (`data/ic_core2`). Suite:
`sbatch scripts/runs/run_suite.sbatch` (the sharding-group timeout is pre-existing, not a regression).

**DO NOT ATTEMPT:** §0 B1 (needs the parallel-session all-on climate run); D3's actual Fortran run (external);
changing a locked decision. **Open question deferred from C1 (for the writeup, non-blocking):** keep Fig 2's
symmetric two-map layout, or restructure §1 around the `c_k`→MLD map with GM as just the adjoint↔EKI bridge?

**Recommended start:** D1 (twin) — cheapest, reuses the grad gate + `calibrate.optimize`.

# ============================================================================
# STATUS — autonomous overnight session 2026-06-14 (PART A infrastructure)
# ============================================================================

**TL;DR: ALL of Part A is DONE. The 6 pure-code tasks are committed + tested green (58 new CPU
tests, all `*_OK` tokens). A7 GPU adjoint-window sweep COMPLETE → `WINDOW_DERISK_OK`, measured
N_max=20 (~10 h) for the TKE→MLD adjoint (fast targets adjoint-reachable, slow GM→T/S ⇒ EKI). Full
regression suite GREEN (ocean 636 + ice 47; sharded timeout is pre-existing, proven not a
regression). A1 obs: WOA18 + EN4 + OSI-SAF staged; only dBM blocked on an IFREMER 503.** No locked
decision was touched. 3 code commits (`9ff999b`, `2a08e8e`, `6671dde`); plan + this prompt
uncommitted + not pushed, per instruction.

## What I built (all 6 high-autonomy tasks DONE, tests green)

| Task | Module + test | Token | Tests |
|---|---|---|---|
| **A3** | `fesom_jax/calibrate.py` + `tests/test_calibrate.py` | `CALIBRATE_SEAM_OK` ✅ | 10/10 |
| **A2** | `fesom_jax/obs_compare.py` + test (the keystone) | `OBS_OPERATOR_OK` ✅ | 11/11 |
| **A2b** | `fesom_jax/obs_ice.py` + test | `OBS_ICE_OK` ✅ | 11/11 |
| **A5** | `fesom_jax/tke_nn.py` + `params.py`/`tke.py` wiring + test | `TKE_NN_OK` ✅ | 9/9 |
| **A4** | `fesom_jax/eki.py` + test | `EKI_OK` ✅ | 8/8 |
| **A6** | `scripts/write_namelist.py` + `scripts/tests/test_write_namelist.py` | `FORTRAN_TRANSFER_OK` ✅ | 9/9 |

The standing **`params=None` / NN-off bit-identical invariant** holds: `tke_nn=None` adds no
pytree leaves (`len(leaves)==8` test still passes); `mixing_tke` with a zero-last-layer NN is
**bit-for-bit** identical to NN-off (asserted in `test_tke_nn.py`). The `params.py` + `tke.py`
edits are byte-identical on the existing path by construction.

## Suite result (the regression guard) — GREEN for these changes ✅

`run_suite` job 25581770: **ocean 636 passed** (all 49 new tests + 587 existing ocean tests — NO
regression), **ice 47 passed**. The **sharding group hit the job time limit** (cancelled, not
failed) — the **pre-existing** sharded-suite timeout (memory `sharded-suite-slow-phase8b`; `shard_map`
compile is slow). A standalone sharding-only confirm (job 25582933, 50-min cap) ALSO timed out in
compile — confirming it's pure slowness, not my change.

**Why the sharded timeout is NOT a regression (structural proof, no rerun needed):** the only changed
runtime code is the `mixing_tke` branch `if params.tke_nn is not None:`. Every existing test (dense AND
sharded) uses `tke_nn=None` ⇒ the branch is skipped ⇒ `mixing_tke` is byte-identical to before. That
byte-identity is *already proven* by the ocean group's **`test_tke_replay.py`** (asserts byte-exact TKE
replay values — **passed**). The sharded path calls the *same* `mixing_tke`, so it cannot differ.
(If you want a concrete sharded green anyway, it's just slow: `sbatch` a single-file
`test_step_sharded.py` run with a ≥60-min cap.)

## A7 adjoint-window sweep — DONE ✅ `WINDOW_DERISK_OK`, **N_max = 20 (~0.42 d / ~10 h)**

`scripts/core2_adjoint_window_sweep.{py,sbatch}` + `scripts/paper/fig_window_snr.py` → `scripts/fig_window_snr.png`.
Measured `d(mean MLD)/d(tke_c_k)` (zstar+TKE, GM/ice off) on a full 80.8 GB A100 (final run, job 25583047):

| N | window | peak GB | d(MLD)/d(c_k) | FD↔AD plateau | verdict |
|---|---|---|---|---|---|
| 4 | 0.08 d | **37** | +0.090 | 2.3e-5 | clean ✓ |
| 20 | 0.42 d | **52** | +3.55 | 6.2e-7 | clean ✓ ← **N_max** |
| 50 | 1.04 d | single 70 GiB alloc | — | — | **OOM** (won't fit 80 GB + working set) |
| 100, 200 | — | — | — | — | OOM (executable too large) |

**Findings:** the gradient is REAL and clean (FD==AD, plateau 2e-5→6e-7) at the reachable windows;
peak ≈ **33 + 0.95·N GB**. N=50's backward needs a single **70 GiB** tensor that OOMs even on the full
80 GB card (with the working set) — so **N_max = 20**, NOT the ~50 I extrapolated from the linear fit
(corrected by the `mem_fraction=0.95` rerun — measure the ceiling, don't extrapolate it).
**Decision:** fast targets (MLD/SST) are adjoint-reachable at **~10-hour batched windows** (feeds D2a/E2
window sizing); slow GM→T/S (multi-year) ⇒ **EKI** (`fesom_jax.eki`). Reconciles the inherited
37.8 GB-at-N=20 GM figure (TKE+zstar+MLD is heavier: 52 GB at N=20). Two infra fixes en route: the
40 GB-node OOM (`--gres=gpu:a100_80:1`) and the prealloc/mem-fraction interaction (in the committed sbatch).

## A1 obs staging — `scripts/tools/stage_obs.sh` + `docs/OBS_DATASETS.md`

Login node HAS internet (the conda `CURL_CA_BUNDLE` points at a missing cert → set it to
`/etc/ssl/certs/ca-bundle.crt`; baked into `stage_obs.sh`). Staged to `/work/ab0995/a270088/port_jax/obs`:
- **WOA18** 1° annual T/S — downloaded (185 MB each), sanity-loaded (°C, 102 depths, 180×360).
- **EN4** thetao/so + **OSI-SAF** siconc — **symlinked** from Levante `…/a270301/cmpitool/obs`
  (seasonal; ⚠️ EN4 is KELVIN, OSI-SAF is %). No re-download needed.
- **de Boyer Montégut MLD** (the dBM target) — ⚠️ **BLOCKED**: the IFREMER cerweb server returns
  503 (maintenance). Re-run `stage_obs.sh` later; alt = SEANOE doi:10.17882/91774. **The machinery
  does not need it** (operators unit-tested on synthetic fixtures).
- Native polar-stereo NSIDC/OSI-450 not located as raw files this pass (the cmpitool regular-grid
  OSI-SAF covers §0 via `obs_compare.to_obs_surface`) — TODO noted in OBS_DATASETS.md.

## Not attempted (per the prompt's DO-NOT list)

§0 B1 (needs the parallel-session all-on climate run), D3 Fortran run (external), anything changing
a locked decision. No locked decision seemed wrong.

## Recommended next step

**Part A is DONE** (all 6 modules + tests green, A7 measured: N_max=20, obs staged). Start the pillars:
1. **§1 C1 sensitivity maps** — the cheapest next pillar, no new dependency: reuse the existing grad
   gates (`d/d(k_gm)`, TKE `c_k`) + promote `k_gm`/`c_k` to a `[nod2D]` **field leaf** (the seam already
   differentiates array leaves — see `calibrate.build_params`) → one backward pass → `∂(MLD-bias)/∂(c_k
   field)`. Use the **N_max=20 (~10-h) window** A7 just measured. Label HONESTLY as fast/instantaneous.
2. **§0 B1 foundation** — once the parallel-session all-on (zstar+TKE+mEVP) climate run lands; consume it
   with the new `obs_compare`/`obs_ice` operators (T/S-vs-WOA, MLD-vs-dBM, sea-ice-vs-OSI-SAF).
3. **§2 D2a** TKE→MLD calibration uses the N_max=20 batched-window adjoint; **D2b** GM→T/S uses EKI.
4. Re-run `stage_obs.sh` for dBM when IFREMER is back up (it returned 503 all night).

## Commits (code only — plan + this prompt deliberately uncommitted, NOT pushed)

- **`9ff999b`** `paper: PART A shared infra …` — all 21 Part-A code/test/script/doc files + the
  params/tke wiring + the first 6 PORTING_LESSONS entries.
- **`2a08e8e`** `paper/A7: use full 80GB A100 (mem_fraction=0.95) …` — sbatch GPU fixes + A7 lesson.
- **`6671dde`** `paper/A7: correct the measured adjoint window to N_max=20 (not ~50)` — the rerun
  showed N=50 OOMs even on a full card; lesson + numbers corrected.

Working tree otherwise clean; `docs/plans/20260614-…md` + `docs/NEXT_SESSION_PROMPT.md` left
uncommitted (per instruction); sweep `.jsonl`/`.png` gitignored.

# ============================================================================

# Next session — AUTONOMOUS overnight: build PART A of the paper-experiments plan

**MODE: MAXIMUM AUTONOMY. The user (koldunovn) is ASLEEP and unavailable.** Do as much as possible
**without human input**. Do NOT ask a question and wait — make a reasonable, *documented* decision and keep
going. Optimize for durable, reviewable progress by morning. If you hit a hard blocker on one task, skip it,
note why, and move to the next autonomous task.

---

## The plan

`docs/plans/20260614-fesom-jax-paper-experiments.md` — the experiments for a **JAMES "first capabilities"
paper** on the differentiable FESOM2→JAX model. Review-hardened (a plan-review pass closed 6 MAJOR
methodology issues — read its **Revision Log** + the **"Two tools" methodology spine** in the Overview
before coding). One differentiable global model (zstar+TKE+mEVP); three capability pillars (sensitivity,
calibration, hybrid-ML), each *perfect-model proof → obs application*; obs targets are OMIP-style and
obs-based (MLD vs de Boyer Montégut, T/S vs WOA/EN4, sea-ice vs NSIDC/OSI-SAF — NOT reanalyses).

⚠️ **DO NOT COMMIT THE PLAN FILE** (`...20260614-fesom-jax-paper-experiments.md`) — user's explicit
instruction; it's a working draft. **DO NOT COMMIT this prompt either.** You **MAY** commit *code* (new
modules + tests + scripts) locally with clear messages for durability — but **NOT** the plan, **NOT** this
prompt, and **DO NOT push** to any remote. (Use selective `git add <files>`, never `git add -A`.)

---

## Your job tonight: execute PART A (shared infra) — it has NO parallel-session dependency

Part A is mostly **pure-Python machinery + CPU pytest** — ideal for autonomous work. Build each module,
write its unit test, run it green, assert the `params=None`/cfg-off **bit-identical invariant** where
relevant, keep the suite green, and append a one-line lesson to `docs/PORTING_LESSONS.md` (project
convention). Mark the plan's checkboxes `[x]` as you finish each item.

**CRUCIAL — do not block on obs downloads:** build + unit-test the operators against **synthetic fixtures**
(a known analytic field regridded by `scipy`/`xarray` as the reference, ≤1e-10) — the real obs data (A1) is
NOT needed for the machinery or its tests.

### Do these FIRST — high autonomy (pure code + CPU pytest, no GPU, no obs data):

1. **A3 `fesom_jax/calibrate.py`** (+ `tests/test_calibrate.py`) — easiest, fully specified by the Phase-7a
   design in `docs/plans/20260607-fesom-jax-paramtune.md` §1 (`optimize`/`grid_scan`/`build_params`).
   Token `CALIBRATE_SEAM_OK`.
2. **A2 `fesom_jax/obs_compare.py`** (+ test) — the keystone. Host-precompute the **horizontal** node→cell
   map ONLY; compute the **vertical interp from the LIVE zstar geometry inside `to_obs`** (must be
   through-differentiated — verify a nonzero FD gradient w.r.t. layer thickness). AD-safe MLD
   (0.03 kg/m³, linear crossing, NOT argmax). Empty-cell 0/0 → finite via the `fesom_jax/ops.py`
   sentinel-mask precedent. First-class temporal `aggregate_windows`. Token `OBS_OPERATOR_OK`.
3. **A2b `fesom_jax/obs_ice.py`** (+ test) — 2D node→polar-stereo concentration map, ice mask + pole-hole;
   forward-only (need not be differentiable). Token `OBS_ICE_OK`.
4. **A5 `fesom_jax/tke_nn.py`** (+ test; modify `params.py` optional `tke_nn` leaf + `tke.py` consume site) —
   pure-JAX small MLP, **bounded multiplier ⇒ positive-definite diffusivities**, **zero last layer ⇒
   multiplier=1 ⇒ default TKE bit-identical** (assert it). FD↔AD through weights clean. Token `TKE_NN_OK`.
5. **A4 `fesom_jax/eki.py`** (+ test) — `eki_step`/`eki_run`, recover a known scalar from a noisy analytic
   forward (no adjoint). Token `EKI_OK`. (The real-model EKI budget is in the plan; the unit test is
   analytic.)
6. **A6 `scripts/write_namelist.py`** (+ test) — patch `K_GM_max`/`Redi_Kmax`/TKE constants into a
   `namelist.oce` template, round-trip safe. Token `FORTRAN_TRANSFER_OK`.

### Then — medium autonomy (GPU queue or internet):

7. **A1 obs staging** (`scripts/tools/stage_obs.sh`, `docs/OBS_DATASETS.md`) — downloads need login-node internet.
   Attempt WOA18/23, EN4.2.x, de Boyer Montégut 2023; locate NSIDC/OSI-SAF on Levante. **If no internet /
   it fails, document and SKIP** — the machinery above doesn't need it.
8. **A7 adjoint-window de-risking** (`scripts/archive/core2_adjoint_window_sweep.py` + `.sbatch`) — the single most
   scientifically valuable autonomous result: a single-GPU N-sweep (N=4,20,50,100,200,…) measuring CORE2
   backward **peak memory** + **gradient SNR/FD-agreement** of `d(MLD-misfit)/d(c_k)`, to find **N_max** and
   the adjoint↔EKI boundary (this de-risks the WHOLE plan and reconciles the inherited 37.8 GB-at-N=20
   figure). Model it on `scripts/archive/core2_tke_grad_gate.py` + `scripts/archive/core2_gm_grad_gate.py`. **Submit the
   sbatch and poll**; if the GPU queue is slow, keep building CPU machinery. Token `WINDOW_DERISK_OK` +
   the gradient-SNR-vs-N supplementary figure.

### DO NOT ATTEMPT (blocked on human / parallel session):
- **§0 B1 foundation** — needs the parallel-session all-on (zstar+TKE+mEVP) climate run.
- **D3 Fortran run** — external Fortran model/build.
- Anything that **changes a locked decision** (Overview §"Locked decisions"). If a locked decision seems
  wrong, write it up in a STATUS note for the user — do NOT act on it.

---

## Guardrails (non-negotiable)

- **Single-GPU for ALL adjoint experiments** (the sharded gradient has a pre-existing ragged-halo AD bug —
  forward-only safe). See `docs/JAX_RAGGED_A2A_BUG.md`.
- **AD masked-NaN rule:** masked lanes must compute a **finite** value (a forward `where` does NOT stop a
  backward `0·inf`). Applies to `obs_compare`, the MLD diagnostic, and `tke_nn`.
- **Keep the suite green** after every task: `sbatch scripts/runs/run_suite.sbatch` (CPU; runs in two chunks —
  the full set in one process exceeds login-node RAM). The `params=None`/cfg-off **bit-identical** invariant
  is the regression guard.
- **House style:** mirror the existing `fesom_jax/tests/test_*.py` and `docs/plans/*.md`; append a
  `PORTING_LESSONS.md` line per task.

## Env / provenance

- Python: `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python` (run pytest via this; float64,
  `jax_enable_x64`).
- GPU jobs: SLURM `-A ab0995_gpu -p gpu --gres=gpu:1` (A100-80GB). Cheap CPU jobs: `-p compute --time=30:00`.
  Large files on `/work`, NOT `/home`.
- IC-partition provenance is per-oracle (16r→`data/ic_core2_dist16`, 864r→`data/ic_core2_dist864`).

## Report before you stop

- Mark every completed plan checkbox `[x]`; add ➕ for new tasks, ⚠️ for blockers.
- Append a **STATUS block to the top of THIS file** (`docs/NEXT_SESSION_PROMPT.md`): what you built, which
  tokens/tests passed, what's committed (code only) vs left uncommitted, what's blocked, and the single
  recommended next step. Leave the working tree reviewable (`git status` clean of stray junk; plan + this
  prompt uncommitted).
- If you committed code: list the commit hashes + one-line messages in the STATUS block.

**Recommended start:** A3 (calibrate, ~30 min) → A2 (obs operator, the keystone) → A2b → A5 → A4 → A6,
then submit A7. That's a full night of durable, review-ready infrastructure with zero human input.
