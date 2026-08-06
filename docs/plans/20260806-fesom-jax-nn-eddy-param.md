# NN-GM: learning FESOM's eddy parameterization from NG5

**Created:** 2026-08-06 (supersedes the same-day pre-brainstorm draft; decisions
user-validated in the brainstorm dialogue; revised same day per plan-review — see Revision
Log). **Status:** PLANNED.
**Venue target:** JAMES-class paper (paper 3).
**Authoritative copy:** THIS file in the main checkout (`~/port_jax/docs/plans/`); the
worktrees hold code only — progress markers are updated here, nowhere else.

## Overview

Replace GM/Redi mesoscale eddy transport in FESOM with a neural network, trained end-to-end
through the differentiable FESOM2-JAX model, taught by a coarse-grained eddy-resolving NG5
Fortran run, polished on slow targets by EKI, and evaluated as **general hydrographic bias
reduction** against the Scholz et al. (2019) FESOM2 assessment biases.

Novelty (web-verified 2026-08-06): (a) first NN parameterization in FESOM / any
unstructured-mesh ocean model; (b) first **online (a-posteriori) trained** eddy
parameterization in a realistic global ocean–sea-ice model — prior art is offline-trained
(Perezhogin et al. 2025, arXiv:2505.08900, MOM6; Zhang et al. 2023 JAMES); online training
exists only in idealized settings (Frezat et al. 2022; arXiv:2412.06393). Only a
differentiable global model can close this — the port's founding purpose
(`docs/plans/20260605-fesom-jax-port.md`).

Lessons imported from the idealized/offline literature: small local MLPs suffice; physics
constraints (guaranteed APE sink) are essential; local nondimensionalization drives
generalization; offline skill does not predict online skill; global deployments needed
taming near coasts/equator.

## Context (from discovery)

- **Meshes:** CORE2 126,858 nodes / 48 lev / dt=1800 (student); NG5 7,402,886 / 70 lev /
  5 km (teacher; ⚠️ NOT yet exported for the Python side — `docs/FORCING_MESHES.md` "needs
  export"); farc 638k Arctic-refined (zero-shot test); FORCA20 2.13M (scaffolding only).
- **GM/Redi:** `fesom_jax/gm.py` + `gm_redi.py`, ON in `configs/core2_full.yaml`
  (`gm: {}`), κ defaults 1000 m²/s × resolution-scaled formula; `k_gm`/`redi_kmax` are
  differentiable `Params` leaves. 63-yr v3 hindcast = default-GM baseline, already run.
- **NN seam precedent:** `fesom_jax/tke_nn.py` (bounded-multiplier MLP, `zero_last` ⇒
  bit-exact identity; ⚠️ it is a `Params` leaf ONLY — it has NO config/run_from_config
  wiring, and `calibrate.py:124` hardcodes its pytree pass-through); `calibrate.py` (optax;
  scalar/field/NN leaves), `eki.py`, `remat_blocks`/`remat_segments`, batched-window
  scripts under `scripts/paper/`, perfect-model twin precedent
  `fesom_jax/tests/test_nn_twin.py` (paper §3 E1).
- **Missing pieces:** no FESOM→FESOM coarse-graining/remap exists; `run_from_config` takes
  no trained-`Params` input (deployment path must be built, Task 7).
- **Fortran:** FESOM2 clone at `/home/a/a270088/fesom27/fesom2` (built `fesom.x`, CORE2 R2
  provenance). FESOM2 output has **no snapshot mode and no same-field multi-frequency** —
  patch required. ⚠️ FESOM2 `u,v` are ELEMENTAL (CORE2 elem2D=244,659 vs nod2D=126,858,
  ×1.93 size); node collocation happens in our pipeline, not in Fortran. FORCA20 run
  (`/work/ab0995/a270088/fesom2_forca20`, 3 yr of MEAN output) = mechanics-pilot data only.
- **Diagnostics to port:** pyfesom2 `xmoc` (w-based MOC, Sidorenko et al. 2020 GMD 13,3337)
  + section transports (Drake/ACC). Analysis-only ⇒ scripts, zero library surface (the
  long-window plan's precedent).
- **Adjoint/limiter facts:** window-mean targets clean ≤ ~48 steps, blow-up by ~96 (longwindow
  campaign) — the horizon for THIS loss is unknown until measured (Task 2). FCT limiter
  gradients are subgradients — finite and FD-consistent where smooth
  (`docs/LIMITER_GRADIENTS.md`), with a one-line `stop_gradient(ae)` escape hatch if
  training signal proves noisy.
- **House rules:** JRA55-do always; never year 2020; no heavy pytest/runs on login node
  (suite via `scripts/runs/run_suite.sbatch`); NEVER-DELETE in unattended chains; never
  blanket `git add <dir>`; env python
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`; GPU `-A ab0995_gpu`.

## Development Approach

- **Testing approach: Regular + hard gates** (house style): implement → pytest units in
  `fesom_jax/tests/` + oracle/bit-identity/conservation gates; suite on a compute node.
- All code work in **dedicated worktrees** (Task 0); running chains and current dev untouched.
- Every task ends with its tests/gates green before the next starts; library-touching tasks
  end with the **full suite green**; checkboxes updated immediately; scope changes go to the
  Revision Log.
- Backward-compatibility invariants: `gm_nn=None` ⇒ JAX model bit-identical (suite green);
  snapshot-flag-off ⇒ Fortran output byte-identical to the unpatched binary.

## Testing Strategy

- Unit tests per task (success + edge cases) in `fesom_jax/tests/`; script-level tests live
  next to their scripts and are wired into pytest via a path fixture.
- Gates are inline per task: conservation ~1e-12, bit-identity when disabled, pyfesom2
  parity, stability gates for runs, and the three campaign gates HORIZON_OK / M3 / M4 /
  anti-E2.
- "e2e" = short sbatch smoke runs of the real models.

## Progress Tracking

- `[x]` immediately when done; ➕ for discovered tasks; ⚠️ for blockers; keep this file in
  sync; move to `docs/plans/completed/` at the end.

## What Goes Where

- **Implementation Steps:** everything achievable in the worktrees (code, patches, runs we
  launch, analysis, figures).
- **Post-Completion:** paper writing, quota routines, possible upstreaming of the Fortran
  patch, follow-up legs.

## Implementation Steps

### Task 0: Workspace — dedicated worktrees

**Files:**
- Create: `/home/a/a270088/nn_eddy/` (worktree container; data root
  `/work/ab0995/a270088/nn_eddy/`)
- Create: worktree `/home/a/a270088/nn_eddy/port_jax` (branch `nn-eddy` off `main`)
- Create: worktree `/home/a/a270088/nn_eddy/fesom2` (branch `nn-eddy-output` off the
  `/home/a/a270088/fesom27/fesom2` checkout's current commit)

- [x] create both worktrees + data root; record layout in a README there (incl. "plan file
      authoritative in main checkout" note) — done 2026-08-06: `~/nn_eddy/{port_jax,fesom2}`
      + `/work/ab0995/a270088/nn_eddy/` + README/CLAUDE.md/NEXT_SESSION_PROMPT.md
- [x] verify FESOM2 builds unchanged in its worktree (same toolchain as the R2 binary)
      — done 2026-08-06: `configure.sh` (env shell.intel, same mpif90 path as the R2
      CMakeCache), 0 errors, `bin/fesom.x` produced; size differs from production only
      by embedded build metadata (2,227,832 vs 2,227,896 B). Version provenance
      verified — see Revision Log (d)
- [ ] verify port_jax suite green in its worktree (sbatch, compute node) — job 26743651
      running 2026-08-06
- [x] gate: `git worktree list` clean in both repos; running chains untouched — verified
      2026-08-06 (fesom27: main + nn-eddy-output; port_jax: 4 worktrees incl. nn-eddy)

### Task 1: Fortran snapshot-output patch (minimal)

**Files:**
- Modify: `io_meandata.F90` (+ namelist plumbing) in the fesom2 worktree

- [x] add a per-entry snapshot flag to the output list: write instantaneous values at the
      listed interval, bypassing mean accumulation; no multi-frequency machinery — done
      2026-08-06 (commit cda32816): optional `&nml_snap` group defines ADDITIONAL
      `<id>_snap` streams (io_list format; absent ⇒ off); reuses the io_list definition
      table so any mean-able field is snapshot-able; mean+snapshot of one field coexist.
      Caveats: vec_autorotate does not apply to `*_snap` (native rotated frame, warned;
      R2/CORE2 production runs vec_autorotate=.true. — pipeline unrotates); 0D refused
- [x] wire namelist control + document usage in the worktree README — `SNAPSHOT_OUTPUT.md`
- [x] test: 1-day CORE2-mesh Fortran smoke run (compute node) — 12/12 PASS (job 26745349,
      dist_128, cold start 1958): `sst_snap[0] == sst[47]` BITWISE where sst is an
      every-step mean (a 1-step mean ≡ instantaneous state, subsumes the constant-field
      case); evolving fields snap≠mean (temp max 2.6, ssh max 1.0); r8 path; elemental
      (u_snap) + ice shapes; fill patterns identical; time-axis metadata identical
- [x] **gate (bit-identity-when-disabled): PASS** — via the rebuilt-unpatched control
      (Revision Log (d) fallback): unpatched-4ccdd35a run D vs patched-flag-off run B =
      **0/33 files differ** (all outputs + oce/ice restarts + clock). ⚠️ vs the Feb-22
      production R2 binary (run A): 30/33 differ FROM STEP 1 (sst max 4.5e-2 at step 1,
      polar-clustered) — the R2 binary's source is NOT today's main and NOT recoverable
      from git (the llm-fix-3 ice_fct.F90 mangling); see Revision Log (h)
- [x] gate: smoke-run I/O overhead <~5% step time at the test cadence — **PASS: +0.12%**
      total (7.8826→7.8922 s; output component 0.48→0.77 s for 6 extra snapshot streams
      incl. two 3D nodal + one 3D elemental, daily cadence)

### Task 2: Gradient-horizon scan — BEFORE the teacher launch (GATE HORIZON_OK)

Needs **no teacher data**: perfect-model twin on CORE2 with existing machinery — generate a
"truth" trajectory with the default model, use the Stage-1 loss form (per-level
variance-weighted T,S+SSH at the candidate target marks) and measure gradient health
(finite, FD-consistent, informative) vs window length 1→10 days.

**Files:**
- Create: `scripts/nn_eddy/horizon_scan.py` (+ sbatch)

- [x] implement the scan on the exact Stage-1 loss form; include the
      `stop_gradient(ae)` limiter escape hatch as a scan variant if signal is noisy —
      done 2026-08-06 (commit 156a0ae, `scripts/nn_eddy/horizon_scan.py`): perfect-model
      GM-κ-multiplier twin (truth m_true=1.25, student m=1), per-level inverse-variance
      T,S + SSH loss, endpoint-mark primary probe (`--marks daily` variant), frozen-ice
      adjoint (= the Stage-1 training config), `loss(m_true)==0` in-process harness
      check; the ae-hatch is documented-deferred (not config-wired; wire only if noisy)
- [ ] run 1→10-day windows on A100; record usable horizon H (longwindow prior: expect
      1–5 days; window-mean variant as fallback axis) — job 26746271 running 2026-08-06
      (winterNH seed, N=1,2,3,5,7,10 d, endpoint marks)
- [x] tests: scan harness determinism on CPU; loss-form units — 7/7
      (`scripts/tests/test_horizon_scan.py`; in-model check `loss_at_truth` per process)
- [ ] **GATE HORIZON_OK:** usable H established. **Protocol rule:** teacher 3D snapshot
      cadence = min(H, 5) days (this SETS Task 3's namelist). **Pivot rule:** if H < 1 day
      → switch Stage-1 loss to window-MEAN matching (longwindow-validated) and set cadence
      for it; if that also fails → surface to user before any NG5 launch

### Task 3: NG5 teacher campaign — setup and launch (LONG POLE, launch right after Task 2)

**Files:**
- Create: run dir `/work/ab0995/a270088/nn_eddy/ng5_teacher/` (namelists, sbatch chain,
  monitor script)

- [x] ⚠️ BLOCKING prerequisite: obtain the AWI NG5 namelist set + `dist_*` partitions
      (user provides the source setup) — done 2026-08-06: user-provided source setup
      `/home/a/a270088/DestinE/fesom2.6/work_NG5_1950_Rohit` (⚠️ fesom2.6/DestinE, ERA5
      forcing, yearnew=1945, mEVP α=β=500, icepack namelist present — full coefficient
      audit + JRA55-do forcing swap required, see Revision Log (d)); partitions already
      at `/work/ab0995/a270088/meshes/NG5/dist_{1024,1920,2048,2560,3840,4096,7680,8192}`
- [ ] adapt namelists: cvmix-TKE + mEVP + zstar, GM/Redi OFF, JRA55-do, PHC IC, **cold
      start 1958-01-01** (same transient as the student hindcast). LOCKED (user
      2026-08-06): **mEVP α=β=500** (the Rohit/template NG5 value, resolution-adjusted
      from CORE2's 250) and **icepack OFF** (our fesom27 binary has USE_ICEPACK=OFF —
      structurally off; the Rohit dir's namelist.icepack is inert). **dt ramp DECIDED
      (user 2026-08-06):** cold start at dt=120 s (step_per_day=720, the sweep-proven
      cold-start value) for the first 1–2 months, **monthly restarts from step one**,
      then stop+restart at a month boundary onto **dt=240 s (step_per_day=360)** for the
      rest of the campaign (Fortran cannot change dt in-run — FORCA20 precedent).
      ✅ CONFIRMED by user 2026-08-06: 120/240 are dt-SECONDS ⇒ step_per_day 720→360
      (both stages more conservative than DestinE's dt=360 s / spd=240)
      **STAGED 2026-08-06** in `/work/ab0995/a270088/nn_eddy/ng5_teacher/` (not launched):
      config/oce/ice/tra/dyn/cvmix/forcing built FROM CORE2-R2 (JRA55-do + student-matched
      physics on our binary) with only resolution-dependent settings taken from the NG5
      side, per the rule "numerics may differ with resolution, physics may not" —
      full three-way audit in `fesom2/NG5_TEACHER_SETUP.md` (commit 8a1e82ef).
      ⚠️ TRAP FOUND AND AVOIDED: DestinE-NG5 runs ice albedos albi/albim=0.65/0.55 vs
      the FESOM2-default and student 0.70/0.68 — an ERA5-co-tuned physics constant that
      would have imposed a systematic polar surface-flux bias the NN would absorb into
      the closure. Teacher keeps 0.70/0.68. Inert differences recorded there too
      (visc_gamma2 0.0285 vs 0.285 — only read for opt_visc 5/8, we run 7; dead
      /work/ollie + /albedo IDEMIX paths — inert under cvmix_TKE).
      **User directive 2026-08-06: sea-ice settings follow FESOM/JAX defaults** (the
      DestinE ice values are ERA5-co-tuned). Full ice audit: DestinE deviates from the
      FESOM template in exactly 4 keys and specifies nothing else ⇒ no other hidden
      ERA5 tuning. Resolved: albi/albim → 0.70/0.68 (student); ice_diff → 10.0 (JAX
      student value; FESOM template ships 0.0); whichEVP=1; alpha/beta_evp=500 kept
      (resolution-scaled numerics, user-locked)
- [ ] output protocol (cadence per GATE HORIZON_OK): 3D snapshots T,S,w (nodal) + u,v
      (native ELEMENTAL); **2D snapshots at the same cadence: SSH, a_ice, m_ice, uice,
      vice** (window-init + η loss targets — means are NOT sufficient); daily 2D means
      (SSH, ice, surface fluxes); monthly 3D means T,S,u,v,w; monthly restarts
- [ ] replay runbook (the dense-target mechanism): documented procedure to rerun any month
      from its restart at daily cadence (~0.4–0.5 TB/replayed month); test it on the
      shakedown month
- [ ] self-resubmitting chain — ⚠️ NEVER-DELETE rule: no rm/overwrite branches anywhere;
      **test the chain: kill + resubmit resumes correctly and touches no existing output**
- [ ] monitor script + cadence (clock/quota/output-volume checks, read-only) — runs for the
      life of the campaign
- [ ] shakedown: 1 month — volumes as budgeted, snapshots readable by our loaders,
      **measure SYPD and node-h/SY** — ⚠️ shakedown month runs at the ramp dt (120 s);
      the budget gate needs the POST-switch rate too (measure a few dt=360 s days
      before sign-off; expected ~3.3 SYPD @ 8192 from the sweep per-step cost)
- [ ] **gate (compute budget):** present measured cost of 10 SY (+ Task-11 rungs) vs the
      ab0995 allocation to the user for sign-off BEFORE the full chain proceeds
- [ ] gate: year 1958 complete; storage on budget (~10.5 TB/decade at 5-daily — see
      Technical Details); record achieved SYPD here

### Task 4: Mesh diagnostics — AMOC + section transports (scripts-only)

**Files:**
- Create: `scripts/nn_eddy/mesh_diag.py`; Create: `scripts/nn_eddy/test_mesh_diag.py`

- [ ] w-based MOC per pyfesom2 `xmoc` (node w → element means × `elem_area` → latitude
      bins → cumsum, basin masks incl. Atlantic; Sidorenko et al. 2020)
- [ ] section volume transport (Drake/ACC) per the pyfesom2 notebook
- [ ] works on teacher (Fortran netCDF), JAX-student, and Fortran-baseline outputs
- [ ] tests: synthetic w with analytically known MOC; bin/basin edge cases
- [ ] gate: parity vs pyfesom2 on one real FORCA20/CORE2 monthly file (plot + numeric diff)

### Task 5: `coarsen` I — mesh-Laplacian scale filter

**Files:**
- Create: `fesom_jax/coarsen.py`; Create: `fesom_jax/tests/test_coarsen.py`

- [ ] iterated mesh-Laplacian filter on a FESOM mesh (fixed diffusion steps; target scale =
      2× **local** CORE2 resolution); area-weighted, mask-aware (coast/bottom)
- [ ] tests: constant preserved exactly; global area-weighted mean conserved ~1e-12; scale
      response on synthetic spectra (small scales damped, large untouched)
- [ ] gate: filtered FORCA20-mean field visually + spectrally sane (mechanics pilot)
- [ ] full suite green (library touched)

### Task 6: `coarsen` II — NG5→CORE2 aggregation + products pipeline

**Files:**
- Modify: `fesom_jax/coarsen.py`; Create: `scripts/nn_eddy/coarsen_stream.py`
- Create: `fesom_jax/tests/test_coarsen_remap.py`

- [ ] ➕ export the NG5 mesh for the Python side (coords, areas, nlvls, resolution — the
      `docs/NEW_MESH.md` flow; currently "needs export")
- [ ] element→node collocation for u,v (area-weighted mean — needed because Fortran saves
      native elemental velocities)
- [ ] conservative Voronoi-area-weighted aggregation teacher-nodes → CORE2 nodes
      (host-precomputed sparse map, `build_h_map` pattern); conservative vertical remap
      70→48 levels, ⚠️ bottom masks (sub-bottom-zero)
- [ ] streaming postprocess: per snapshot emit (a) filtered state on CORE2 mesh and
      (b) diagnosed eddy fluxes filter(u·τ) − filter(u)·filter(τ), τ∈{T,S}
- [ ] tests: conservation through the full chain ~1e-12; bottom-mask cases; idempotence on
      an already-coarse field; collocation units
- [ ] gate (mechanics, on FORCA20 means): remap conservation + spectra. **Gate (flux
      product): on the Task-3 shakedown month's REAL NG5 snapshots** — hotspot maps
      physically sane (ACC/Agulhas/Gulf Stream); pipeline keeps pace with the run rate
- [ ] full suite green

### Task 7: `gm_nn` — network, seam, AND deployment path

**Files:**
- Create: `fesom_jax/gm_nn.py`; Modify: `fesom_jax/params.py` (leaf +
  `register_dataclass` data_fields), `fesom_jax/calibrate.py` (generalize the hardcoded
  `tke_nn` pytree pass-through), `fesom_jax/gm.py`, `fesom_jax/run_config.py` +
  `run_from_config` (trained-params loading); Create: `fesom_jax/tests/test_gm_nn.py`

- [ ] per-column MLP (~16,16 tanh, ~1–2k weights) on locally nondimensionalized features
      (arcsinh): |∇h b|, N², isopycnal slope, f, resolution/Rossby-radius ratio, depth
      fraction, shear/strain invariants
- [ ] outputs per level: m_GM, m_Redi ∈ [1/M, M], M=3 (exp·tanh); κ = formula base × m
      (APE sink by construction); `Params` leaf (NOT in hashable `GMConfig`)
- [ ] **deployment path:** weights (de)serialization + a `params_file`-style config key so
      `run_from_config` runs with trained params, surviving archival-restart leg chaining
      (Task 11 depends on this)
- [ ] tests: `zero_last` ⇒ m≡1 ⇒ full suite bit-identical; bounds under extreme inputs;
      feature scaling units; gradient flows to every weight (masked-NaN rule); params
      round-trip through file + across a restart leg
- [ ] gate: 10-day CORE2 A100 run with random-perturbed NN stable; d(loss)/d(weights)
      finite and nonzero through the assembled model; full suite green

### Task 8: Stage 0 — offline pretrain + a-priori skill (GATE M3)

**Files:**
- Create: `scripts/nn_eddy/offline_fit.py`, `scripts/nn_eddy/apriori_eval.py` (+ sbatch)

- [ ] project diagnosed fluxes onto GM/Redi form → target κ̂(node,level,time); fallbacks:
      flux-divergence fit, one-step tendency matching
- [ ] fit `gm_nn` offline (minibatch over snapshots/regions); regularize toward m=1
- [ ] a-priori evaluation: R² by region/depth/season vs constant-κ GM and vs the formula κ
- [ ] first **skill-vs-teacher-years curve** (also the NG5 stopping rule)
- [ ] tests: projection algebra on synthetic GM-generated fluxes recovers κ exactly
- [ ] **GATE M3:** offline κ-NN beats constant-κ GM a-priori; else pivot to Arm B (ψ_NN —
      scoped in Technical Details, ~+2–3 wk) before proceeding

### Task 9: Stage 1 — online training through the adjoint (GATE M4)

**Files:**
- Create: `scripts/nn_eddy/online_train.py` (+ sbatch)

- [ ] **perfect-model twin control first** (test_nn_twin precedent, run through the FULL
      window driver incl. the coarsen chain on a JAX-generated "teacher"): a known
      perturbed-NN is recovered ⇒ driver + remap attribution is clean before real-teacher
      training
- [ ] window driver: init student from filtered NG5 SNAPSHOT state (T,S,u,v,SSH,ice; TKE
      spun in-window), 6–12 h nudge-in discarded from loss
- [ ] loss: per-level variance-weighted T,S (+SSH) at the cadence marks from HORIZON_OK;
      velocities excluded; training years 1959–1965 ⇒ ~511 candidate windows (1966–67
      held out); replay months supply denser targets if the curriculum needs them
- [ ] training: adam via `calibrate.optimize`, `remat_blocks`, one window per A100
      data-parallel; curriculum short→long windows within H; E2 trust-region reg around
      Stage-0 weights; `stop_gradient(ae)` limiter escape hatch if signal noisy
- [ ] tests: window-driver determinism (CPU); nudge-in mask; loss weights
- [ ] **GATE M4:** 1-yr free run from trained weights **stable** and beats default GM vs
      the filtered teacher on ≥3 of 5 metrics (T bias, S bias, SSH var, MLD, AMOC)

### Task 10: Baseline-ladder rung — adjoint-calibrated static κ(x,z)

**Files:**
- Create: `scripts/nn_eddy/static_kappa_fit.py`

- [ ] calibrate a static per-node/level κ_GM (+κ_Redi) **field** by adjoint (field leaves
      already supported in `calibrate.py`) on the same windows/loss as Stage 1
- [ ] regularize (smoothness + bounds); 1-yr stability run
- [ ] tests: field-leaf gradient shape/masking; bounds
- [ ] gate: converged static-κ run stable — isolates **state-dependence** (NN vs static map)

### Task 11: Evaluation ladder, 63-yr deployment, figures

**Files:**
- Create: `scripts/nn_eddy/eval_ladder.py`, `scripts/nn_eddy/figs/` (+ sbatch)

- [ ] ladder free runs (decadal): (1) default GM [done: v3], (2) scalar-tuned GM,
      (3) static κ(x,z), (4) **NN-GM**, (5) no-GM, (6) offline-only-NN [gap exhibit] —
      via the Task-7 deployment path
- [ ] **headline: full 63-yr hindcast with NN-GM** (trained 1959–65; evaluated 1998–2007
      vs WOA à la Scholz = 4-decade generalization); reuse v3 pipelines
- [ ] figures: Fig-8 twin (30°W S-section, AAIW tongue S<34.7), zonal-mean T/S bias by
      basin/depth, AMOC(φ,z) + ACC, MLD, SST/SSS maps; final skill-vs-teacher-years curve
- [ ] claim tests: held-out teacher years 1966–67; farc zero-shot; interannual SO
      eddy-flux variability (SAM) tracked by NN-κ but not constant-κ; conservation audit
      (volume-mean S drift)
- [ ] gate: pipeline regression — default-GM rung reproduces the known v3 numbers

### Task 12: Stage 2 — EKI polish on slow targets (GATE anti-E2)

**Files:**
- Create: `scripts/nn_eddy/eki_polish.py`

- [ ] freeze NN body; head = last layer or per-basin scalings (O(10–100) dof)
- [ ] EKI (`eki.py`) multi-year forward ensembles vs slow metrics: AMOC, ACC, Fig-8/AAIW
      section score, zonal-mean bias norms, MLD clim; targets teacher-clim and/or obs
      (WOA, RAPID)
- [ ] tests: head parameterization round-trip; ensemble bookkeeping
- [ ] **GATE anti-E2:** the polished improvement **persists** in a fresh multi-decade
      forward run — the deploy-gap test this project exists to pass

### Task 13 (optional): Fortran round-trip

**Files:**
- Modify: fesom2 worktree (GM/Redi coefficient path); Create:
  `scripts/nn_eddy/export_weights.py`

- [ ] export weights + feature-scaling constants to a plain file
- [ ] hand-coded ~100-line Fortran MLP inference where fer_GM/Redi coefficients are set
- [ ] parity gate: JAX↔Fortran κ to fp precision on identical inputs
- [ ] multi-year Fortran CORE2 run with the NN closure vs the JAX NN run and the Fortran
      default baseline
- [ ] gate: parity + stable run ⇒ the "runs unchanged in the production model" section

### Task 14: Verify acceptance criteria

- [ ] all Overview claims implemented or explicitly descoped in the Revision Log
- [ ] full suite green (sbatch); `gm_nn=None` bit-identity re-verified; Fortran
      flag-off byte-identity re-verified
- [ ] gates HORIZON_OK / M3 / M4 / anti-E2 recorded with numbers in this plan
- [ ] data products + run dirs inventoried (paths + sizes) for the paper's data statement

### Task 15: Documentation and closeout

- [ ] append lessons to `docs/PORTING_LESSONS.md` (house rule — per task, as we go)
- [ ] update README "Differentiable capabilities" with `gm_nn`
- [ ] move this plan to `docs/plans/completed/`

## Technical Details

- **Window arithmetic:** dt=1800 ⇒ 48 steps/day; candidate windows start at snapshot marks;
  training years 1959–1965 ⇒ ~511 windows at 5-daily cadence (73/yr); 1966–67 held out.
- **Storage budget (5-daily cadence):** per 3D snapshot: T,S,w nodal ≈ 3×2.1 GB + u,v
  elemental ≈ 2×4.05 GB ⇒ ~14.4 GB; 73/yr ⇒ **~1.05 TB/yr, ~10.5 TB/decade**; 2D snapshots
  negligible; coarse products ~50 GB; replay months ~0.4–0.5 TB each. All ≪ 100 TB budget.
  Raw NG5 thinnable after coarsening (⚠️ manually only, never in automation).
- **Compute budget:** NG5 ~1.5–3 SYPD on 40–80 nodes (estimate) ⇒ ~10 SY ≈ O(10⁶) core-h —
  **measured in the Task-3 shakedown and signed off before the full chain** (gate). Ladder:
  five decadal rungs + one 63-yr CORE2 run (single-GPU-cheap, v3-known cost).
- **Loss:** L = Σ_lev w_lev (‖T−T̂‖² + λ_S‖S−Ŝ‖²)/σ²_lev + λ_η‖η−η̂‖², at HORIZON_OK cadence
  marks, first 6–12 h excluded; w_lev from teacher anomaly variance.
- **Arm B contingency (the M3 pivot, scoped):** ψ_NN — NN predicts a bolus streamfunction
  through the existing `gm.py` bolus machinery (conservative by construction; subsumes GM);
  same features/bounds discipline, new head + operator wiring, est. +2–3 wk. Triggered only
  if the κ-projection a-priori fit fails.
- **References:** Perezhogin et al. 2025 (arXiv:2505.08900); Zhang et al. 2023 JAMES;
  Frezat et al. 2022; arXiv:2412.06393; Ross et al. 2023; Scholz et al. 2019 (GMD 12,4875,
  Fig 8; eval 1998–2007, WOA); Sidorenko et al. 2020 (GMD 13,3337); Grooms et al. 2021
  (GCM-Filters); Gent & McWilliams 1990; Visbeck et al. 1997.

## Post-Completion

**Manual/external:**
- NG5 chain monitoring stays live for the campaign (script from Task 3; read-only).
- Paper writing (own repo, model-paper pattern); co-author circle (AWI GM/backscatter
  expertise natural); data publication via the Zenodo uploader.
- Possible upstreaming of the Fortran snapshot patch to FESOM2.
- Optional later legs: modern-era NG5 cold start for obs-era work; Arm B as a follow-up if
  not triggered; NG5-JAX teacher.

## Open scope question (user-raised 2026-08-06, not yet decided)

**Easy backscatter (`opt_visc=5`) as a JAX capability + ladder rung.** FESOM ships
`opt_visc=5` + `visc_easybsreturn=1.5` as its default and the user notes CORE2 "ideally"
would run it. Findings (detail in `fesom2/NG5_TEACHER_SETUP.md`): (a) the TEACHER cannot —
`check_validviscopt_5` (src/oce_dyn.F90:930) ABORTS opt_visc=5 on eddy-resolving/permitting
meshes (resol/Rossby<1.5) citing spurious boundary currents + Drake <80 Sv, and says
backscatter "was designed to energise coarse resolved configurations"; NG5 stays at 7 (as
DestinE did). (b) CORE2 IS backscatter's design regime, so our opt_visc=7 there is the
deviation — adopted to match the JAX student. (c) The blocker is the port: **JAX implements
only opt_visc=7** (`fesom_jax/config.py` ViscConfig — no backscatter kernel), so switching
the Fortran CORE2 alone would break the JAX↔Fortran match paper 2 rests on. ⇒ This is a
PORT task (backscatter kernel + adjoint + bit-faithfulness cert), not a namelist edit.
Attractive because backscatter is itself a coarse-mesh eddy-energy parameterization ⇒ a
natural extra Task-11 ladder rung (and a competitor/complement to the NN). Est. non-trivial
(new momentum kernel + adjoint + validation vs Fortran). **Decision deferred to the user;
nothing in the current plan depends on it.**

## Revision Log

- 2026-08-06 (a): plan created from the brainstorm dialogue (supersedes the pre-brainstorm
  draft): NG5 teacher (cold start 1958, not FORCA20/2008), snapshot patch scope corrected
  (no multi-frequency; replay-from-restart bursts), w saved for mesh AMOC (pyfesom2 recipe),
  evaluation reframed to general bias reduction anchored on Scholz et al. 2019 Fig 8,
  static-κ ladder rung added, Fortran round-trip added as optional task.
- 2026-08-06 (b): revised per plan-review agent (verdict NEEDS REVISION, 3 blockers + 10
  major + 7 minor — all addressed): horizon scan moved BEFORE the NG5 launch and made a
  gate (HORIZON_OK) that SETS the snapshot cadence, with an explicit pivot rule; SSH/ice/η
  added as 2D snapshots to the teacher protocol; AWI-namelist prerequisite + compute-budget
  sign-off + chain resume test + monitor checkbox added to Task 3; storage corrected for
  ELEMENTAL u,v (~1.05 TB/yr) + element→node collocation added; flux-product pilot
  re-anchored on shakedown NG5 snapshots (FORCA20 means = mechanics only); deployment path
  for trained params (run_from_config + restart-leg carry) added to Task 7 incl. the
  calibrate.py hardcode + register_dataclass; perfect-model twin control added before real
  training; Fortran flag-off byte-identity gate added; window count fixed (511, train
  1959–65); NG5 mesh export added; Arm B pivot scoped; module renamed fesom_filters→
  coarsen.py, MOC to scripts-only; paths fixed (fesom_jax/tests/,
  scripts/runs/run_suite.sbatch); authoritative-copy rule stated.
- 2026-08-06 (c): Task-0 workspace created. Branch points: `nn-eddy` off 8dca7e4
  (perf/cpu-cg-collectives HEAD — where this plan is committed; NOT off main, which
  predates it), `nn-eddy-output` off 4ccdd35a (~/fesom27/fesom2 main HEAD, which already
  carries NG5 partition-sweep configs — the likely namelist starting point for Task 3).
- 2026-08-06 (d): **FESOM version verified per user directive** ("use the fesom version
  the port is based on"): the R2 binary's recorded commit 6b26861a sits on branch
  `llm-fix-3`, NOT an ancestor of main — but its ONLY src delta vs the merge-base
  df1ab8cd is an accidentally *emptied* `src/ice_fct.F90` (that tree calls `ice_TG_rhs`
  with no definition anywhere ⇒ cannot link; the binary must have been built with the
  file present). Main @4ccdd35a has ZERO src changes since the merge-base and its
  `ice_fct.F90` md5-matches the production working tree ⇒ **worktree branch point
  4ccdd35a is the correct source state** (better than literal 6b26861a). Definitive
  behavioral check remains Task 1's gate: rebuilt-unpatched worktree binary vs the R2
  binary's outputs; if THAT differs, the gate control becomes rebuilt-unpatched vs
  rebuilt-patched-flag-off (same toolchain: env `shell.intel`, intel-oneapi 2022.0.1 +
  openmpi 4.1.2 + netcdf-fortran 4.5.3, via `configure.sh`).
- 2026-08-06 (e): **Task-3 inputs materialized** (user provided): NG5 source setup =
  `work_NG5_1950_Rohit` (see Task 3 checkbox); NG5 partitions already on /work. User
  directive: Task-1 output testing starts from the CORE2 setup (`fesom2_core2`
  namelists), NOT NG5. **Measured NG5 cost prior** (partition sweep, OUR fesom27
  binary, dt=120 s, 30-day runs, monthly output): 0.61 SYPD @4096 cores, 0.98 @7680,
  **1.10 SYPD @8192 cores (64 nodes)** — below the plan's 1.5–3 SYPD estimate ⇒ 10 SY
  ≈ 9.1 wall-days ≈ **~1.8 M core-h** at 64 nodes, before snapshot-I/O overhead; feeds
  the Task-3 compute-budget gate. ⚠️ dt: sweep needed 2-min (3-min blew up at mstep=47,
  cold start); the fesom2.6 Rohit run used 6-min (240 steps/day, after a ramp trail
  96→72→45→240 in comments) — teacher dt/ramp strategy is an open Task-3 decision.
- 2026-08-06 (f): **NG5 config decisions locked by user + sweep-SYPD explained**: mEVP
  α=β=500, icepack OFF (see Task 3). User hypothesized the sweep's 1.1 SYPD was
  output-heavy — checked: the sweep FORCED monthly-only output (commit 1c4ba52c) and
  disabled restarts (ff9ca26f); results.csv puts output at ~146 s of 4352 s/rank-mean
  (**~3%**) at n8192 — output is exonerated. The lever is **dt**: sweep ran 2-min
  (720 steps/day); the DestinE NG5 production ran **6-min** (step_per_day=240) ⇒ per-step
  cost ≈ constant ⇒ expect **~3× ≈ 3.3 SYPD @ 8192 cores** at production dt, i.e. 10 SY
  ≈ **~0.6 M core-h** — back inside the plan's O(10⁶) estimate. The sweep's 3-min
  blow-up (mstep=47) was an unramped cold start; teacher needs a cold-start dt ramp
  (FORCA20 precedent 180→240). Real SYPD incl. snapshot I/O = Task-3 shakedown output.
  (Sweep run dirs on /scratch have expired; results.csv + repo commits are the record.)
- 2026-08-06 (g): **teacher dt ramp decided by user**: dt=120 s (spd=720) for months
  1–2 with monthly restarts from the start, then stop+restart onto spd=240 (dt=360 s)
  for the campaign. Consequence for the shakedown: month 1 measures the ramp-dt SYPD
  (~1.1 @ 8192 expected); the compute-budget sign-off additionally needs a few days
  measured at dt=360 s (~3.3 SYPD expected). Snapshot cadence (GATE HORIZON_OK) is in
  DAYS, so the dt switch does not move the output protocol.
- 2026-08-06 (h): **Task 1 COMPLETE + provenance finding.** Snapshot patch cda32816 on
  `nn-eddy-output`; all three gates green (functional 12/12, byte-identity via control,
  overhead +0.12%). The byte-identity investigation PROVED the Feb-22 production R2
  binary (`~/fesom27/fesom2/bin/fesom.x`) computes a trajectory different from ANY
  buildable git state: it diverges from today's main from STEP 1 (sst max 4.5e-2 K,
  73% of >1e-3 diff nodes poleward of 50° ⇒ sea-ice advection), while an independent
  unpatched rebuild of 4ccdd35a is byte-identical to our patched-flag-off binary. The
  R2 binary was built from a working tree whose `ice_fct.F90` state matches no commit
  (llm-fix-3 emptied the file; the content compiled on Feb 22 is unrecoverable).
  Consequences: (1) the Task-1 gate legitimately uses the rebuilt-unpatched control;
  (2) **the worktree binary (main 4ccdd35a + snapshot patch) is THIS project's Fortran
  reference from now on** — teacher + all new Fortran baselines use it (clean git
  provenance); (3) the 63-yr R2 hindcast remains valid for its statistical/clim
  comparisons (paper 2), but pointwise re-use should note the caveat. Control build
  worktree at /scratch/a/a270088/tmp/fesom2_ctrl (throwaway; scratch auto-expires —
  `git worktree prune` in ~/fesom27/fesom2 afterwards). Smoke runs preserved in
  /work/ab0995/a270088/nn_eddy/task1_smoke/.
