# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0, 1, and 2 are COMPLETE
(GATEs 0/1/2 met).** The full single-step pi ocean model is ported, dump-gated, jitted,
and 100-step stable. Next is **Phase 3 — the AD smoke test (the project's
biggest-risk de-risking gate): wrap `step` in a checkpointed `lax.scan` (Task 3.1) and
prove an end-to-end gradient through the whole model incl. the CG `custom_linear_solve`
(Task 3.2).**

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean
model for hybrid ML (trainable NN parameterizations for vertical mixing and mesoscale
eddy fluxes, trained end-to-end). This continues a multi-session effort. Work from
`/home/a/a270088/port_jax`. Use max effort.

## START HERE, in order
1. **Read the plan (source of truth):**
   `/home/a/a270088/port_jax/docs/plans/20260605-fesom-jax-port.md` — decisions, the
   verification ladder, per-task gates, Revision Log. Phase 3 = Tasks 3.1–3.2; keep its
   checkboxes in sync.
2. **Read the lessons log (every session):** `docs/PORTING_LESSONS.md` — esp. the AD
   entries (safe-sqrt, safe-divide / masked-NaN traps, `custom_linear_solve` forward-vs-
   gradient split, the upwind `|vflux|` kink). **STANDING RULE: append a lesson per task.**
3. **Read the project memory:**
   `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.
4. Skim `fesom_jax/step.py` (the `step`/`step_jit`/`run` you'll scan) + `ssh.py`
   (`solve_ssh`/`custom_linear_solve` — the AD-critical solver) + the Phase-3 plan
   section (Tasks 3.1/3.2, incl. the FD step-size-sweep requirement and the
   smooth-regime caveat).

## STATUS — Phases 0/1/2 COMPLETE (GATE 2 met); all committed on `main`
- **Phase 0/1 (GATE 0/1):** env, verify harness, pi mesh export, the C-port per-substep
  dump oracle (`pi_cdump.00000`); `mesh.py`/`state.py`/`ops.py` (AD-verified).
- **Phase 2 (GATE 2):** the full single-step ocean chain, substeps 1–16, each dump-gated:
  `eos`, `pgf`, `pp`, `momentum` (+`forcing`), `ssh` (compute_ssh_rhs / static linfs
  operator + MITgcm precond / `solve_ssh` early-stop CG via `custom_linear_solve`),
  `ale` (`compute_w`/`thickness_linfs`/`commit_thickness`), `tracer_adv` (upwind),
  `tracer_diff` (per-node TDMA). Assembled in **`step.py`** (`step`/`step_jit`/`run`).
  - Step 1 reproduces every per-kernel dump gate; **rest state** machine-precision;
    **`S` exactly 35** multi-step; **100 steps stable** (`max|uv|`=0.075, `|eta|`=0.35 m,
    no NaN); the CG **warm-start** is load-bearing (the `solve_ssh` `rtol_abs` fix).
  - ⚠️ Tight multi-step `T` match is Phase-4 (upwind ≠ the dump's FCT, cascades via density).
- **Full suite: 274 passing** (`JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`; the
  100-step test runs jitted, ~20 s).

## IMMEDIATE WORK — Phase 3 (Tasks 3.1 + 3.2): the AD de-risking gate
This is THE gate that retires the project's biggest risk — that the hard AD patterns
(`lax.scan`+`jax.checkpoint`, the CG `custom_linear_solve`, the upwind/PP kinks) hold
end-to-end on the real model. AD was kept safe-by-construction throughout (every kernel
grad-checked), so this should mostly *confirm* — but expect to hunt one or two masked-NaN
or kink issues over a long window.

### Task 3.1 — checkpointed scan time loop (`fesom_jax/integrate.py` + `tests/test_integrate.py`)
- Wrap `step` in `jax.lax.scan` over N steps; apply `jax.checkpoint` (rematerialization)
  to the per-step fn for backward-pass memory.
- **⚠️ `is_first_step`:** the scan body must be uniform. Cleanest: run step 1 eagerly
  (`is_first_step=True`) OUTSIDE the scan, then `scan` steps 2..N with `is_first_step=False`
  baked in (the flag only flips the AB2 `ff_step`). Don't try to carry it as a traced bool.
- The static SSH `op` + `stress_surf` are loop-invariant → close over them (or pass as
  non-scanned args). `mesh`/`op` are pytrees; fine to close over.
- Gates: scan forward == the Phase-2 `run` loop (climate-close); N=200 backward pass fits
  device memory with checkpointing (memory sanity).

### Task 3.2 — end-to-end gradient (`fesom_jax/tests/test_gradient.py`)
- Scalar loss (e.g. mean SST after N steps). **Pick the param/window to stay in a SMOOTH
  regime** — verify the probe column never goes convective (the PP `max(N²,0)` / convective-
  `max` kinks) and that `S` stays ≳0.5 (the salinity-floor `max` kink) and away from
  `|vflux|=0` (the upwind kink). Keep N modest (the model is mildly chaotic via scatter
  reassociation; long windows amplify).
- **`d(loss)/d(param)`** for a scalar param — the plan names **PP `K_ver`** (background
  vertical diffusivity). ⚠️ `K_ver` is currently a **config constant** (`config.py`,
  `FESOM_PHASE1_K_VER=1e-5`); to differentiate it you must thread it as a traced **arg**
  through `pp.mixing_pp` (→ `step`). This is also the first concrete "ML-hook" seam
  (Phase 7 swaps the mixing here) — do it cleanly (a `params` dict or explicit arg).
- **FD check with a step-size SWEEP** (`h ∈ {1e-4…1e-7}`, relative, central, float64):
  report the FD-convergence plateau and assert `|grad_AD − grad_FD|/|grad_FD| < 1e-4` at
  the plateau — NOT at a single `h` (chaos floor below, truncation above). (See the
  Task-2.10 diffusion-`d/dKv` note: FD underflows at tiny-gradient entries.)
- **Confirm the gradient flows through the CG** `custom_linear_solve` (perturb a param
  affecting the stiffness/RHS — already proven per-step in `test_ssh.py`; now over the scan).
- **grad w.r.t. an IC field** (vector-valued, e.g. `d(loss)/d(T₀)`) sanity check — easy
  (differentiate w.r.t. the initial `State.T`); finite + nonzero.
- Write `test_gradient.py` as the permanent AD gate (re-run at every later gate).

**GATE 3 (DE-RISKING):** end-to-end gradient passes; scan+checkpoint and
`custom_linear_solve` proven on the real model. Then Phase 4 (FCT + opt_visc7 completion
+ pi 1000-step) — at which point the tight multi-step `T/S` dump match becomes available.

## THE PROVEN VERIFICATION RECIPE (still applies)
Dump gate at the pinned probes (node `1001,1500,2000,2500,3000`; elem `1757,2656,3688,
4604,5575`), truncate to `nlevels`, `verify.assert_close(col, rec, kind=…)` (`map`/`gather`
1e-15, `scatter`/`reduction` 1e-12; calibrate `atol`). AD-check with the **double-`where`
safe-sqrt** for `sqrt(x→0)` and **`where(d==0,1,d)` / `where(a>0,a,1)`** for any divide
whose denominator can vanish in a masked lane (the forward `where` does NOT stop a 0·inf
NaN in the backward pass — this bit `tracer_diff`'s `d/dKv`). Append lessons; tick the plan;
commit only when asked.

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`. **For the backward-pass memory
  sanity (N=200) use the GPU** (SLURM `shared`/`gpu`, acct `ab0995`) — CPU is fine for
  correctness/gradient-value gates.
- Exported pi mesh (gitignored): `data/mesh_pi/*.npy`; dump fixture (committed):
  `fesom_jax/tests/fixtures/pi_cdump.00000`.
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/` (branch
  `jax-mesh-export`). Kokkos (`GPU_FIDELITY.md` M5.8/M5.9 long-window chaos notes):
  `/home/a/a270088/port_kokkos/`.

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi).
2. Full-fidelity, bottom-up, not a toy. 3. **AD-safe by construction + early end-to-end
gradient (THIS Phase) re-run at every gate — AD is never deferred.** 4. Mesh = index
gather/scatter over `ops.py`. 5. Single-device + data-parallel now; mesh sharding Phase 8.

## CRITICAL GOTCHAS (verified; full list in PORTING_LESSONS.md)
- **`custom_linear_solve` splits forward (early-stop, dump-matching) from gradient (tight
  `transpose_solve` ⇒ clean implicit-diff `S⁻¹`).** The CG warm-start uses `rtol_abs` =
  `soltol·‖ssh_rhs‖` (the original rhs). The static linfs operator makes the SSH AD clean.
- **AD kinks to keep the smoke test away from:** PP `max(N²,0)` + convective `max`; the
  salinity floor `max(S,0.5)`; the upwind `|vflux|`/`|w|`. Stay smooth + modest-N.
- **IC = constant + Gaussian T-blob** (`ic.initial_state`); **S=35 constant**. `step` needs
  the static SSH `op` + a `stress_surf` (`forcing.surface_stress`, or zeros for rest).
- **AD-safe divides:** masked lanes must compute FINITE values (`where(d==0,1,d)`), not rely
  on a forward `where` to hide a 0·inf in the backward pass.

## WORKFLOW NOTES
- Plan is authoritative; tick `[x]`; keep the Revision Log + lessons current.
- Commit only when asked. C/Fortran edits go on the `port2` branch `jax-mesh-export`.
- All python/pytest via the env python above.

Confirm you've absorbed this, tell me which task you're starting, then proceed.
