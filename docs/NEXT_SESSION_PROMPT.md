# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0, 1, 2, and 3 are COMPLETE
(GATEs 0/1/2/3 met).** The full single-step pi ocean model is ported, dump-gated, jitted,
100-step stable, AND **proven differentiable end-to-end** (checkpointed `lax.scan`, the
CG `custom_linear_solve` gradient, an end-to-end FD-checked `d(loss)/d(K_ver)`, the N=200
backward fits GPU memory). **The project's biggest risk is retired.** Next is **Phase 4 —
pi fully stable: FCT (Zalesak) advection + the limiter-gradient decision (Task 4.1),
complete opt_visc=7 + wsplit (Task 4.2), pi 1000-step stability + AD re-check (Task 4.3).**

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean
model for hybrid ML (trainable NN parameterizations for vertical mixing and mesoscale
eddy fluxes, trained end-to-end). This continues a multi-session effort. Work from
`/home/a/a270088/port_jax`. Use max effort.

## START HERE, in order
1. **Read the plan (source of truth):**
   `/home/a/a270088/port_jax/docs/plans/20260605-fesom-jax-port.md` — decisions, the
   verification ladder, per-task gates, Revision Log. Phase 4 = Tasks 4.1–4.3; keep its
   checkboxes in sync.
2. **Read the lessons log (every session):** `docs/PORTING_LESSONS.md` — esp. the AD
   entries (the eos `bvfreq` `1/zdiff` backward-NaN trap; the safe-sqrt / safe-divide
   masked-NaN rule; `custom_linear_solve` forward-vs-gradient split; the upwind `|vflux|`
   kink; the FD-floor / plateau-at-large-h note). **STANDING RULE: append a lesson per task.**
3. **Read the project memory:**
   `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.
4. Skim `fesom_jax/tracer_adv.py` (the upwind advection you'll extend to FCT) +
   `fesom_jax/integrate.py` (`integrate`/`integrate_jit` — the differentiable entry) +
   `fesom_jax/tests/test_gradient.py` (the permanent AD gate, re-run at every later gate)
   + the Phase-4 plan section (Tasks 4.1–4.3, incl. the limiter-gradient `docs/LIMITER_GRADIENTS.md`
   research item).

## STATUS — Phases 0/1/2/3 COMPLETE (GATE 3 met); all committed on `main`
- **Phase 0/1/2 (GATE 0/1/2):** env, verify harness, pi mesh export, the C-port per-substep
  dump oracle (`pi_cdump.00000`); `mesh.py`/`state.py`/`ops.py`; the full single-step ocean
  chain (substeps 1–16) in **`step.py`** (`step`/`step_jit`/`run`), each dump-gated; step 1
  reproduces every per-kernel gate, rest state machine-precision, `S` exactly 35, 100 steps
  stable.
- **Phase 3 (GATE 3 — the AD de-risking gate):**
  - **`params.py`** — a `Params` pytree (`k_ver`, `a_ver`) threaded `step(...,params) →
    pp.mixing_pp`; `params=None ⇒ defaults` (numerically transparent). **The ML-hook seam**
    (Phase 7 swaps the mixing here).
  - **`integrate.py`** — `step` in a `jax.checkpoint`-ed `lax.scan`. **Step 1 eager
    (`is_first_step=True`) OUTSIDE the scan**, steps 2..N scanned with `is_first_step=False`
    baked in. Forward == the Phase-2 `run` loop **bit-identical**; checkpoint forward-transparent.
  - **`test_gradient.py`** — the permanent AD gate: `d(mean SST)/d(k_ver)` AD↔FD sweep
    (plateau 5.9e-7 ≪1e-4), gradient **flows through the CG**, `d(loss)/d(T₀)` finite, smooth
    regime certified.
  - **⚠️ Fixed an eos `bvfreq` backward-NaN trap** (`zdiff=0` at the bottom-padding lane →
    `0·inf` in the backward pass → `d/d(T₀)` NaN; the forward was always fine). Fix:
    `where(zdiff==0,1,zdiff)`. The **IC-field gradient is the stronger masked-NaN probe**.
  - **GPU memory sanity (job 25378918):** N=200 backward = **4.23 GB checkpointed** vs **48.7 GB
    OOM** without ⇒ checkpointing load-bearing.
  - ⚠️ Tight multi-step `T` match is still Phase-4 (upwind ≠ the dump's FCT, cascades via density).
- **Full suite: 286 passing** (`JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`; ~3 min;
  the 100-step test + the gradient FD sweeps dominate the time).

## IMMEDIATE WORK — Phase 4 (Tasks 4.1 + 4.2 + 4.3): pi fully stable
This makes pi match the dump's *live* physics (FCT, full opt_visc=7) and pushes to 1000-step
stability — at which point the **tight multi-step `T/S` dump match** (deferred since Phase 2)
finally becomes available, and the Phase-3 AD gate is **re-run with the full pi physics**.

### Task 4.1 — FCT (Zalesak) advection + the limiter-gradient decision
- Port high-order + Zalesak limiter (`fct_plus/minus`, local min/max bounds, sign-dependent
  flux selection) — ref `fesom_tracer_adv.c:814+`, FRESH_START §12. Extend `tracer_adv.py`.
- **⚠️ RESEARCH ITEM — `docs/LIMITER_GRADIENTS.md`:** the Zalesak limiter has `min`/`max`/
  sign-select **kinks** that AD must handle. Decide + document one of: (a) subgradient as-is,
  (b) smooth min/max relaxation, (c) `stop_gradient` on the limiter coefficients (treat the
  limited-flux mask as fixed in the backward pass). Implement the choice. This is the new
  **AD-hard** item (the same risk class as the CG, now retired) — keep `test_gradient.py`
  finite + FD-consistent **where smooth** under the chosen strategy.
- Forward gate: FCT `T/S` vs the C dump substep 15 (≤1e-12) — now a TIGHT gate (the dump runs
  FCT, so the upwind−FCT gap closes); `S` already bit-exact.

### Task 4.2 — complete opt_visc=7 (flow-aware biharmonic) + wsplit
- Complete the flow-aware terms of `visc_filt_bidiff` (opt_visc=7, `visc_gamma0=0.003`) to
  match the live C kernel; port `use_wsplit` vertical-velocity splitting (`wsplit_maxcfl=1.0`).
  Modify `momentum.py`/`ale.py`.
- Forward gate vs dumps (substep-13 `w` node-direct; substep-6 `uv_rhs` element-direct).

### Task 4.3 — pi 1000-step stability + AD re-check
- Run pi 1000 steps at dt=100; assert stable; snapshot climate-close to C.
- **Re-run the Phase-3 gradient gate (`test_gradient.py`) with FCT + opt_visc7 active** — the
  AD must still pass with the limiter-gradient strategy live (this is why AD is never deferred).

**GATE 4:** pi 1000 steps stable & climate-close; gradient check still passes with full pi
physics. Then Phase 5 (CORE2 single-device) — expand into `docs/plans/<date>-fesom-jax-core2.md`.

## THE PROVEN VERIFICATION RECIPE (still applies)
Dump gate at the pinned probes (node `1001,1500,2000,2500,3000`; elem `1757,2656,3688,
4604,5575`), truncate to `nlevels`, `verify.assert_close(col, rec, kind=…)` (`map`/`gather`
1e-15, `scatter`/`reduction` 1e-12; calibrate `atol`). AD-check with the **double-`where`
safe-sqrt** for `sqrt(x→0)` and **`where(d==0,1,d)` / `where(a>0,a,1)`** for any divide whose
denominator can vanish in a masked lane (the forward `where` does NOT stop a 0·inf NaN in the
backward pass — this bit `tracer_diff`'s `d/dKv` AND the eos `bvfreq` `d/dT₀`). **Re-run
`test_gradient.py` at every gate**, and always grad w.r.t. a full IC field (the strongest
masked-NaN probe). For end-to-end FD checks, sweep `h` and assert the **plateau** (round-off
floor below, truncation above) — and remember the plateau may be at LARGE `h` for a
near-linear loss. Append lessons; tick the plan; commit only when asked.

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`. **For GPU runs** (the N=200 backward
  memory gate, future 1000-step timings) use SLURM: `sbatch scripts/phase3_grad_memory.sbatch`
  (acct `ab0995_gpu`, `-p gpu`/`gpu-devel`, A100-40). CPU is fine for correctness/gradient-value.
- Exported pi mesh (gitignored): `data/mesh_pi/*.npy`; dump fixture (committed):
  `fesom_jax/tests/fixtures/pi_cdump.00000`.
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/` (branch
  `jax-mesh-export`). Kokkos (`GPU_FIDELITY.md` M5.8/M5.9 long-window chaos notes):
  `/home/a/a270088/port_kokkos/`.

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi).
   The seam is now real: `fesom_jax/params.py` (`Params` pytree, threaded through `step`).
2. Full-fidelity, bottom-up, not a toy. 3. **AD-safe by construction + early end-to-end
   gradient (Phase 3, DONE) re-run at every gate — AD is never deferred.** 4. Mesh = index
   gather/scatter over `ops.py`. 5. Single-device + data-parallel now; mesh sharding Phase 8.

## CRITICAL GOTCHAS (verified; full list in PORTING_LESSONS.md)
- **AD masked-NaN rule (bit us 3×):** make masked-off lanes compute a FINITE value
  (`where(d==0,1,d)` / double-`where` safe-sqrt) — a forward `where`/clip/mask does NOT stop a
  `0·inf` NaN in the backward pass. Latest: eos `bvfreq`'s bottom-padding `1/zdiff`.
- **`custom_linear_solve` splits forward (early-stop, dump-matching) from gradient (tight
  `transpose_solve` ⇒ clean implicit-diff `S⁻¹`).** The static linfs operator makes the SSH AD
  clean. The gradient is now proven to flow through it end-to-end over the scan.
- **AD kinks to keep the smoke test away from:** PP `max(N²,0)` + convective `max`; the
  salinity floor `max(S,0.5)`; the upwind `|vflux|`/`|w|`; **NEW in Phase 4: the FCT/Zalesak
  `min`/`max`/sign-select limiter** (the Task-4.1 research item). Stay smooth + modest-N.
- **IC = constant + Gaussian T-blob** (`ic.initial_state`); **S=35 constant**. `step`/`integrate`
  need the static SSH `op` (`ssh.build_ssh_operator`) + a `stress_surf` (`forcing.surface_stress`,
  or zeros for rest) + optional `params` (`Params`, default = config constants).
- **Differentiable time loop = `integrate.integrate` (checkpointed `lax.scan`)**, NOT the
  Python `run` loop. Step 1 eager + scan 2..N; close over loop-invariants.

## WORKFLOW NOTES
- Plan is authoritative; tick `[x]`; keep the Revision Log + lessons current.
- Commit only when asked. C/Fortran edits go on the `port2` branch `jax-mesh-export`.
- All python/pytest via the env python above.

Confirm you've absorbed this, tell me which task you're starting, then proceed.
