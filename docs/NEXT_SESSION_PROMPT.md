# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0–4 COMPLETE (GATEs 0/1/2/3/4).**
The full single-step pi ocean model is ported, dump-gated, jitted, **proven differentiable
end-to-end** (checkpointed `lax.scan`, CG `custom_linear_solve` gradient, FD-checked
`d(loss)/d(K_ver)`, N=200 backward fits GPU memory), runs the **live pi physics** — **FCT
(Zalesak) advection** + **opt_visc=7 flow-aware biharmonic** + the **wsplit machinery**
(off, matching the reference config) — matches the per-substep C dump tightly (FCT `T`
1.8e-15; step-2 element viscosity ~1e-17), is **pi 1000-step stable** (`S` exactly 35, bounded
uv/eta, no NaN), and the gradient gate stays green with the full physics live (plateau 5.70e-7).
**Full suite 313 passing.** The next milestone is **Phase 5 — CORE2 single-device**, which the
plan says to **expand into its own sub-plan** (`docs/plans/<date>-fesom-jax-core2.md`) before
coding.

**All of Phase 4 (Tasks 4.1 FCT, 4.2 opt_visc7+wsplit, 4.3 1000-step) is committed on `main`**
(the "Phase 4 complete (GATE 4)" commit). The `port2` C source is clean/committed (the temp
FCT-intermediate dumps in `fesom_step.c` were reverted — working tree matches the committed
`d51dca4` dump writer); `jobs/jax_cdump_dbg.sh` (pure-advection `FESOM_NO_TRDIFF=1` debug dump)
remains **untracked** in `port2` (kept as a working-tree helper, not committed among port2's
other dev artifacts).

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean
model for hybrid ML (trainable NN parameterizations for vertical mixing and mesoscale
eddy fluxes, trained end-to-end). This continues a multi-session effort. Work from
`/home/a/a270088/port_jax`. Use max effort.

## START HERE, in order
1. **Read the plan (source of truth):**
   `/home/a/a270088/port_jax/docs/plans/20260605-fesom-jax-port.md` — decisions, the
   verification ladder, per-task gates, Revision Log. **Phase 4 is done (GATE 4 met);
   Phase 5 is the next section** (currently an outline to expand into a sub-plan).
2. **Read the lessons log (every session):** `docs/PORTING_LESSONS.md` — esp. the AD
   entries (the eos `bvfreq` `1/zdiff` backward-NaN trap; the safe-sqrt / safe-divide
   masked-NaN rule; `custom_linear_solve` forward-vs-gradient split; the FCT subgradient
   `flux_eps`; the "verify which BRANCH the test inputs exercise" viscosity lesson; the
   wsplit-off-is-the-reference-config note). **STANDING RULE: append a lesson per task.**
3. **Read the project memory:**
   `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.
4. For Phase 5: skim `port2/FRESH_START.md` §2/§4/§6/§9/§14 (CORE2 mesh rotation +
   CW orientation + partial cells; zlevel ALE; JRA55 + bulk; CORE2 default params), and
   the C modules `fesom_ale.c` (zlevel branch), `fesom_phc.c`, `fesom_jra55.c`,
   `fesom_bulk.c`, `fesom_sss_runoff.c`. Skim `fesom_jax/integrate.py` +
   `fesom_jax/tests/test_gradient.py` (the permanent AD gate — re-run at every later gate).

## STATUS — Phases 0/1/2/3/4 COMPLETE (GATEs 0/1/2/3/4); all committed on `main`
- **Phase 0/1/2 (GATE 0/1/2):** env, verify harness, pi mesh export, the C-port per-substep
  dump oracle (`pi_cdump.00000`); `mesh.py`/`state.py`/`ops.py`; the full single-step ocean
  chain (substeps 1–16) in **`step.py`** (`step`/`step_jit`/`run`), each dump-gated.
- **Phase 3 (GATE 3 — the AD de-risking gate):** `params.py` (the `Params` ML-hook seam),
  `integrate.py` (checkpointed `lax.scan`, step-1 eager outside), `test_gradient.py` (the
  permanent AD gate); eos `bvfreq` backward-NaN fix; N=200 backward = 4.23 GB checkpointed.
- **Phase 4 (GATE 4 — pi fully stable) — DONE THIS BLOCK:**
  - **Task 4.1 (FCT):** `tracer_adv.advect_one_fct` + `zalesak_limit` (MFCT/QR4C HO,
    `compute_fct_lo`, `fill_up_dn_grad`); limiter AD = **subgradient** (`docs/LIMITER_GRADIENTS.md`,
    NaN-safe via `flux_eps`). FCT `T` vs dump **1.8e-15** after the step-1 `T_old`=pre-blob-base
    IC fix (`ic.py`).
  - **Task 4.2 (opt_visc7 + wsplit):** the flow-aware biharmonic was **already fully ported in
    2.5** — this CLOSED its verification gap (the dump never reaches the flow-aware regime:
    edge-velocity-diff ≤8e-4 ≪ the |du|>0.03 onset; added a strong-flow synthetic test for BOTH
    the γ1 and quadratic γ2 branches + a step-2 substep-6 `uv_rhs` dump gate ~1e-17). Ported
    `ale.compute_cfl_z` + `compute_wvel_split` (CORE2-ready); **`use_wsplit=0` in the reference
    config** ⇒ the split is the identity (`w_e=w, w_i=0`, transparent), wired into `step.py`
    (populates `State.cfl_z`). ⚠️ `impl_vert_visc`'s `w_i` advective terms stay dropped (a Phase-5
    item, only needed at `use_wsplit=1`).
  - **Task 4.3 (1000-step + AD re-check):** `test_1000_step_stability` — pi 1000 steps stable
    (~48 s; no NaN, max|uv|=0.17, max|eta|=0.63 m, `S` exactly 35, T∈[10.0,14.98], max
    `cfl_z`=2.8e-3 ≪ maxcfl). `test_gradient.py` green with the full physics live (plateau 5.70e-7).
- **Full suite: 313 passing** (`JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`; ~5 min;
  the 1000-step + 100-step tests + the gradient FD sweeps dominate the time).

## IMMEDIATE WORK — Phase 5 (CORE2 single-device): FIRST expand the sub-plan
The plan's Phase 5 is an **outline** — per the project's discipline, **expand it into
`docs/plans/<date>-fesom-jax-core2.md` first** (task granularity + per-task gates), then execute.
Outline (from the main plan) — port, each verified against a CORE2 reference, AD re-checked:
- **CORE2 mesh specifics:** rotation auto-detect, CW element orientation (`test_tri`), partial
  cells, `nlevels_nod2D_min` (K_v⁻) — FRESH_START §2/§4/§14.
- **zlevel ALE** (surface-layer thickness change; local-zstar fallback) — `fesom_ale.c`. This
  also brings back the **`w_i` advective terms in `impl_vert_visc`** if `use_wsplit` is on.
- **PHC initial conditions** (bilinear interp + extrap + vertical fill) — `fesom_phc.c`.
- **JRA55 forcing** reader (bilinear→mesh, time interp, L&Y09 bulk) — `fesom_jra55.c`,
  `fesom_bulk.c`, FRESH_START §9.
- **SSS restoring + runoff** (additive virtual freshwater flux) — `fesom_sss_runoff.c`.
- **GATE 5:** CORE2 1-day (172 steps, dt=500) and 10-day climate-close to C; gradient check
  on a CORE2 slice. (Will likely need a new C-port dump + a CORE2 mesh export, like Tasks 0.3/0.4.)

**Open question for the user before Phase 5:** confirm the CORE2 reference path — do we extend
the `port2` C dump writer to a CORE2 config (the tightest gate, mirrors Path A), or compare to
an existing Fortran CORE2 run (climate-level only)? This drives the verification ladder for Phase 5.

## THE PROVEN VERIFICATION RECIPE (still applies)
Dump gate at the pinned probes (node `1001,1500,2000,2500,3000`; elem `1757,2656,3688,
4604,5575`), truncate to `nlevels`, `verify.assert_close(col, rec, kind=…)` (`map`/`gather`
1e-15, `scatter`/`reduction` 1e-12; calibrate `atol`). AD-check with the **double-`where`
safe-sqrt** for `sqrt(x→0)` and **`where(d==0,1,d)` / `where(a>0,a,1)`** for any divide whose
denominator can vanish in a masked lane (the forward `where` does NOT stop a 0·inf NaN in the
backward pass). **Re-run `test_gradient.py` at every gate**, and always grad w.r.t. a full IC
field (the strongest masked-NaN probe). For end-to-end FD checks, sweep `h` and assert the
**plateau**. ⚠️ **NEW (Task 4.2): instrument WHICH data-dependent branch a test input exercises**
— a "ported + tested" kernel can silently skip a coefficient regime if the test inputs are too
mild (the flow-aware viscosity γ2 branch). For a deferred "trivial-at-rest" element gate, re-gate
it at step ≥2 once the trajectory is dump-tight. Append lessons; tick the plan; commit only when asked.

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`. **For GPU runs** (the N=200 backward
  memory gate, 1000-step timings, future CORE2) use SLURM: `sbatch scripts/phase3_grad_memory.sbatch`
  (acct `ab0995_gpu`, `-p gpu`/`gpu-devel`, A100-40). CPU is fine for correctness/gradient-value
  (pi 1000 steps = ~48 s on CPU).
- Exported pi mesh (gitignored): `data/mesh_pi/*.npy`; dump fixture (committed):
  `fesom_jax/tests/fixtures/pi_cdump.00000`.
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/` (branch
  `jax-mesh-export`). Kokkos (`GPU_FIDELITY.md` M5.8/M5.9 long-window chaos notes):
  `/home/a/a270088/port_kokkos/`.

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi).
   The seam is real: `fesom_jax/params.py` (`Params` pytree, threaded through `step`).
2. Full-fidelity, bottom-up, not a toy. 3. **AD-safe by construction + early end-to-end
   gradient re-run at every gate — AD is never deferred.** 4. Mesh = index gather/scatter
   over `ops.py`. 5. Single-device + data-parallel now; mesh sharding Phase 8.

## CRITICAL GOTCHAS (verified; full list in PORTING_LESSONS.md)
- **AD masked-NaN rule (bit us 4×):** make masked-off lanes compute a FINITE value
  (`where(d==0,1,d)` / double-`where` safe-sqrt) — a forward `where`/clip/mask does NOT stop a
  `0·inf` NaN in the backward pass. (eos `bvfreq`, `tracer_diff`/qr4c `1/zdiff`, the visc safe-sqrt.)
- **`custom_linear_solve` splits forward (early-stop, dump-matching) from gradient (tight
  `transpose_solve`).** The static linfs operator makes the SSH AD clean; proven end-to-end.
- **AD kinks to keep the smoke test away from:** PP `max(N²,0)` + convective `max`; the
  salinity floor `max(S,0.5)`; the upwind `|vflux|`/`|w|`; the FCT/Zalesak limiter (subgradient,
  NaN-safe via `flux_eps`). Stay smooth + modest-N (N=20 in `test_gradient.py`).
- **config = the pi reference run:** linfs ALE, PP mixing, FCT tracers, opt_visc=7
  (γ0/γ1/γ2=0.003/0.1/0.285, Laplacian add-on γ_h=0), **`use_wsplit=0`** (⇒ `w_e=w`, `w_i=0`),
  CG SSH (`α=1`), analytical wind (double-averaged), dt=100, constant T=10 + Gaussian T-blob,
  S=35. `step`/`integrate` need the static SSH `op` (`ssh.build_ssh_operator`) + a `stress_surf`
  (`forcing.surface_stress`, or zeros for rest) + optional `params` (`Params`).
- **Differentiable time loop = `integrate.integrate` (checkpointed `lax.scan`)**, NOT the
  Python `run` loop. Step 1 eager + scan 2..N; close over loop-invariants.

## WORKFLOW NOTES
- Plan is authoritative; tick `[x]`; keep the Revision Log + lessons current. Expand Phases 5–8
  into sub-plans when reached (Phase 5 is next).
- Commit only when asked. C/Fortran edits go on the `port2` branch `jax-mesh-export`.
- All python/pytest via the env python above.

Confirm you've absorbed this, tell me which task you're starting, then proceed.
