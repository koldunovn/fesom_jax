# Next-session prompt — FESOM2 → JAX port (Phase 7a: differentiable parameter tuning / calibration)

> **⚡ MILESTONE (2026-06-07): GATE 6C MET — the FULL functioning FESOM2 CORE2 model now runs in JAX,
> differentiable end-to-end.** KPP vertical mixing (the *real* CORE2 default) + GM/Redi eddies +
> prognostic sea ice. Phase 6C (KPP) is COMPLETE: `fesom_jax/kpp.py` (KppConfig + the 7 kernels +
> `mixing_kpp` driver) wired at the `step.py` mixing seam behind `kpp_cfg` (`None ⇒ PP byte-identical`).
> **29 KPP tests:** K.2–K.7 per-kernel controlled-replay bit-faithful vs the C; **K.8 the assembled step
> bit-faithful** (Kv/Av@probes 1.7e-21, all-nodes 4e-12 — the JAX forcing is a validated 1:1 port of the
> C's, so there is no JAX↔C transient); **K.9** KPP+GM+ice 10-day stable + climate distinct from PP
> (SST RMS 0.13 °C); **K.10** masked-NaN `d(SST)/d(T0)` clean + additive `d/d(K_bg)` plateau 1.1e-11.
> Suite 483 green. **Phase 6C plan `docs/plans/20260607-fesom-jax-kpp.md` (K.0–K.11 `[x]`) + memory
> `[[fesom-jax-port]]` + `docs/PORTING_LESSONS.md` are the record.** **RESUME AT Phase 7a Task 7a.1**
> (the `calibrate.py` seam + the perfect-model `k_gm` twin) — the §0–§6 below is the full briefing.

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for hybrid ML
(trainable NN parameterizations + parameter calibration). Multi-session effort. Work from
`/home/a/a270088/port_jax`. Max effort. **The forward model is now COMPLETE** (pi + CORE2 + sea ice +
GM/Redi + KPP); this phase turns differentiability into a **calibration** capability.

## START HERE, in order
1. **Phase 7a sub-plan (source of truth — READ FULLY):** `docs/plans/20260607-fesom-jax-paramtune.md` —
   §0 the already-verified de-risking (DON'T re-check: `optax 0.2.8` installed clean / jax untouched; the
   `k_gm` twin is well-posed — `gm.py` `k_top=max(scaling·k_gm, k_gm_min)` is a lower floor ONLY, no
   upper clamp ⇒ `fer_K ∝ k_gm` linear, smooth bowl over [800,1500]; tune a single θ driving BOTH
   `k_gm=θ` and `redi_kmax=θ`, the C auto-sync). §1 the `calibrate.py` seam (`optimize` + `grid_scan`).
   §2 **Task 7a.1 — the perfect-model `k_gm` twin** (the first experiment). §3 the task ladder
   7a.1→7a.4 + the Fortran transfer. §4 GATE 7a.
2. **The seam this calibrates:** `fesom_jax/params.py` (`Params` pytree, the ML-hook leaves) +
   `scripts/core2_gm_grad_gate.py` (the `d/d(k_gm)` plateau 3.5e-6 — the gradient 7a descends) +
   `fesom_jax/integrate.py` (the checkpointed differentiable `lax.scan`). The KPP tunables
   (`Ricr`/`visc_sh_limit`/`K_bg`) are NEW 7a targets — K.10 already showed `d/d(K_bg)` is clean
   (plateau 1.1e-11); to make a `KppConfig` field trainable, lift it from the static config into the
   traced `Params` (the K.10 `_replace(k_bg=·)` trick is the preview).
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the Phase-7-relevant ones: FD↔AD
   plateaus are only well-posed in SMOOTH regimes (validate at N=1 / isolated seams — the forced
   multi-step model is non-smooth in the physics params); **do NOT backprop through the multi-decade
   spin-up** (memory + chaotic-gradient blow-up). **STANDING RULE: append a lesson per task.**
4. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`
   ([[fesom-jax-port]], [[porting-lessons-log]], [[hpc-job-file-conventions]]).

## STATUS
- **Phases 0–6 + 6B + 6C ALL COMPLETE (GATEs 0–6, 6B, 6C).** The full CORE2 model (PP **and** KPP
  mixing, GM/Redi, prognostic ice, linfs/FCT/opt_visc7) runs + is differentiable. Suite = ocean 436 +
  ice 47 = **483 green** (`sbatch scripts/run_suite.sbatch`, two chunks; full set in one process OOMs
  the login node — a known pytest+JAX jit-cache pattern, not a bug).
- **Both ML hooks live + FD-verified:** 1st (mixing) `k_ver`/`a_ver`; 2nd (eddy) `k_gm`/`redi_kmax`.
  KPP adds `Ricr`/`visc_sh_limit`/`K_bg` as further mixing-seam targets (`d/d(K_bg)` plateau 1.1e-11).
- **`optax 0.2.8` installed clean** (jax untouched). The calibration loop is "an objective + an
  optimizer loop + a target" away — a small extension, not a restructure.

## IMMEDIATE WORK — Phase 7a Task 7a.1: `calibrate.py` + the perfect-model `k_gm` twin
1. **New `fesom_jax/calibrate.py`** — the optimizer half of the ML-hook seam, generic over a pytree of
   tunable leaves: `optimize(loss_fn, init, optimizer, *, n_iters, on_step, stop_fn)` (jit
   `value_and_grad` once, host loop) + `grid_scan(loss_fn, base, leaf, values)` (the forward-only
   misfit-bowl probe — CONFIRM the minimum sits at the injected θ before trusting the descent). The exact
   signatures are in §1 of the plan. **Unit test** `tests/test_calibrate.py` (fast, no CORE2 — a 1-D
   quadratic bowl: Adam recovers the known minimum; keeps the suite green).
2. **The perfect-model `k_gm` twin (the first experiment, a script not a test):** generate a short CORE2
   reference trajectory at `k_gm=θ*` (e.g. 1200), then from a wrong `k_gm₀` (e.g. 800) descend
   `d(misfit)/d(θ)` with `optax.adam` until θ→θ*. First `grid_scan` to confirm the bowl minimum is at θ*
   (well-posed, unclamped). GPU job (mirror `core2_gm_grad_gate.sbatch`). **Gate:** θ recovered to a few
   %, monotone misfit decrease.
3. Then **7a.2** (the `Params`-expansion pattern for N tunables + a real-obs misfit), **7a.3** (short-
   window-adjoint-AT-EQUILIBRIUM tuning + a gradient-free **EKI** baseline for the slow mean — NOT a
   spin-up backprop), **7a.4** (export the optimum to the Fortran `namelist.oce` `K_GM_max`/`Redi_Kmax`
   and confirm the Fortran run improves). See §3.

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`.
- **Env python (ALL python/pytest):** `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest`. ⚠️ **Heavy / full-suite + any CORE2 BACKWARD → `sbatch
  scripts/run_suite.sbatch` (compute node) or a GPU job** — CORE2 backprop HANGS on the login node (RAM
  thrash). Quick CPU forward-smokes (≤ few steps) + small isolated backwards run on the login node.
- GPU via SLURM: `-A ab0995_gpu -p gpu --gres=gpu:1` (A100-80GB; KPP+GM+ice ~0.10 s/step; the K.10
  N=4 KPP backward = 28 GB / 44 %). Stream forcing per step (don't stack a long trajectory → OOM);
  one N-step backward per process (`jax.clear_caches()`).
- The full model is selected by passing `kpp_cfg=KppConfig()` + `gm_cfg=GMConfig()` + `ice_cfg=IceConfig()`
  together to `step`/`integrate` (the real CORE2 production config); `…=None` ⇒ that feature off,
  byte-identical. The pi path keeps PP (KPP needs forcing → raises on the pi path).
- C/Fortran SoT: `/home/a/a270088/port2/fesom2_port/src/`. **C edits → port2 `jax-mesh-export`, NEVER
  port2 main.** Fortran KPP validation run: `/scratch/a/a270088/fortran_2yr_dt1800`. CORE2 mesh
  `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC `…/phc3.0_winter.nc`; JRA55 `…/JRA55-do-v1.4.0/`.
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu).

## LOCKED DECISIONS (do NOT re-litigate)
1. Hybrid-ML use case; the calibration seam = `params.py` (the SAME ML-hook, trainable leaves = physical
   constants instead of NN weights). 2. Full-fidelity model is DONE — 7a is a small extension (objective
   + optimizer + target), not a restructure. 3. Tune θ→BOTH `k_gm`/`redi_kmax` (the C auto-sync); the
   twin is unclamped/well-posed. 4. `params=None`/`kpp_cfg=None`/`gm_cfg=None`/`ice_cfg=None` ⇒
   bit-identical. 5. **Do NOT backprop through the multi-decade spin-up** — spin up FORWARD (no AD) →
   short-window adjoint at equilibrium → gradient-free EKI for the slow mean. 6. FD↔AD only in smooth
   regimes (N=1 / isolated seams). 7. The end goal = push the optimum into the operational Fortran
   `namelist.oce` (scalar transfer = zero Fortran code).

## WORKFLOW NOTES
- Tick `[x]` in the 7a sub-plan, keep the Revision Logs + lessons current. **Commit per-task on `main`
  when asked.**
- The KPP constants (`Ricr`/`visc_sh_limit`/`K_bg`/backgrounds) are now additional calibration targets
  alongside `k_gm`/`k_ver` — lift a `KppConfig` field into `Params` to make it trainable (K.10 preview).
- See memory [[fesom-jax-port]], [[porting-lessons-log]], [[hpc-job-file-conventions]].

Confirm you've absorbed this; then proceed with Phase 7a Task 7a.1 (`calibrate.py` + the perfect-model
`k_gm` twin). If anything about the seam or the twin set-up is ambiguous, read the plan §0–§2 and ask.
