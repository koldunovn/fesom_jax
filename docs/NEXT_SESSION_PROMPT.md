# Next-session prompt — FESOM2 → JAX port (Phase 6B COMPLETE → Phase 7a parameter tuning)

Paste the block below to start the next session. **Phases 0–6 COMPLETE (GATEs 0–6) + Phase 6B
GM/Redi COMPLETE (GATE 6B met) — the ENTIRE GM/Redi physics is ported, assembled into the live
step, and gated; the 2nd ML-hook seam is established.** Both ML hooks (`k_ver`/`a_ver` mixing +
`k_gm`/`redi_kmax` eddy) are training-ready with FD-verified gradients. **The next step the user
chose is Phase 7a — differentiable PARAMETER TUNING** (calibrate physics params via the diff. port
→ push to the Fortran namelist), starting with the **perfect-model `k_gm` twin**. Phase 6C (KPP) is
the other pending phase.

GATE 6B results (all committed on `main`): the assembled GM step is **bit-exact** vs the C GM-ON
dump (post-step T 7.1e-15 / S 2.1e-14 — GM is deterministic, no ice-EVP floor; this closed K33's
gate); **10 days stable** (GM+ice, max|vel| 2.84, SST capped −1.91) with **GM smoothing fronts**
(front|∇T| 7.42e-6 ON vs 7.89e-6 OFF); **gradient gate `GM_GRAD_GATE_OK`** — `d/d(k_gm)` plateau
**3.5e-6 (well-conditioned)**, `d/d(k_ver)` 5.8e-10, masked-NaN `d/d(T0)` clean; **full suite 453
green** (`gm_cfg=None` ⇒ bit-identical).

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for hybrid
ML (trainable NN parameterizations + parameter calibration, trained/tuned end-to-end). Multi-session
effort. Work from `/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Parent plan (source of truth across phases):** `docs/plans/20260605-fesom-jax-port.md` — the
   Revision Log (Phase 6B COMPLETE + Phase 7a ADDED 2026-06-07); the **Phase 7a — Differentiable
   Parameter Tuning** section (the task ladder 7a.1–7a.4, the 3 Fortran-transfer tiers, the
   decades-spin-up strategy, GATE 7a); Phase 6C (KPP) outline.
2. **GM/Redi sub-plan (COMPLETE):** `docs/plans/20260607-fesom-jax-gmredi.md` — the §"Forward
   pointer — GM/Redi as a parameter-tuning target" (the perfect-model twin recipe + the
   spin-up/short-window caveat). G.1–G.7 all `[x]`.
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 6B Task G.7** entries
   (the assembled-GM bit-exact fidelity class, the pre-step-tracer Redi threading, the
   well-conditioned 2nd-hook gradient, the GM front-smoothing). **STANDING RULE: append a lesson
   per task.**
4. **The seam:** `fesom_jax/params.py` (the `Params` pytree — `k_ver`/`a_ver`/`k_gm`/`redi_kmax`,
   `params=None ⇒ defaults` bit-identical) + `fesom_jax/integrate.py` (the checkpointed
   differentiable `lax.scan`; closes over `params`, accumulates `d(loss)/d(params)`).
5. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.

## STATUS
- **Phases 0–6 + 6B (GATEs 0–6 + 6B):** full pi + CORE2 + sea ice + GM/Redi, committed on `main`.
- **GM/Redi modules:** `eos.compute_sw_alpha_beta`; `gm.py` (`GMConfig`, `gm_diagnostics` driver);
  `gm_redi.py` (G7a/G7b/K33). Wired into `step.py`/`integrate.py` behind static `gm_cfg=None`.
- **Gates/scripts:** `tests/test_gm_step.py` (assembled), `scripts/core2_gm_stability_run.py`
  (+`_gpu.sh`), `scripts/core2_gm_grad_gate.py` (+`.sbatch`). Full suite: `sbatch
  scripts/run_suite.sbatch` (compute node, 453 tests).

## IMMEDIATE WORK — Phase 7a Task 7a.1: the perfect-model `k_gm` twin (the user's choice)
Prove the diff.-port **calibration loop** end-to-end with a self-consistent twin (no real obs yet):

1. **Twin target:** run CORE2 forward (GM+ice or GM-only) over a SHORT window with `k_gm=1500`
   (a `Params(k_gm=1500., redi_kmax=1500.)`); save the post-window SST/SSS as the target. ⚠️ This
   is a SHORT-WINDOW MACHINERY test (the injected `k_gm` difference shows in the short-window
   tendency) — NOT an equilibrium tune. Do **not** try to spin up decades under AD.
2. **Recover:** from `k_gm=800`, define `misfit(k_gm) = ‖SST(k_gm) − SST_target‖²` over the same
   window, and `optax` (Adam or L-BFGS) descent on `grad(misfit)(k_gm)` → recover ~1500. Confirm
   convergence + that the recovered value reproduces the target misfit→0.
3. **Generalize the seam:** factor the optimizer/loss into `scripts/core2_param_tune.py` (or a
   `fesom_jax/calibrate.py`) so adding a 2nd tunable (e.g. `k_ver`, or a `[nod2D]` field leaf —
   array leaves already differentiate) is the established `Params`-expansion pattern.
4. **(7a.4 stretch) Fortran transfer:** write the recovered `k_gm` into `namelist.oce`
   (`K_GM_max`/`Redi_Kmax`) and confirm the C/Fortran run reproduces the JAX-predicted change
   (config-match caveat: the port is the reduced default-namelist physics — tune for the config you
   run). Scalar transfer = ZERO Fortran code.

**⚠️ The decades-spin-up strategy (locked — do NOT backprop through decades):** memory (≈630k
steps/decade → >150 GB even O(√N)) AND chaotic gradient blow-up both forbid it. For
physically-meaningful (equilibrated) tuning: **spin up FORWARD, no AD** (the stability run / Fortran)
→ **short-window adjoint anchored at the equilibrated state** → **gradient-free EKI** (Ensemble
Kalman Inversion, forward runs only, `vmap`-parallel) for the slow mean. The perfect-model twin
(7a.1) only needs the short-window adjoint.

**GATE 7a (acceptance):** a tuned scalar (e.g. `k_gm`) measurably reduces a defined misfit in JAX
AND, written to the namelist, in Fortran; the perfect-model twin recovers the injected value;
masked-NaN clean; suite green. **Then Phase 6C = KPP** (`fesom_kpp.c` first; `pp.py` ↔ a new
`kpp.py` behind the mixing seam) if not already done.

## WELL-CONDITIONED TUNING TARGETS (clean gradients shown)
`k_gm`/`redi_kmax` (GM eddy — ACC/stratification), `k_ver`/`a_ver` (mixing — SST/MLD), GM
depth/resolution scalings (`GMzexp_zref`, `refscalresol`). Ice **thermo** params OK; ⚠️ EVP
**rheology** stiff (`1/delta_min` ~1e16 — use EKI / `stop_gradient`).

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`.
- **Env python (ALL python/pytest):** `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest`. ⚠️ **Heavy / full-suite + any CORE2 BACKWARD →
  `sbatch scripts/run_suite.sbatch` (compute node) or a GPU job** — the CORE2 backprop HANGS on the
  login node (RAM thrash); the login node runs ONE CPU-JAX process at a time. Quick CPU
  forward-smokes (≤ a few steps) are fine on the login node.
- GPU via SLURM: `-A ab0995_gpu -p gpu --gres=gpu:1` (A100; got an **A100-80GB** last run). ⚠️
  stream forcing per step (don't `cf.stack` the whole trajectory → OOM); one N-step backward per
  process (`jax.clear_caches()`); GPU steady-state ~0.09 s/step (10-day run ~4 min).
- C/Fortran (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/`. Build:
  `bash -lc 'cd …/port2/fesom2_port && source env.sh && make -C build fesom_port'`. **C edits →
  port2 `jax-mesh-export`, NEVER port2 main** (the user's strict rule).
- CORE2 mesh `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC `…/phc3.0_winter.nc`; JRA55
  `…/JRA55-do-v1.4.0/`; `data/` symlink → `/work/.../port_jax/data`.
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu).

## LOCKED DECISIONS (do NOT re-litigate)
1. Hybrid-ML use case; seam = `params.py`. Both ML hooks LIVE (`k_ver`/`a_ver` + `k_gm`/`redi_kmax`),
   gradients FD-verified. 2. Full-fidelity, match the C 1:1, dump-gate. 3. AD-safe by construction +
   gradient re-run at every gate. 4. `gm_cfg=None`/`ice_cfg=None`/`params=None` ⇒ bit-identical.
   5. GM/Redi is STATELESS. 6. Full-cell linfs ⇒ static vertical geometry. 7. **Parameter tuning
   (7a):** scalar→namelist (zero Fortran); field→netCDF; NN→Fortran inference. 8. **No multi-decade
   adjoint** — forward spin-up + short-window adjoint + EKI for the slow mean. 9. EVP rheology
   gradient is stiff (EKI/`stop_gradient` it).

## WORKFLOW NOTES
- Tick `[x]`, keep the Revision Logs + lessons current. **Commit per-task on `main` when asked.**
- The user is interested in using the diff. port to **tune Fortran-deployable parameters** — keep the
  Fortran-transfer tiers (scalar→namelist first) front of mind.
- See memory [[fesom-jax-port]], [[porting-lessons-log]], [[hpc-job-file-conventions]].

Confirm you've absorbed this; then proceed with Phase 7a Task 7a.1 (the perfect-model `k_gm` twin):
build the misfit + optax loop over a short window, recover the injected `k_gm`, factor the calibration
seam, and (stretch) write the optimum into `namelist.oce`. Or, if the user prefers, start Phase 6C
(KPP). Confirm which with the user if ambiguous.
