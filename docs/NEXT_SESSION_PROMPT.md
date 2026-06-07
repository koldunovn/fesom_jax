# Next-session prompt — FESOM2 → JAX port (Phase 6C: KPP vertical mixing — finish the full model)

Paste the block below to start the next session. **Phases 0–6 COMPLETE (GATEs 0–6) + Phase 6B GM/Redi
COMPLETE (GATE 6B).** The user's 2026-06-07 decision: **finish the full functioning model — port KPP
(Phase 6C) — BEFORE the Phase-7a parameter-tuning on-ramp** (which is scoped + deferred, design saved
in `docs/plans/20260607-fesom-jax-paramtune.md`). **KPP is the *real* FESOM2 CORE2 default mixing
scheme; the JAX port currently runs the opt-in PP (`pp.py`)** — so porting KPP brings the model to the
actual production config. The C KPP port is **already done + Fortran-validated**, so we port from it and
dump-gate against it.

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for hybrid ML
(trainable NN parameterizations + parameter calibration). Multi-session effort. Work from
`/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Phase 6C sub-plan (source of truth — READ FULLY):** `docs/plans/20260607-fesom-jax-kpp.md` — §0
   scope (why KPP = the real default; the 8-stage driver map), §1 the verified CORE2 KPP config, §2 the
   seam + the input-availability audit (`dbsfc` is the one thing to add), §3 the **controlled-replay**
   validation strategy, §4 the **AD-safety kink inventory + treatments** (the crux), §5 the **K.0–K.11
   task ladder**, §6 GATE 6C.
2. **The C source (algorithmic SoT — port from this):** `/home/a/a270088/port2/fesom2_port/src/fesom_kpp.c` (1046 lines) + `fesom_kpp.h`. The driver `fesom_kpp_mixing` (`:770-924`); the C plan
   `port2/.../docs/plans/completed/20260524-kpp-vertical-mixing.md` (K0–K11 — the proven decomposition
   this mirrors; the controlled-replay technique is documented there).
3. **The reference config:** `port2/.../docs/kpp_reference_namelists/` (`namelist.oce`/`.tra` +
   `PROVENANCE.md`) — `mix_scheme='KPP'`, `Ricr=0.3`, `visc_sh_limit=diff_sh_limit=5e-3`, etc.
4. **The seam to mirror:** `fesom_jax/pp.py` (the PP path KPP replaces) + `fesom_jax/step.py:130` (the
   mixing call site) + how `gm_cfg`/`ice_cfg` are threaded as static args (`step.py:53,227`,
   `integrate.py:46,110`) — KPP copies this `kpp_cfg=None ⇒ PP bit-identical` pattern.
5. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the GM/Redi Task G.* entries +
   the new KPP-pivot entry. **STANDING RULE: append a lesson per task.**
6. **Deferred (do NOT start):** `docs/plans/20260607-fesom-jax-paramtune.md` (Phase 7a — after GATE 6C).
   **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.

## STATUS
- **Phases 0–6 + 6B (GATEs 0–6 + 6B):** full pi + CORE2 + sea ice + GM/Redi, committed on `main`,
  **453-test suite green.** Both ML hooks live (`k_ver`/`a_ver` mixing + `k_gm`/`redi_kmax` eddy).
- **Mixing today:** `pp.mixing_pp` (`step.py:130`) → `(Kv, Av, uvnode)`; `Kv`→tracer diff (+GM K33),
  `Av`→momentum. This is the **opt-in PP** scheme. KPP slots **exactly here** behind `kpp_cfg`.
- **`optax` 0.2.8 is installed** in the env (clean — jax untouched; for Phase 7a, not needed for KPP).

## IMMEDIATE WORK — Phase 6C Task K.0: scaffold `kpp_cfg` + generate the C KPP reference dumps
Start with the no-behavior-change scaffolding (the §5 K.0 task):
1. **New `fesom_jax/kpp.py`** with `KppConfig(NamedTuple)` holding the §1 constants (port the derived
   `Vtc/cg/deltaz/deltau` **verbatim from `fesom_kpp.c:130-138`** — research had a minor disagreement on
   their form; trust the C). Mirror `GMConfig` (`gm.py:42`).
2. **Thread `kpp_cfg=None`** through `step.py` (`:53`, `:227` static_argnames) and `integrate.py`
   (`:46`, `:110`, the eager step-1 + scan body) — exactly like `gm_cfg`. **`kpp_cfg=None` ⇒ the PP
   path, byte-identical** (regression: `sbatch scripts/run_suite.sbatch`, 453 green).
3. **Generate the C KPP reference dumps:** a `port2/jobs/jax_kpp_dump_core2.sh` (mirror
   `jax_gm_dump_core2.sh`) run with `FESOM_MIX_SCHEME=KPP` + `FESOM_KPP_DUMP_DIR` (per-node `hbl/bfsfc/
   ustar/Bo/kbl` + element `viscAE`) + the main `DUMP_SUB_MIXING=4` (`Kv`/`Av` at probe nodes) + the
   per-kernel replay inputs (`FESOM_KPP_REPLAY_DIR`) → `data/kpp_dump_core2/…`. The C dump+replay
   harness already exists (validated K0–K11) — this is **running** it, not writing C. ⚠️ **C edits →
   port2 `jax-mesh-export`, NEVER port2 main** (the user's strict rule).

Then **K.1–K.7 port each kernel AD-safe + controlled-replay dump-gate vs C** (`init`→`wscale`→
`ri_iwmix`→`ddmix` gate→`bldepth`→`blmix`→`enhance`/assembly), **K.8** wire into the step, **K.9**
climate/stability, **K.10** the gradient gate. See the sub-plan §5.

## VALIDATION = CONTROLLED REPLAY (the load-bearing technique — sub-plan §3)
A **live-run** KPP dump diffs at **~52 % of nodes** vs the reference — NOT an algebra bug, but the
**step-1 surface-forcing transient** (a known C↔Fortran flux mismatch) perturbing `bfsfc`/`ustar` at
nearly every node, which `blmix` amplifies (`f1 ∝ bfsfc/u*⁴`). So **don't** trust a whole-field live
diff. **Per kernel (K.2–K.7): inject the C-dumped INPUTS into the JAX kernel, compare its OUTPUTS to
the C-dumped outputs** (~1e-12; the C hit 3.18e-13 libm-ULP). The end-to-end check is the **climate
gate (K.9):** JAX-KPP climate ≈ C-KPP (RMS 0.005–0.013 °C class) AND distinct from JAX-PP (the genuine
scheme difference ≈ 0.085 °C, ~18× the residual).

## AD-SAFETY — KPP is the kink-heaviest scheme (sub-plan §4 has the full inventory + treatments)
Bar = **no NaN/Inf in the backward, finite everywhere incl. masked lanes** (hard) + a well-conditioned
gradient where one physically exists (bonus). Top treatments: (1) **`ustar = sqrt(sqrt(|τ|/ρ₀))`** →
project `_safe_sqrt` (∞ backward slope at zero wind; `ustar` is in many denominators — the #1 AD
priority); (2) the **discrete OBL level `kbl`** (bulk-Ri threshold search) → vectorize as a masked
first-crossing, **`stop_gradient` the integer index** but keep the **`hbl` interpolation weight
differentiable**; same for the `wscale` `int()` bin index + `caseA`; (3) replace the **`EPSLN=1e-40`
denominators** that sit on *physically-small* quantities (the `hbl` interp `/(Rib_k−Rib_km1+ε)`,
`f1=bfsfc/u*⁴`, `hekman/max(|f|,ε)`) with **physical floors** (1e-40 stops Inf but not gradient
blow-up). **Gradient gate (K.10):** masked-NaN-clean `d(loss)/d(T0)` through the assembled KPP model +
`d/d(visc_sh_limit)` or `d/d(K_bg)` (additive ⇒ clean plateau). Don't require a smooth plateau through
`kbl`.

## GATE 6C (acceptance)
KPP selectable via `kpp_cfg`; **PP byte-identical when `kpp_cfg=None`** (suite green); **every kernel
K.2–K.7 controlled-replay bit-faithful (~1e-12) vs the C**; assembled CORE2 **KPP+GM+ice stable** + the
**climate matches the C KPP** and is distinct from PP; **masked-NaN-clean gradient** + a well-conditioned
KPP-tunable gradient. **Then the full functioning model is complete → Phase 7a** (parameter tuning).

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`.
- **Env python (ALL python/pytest):** `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest`. ⚠️ **Heavy / full-suite + any CORE2 BACKWARD → `sbatch
  scripts/run_suite.sbatch` (compute node) or a GPU job** — CORE2 backprop HANGS on the login node (RAM
  thrash). Quick CPU forward-smokes (≤ few steps) + the **per-kernel replay gates** (small, isolated)
  run on the login node.
- GPU via SLURM: `-A ab0995_gpu -p gpu --gres=gpu:1` (A100-80GB last run). Stream forcing per step
  (don't `cf.stack` a long trajectory → OOM); one N-step backward per process (`jax.clear_caches()`);
  GPU steady-state ~0.09 s/step.
- C/Fortran (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/` (`fesom_kpp.c`/`.h`). Build:
  `bash -lc 'cd …/port2/fesom2_port && source env.sh && make -C build fesom_port'`. KPP dump env:
  `FESOM_MIX_SCHEME=KPP`, `FESOM_KPP_DUMP_DIR`, `FESOM_KPP_REPLAY_DIR`. **C edits → port2
  `jax-mesh-export`, NEVER port2 main.**
- KPP reference namelists: `port2/.../docs/kpp_reference_namelists/`. CORE2 mesh
  `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC `…/phc3.0_winter.nc`; JRA55
  `…/JRA55-do-v1.4.0/`; `data/` symlink → `/work/.../port_jax/data` (dumps live here).
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu).

## LOCKED DECISIONS (do NOT re-litigate)
1. Hybrid-ML use case; mixing seam = `pp.py` ↔ `kpp.py` behind `kpp_cfg` (KPP = the 1st ML-hook's
   physics alternative). 2. Full-fidelity, match the C 1:1, dump-gate (controlled replay for KPP).
   3. AD-safe by construction + a masked-NaN gradient gate (KPP is kink-heavy — `stop_gradient` the
   discrete `kbl`, safe-sqrt `ustar`, physical floors over `EPSLN`). 4. `kpp_cfg=None`/`gm_cfg=None`/
   `ice_cfg=None`/`params=None` ⇒ bit-identical. 5. KPP is **STATELESS** (recomputed each step, no new
   `State` fields — like GM). 6. CORE2-faithful = port only `mix_scheme_nmb==1`: **double diffusion +
   nonlocal flux GATED OFF** (port the gate, defer the body). 7. KPP is a **CORE2 forced-path feature**
   (needs surface forcing → pi path keeps PP). 8. Full-model run = `kpp_cfg + gm_cfg + ice_cfg` together
   (the real CORE2 production config). 9. **Phase 7a is DEFERRED** until GATE 6C.

## WORKFLOW NOTES
- Tick `[x]` in the sub-plan, keep the Revision Logs + lessons current. **Commit per-task on `main`
  when asked.**
- After GATE 6C, Phase 7a (`…-paramtune.md`) resumes — KPP's `Ricr`/`visc_sh_limit`/backgrounds become
  additional mixing-seam tuning targets.
- See memory [[fesom-jax-port]], [[porting-lessons-log]], [[hpc-job-file-conventions]].

Confirm you've absorbed this; then proceed with Phase 6C Task K.0 (scaffold `kpp_cfg` bit-identical +
generate the C KPP reference dumps), then port the kernels K.1→K.10 in data-flow order, controlled-replay
dump-gating each. If anything about the C source or the seam is ambiguous, read the C and ask.
