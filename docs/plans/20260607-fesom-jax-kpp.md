# FESOM2 → JAX Port — Phase 6C: KPP vertical mixing (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 6 outline — KPP = 6C).
**Predecessors:** `docs/plans/20260606-fesom-jax-core2.md` (Phase 5 ocean, GATE 5) +
`docs/plans/20260606-fesom-jax-phase6-seaice.md` (sea ice, GATE 6) +
`docs/plans/20260607-fesom-jax-gmredi.md` (GM/Redi, GATE 6B).
**Created:** 2026-06-07. **Status:** ⏳ NOT STARTED (K.0–K.11 planned; this is the next session's work).
**Scope (user-confirmed 2026-06-07):** **KPP only** — the K-Profile Parameterization vertical mixing,
i.e. the *real* FESOM2 CORE2 default mixing scheme. This **completes the full functioning model**
(the user's stated goal: finish the physically-complete model before the Phase-7a parameter-tuning
on-ramp). Differentiable-parameter tuning (Phase 7a) is DEFERRED and its design is preserved in
`docs/plans/20260607-fesom-jax-paramtune.md`.

**Decisions (locked, from the project conventions):** (1) **Mirror the C/Fortran 1:1, dump-gate each
kernel** — the algorithmic source of truth is the C port's `fesom_kpp.c` (1046 lines), which is itself
a **completed + validated** line-by-line port of the Fortran `oce_ale_mixing_kpp.F90`
(see `port2/.../docs/plans/completed/20260524-kpp-vertical-mixing.md`, tasks K0–K11, climate RMS
0.005–0.013 °C vs Fortran). (2) **Config-gate exactly like GM/Redi/ice** — a static `KppConfig`
threaded as `kpp_cfg`; `kpp_cfg=None ⇒ the existing PP path, bit-identical` (all current gates
preserved). (3) **AD-safe by construction + a gradient gate at GATE 6C** — KPP is the kink-heaviest
scheme yet; safe-sqrt / meaningful floors / `stop_gradient` on the discrete level-selection, masked
lanes finite. (4) **CORE2-faithful = port only what `mix_scheme_nmb==1` uses** — double diffusion +
nonlocal flux are GATED OFF (port the gate, defer the unused body).

---

## 0. Scope (READ FIRST — what the C KPP port actually is, and why it matters)

### Why KPP now — it is the *real* default mixing scheme

The JAX port currently runs **PP** (Pacanowski–Philander, `pp.py`) — but that is `mix_scheme_nmb==2`,
the **opt-in** scheme. The FESOM2 CORE2 **production default is `mix_scheme='KPP'` (`nmb==1`)**, and
the C port made KPP its default at `8d0cdbc` (2026-05-25); PP is reachable only via
`FESOM_MIX_SCHEME=PP`. Every JAX gate so far has therefore been **PP-vs-PP** (the GM/Redi sub-plan
notes this explicitly: "the GM dump runs on the same PP path the JAX port uses; KPP is the C default
but is Phase 6C", `20260607-fesom-jax-gmredi.md:49-51`). Porting KPP brings the JAX mixing up to the
**real default config** → the model is then physically complete and matches what an operational FESOM2
CORE2 run actually does. **This is the headline goal: forward fidelity to the C/Fortran KPP.**

### What KPP IS (one driver, eight stages, one mixing-seam output)

KPP replaces PP's local Richardson-number `Kv/Av` with a **boundary-layer (OBL) profile**: it finds the
ocean-boundary-layer depth `hbl` from a bulk Richardson criterion, builds a cubic shape function for
the diffusivity inside the OBL (matched to the interior at the base), and uses shear-Ri + background
mixing in the interior — producing the **same two outputs PP does**: `Kv` (node, tracer diffusivity)
and `Av` (element, momentum viscosity). It is driven by surface forcing (wind stress → `u*`,
heat/freshwater flux → surface buoyancy `Bo`), so — like ice — **KPP is a CORE2 forced-path feature**
(in the pi analytical path there is no real surface forcing; that path keeps PP).

**Top-level driver** `fesom_kpp_mixing` (`fesom_kpp.c:770-924`), stage order:

| # | Stage | C function / lines | Output |
|---|-------|--------------------|--------|
| 1 | `dVsq` shear-vs-surface | `fesom_kpp.c:792-809` | `dVsq[N,nl]` |
| 2 | pre-step: `ustar`, `Bo` surface forcing | `fesom_kpp.c:811-821` | `ustar[N]`, `Bo[N]` |
| 3 | **interior** shear-Ri + background mixing | `kpp_ri_iwmix` `:219-274` (call `:824`) | `viscA`, `diffKt/s` |
| 4 | double diffusion | `ddmix` — **GATE ONLY** (`#error` if on, `:828-831`) | (no-op CORE2) |
| 5 | **OBL depth** `hbl`/`kbl` (HIGHEST RISK) | `kpp_bldepth` `:317-435` (call `:834`) | `hbl[N]`, `kbl[N]`, `bfsfc`, `caseA` |
| 6 | **BL coeffs** + cubic shape + `dkm1` + `ghats` | `kpp_blmix` `:449-579` (call `:845`) | `blmc[3,N,nl]`, `dkm1[3,N]`, `ghats` |
| 7 | `enhance` (blend at `kbl-1`) | `kpp_enhance` `:588-621` (call `:856`) | blended `blmc` |
| 8 | `smooth_blmc` (3-sweep) + combine + node→elem | `:863-918` | `aux->Kv[node]`, `aux->Av[elem]` |

Helper `kpp_wscale` (turbulent velocity scales `wm`/`ws`, `:173-210`) is used by stages 5 & 6.
`mo_convect` runs **after** KPP (shared with PP, `fesom_step.c:264`) — **already ported** in
`pp.py:mo_convect`, reuse it. **KPP is STATELESS** (every field recomputed each step from T/S/N²/forcing
— no new `State` fields), exactly like GM/Redi → simple to thread and to AD.

### What is ABSENT (out of scope — CORE2 default disables it; port the gate, defer the body)

- **Double diffusion** (`ddmix` — salt fingering / diffusive convection): `double_diffusion=.false.`,
  `KPP_DOUBLE_DIFFUSION=0`. The driver `#error`s if enabled (`fesom_kpp.c:828-831`). Port the gate
  (no-op), document the gap.
- **Nonlocal transport flux** (`ghats`→tracer): `use_kpp_nonlclflx=.false.`. `ghats` is **computed**
  in `blmix` (`:561-562`) but **never consumed** in CORE2 (`fesom_kpp.h:42,67`). Port the computation
  (it is cheap and keeps `blmix` faithful) but **do not** wire it into the tracer flux.
- **`smooth_hbl`**, **`smooth_Ri`** = `.false.` (skipped). **`smooth_blmc` = `.true.`** (3-sweep
  smoothing of the BL coeffs IS applied — `:863-864`; this is a linear halo-stencil op, AD-safe).
- **Convective adjustment is NOT in `fesom_kpp.c`** — `mo_convect` (shared, already ported) does it.
  The only convective signal inside KPP is `ri_iwmix` clamping `N²≥0` (static instability → Ri=0 →
  max shear mixing).

---

## 1. CORE2 KPP reference configuration (VERIFIED — the values to port)

Source: `port2/.../docs/kpp_reference_namelists/{namelist.oce,namelist.tra}` (+ `PROVENANCE.md`),
matching the Fortran validation run `/scratch/a/a270088/fortran_2yr_dt1800` (KPP, 2 yr, dt=1800).
The KPP module constants are **hardcoded `#define`s** in `fesom_kpp.c:30-52` (not namelist):

**Namelist (the only KPP-relevant tunables):** `mix_scheme='KPP'`; `Ricr=0.3` (critical bulk Ri);
`A_ver=1e-4` (bg viscosity); `K_ver=1e-5` (bg diffusivity, `Kv0_const=.true.`); `visc_sh_limit=5e-3`;
`diff_sh_limit=5e-3`; `concv=1.6`; `double_diffusion=.false.`; `use_instabmix=.true.` (→ `mo_convect`);
`use_sw_pene=.true.`.

**Hardcoded `#define` constants** (`fesom_kpp.c:30-52`): `EPSLN=1e-40` (denom floor — **NOT** a
physical ε; see §4), `EPSILON=0.1` (surface-layer fraction), `VONK=0.4`, `CONC1=5.0`, `ZMIN=-4e-7`,
`ZMAX=0.0`, `UMIN=0.0`, `UMAX=0.04`, `RICR=0.3`, `CONCV=1.6`, `VISC_SH_LIMIT=5e-3`,
`DIFF_SH_LIMIT=5e-3`, `RIINFTY=0.8`, `MINMIX=3e-3` (surface elem-visc floor), `CEKMAN=0.7`,
`CMONOB=1.0`. **Table-build constants** (`:125-127`): `cstar=10`, `conam=1.257`, `concm=8.380`,
`conc2=16`, `zetam=-0.2`, `conas=-28.86`, `concs=98.96`, `conc3=16`, `zetas=-1.0`. **Table dims**
`NNI=890`, `NNJ=480` (`fesom_kpp.h:54-55`). **Derived scalars** `Vtc`, `cg`, `deltaz`, `deltau`
(`:130-138`) — ⚠️ **port the C expressions verbatim** (research surfaced minor disagreement on the
derived Vtc/cg forms; trust `fesom_kpp.c:130-138`, do not re-derive from a paper). External
constants: `g=9.81`, `rho0=1030`, `VCPW=4.2e6`, bg `A_ver=1e-4`/`K_ver=1e-5`.

---

## 2. The mixing seam + integration (mirror gm_cfg / ice_cfg exactly)

**JAX seam today** (`step.py:130`): `Kv, Av, uvnode = pp.mixing_pp(mesh, st.uv, bvfreq, k_ver=params.k_ver, a_ver=params.a_ver)`. `Kv` → tracer vertical diffusion (`step.py:185,197`
`Kv_eff = Kv (+ GM K33)` → `tracer_diff.impl_vert_diff`); `Av` → momentum (`momentum.impl_vert_visc`,
`step.py:142`); `uvnode` reused downstream. KPP slots **exactly here** (substep 4, where PP is).

**Config-gate pattern (copy gm_cfg/ice_cfg):** a `KppConfig(NamedTuple)` of static constants (like
`GMConfig` `gm.py:42`); add `kpp_cfg=None` to `step(...)` (`step.py:53`) and `integrate(...)`
(`integrate.py:46`) and to both `static_argnames` (`step.py:227`, `integrate.py:110`); thread it to
the eager step-1 + the scan body (`integrate.py:87-98`). **`kpp_cfg=None` ⇒ the PP branch, byte-identical** to today (regression: full suite stays green). The full-model run = `kpp_cfg=KppConfig()`
**+** `gm_cfg=GMConfig()` **+** `ice_cfg=IceConfig()` together (the real CORE2 production config).
KPP sets `Kv/Av`; GM's K33 still augments `Kv` afterward (same as PP+GM today, `step.py:196`);
`mo_convect` runs after KPP (`pp.mo_convect`, shared).

**Inputs KPP needs at the seam — availability audit:**

| Input | KPP use | JAX source | Status |
|-------|---------|-----------|--------|
| `bvfreq` (N²) | ri_iwmix, bldepth, Vtsq | `eos.compute_pressure_bv` (`step.py:110`) | ✅ at seam |
| `sw_alpha`, `sw_beta` | `Bo`, `bfsfc`, `dbsfc` | `eos.compute_sw_alpha_beta` (`eos.py:211`) | ✅ exists (GM added it) |
| `uvnode` | dVsq, ri_iwmix shear | `pp.compute_vel_nodes` (`step.py:129`) | ✅ reuse |
| `S` surface | `Bo` haline term | `st.S[:,0]` | ✅ |
| `heat_flux`, `water_flux` | `Bo` (surface buoyancy flux) | `StepForcing` (`core2_forcing.py:95-96`) | ⚠️ thread to seam |
| `stress_node_surf` | `ustar = sqrt(sqrt(\|τ\|/ρ₀))` | `bulk.stress_node_surf` (`core2_forcing.py:137`) | ⚠️ thread to seam |
| `sw_3d` | `bfsfc` shortwave penetration | forcing `cal_shortwave_rad` (`core2_forcing.py:147`) | ⚠️ thread to seam |
| **`dbsfc`** | `Ritop = zk·dbsfc(nz)` (bldepth) | **not computed under PP** (`eos.py:7`) | ❌ **ADD** (K.5) |

So K.8 threads `heat_flux`/`water_flux`/`stress_node_surf`/`sw_3d` (already produced by the CORE2
forcing/ice path — `step.py:96-107`) into the mixing call, and **K.5 adds `dbsfc`** (the surface
buoyancy difference, from the EOS α/β + T/S — `fesom_eos.c` stores it; mirror that). These come from
the forced path only → KPP's `kpp_cfg` branch is wired in the CORE2 path; the pi path stays PP.

**Outputs:** `Kv` `[nod2D,nl]` node (= `diffK` channel 0, the T-diffusivity, used for **both** T&S in
CORE2, `fesom_kpp.c:918`); `Av` `[elem2D,nl]` element (node `viscA` → 3-vertex mean → bottom-fill →
surface `minmix=3e-3` floor, `:896-910`). Same shapes/locations as PP — drop-in.

---

## 3. Validation strategy — CONTROLLED REPLAY is the key technique

The C port's hardest lesson (K6/K7): a **live-run** KPP dump diffs at **~52 % of nodes** vs Fortran —
NOT because the algebra is wrong, but because the **step-1 surface-forcing transient** (a known
pre-existing C↔Fortran flux mismatch, `[[project_forcing_step1_diff]]`) perturbs `bfsfc`/`ustar` at
nearly every node, and `blmix` **amplifies** it (`f1 ∝ bfsfc/u*⁴`, the `wscale` table, the
`stable`-flips). A whole-field live diff is therefore **uninterpretable**. The fix — **controlled
replay**: inject the *reference-dumped inputs* into the kernel under test, run only that kernel's
algebra, and diff its outputs. This isolates the algebra from forcing noise → the C achieved **worst
max|Δ| = 3.18e-13** (libm last-ULP) on `blmix` and `enhance`.

**Adopt the same for JAX**, gating against the **C** (the JAX port's algorithmic SoT, already
Fortran-validated):

1. **Per-kernel controlled replay (the trustworthy gate).** For each kernel K.2–K.7, feed the JAX
   kernel the **C-dumped inputs** for that kernel and compare its outputs to the **C-dumped outputs**,
   node-by-node, to ~1e-12. The C already has the dump + replay harness (`FESOM_KPP_DUMP_DIR`,
   `FESOM_KPP_REPLAY_DIR`, `fesom_kpp_dump_nodes/elems`) — K.0 generates these reference files.
2. **Assembled step-1 sanity (expected to show the forcing transient).** Run the JAX KPP forward one
   step and compare `Kv`/`Av`/`hbl` at the probe nodes to the C KPP live dump. Like the C, expect
   diffs from the step-1 forcing transient at a fraction of nodes — this is a **sanity** gate, not a
   bit-exact one; the per-kernel replay (1) is the bit-faithful proof.
3. **End-to-end climate/stability (the real end-to-end gate, K.9).** Assembled CORE2 **KPP+GM+ice**
   forward over N days: stable, and SST/SSS climate **matches the C KPP** (the C hit SST/SSS RMS
   0.005–0.013 °C vs Fortran) and **differs from the JAX PP** run by the genuine scheme difference
   (the C measured C-PP vs Fortran-KPP ≈ 0.085/0.093 °C — ~18× the KPP-KPP residual). So "JAX-KPP ≈
   C-KPP ≪ JAX-PP↔KPP gap" is the discriminating check.

**Dump sourcing (K.0):** mirror the existing per-feature dump jobs (`port2/jobs/jax_gm_dump_core2.sh`,
`jax_ice_*_dump_core2.sh`) → a new `port2/jobs/jax_kpp_dump_core2.sh` run with `FESOM_MIX_SCHEME=KPP`
+ `FESOM_KPP_DUMP_DIR` + the main `DUMP_SUB_MIXING=4` (Kv/Av at probe nodes) → `data/kpp_dump_core2/…`.
The existing CORE2 gate convention is `data/step_dump_core2/core2_cdump.00000` from
`port2/jobs/jax_step_dump_core2.sh` (`test_core2_step.py:5-6,40`) — follow it. **C edits → port2
`jax-mesh-export`, NEVER port2 main** (the user's strict rule). The C KPP dump harness already exists
(validated K0–K11), so K.0 is mostly *running* it with the right env, not writing C.

---

## 4. AD-safety strategy (KPP is the kink-heaviest scheme — the crux)

KPP has **structural discreteness** PP/GM did not: an integer OBL level `kbl` chosen by a thresholded
search, a piecewise `hbl` interpolation, a `wscale` table `int()` lookup, `copysign` step functions.
Making *all* of this smoothly differentiable is not the goal; the project's bar is **(a) no NaN/Inf in
the backward, finite everywhere incl. masked lanes** (the hard requirement, like every prior gate) and
**(b) a well-conditioned gradient where one physically exists** (a bonus, e.g. through the additive
`visc_sh_limit·frit + K_bg` interior term). The discrete level-selection is treated the way the model
already treats the FCT limiter / upwind kinks (locked: "the multi-step forced trajectory is non-smooth,
Task 5.8") — `stop_gradient` the *index*, differentiate the *continuous* parts. Treatments, by Agent-A
hotspot category:

- **A. Discrete index selection** (`kbl` bulk-Ri search + `EXIT` `:343-385`; second `kbl` relocation
  `:407-413`; `caseA`→`kn` `:506`; `wscale` `int()` `:178,186`; blmix loop bound `:548`). Port the
  per-node early-exit search as a **vectorized cumulative threshold-crossing** (mask `Rib_k>Ricr`,
  take the first-true level via a masked `argmax`/searchsorted), `stop_gradient` the resulting integer
  `kbl`/`kn`/bin-index, but keep the **interpolation weight differentiable** — e.g.
  `hbl = zkm1 + (zk−zkm1)·(Ricr−Rib_km1)/(Rib_k−Rib_km1+ε)` is smooth given fixed `kbl`. Physically:
  "which level" is discrete; "where within the level" is smooth.
- **B. Sign-step / Heaviside** (`stable = 0.5+copysign(0.5,bfsfc)` `:351,381,426`; `caseA` `:433`).
  Replace `copysign` with a `jnp.where(x>=0, …)` mask; `stop_gradient` the boolean (it's a regime
  switch, not a smooth knob).
- **C. min/max/clamp/abs** (`max(N²,0)` `:244`; `max(Ri,0)` `:258`; `min(ratio,1)` `:260`;
  `min(hekman,hmonob)` + `hbl` clamps `:393-396`; one-sided slopes `dvdz+|dvdz|=2·max(dvdz,0)`
  `:518-527`; `dat1=min(dat1,0)` `:537-543`; `fmin(sig,ε)` `:550,568`; `fmax(interior,blmc)` combine
  `:878-880`; `minmix` floor `:908`). All are `jnp.maximum/minimum/abs` — **AD-safe** (kinks have a
  defined subgradient); no special handling beyond using the jnp ops. `fabs` on depths is benign
  (fixed sign); `fabs(N²)`/`fabs(dvdz)` are genuine kinks at 0 but finite.
- **D. sqrt (∞ backward slope as arg→0; NaN for arg<0)** — the **critical NaN sources**:
  - **`ustar = sqrt(sqrt(|τ|/ρ₀))`** (`:815`): `d(ustar)/d(τ)→∞` as wind→0, and `ustar` sits in many
    denominators (`u*⁴` in `f1`, `u*³` in `hmonob`, `wscale`). Use the project's `_safe_sqrt`
    idiom (the GM `eos`/`gm` pattern: `sqrt(max(x,0))` for the value + a guarded/clamped backward, or
    `sqrt(x+δ)` with a **physical** `δ` ≫ `EPSLN`). This is the #1 AD priority.
  - `Vtsq ∝ sqrt(|N²|)` (`:360`), `_safe_sqrt`.
  - Fractional `pow(·,1/4|1/2|1/3)` in the **table build** (`:152-160`) and `cg` (`:134`): the
    `(conas·u³−concs·zehat)^{1/3}` base can go negative → clamp `≥0` (the table is a **constant** built
    once → can even build it in numpy and freeze it; no grad needed through the table values).
- **E. Divisions with `EPSLN=1e-40` floors** — `EPSLN` prevents `Inf` but **not gradient blow-up**
  (`d(1/den)/d(·) ~ 1/den²` is enormous at `den≈1e-40`). Replace `EPSLN` with a **physically
  meaningful floor** on the *physically-small* denominators: the `hbl` interp `/(Rib_k−Rib_km1+ε)`
  (`:368`, two near-equal bulk-Ri), `f1 = bfsfc/(u*⁴+ε)` (`:533`), `hekman /max(|f|,ε)` (equatorial
  f→0, `:390`), `hmonob /(bfsfc+ε)` (`:392`), `gat1/dat1 /(w+ε),/(hbl+ε)` (`:535-542`),
  `ghats /(ws·hbl+ε)` (`:562`). The shear `Ri=N²/(shear+ε)` (`:245`) — a small floor.
- **F. Formula-family switches** (`wscale` table-vs-analytic `zehat≤zmax` `:176`; table-build branches
  `:147-160`; Ekman/MO gate `bfsfc>0 && nzmin==0` `:389`; combine `nz<kbl` `:877`). `jnp.where`;
  `stop_gradient` the boolean where it is a regime switch.
- **G. Non-AD but port-relevant** (AD-safe, just port carefully): `smooth_blmc` 3-sweep
  (`fesom_smooth_nod3D`, linear stencil + halo, `:863-864`); node→element averaging (fixed linear
  scatter via `elem_nodes`, `:896-910`). Upstream `bvfreq`/`dbsfc`/`sw_alpha`/`sw_beta` must be
  differentiable (EOS — already are / `dbsfc` added AD-safe in K.5).

**Gradient gate (K.10):** `d(loss)/d(T0)` finite everywhere incl. masked lanes on the **assembled KPP
model** (the masked-NaN probe — the same one that has caught every prior backward-NaN trap) + a
well-conditioned KPP-tunable gradient where one exists (e.g. `d/d(visc_sh_limit)` or `d/d(K_bg)` — both
enter the interior `Kv/Av` **additively** → clean plateau, the cleanest analog of `k_ver`). Do **not**
require a smooth plateau through the discrete `kbl`.

---

## 5. Task ladder K.0–K.11 (data-flow order; mirrors the proven C decomposition)

Each kernel: **port AD-safe → controlled-replay dump-gate vs C → tick + lesson**. (The C plan's K0–K11
is the template; `port2/.../docs/plans/completed/20260524-kpp-vertical-mixing.md`.)

- [x] **K.0 — Scaffolding + reference dumps (NO behavior change).** ✅ New `fesom_jax/kpp.py` with
  `KppConfig(NamedTuple)` (all §1 constants + derived `Vtc/cg/deltaz/deltau` ported verbatim from
  `fesom_kpp.c:130-138` — **bit-match the C dump**). Threaded `kpp_cfg=None` through
  `step.py`/`integrate.py` (+ both `static_argnames` + the pi/CORE2 eager+scan bodies), mirroring
  `gm_cfg` — verified `kpp_cfg=None ⇔ KppConfig()` bit-identical (max|Δ|=0) through `step_jit` +
  `integrate`. Generated the C KPP reference dumps: `port2/jobs/jax_kpp_dump_core2.sh` (KPP, GM OFF,
  ice OFF; 57 s, ~890 MB text) → `data/kpp_dump_core2/…` (per-kernel `*_rank0.txt` + `kpp_init` +
  `kpp_wscale`); reader added (`io_dump.{read_kpp_table,load_kpp_dump,load_kpp_init,
  load_kpp_wscale_sweep}`). **Bonus:** the wm/ws 892×480 table + the 4 derived scalars recompute
  bit-exactly (max|Δ|=0) vs the dump ⇒ K.1's builder is pre-validated. **Gate:** reference dumps exist
  + load ✅; full suite green (`sbatch run_suite.sbatch`) — *(C side already built/validated — this was
  running it, not writing C.)*
- [x] **K.1 — `init`: lookup tables + derived constants.** ✅ `kpp.build_wscale_tables` (the `wmt`/`wst`
  892×482 tables, `lru_cache`d host-numpy constant → jnp; `^{1/3}` base clamped ≥0 in discarded lanes,
  exact) + `Vtc,cg,deltaz,deltau` in `KppConfig`. **Gate:** tables max|Δ|=1.7e-18, scalars BIT-match
  the C init dump (`test_kpp_wscale.py`).
- [x] **K.2 — `wscale`: turbulent velocity scales.** ✅ `kpp.wscale` (`:173-210`): bilinear lookup
  (`jnp.trunc` bin index = zero-grad discrete; **`ufrac`/`zfrac` use the UNCLAMPED numerator** ⇒ ustar
  beyond UMAX extrapolates, the key fidelity subtlety) + the stable-analytic branch (`jnp.where`, safe
  denom). **Gate (controlled replay):** C sweep (201×101) → `wm,ws` max|Δ|=4.3e-13/8.6e-13 (stable
  region exact); AD finite incl. the ustar=0 column.
- [x] **K.3 — `ri_iwmix`: interior mixing.** ✅ `kpp.ri_iwmix` (`:219-274`): shear `Ri=max(N²,0)/
  (shear+δ)` → `frit=(1−min(Ri/Riinfty,1)²)³` → `viscA=visc_sh_limit·frit+A_bg`, `diffKt=diffKs=
  diff_sh_limit·frit+K_bg`, edge copies nzmin/nzmax. **Gate (replay):** vs C dump max|Δ|=0 (BIT-EXACT,
  shear=0 step-1) + synthetic cubic-frit (3e-18) + AD finite. (`test_kpp_ri_iwmix.py`.)
- [x] **K.4 — `ddmix`: double diffusion — GATE ONLY.** ✅ `kpp.assert_no_double_diffusion` — no-op for
  CORE2 (`double_diffusion=.false.`), raises if enabled (the C `#error` analog). `ghats` computed in
  K.6 but `use_kpp_nonlclflx=False` ⇒ not wired. **Gate:** confirmed no-op + raises-if-on.
- [x] **K.5 — pre-step (`dVsq`,`ustar`,`Bo`) + `dbsfc` + `bldepth` (HIGHEST RISK).** ✅
  `kpp.prestep`: `dVsq` (=0 at cold-start step 1), `ustar` = `sqrt(sqrt(τx²+τy²)/ρ0)` (both nested
  sqrts `_safe_sqrt`, τ=0 AD finite), `Bo` (formulas bit-exact). `eos.compute_dbsfc` (the missing EOS
  input) bit-exact (max|Δ|=0) vs the C dump (gated via the GM dump's step-1 T/S). `kpp.bldepth`:
  two masked first-crossings (`Rib_k>Ricr→kbl1`+hbl interp; `|zbar|>hbl→kbl`), `stop_gradient` the
  integer kbl + differentiable hbl weight, Ekman/MO clamp, `caseA`. **Extended the C dump
  (jax-mesh-export) with `sw_3d`+`sw_alpha`** for the replay. **Gate (controlled replay):** hbl 6.5e-12,
  **kbl 0/126858 mismatches**, bfsfc 8.6e-23, stable/caseA EXACT; AD finite. (`test_kpp_bldepth.py`.)
- [x] **K.6 — `blmix`: BL coeffs + cubic shape + `dkm1` + `ghats`.** ✅ `kpp.blmix` (`:449-579`):
  per-node scalars (kn stop-grad, one-sided slope `max(dvdz,0)`, `gat1/dat1` w/ `dat1≤0`,
  `f1=bfsfc/(u*⁴+δ)`) → cubic `G(σ)` over `[nzmin+1,kbl-1]` → `blmc[3]`; `dkm1` at `kbl-1` (σ from
  zbar); `ghats` (computed, gated off). hnode passed in (static linfs ⇒ GM-dump hnode for replay).
  **Gate (controlled replay):** blmc **1.9–3.0e-13** (C hit 3.18e-13), dkm1 2.6e-14, ghats rel ~3.7e-14
  (huge-range → relative gate); AD finite. (`test_kpp_blmix.py`.)
- [x] **K.7 — `enhance` + `smooth_blmc` + combine + node→elem.** ✅ `kpp.enhance` (blend at `kbl-1`,
  masked single-interface update) + `kpp.assemble_mixing` (`smooth_blmc`=`eos.smooth_nod3D(·,3)`;
  combine `max(interior,blmc)` within BL + zero `ghats` below; node→elem `Av` + bottom-fill + `minmix`;
  `Kv=diffKt`). **Gate (controlled replay):** viscA/viscAE **2.2e-16**, diffKt/diffKs **6.7e-16**,
  ghats exact, Kv==diffKt; AD finite. **K.1→K.7 = the complete KPP forward chain.** (`test_kpp_enhance.py`.)
- [ ] **K.8 — wire KPP into the step (single Kv; nonlocal GATED OFF).** Gate `step.py:130`:
  `if kpp_cfg is not None: Kv,Av = kpp.mixing_kpp(…)` else the PP path. Thread `heat_flux/water_flux/
  stress_node_surf/sw_3d/sw_alpha/sw_beta/dbsfc/uvnode` to the call; `mo_convect` after (shared).
  `Kv→tracer diff (+GM K33)`, `Av→momentum`. **Gate:** PP byte-identical when `kpp_cfg=None`
  (regression); assembled step-1 `Kv/Av/hbl` vs C KPP dump = the expected forcing-transient sanity
  match.
- [ ] **K.9 — end-to-end climate + stability.** Assembled CORE2 **KPP+GM+ice** forward (extend
  `core2_gm_stability_run.py` with `--mixing kpp`, or a `core2_kpp_stability_run.py`): N-day stable;
  SST/SSS climate **matches the C KPP** and is **distinct from JAX PP** (the discriminating check, §3).
  GPU job (mirror `core2_gm_stability_gpu.sh`). **Gate:** stable + climate ≈ C KPP ≪ PP↔KPP gap.
- [ ] **K.10 — AD-safety gradient gate + acceptance (GATE 6C AD half).** Masked-NaN gradient gate on
  the assembled KPP model (`d(loss)/d(T0)` finite everywhere incl. masked lanes — mirror
  `core2_gm_grad_gate.py`, add `--mixing kpp`) + a well-conditioned KPP-tunable gradient
  (`d/d(visc_sh_limit)` or `d/d(K_bg)`, additive). **Gate:** `KPP_GRAD_GATE_OK` (finite/nonzero,
  masked-NaN clean) + suite green.
- [ ] **K.11 — docs + memory + commit + next-session.** Tick this plan; Revision Log; per-task lessons
  in `PORTING_LESSONS.md`; update parent-plan Phase 6C → COMPLETE; refresh memory; write the next
  `NEXT_SESSION_PROMPT.md` (→ Phase 7a parameter tuning, the preserved `…-paramtune.md` plan).

**Compute notes (same as GM):** heavy / full-suite / any CORE2 BACKWARD → `sbatch` (compute node) or
a GPU job — the login node hangs on CORE2 backprop (RAM thrash). Quick CPU forward smokes (≤ few steps)
+ the per-kernel replay gates (small, kernel-isolated) run on the login node. GPU via SLURM
`-A ab0995_gpu -p gpu --gres=gpu:1`; stream forcing per step (don't stack a long trajectory → OOM);
one N-step backward per process (`jax.clear_caches()`).

---

## 6. GATE 6C (acceptance)

**Forward fidelity (the headline):** KPP selectable via `kpp_cfg`; **PP byte-identical when
`kpp_cfg=None`** (suite green); **every ported kernel K.2–K.7 controlled-replay bit-faithful to ~1e-12
vs the C** (`wscale`, `ri_iwmix`, `bldepth`, `blmix`, `enhance`/assembly); assembled CORE2
**KPP+GM+ice** stable over a multi-day run; **end-to-end climate matches the C KPP** (SST/SSS RMS
~0.005–0.013 °C class) and is **distinct from the JAX PP** trajectory by the genuine scheme difference.
**Differentiability (the standing rule):** masked-NaN-clean `d(loss)/d(T0)` through the assembled KPP
model + a well-conditioned KPP-tunable gradient (additive `visc_sh_limit`/`K_bg`). **Then the full
functioning model is complete** → Phase 7a (differentiable parameter tuning,
`docs/plans/20260607-fesom-jax-paramtune.md`) becomes the next phase, with KPP's own constants (`Ricr`,
`visc_sh_limit`, the background diffusivities) as additional mixing-seam tuning targets.

---

## Revision Log

- **2026-06-07 — K.6–K.7 DONE (blmix + enhance/assembly) → the complete KPP forward chain.** `kpp.blmix`
  matched the C's hardest replay (blmc 1.9–3.0e-13, dkm1 2.6e-14, ghats rel ~3.7e-14); `kpp.enhance` +
  `kpp.assemble_mixing` (smooth_blmc 3-sweep + combine + node→elem Av) BIT-EXACT (viscA/viscAE 2.2e-16,
  diffKt/diffKs 6.7e-16, ghats exact, Kv==diffKt). 8 more tests green (`test_kpp_blmix.py`,
  `test_kpp_enhance.py`) — **22 KPP tests total, K.1→K.7 all controlled-replay bit-faithful + AD-finite.**
  All gateable with the existing dumps (no new C edits). **Next: K.8 — wire `kpp.mixing_kpp` into
  `step.py`** (assemble blmix→enhance→combine into a single driver, thread the forcing/dbsfc inputs;
  `Kv→tracer diff +GM K33`, `Av→momentum`; PP byte-identical when `kpp_cfg=None`), then K.9 climate +
  K.10 grad gate (both need SLURM GPU/compute jobs — a fresh-session boundary).

- **2026-06-07 — K.1–K.5 DONE (tables → wscale → ri_iwmix → ddmix gate → pre-step/dbsfc/bldepth).**
  All controlled-replay dump-gated vs the C + AD-finite, 14 new tests green (`test_kpp_wscale.py`,
  `test_kpp_ri_iwmix.py`, `test_kpp_bldepth.py`). Highlights: the wm/ws table + 4 scalars bit-match
  (1.7e-18); wscale forward bit-faithful (4e-13, the `ufrac` beyond-UMAX extrapolation subtlety) + AD
  finite incl. ustar=0; ri_iwmix BIT-EXACT (max|Δ|=0, shear=0 step-1) + synthetic cubic frit; ddmix a
  no-op gate; **bldepth (the highest-risk OBL search) matched the C on the FIRST try — hbl 6.5e-12, kbl
  0/126858 mismatches, bfsfc 8.6e-23, stable/caseA exact** (two masked first-crossings, stop-grad kbl +
  differentiable hbl interp); `eos.compute_dbsfc` bit-exact (the one missing EOS input). The C KPP dump
  was extended (jax-mesh-export) with `sw_3d`+`sw_alpha` for the bldepth replay (rebuilt + rerun).
  **Next: K.6 `blmix`** (BL coeffs + cubic shape + dkm1 + ghats — the C's hardest replay, hit 3.18e-13).

- **2026-06-07 — K.0 DONE (scaffolding + reference dumps).** `fesom_jax/kpp.py` + `KppConfig` (all §1
  constants + the 4 derived scalars, bit-matching the C dump). `kpp_cfg=None` threaded through
  `step.py`/`integrate.py` exactly like `gm_cfg` — bit-identical regression verified locally
  (`kpp_cfg=None ⇔ KppConfig()`, max|Δ|=0 through `step_jit` + the checkpointed `integrate`). The C KPP
  reference dump generated (`jax_kpp_dump_core2.sh`: KPP / GM-OFF / ice-OFF, single-rank, 57 s, ~890 MB
  text) and the `io_dump` readers added + validated against the real data: init scalars bit-match
  `KppConfig`, the 892×480 wm/ws table recomputes at max|Δ|=0 (pre-validates K.1), the wscale sweep +
  per-kernel tags load with single-rank gid-identity (JAX node `i` ↔ gid `i+1`), sane values
  (ustar≥0, hbl∈[0.04,790] m, caseA/stable∈{0,1}). Full-suite green pending the compute-node job.
  Lessons appended (`PORTING_LESSONS.md`, "Phase 6C — KPP (Task K.0)").

- **2026-06-07 — Created.** Phase 6C (KPP) sub-plan written from a full research pass: `fesom_kpp.c`
  (1046 lines) + `.h` algorithmic breakdown; the mixing seam on both sides (`pp.py`→`step.py`→solvers;
  the `gm_cfg`/`ice_cfg` gate pattern); the CORE2 KPP reference namelists + the **completed/validated**
  C KPP port (`port2/.../20260524-kpp-vertical-mixing.md`, K0–K11, climate RMS 0.005–0.013 °C). Key
  framings established: **KPP is the real CORE2 default** (JAX currently runs the opt-in PP); **controlled
  replay** is the load-bearing validation technique (live-run diffs at ~52 % of nodes from the step-1
  forcing transient); the **AD-kink inventory + treatments** (safe-sqrt `ustar`, `stop_gradient` the
  discrete `kbl`/bin-index with a differentiable `hbl` interp, meaningful floors over `EPSLN=1e-40`);
  `dbsfc` is the one missing EOS input to add. Task ladder K.0–K.11 mirrors the proven C decomposition.
