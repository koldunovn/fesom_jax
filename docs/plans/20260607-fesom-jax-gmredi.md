# FESOM2 → JAX Port — Phase 6B: GM/Redi (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 6 outline — GM/Redi = 6B).
**Predecessors:** `docs/plans/20260606-fesom-jax-core2.md` (Phase 5 ocean, GATE 5) +
`docs/plans/20260606-fesom-jax-phase6-seaice.md` (sea ice, GATE 6).
**Created:** 2026-06-07. **Status:** DRAFT (no tasks started).
**Scope (user-confirmed 2026-06-07):** **GM/Redi only** (the mesoscale eddy parameterization).
KPP is the separate Phase-6C sub-plan. **Decisions (user-confirmed):** (1) **thread the GM/Redi
eddy diffusivities through `params.py` now** — the parent plan's 2nd ML-hook seam, default=config
so bit-identical (exactly like `k_ver`/`a_ver`); Phase 7 swaps in the NN. (2) **7-task ladder in
data-flow order**, each kernel dump-gated bit-exact (mirrors the ice plan).

---

## 0. Scope (READ FIRST — what the C GM/Redi port actually is)

Phase 6B ports the FESOM2 **mesoscale eddy parameterization** on the CORE2 mesh, on top of the
completed Phase-5 ocean + Phase-6 sea ice. The algorithmic source of truth is the C port's
`fesom_gm.c` (1077 lines) + its integration in `fesom_step.c` / `fesom_ale.c` /
`fesom_tracer_diff.c` / `fesom_eos.c` (all read in full this session). As everywhere, the C is a
**deliberately reduced** model — only the default-namelist branches are live; match THAT.

**What GM/Redi IS** = two coupled eddy effects, both functions of the current density field:
- **GM (Gent–McWilliams)** — an eddy-induced **bolus advection**: a streamfunction `Γ` solved per
  column from the neutral slopes, reconstructed into a bolus velocity `fer_uv` (element) +
  `fer_w` (node), which **augments the advecting velocity** of the tracers (only — it is added
  before tracer advection and subtracted back after, so momentum never sees it).
- **Redi** — **neutral (isopycnal) diffusion**: rotates the tracer diffusion tensor along neutral
  surfaces. Enters as two **explicit** flux terms (vertical-projection G7a + horizontal-edge G7b)
  plus an **implicit K33** diagonal augmentation of the vertical tracer diffusivity.

**⚠️ Key structural finding — GM/Redi is NOT one substep-14 kernel; it threads the step at 6
points** (the parent plan's "substep 14 gm_bolus" is the *bolus reconstruction* only):

| # | Where in the step | What | C site |
|---|---|---|---|
| 1 | after EOS (substep 2) | **`sw_alpha_beta`** — McDougall(1987) α/β (deferred from Phase 2; GM+KPP read it) | `fesom_eos.c:323-375` |
| 2 | post-EOS "gm_bolus" block | `compute_sigma_xy → compute_neutral_slope → init_redi_gm → fer_solve_gamma → fer_gamma2vel` → **`fer_uv`** | `fesom_step.c:118-130`, `fesom_gm.c` |
| 3 | ALE `compute_w` | **`fer_w`** bolus vertical velocity, same edge scatter as `w` but driven by `fer_uv` | `fesom_ale.c:91-171` |
| 4 | around tracer adv/diff | **bolus wrap**: `uv += fer_uv; w,w_e += fer_w` before advection, `-=` after diffusion | `fesom_step.c:312-332, 416-434` |
| 5 | tracer solve, per T & S | **Redi explicit**: `diff_ver_part_redi_expl` (G7a) + `diff_part_hor_redi` (G7b, 5 branches) | `fesom_step.c:344-356`, `fesom_gm.c:646-1022` |
| 6 | implicit vert diffusion | **Redi K33** diagonal augmentation `slope_tapered²·Ki` added to the TDMA | `fesom_tracer_diff.c:167-246` |

**⚠️ GM/Redi is STATELESS** — every field (`sigma_xy`, `neutral_slope`, `slope_tapered`,
`fer_tapfac`, `fer_K`, `Ki`, `fer_C`, `fer_scal`, `fer_gamma`, `fer_uv`, `fer_w`, `tr_xy`,
`tr_z`) is **recomputed each step** from the current T/S/N². **No new `State` fields** (unlike
ice's prognostic σ). It's a pure diagnostic of the density state → simpler to thread and to AD.

**Mixing-scheme independence:** GM reads only `bvfreq` + α/β (EOS), **never** `Kv`/`Av`. So the GM
dump runs on the **same PP path** the JAX port uses (`FESOM_MIX_SCHEME=PP`); KPP is the C default
but is Phase 6C. The Redi K33 *adds* to whatever `Kv` the mixing scheme produced (here PP's).

**What is ABSENT (out of scope — the default namelist disables it):** `scaling_LDD97` (c2≡1),
`scaling_Rossby`, `FESOM14`, `GINsea`, `Ferreira`/MLD-reference (`K_GM_bvref`), `K_GM_Ktaper`,
`K_GM_rampmax/min`, the `K_hor` Redi-without-GM branch, cavities (`ulevels>1` skip),
partial cells (full-cell linfs: `zbar_n=zbar`, `Z_n=Z`). Active sub-features only:
`Fer_GM=Redi=Redi_Ktaper=scaling_ODM95=scaling_resolution=scaling_GMzexp=T`,
`K_GM_resscalorder=2`.

**ML-hook note (decided — thread now):** GM/Redi is the parent plan's **2nd ML-hook seam**
(eddy fluxes). Add the eddy diffusivities to `params.py` as traced leaves — `k_gm`
(=`K_GM_max`, the GM thickness diffusivity ceiling) and `redi_kmax` (=`Redi_Kmax`, the Redi
ceiling; auto-synced to `k_gm` in the C). Default = the config constants (1000 m²/s) ⇒
numerically transparent (the 423-test suite stays bit-identical). The GATE-6B gradient gate adds
`d(loss)/d(k_gm)` as the 2nd hook's gradient-path proof (exactly as `d/d(k_ver)` proved the
mixing hook in Phase 3). The richer spatial NN (predicting `fer_uv`/slopes/spatial K) is Phase 7;
this establishes the scalar seam + proves the gradient flows.

## 1. Reference path — Path A (GM-ON dump + a new `fesom_gm_dump` hook)

Exactly as Phases 0/5/6: a per-substep C-port dump at the JAX-matched config, so JAX↔C diffs are
pure FP reassociation (the tightest gate).

- **Config switch:** drop `FESOM_NO_GMREDI=1` (→ GM active), keep `FESOM_MIX_SCHEME=PP`, and turn
  **ice OFF** (`FESOM_NO_ICE_*`) for the GM kernel dumps — GM is ice-independent, and an ice-off
  T/S trajectory is the one the GM gate reproduces in isolation. (G.7 then confirms GM+ice
  together.) `FESOM_BULK_FIXED_ITERS=1` stays on (Phase-5 finding).
- **New additive C hook `fesom_gm_dump`** (port2 branch `jax-mesh-export`, env-gated
  `FESOM_GM_DUMP_DIR`), `fesom_bulk_dump`-style: re-runs the GM block on copies → **all-node /
  all-element** inputs+outputs for the GM intermediates the per-substep `fesom_dump.c` doesn't
  carry: `sw_alpha`, `sw_beta`, `sigma_xy`, `neutral_slope`, `slope_tapered`, `fer_tapfac`,
  `fer_K`, `Ki`, `fer_C`, `fer_scal`, `fer_gamma`, `fer_uv`, `fer_w`, `tr_xy`, `tr_z`. All-node
  dumps verified bit-exact over all 126858 nodes (the ice-thermo precedent) — no probe luck.
- **Optional 1-line staging knob `FESOM_GM_BOLUS_ONLY`** (skip G7a/G7b/K33) → a bolus-only T/S
  dump that isolates G.5 from the Redi terms (mirrors the ice `FESOM_NO_ICE_*` staging). The full
  post-Redi T/S match is the G.6 gate.
- **Probe re-pin** (env-only `FESOM_DUMP_PROBES`, no C edit) for GM-active regions: a strong-slope
  baroclinic node (western boundary current / ACC), a weakly-stratified node (taper ≈ 0), a
  deep node (the F2 `exp(-|z|/zref)` depth scaling), a level-mismatch edge (the G7b A/B/D/E
  branches), and a dry/masked node (the AD masked-NaN probe).

C-side discipline: **C edits → port2 `jax-mesh-export`, NEVER port2 main**; job scripts untracked.
Cheap dumps → `-p compute --time=00:30:00`. New data → `/work` (the `data` symlink); cache under
`data/gm_dump_core2/`.

## 2. Verification ladder (unchanged classes)

Per-kernel probe-column / all-node dump, truncate to `nlevels`, `verify.assert_close(col, rec,
kind=…)`: **map/gather 1e-15, scatter/reduction 1e-12** (calibrate `atol`). Class map for GM:
- **map-class (~1e-15, often bit-exact):** `sw_alpha_beta` (pure per-node polynomial, like
  `density`); the `fer_gamma2vel` interface-difference; the K33 augmentation.
- **scatter/reduction-class (~1e-12):** `sigma_xy` (element→node area-weighted gradient),
  `fer_w` (edge→node + cumsum), G7a (element→node), G7b (edge→node, the 5 branches).
- **TDMA-class:** `fer_solve_gamma` (per-node Thomas sweep, reuse `ops.tdma` — sequential, near
  map-class).

**Re-run the gradient gate at GATE 6B.** New AD surfaces: the slope `√(sx²+sy²)` + `√c1` +
`√bv` + `√tapfac` (safe-sqrt), the `tanh` taper (smooth), the `|√tapfac−1|` + `min`/`max` clips
(subgradient), the `fer_solve_gamma` TDMA (grad-verified primitive), and every `1/(z-diff)`,
`1/helem`, `1/(areasvol·hnode_new)` safe-divide on masked lanes. **AD rule (bit us 5× already):**
any divide/sqrt whose denom/arg can vanish in a masked (dry / weakly-stratified) lane must compute
a FINITE value (`where(d==0,1,d)` / double-`where` safe-sqrt) — a forward `where` does NOT stop a
`0·inf` backward NaN. **Plus the new 2nd-hook target:** `d(loss)/d(k_gm)` FD↔AD plateau in a
smooth regime.

## 3. Config (the CORE2 GM-ON reference run)

Everything in Phase-5 §3 + Phase-6 §3 (PP/linfs/FCT/opt_visc7, dt=500, PHC IC, JRA55+SSS+runoff,
full-cell, CG α=1, sea ice), **plus** GM/Redi (default `namelist.oce`, active branches only;
constants cross-checked in `fesom_gm.c`):

- **Master:** `Fer_GM=T`, `Redi=T`, `Redi_Ktaper=T`, `isredi=1`.
- **Diffusivities:** `K_GM_max=1000`, `K_GM_min=2`, `Redi_Kmin=100`, `Redi_Kmax=K_GM_max=1000`
  (auto-sync), `K_GM_cmin=0.1`, `K_GM_cm=3`.
- **Resolution scaling:** `K_GM_resscalorder=2` (real ⇒ `/2`=0.5 exponent → `√`),
  `scaling_resolution=T`, `refscalresol=100 km` (`inv²=1e-10`).
- **Depth scaling:** `scaling_GMzexp=T`, `GMzexp_zref=500`, `GMzexp_smin=0.6`.
- **Slope tapering:** `scaling_ODM95=T` (c1 tanh), `scaling_LDD97=F` (c2≡1); `ODM95_Scr=0.2e-2`,
  `ODM95_Sd=1.0e-3`; slope `eps=5e-6` (`eps²=2.5e-11`).
- **Floors:** `bvfreq` floor in the Γ solve = `1e-8`; constants `g=9.81`, `ρ0=1030`, `π` (truncated).

---

## Implementation Steps

> **Module layout:** new `fesom_jax/gm.py` (the coefficient + bolus pipeline: `sw_alpha_beta` is
> already eos territory; `sigma_xy`/`neutral_slope`/`init_redi_gm`/`fer_solve_gamma`/
> `fer_gamma2vel`) and `fesom_jax/gm_redi.py` (the tracer-side G7a/G7b + the K33 helper).
> `GMConfig` (static constants) + the `params.py` extension. Wired into `step.py`/`integrate.py`
> behind a static `gm_cfg=None` arg, mirroring `ice_cfg` (None ⇒ pi/Phase-5/ice paths
> bit-identical). **STANDING RULE: append a lesson per task to `docs/PORTING_LESSONS.md`.**

### Task G.1: `sw_alpha_beta` + `GMConfig` + the `params.py` seam + the GM dump hook

**Files:** modify `fesom_jax/eos.py` (add `compute_sw_alpha_beta`), `fesom_jax/params.py` (add
`k_gm`/`redi_kmax`), create `fesom_jax/gm.py` (`GMConfig` skeleton). C (`port2`,
`jax-mesh-export`): the NEW `fesom_gm_dump` hook (`FESOM_GM_DUMP_DIR`) + a `jobs/jax_gm_dump_core2.sh`
(GM-ON, ice-OFF, PP). Create `tests/test_sw_alpha_beta.py`.

> **✅ DONE 2026-06-07.** `eos.compute_sw_alpha_beta` ported (verbatim McDougall, ⚠️ `Z` is
> `[nl-1]` → pad to `[nl]` like `pressure_bv`); **bit-exact vs the C GM-ON dump** (max|Δ|=**0.0**
> for both sw_alpha/sw_beta over all 3.7M wet lanes / 126858 CORE2 nodes — a pure pointwise map,
> like `density`), + bit-exact vs an independent numpy transcription (<1e-18 blob IC, <1e-15
> synthetic varied-S exercising the `s35` terms) + physical-range + below-bottom-masked + AD-finite
> (`d/d(T)`,`d/d(S)` finite everywhere, 0 on masked). `params.py` += `k_gm`/`redi_kmax` (the 2nd
> ML-hook seam; **defaults so the old `Params(k_ver=,a_ver=)` 2-arg construction is unchanged** —
> verified). `gm.GMConfig` skeleton (the §3 static bundle). `config.py` += `K_GM_MAX`/`REDI_KMAX`.
> **The C `fesom_gm_dump` hook** (`fesom_step.c`, env-gated `FESOM_GM_DUMP_DIR`, all-node/element,
> stateless snapshot) + `jobs/jax_gm_dump_core2.sh` (GM-ON, ice-OFF, PP; job 25397273, 31 s) →
> `data/gm_dump_core2/` (T/S/bvfreq/hnode/hnode_new + all GM outputs; **seeds G.2-G.4**). The
> reader is `io_dump.load_gm_dump`. **Bit-identity preserved** (test_gradient+integrate green, AD
> plateau unchanged). `test_sw_alpha_beta.py` = **6 passed**.

- [x] **`sw_alpha_beta`** (`fesom_eos.c:323-375`, McDougall 1987): per node/level, `t1=T·1.00024`,
  `s1=S`, `p1=|Z[nz]|` (linfs full-cell ⇒ `Z_3d_n=Z`; ⚠️ `mesh.Z` is `[nl-1]` → pad to `[nl]`);
  the two term-by-term polynomials `beta` (10 terms) and `a_over_b` (11 terms); outputs
  `sw_beta=beta`, `sw_alpha=a_over_b·beta`. Pure per-node MAP (bit-exact vs the numpy ref). AD:
  smooth, no guards. Masked to `node_layer_mask`.
- [x] **`GMConfig`** (NamedTuple of the §3 constants). Static, closed over the step; separate from
  `params.py`. (Kernel fns land G.2-G.4.)
- [x] **`params.py` seam:** added `k_gm` (=`K_GM_max`) + `redi_kmax` (=`Redi_Kmax`) leaves
  (defaults = config); `Params.defaults()` → 4 leaves. The GM coefficient builder (G.3) reads
  these so `d/d(k_gm)` flows. **Bit-identical** (17-test seam check green).
- [x] **C hook `fesom_gm_dump`** (`fesom_step.c`, env-gated `FESOM_GM_DUMP_DIR`, all-node/element;
  reads the already-computed `gm->*`/`dyn->fer_uv`/`aux->sw_*` arrays after the GM block — GM is
  stateless, no re-run needed). Built clean on `jax-mesh-export`; `jobs/jax_gm_dump_core2.sh`
  (GM-ON, ice-OFF, PP) → `data/gm_dump_core2/` (job 25397273, N=126858/E=244659/nl=48). Reader
  `io_dump.load_gm_dump`. **Dumps the INPUTS too** (T/S/bvfreq/hnode/hnode_new) ⇒ seeds G.2-G.4.
- [x] **Probe coverage:** the all-node dump (no probe luck — like the ice-thermo precedent)
  supersedes a probe re-pin for the per-node/element GM fields; the per-substep `FESOM_DUMP_FILE`
  (probes) is kept for the G.6/G.7 post-Redi T/S.
- [x] **Gate:** `sw_alpha`/`sw_beta` **bit-exact (max|Δ|=0)** vs the dump over all wet lanes; pi +
  Phase-5 + ice paths bit-identical (`gm_cfg=None`, `params` GM-leaves default).
- [x] **AD:** `d(Σsw_alpha)/d(T)` / `d/d(S)` finite everywhere incl. masked lanes (the masked-NaN
  baseline).
- [x] run — **DONE** (test_sw_alpha_beta 6 passed; ocean suite re-run for the params-seam change).
  **Lesson:** appended (the α/β bit-exact map + the `Z`-is-`nl-1` padding trap + the
  params-seam-defaults backward-compat + the stateless-GM-dump-snapshot pattern).

### Task G.2: Neutral slopes — `compute_sigma_xy` + `compute_neutral_slope`

**Files:** `fesom_jax/gm.py`; `tests/test_gm_slopes.py`. C: gated by the G.1 dump.

> The density-gradient → neutral-slope → ODM95-taper chain. `sigma_xy` is the first GM
> element→node area-weighted gradient (reuse the `nod_in_elem2D` CSR / `gradient_sca` pattern
> from the bvfreq smoother + momentum advection).

> **✅ DONE 2026-06-07.** `gm.compute_sigma_xy` (the smoother-style element→node area-weighted
> ∇T/∇S scatter, ÷Σarea, one (E,nl,5) scatter for tx/ty/sx/sy/vol) + `gm.compute_neutral_slope`
> (ro_z_inv from N², the ODM95 tanh taper + the `bv≤0` where-mask, double-`where` safe-sqrt on
> `|s|` and `√c1`). Verified vs the G.1 GM-ON dump (all-node, CORE2): **`sigma_xy` bit-exact**
> (max|Δ|=0, eager), **`neutral_slope` map-class** (eager bit-exact; ⚠️ huge dynamic range to
> ~1e5-1e6 where N²→the floor ⇒ gated RELATIVE rtol=1e-12), **`slope_tapered`** within an FMA
> floor (atol 1e-9 — the `huge×taper→0 ≈ 0` lane carries the huge factor's FMA noise; eager
> bit-exact), **`fer_tapfac`** 3.9e-16. AD `d(Σslope_tapered)/d(T)` finite + nonzero.
> `test_gm_slopes.py` = **4 passed**. 3 lessons appended.

- [x] **`compute_sigma_xy`** (`fesom_gm.c:124-202`): per node/level, area-weighted mean of the
  per-element `∇T`/`∇S` (`gradient_sca` 6-pack: `[0..2]=∂N/∂x`, `[3..5]=∂N/∂y`) over surrounding
  elements (÷`Σ elem_area`), then `sigma_xy[c] = (-α·∇_c T + β·∇_c S)·ρ0`, c∈{x,y}. el-range ⊆
  node-range ⇒ `elem_layer_mask` suffices. **Bit-exact** vs the dump (eager). `inv_vol = where(vol>0,1/vol,0)`.
- [x] **`compute_neutral_slope`** (`fesom_gm.c:223-310`): per node/level,
  `bv_sum=bvfreq[nz]+bvfreq[nz+1]`, `denom=max(bv_sum, eps²=2.5e-11)`, `ro_z_inv=2g/ρ0/denom`;
  `sx,sy = sigma_xy·ro_z_inv`; `sm=√(sx²+sy²)` (**safe-sqrt**). ODM95 `c1=0.5(1+tanh((Scr−sm)/Sd))`,
  forced 0 where `bvfreq[nz]≤0 ∨ bvfreq[nz+1]≤0` (**where-mask**). `fer_tapfac=c1`;
  `slope_tapered = neutral_slope·√c1` (**safe-sqrt on c1**). `neutral_slope=[sx,sy,sm]`.
- [x] ⚠️ **AD-safe guards:** `denom` clamp (already a `max`); the `sm` safe-sqrt; the `√c1`
  safe-sqrt; the `bv≤0` where-mask. All masked/dry lanes finite (AD test green).
- [x] **Gate:** `sigma_xy` (2c, bit-exact), `neutral_slope` (3c, relative map class — huge
  dynamic range), `slope_tapered` (3c, FMA floor atol 1e-9), `fer_tapfac` (3.9e-16) vs the G.1
  dump at all nodes.
- [x] **AD:** `d(Σslope_tapered)/d(T)` finite incl. weakly-stratified + masked lanes; nonzero on wet.
- [x] run — **DONE** (test_gm_slopes 4 passed). **Lesson:** appended (the area-weighted gradient,
  the huge-dynamic-range relative gate, the FMA-floor-on-huge×tiny + the eager-bit-exact/jit-FMA split).

### Task G.3: GM/Redi coefficients — `init_redi_gm`

**Files:** `fesom_jax/gm.py`; `tests/test_gm_coeffs.py`. C: G.1 dump.

> The per-step coefficient builder: `fer_K` (GM thickness diff), `Ki` (Redi diff), `fer_C`
> (baroclinic-wave-speed²), `fer_scal` (resolution scaling). Two passes (F1 horizontal scalar,
> F2 vertical/level). **This is where `params.k_gm`/`redi_kmax` enter** (the ML-seam).

> **✅ DONE 2026-06-07.** `gm.init_redi_gm(mesh, bvfreq, hnode_new, fer_tapfac, params, cfg)`.
> Verified vs the G.1 dump (all-node): **`fer_K`** (iface range) map-class (max|Δ|=1.1e-13 @ scale
> 1000), **`Ki`** (layer range) map-class (2.3e-13 @ 990), **`fer_C`** 2e-15 (the cm depth
> reduction), **`fer_scal`** bit-exact (0.0). ⚠️ F1 uses CONSERVATIVE bounds, F2 REGULAR; `fer_K`
> on `node_iface_mask`, `Ki` on `node_layer_mask`. **2nd ML-hook LIVE:** `d(Σfer_K)/d(k_gm)=2.03e6`
> (finite, positive — the eddy-flux gradient path proven, like `k_ver` for mixing). AD finite
> (`d(ΣKi)/d(bvfreq)`, `d(Σfer_C)/d(hnode_new)`). `test_gm_coeffs.py` = **5 passed**. 2 lessons.

- [x] **Pass 1 — F1** (`fesom_gm.c:371-414`), conservative bounds: `cm = max(Σ hnode_new·0.5(√bv0+
  √bv1)/π/K_GM_cm, K_GM_cmin)` (safe-sqrt); `scaling = min(√(area·2·inv_refscalresol²), 1)`;
  `fer_scal=scaling`; `fer_K_top=max(scaling·k_gm, K_GM_min)`; `fer_C=cm²`;
  `Ki_top=max(scaling·redi_kmax, K_GM_min)`. (`k_gm`/`redi_kmax` from `params`.)
- [x] **Pass 3 — F2** (`fesom_gm.c:416-460`), regular bounds: `zscaling=clip(smin+(1−smin)·
  exp(−|zbar_3d_n|/zref), smin, 1)`; `fer_K=fer_K_top·zscaling` (iface mask);
  `Ki=Ki_top·0.5(zscaling[nz]+zscaling[nz+1])` (layer); **Redi_Ktaper**
  `Ki = Ki·√tapfac + Redi_Kmin·|√tapfac−1|` (safe-sqrt + subgradient abs).
- [x] ⚠️ **AD-safe guards:** `√bv`, `√tapfac`, `√area` (safe-sqrt); `min`/`max` subgradient;
  `|√tapfac−1|` subgradient. F1/F3 level-bound masks separate (`cons_mask` vs node iface/layer).
- [x] **Gate:** `fer_K`/`Ki`/`fer_C`/`fer_scal` map-class vs the dump; `d(Σfer_K)/d(k_gm)=2.03e6`
  (the seam live + correct).
- [x] **AD:** masked-NaN finite (`√bv` at bv=0, `√tapfac` at tapfac=0); `d(Σfer_K)/d(k_gm)` finite.
- [x] run — **DONE** (test_gm_coeffs 5 passed). **Lesson:** appended (the F1/F2 conservative-vs-
  regular bounds, the iface-vs-layer fer_K/Ki, the live 2nd ML-hook gradient).

### Task G.4: Streamfunction + bolus velocity — `fer_solve_gamma` + `fer_gamma2vel`

**Files:** `fesom_jax/gm.py`; `tests/test_gm_bolus.py`. C: G.1 dump.

> The GM core: a per-node TDMA for the streamfunction `Γ` (`∂z(C·∂z Γ) − N²·Γ = (g/ρ₀)·∇σ·K_GM`),
> 2 components sharing the matrix; then the element bolus velocity. Reuse `ops.tdma` (grad-verified).

> **✅ DONE 2026-06-07.** `gm.fer_solve_gamma` + `gm.fer_gamma2vel`. ⚠️ Full-cell linfs ⇒
> `zbar_n=zbar`, `Z_n=Z` are STATIC (verified `hnode_new == zbar thickness`, max|Δ|=0), so the
> tridiagonal geometry is a precomputed constant × `fer_C`; the 2 components share `(a,b,c)` (two
> `ops.tdma` calls); body on the conservative inner bounds, Dirichlet/padding rows `b=1` →
> full-column `ops.tdma` reproduces the C's bounded Thomas. Verified vs the G.1 dump (all-node):
> **`fer_gamma` ~8.9e-15** (sequential Thomas ≈ bit-exact), **`fer_uv` ~1.1e-16**, chained ~2e-16.
> AD through the TDMA finite + nonzero. `test_gm_bolus.py` = **4 passed**. 2 lessons. **The whole
> GM coefficient+bolus pipeline (G.1-G.4) is now done** — `fer_uv` from T/S.

- [x] **`fer_solve_gamma`** (`fesom_gm.c:492-612`): static `zbar_n=zbar`/`Z_n=Z` geometry; tridiag
  on the inner (conservative) bounds — Dirichlet `b=1` at endpoints, body `a=fc·zinv_if[nz-1]·
  zinv_mid`, `c=fc·zinv_if[nz]·zinv_mid`, `b=−a−c−max(bvfreq,1e-8)`; RHS `tr=(g/ρ0)·0.5(σ_up+σ_dn)·
  fer_K`. Both components via two `ops.tdma` (shared matrix). Γ=0 on degenerate/below-bottom.
- [x] **`fer_gamma2vel`** (`fesom_gm.c:1035-1077`): `fer_uv[c,nz,el]=(1/3)·Σ_v(Γ[c,nz,v]−
  Γ[c,nz+1,v])/helem` (gather Γ→vertices, interface-difference, safe ÷helem). `helem=⅓Σ_v hnode`.
- [x] ⚠️ **AD-safe guards:** `bvfreq` floor; the static geometry (finite); the `1/helem` guard.
  TDMA grad is the ops.tdma-verified primitive.
- [x] **Gate:** `fer_gamma` ~8.9e-15 + `fer_uv` ~1.1e-16 vs the dump (Thomas≈bit-exact, gather).
- [x] **AD:** `d(Σfer_uv²)/d(sigma_xy)` through the TDMA finite + nonzero.
- [x] run — **DONE** (test_gm_bolus 4 passed). **Lesson:** appended (the static-geometry TDMA, the
  shared-matrix 2-component solve, the gamma2vel ÷helem).

### Task G.5: Bolus advection wiring + `fer_w`

**Files:** modify `fesom_jax/ale.py` (add `fer_w` to `compute_w`), `fesom_jax/step.py` (the bolus
wrap). `tests/test_gm_bolus_adv.py`. C: G.1 dump + the optional `FESOM_GM_BOLUS_ONLY` T/S dump.

> Wires the GM bolus into the tracer transport. In JAX (functional, no in-place add/sub) the
> C's "`uv += fer_uv` … `uv −= fer_uv`" becomes simply *passing the augmented velocity into
> tracer advection* — the original `uv`/`w_e` are untouched, so the subtract-back is automatic.

> **✅ DONE 2026-06-07.** ⚠️ **`fer_w = ale.compute_w(fer_uv)` — a PURE REUSE** (the C's fer_w uses
> the byte-identical scatter+cumsum+÷area as `w`, just driven by fer_uv; verified by composition +
> the no-flux BC + activity ~3e-4). The new piece is `gm.gm_diagnostics` (the GM driver composing
> G.1-G.4 from the state → `GMDiag(fer_uv, slope_tapered, Ki, fer_K, fer_C)`): fed the C's
> T/S/bvfreq/hnode_new it reproduces **`fer_uv` END-TO-END at 2.2e-16** (essentially bit-exact, the
> whole chain all-node). The bolus wrap = pass `uv+fer_uv`, `w_e+fer_w` to the advection (no
> subtract). `test_gm_bolus_adv.py` = **3 passed**. 2 lessons. ⚠️ The step.py integration (the
> `gm_cfg` arg) + the tight bolus-effect-on-T/S gate move to **G.7** (the C dump has bolus+Redi
> together; the assembled full-GM step gate covers both). The FCT vertical uses `w_e` (not `w`).

- [x] **`fer_w`** = `ale.compute_w(mesh, fer_uv, helem)` — pure reuse (the C ÷area's fer_w like w,
  `fesom_ale.c:181-183`). No new kernel; verified by composition + BC + activity.
- [x] **Bolus wrap** = `uv_adv=uv+fer_uv`, `w_e_adv=w_e+fer_w` → `advect_one_fct` (functional, no
  subtract). The FCT vertical uses `w_e`. The actual step.py wiring is **G.7**.
- [x] **`gm_diagnostics` driver** (the real G.5 deliverable) — composes G.1-G.4; `fer_uv`
  end-to-end 2.2e-16 vs the dump; AD through the full chain finite + nonzero.
- [x] ⚠️ **AD/consistency:** `fer_w` linear in `fer_uv`; no-flux BC exact (divergence-free bolus).
  The constant-tracer `S=35`-under-bolus + the tight bolus T/S vs C are folded into the G.7
  assembled gate (needs the FCT in the loop / the FESOM_GM_BOLUS_ONLY knob to isolate).
- [x] run — **DONE** (test_gm_bolus_adv 3 passed). **Lesson:** appended (the functional add-no-subtract, the
  fer_w÷area, the constant-tracer-under-bolus check).

### Task G.6: Redi tracer terms — G7a (vertical-explicit) + G7b (horizontal) + K33

**Files:** create `fesom_jax/gm_redi.py` (G7a + G7b), modify `fesom_jax/tracer_diff.py` (the K33
augmentation in `impl_vert_diff`). `tests/test_gm_redi.py`. C: G.1 dump (full GM-ON T/S).

> The Redi neutral-diffusion terms. ⚠️ **Threading:** G7a/G7b read `valuesold` (pre-step T) to
> build `tr_xy`/`tr_z`, and add the flux to `values` (post-advection T) ÷`hnode_new` (the C
> composes the Fortran `del_ttf` accumulation + `ale_reconstruct`). The 5 partial-cell branches
> of G7b collapse to masked per-level sums (the ocean-upwind-5-zones precedent).

> **✅ DONE 2026-06-07.** `fesom_jax/gm_redi.py` (G7a + G7b + K33). NEW C `fesom_redi_blob` hook
> (`FESOM_REDI_DUMP_DIR`, `fesom_step.c`) dumps `T_old/T_pre/T_g7a/T_g7b/tr_xy/tr_z` all-node
> (job 25401747). Verified vs the dump: **`tr_xy` bit-exact**, **G7a 1.78e-15** (moved T 3.1e-3),
> **G7b 1.07e-14** (the 5-branch edge loop, moved T 1.2e-3 — the 5→3-case `(in1,in2)` collapse
> worked first try). **K33 = augment Kv** (`Ty(nz)==Ty1(nz-1)` ⇒ one per-interface value; `impl_vert_diff`
> already does `a∝Kv[nz]`/`c∝Kv[nz+1]` so `Kv+K33_aug` reproduces the C with NO diffusion change)
> — sanity-checked (finite/active/AD-finite; tight gate at G.7). `test_gm_redi.py` = **4 passed**.
> 3 lessons. **The entire GM/Redi physics is now ported + verified.** `dt=fesom_phase1_dt=500`.

- [x] **G7a `diff_ver_part_redi_expl`** (`fesom_gm.c:646-790`): `tr_xy`=∇(T_old); area-weighted →
  `tr_xynodes` (÷3·areasvol); `vd_flux` from `slope_tapered·tr_xynodes·Ki` (static geometry);
  `ΔT=(vd_flux[nz]−vd_flux[nz+1])·dt/(areasvol·hnode_new)`. **1.78e-15 vs the dump.**
- [x] **G7b `diff_part_hor_redi`** (`fesom_gm.c:824-1022`): `tr_z`=∂(T_old)/∂z; edge loop with the
  **5 branches → 3 cases** (`both/only1/only2` from `elem_layer_mask[el1/el2]`); `c=(CX·Fx+CY·Fy)·dz`;
  antisymmetric `+c→e1,−c→e2` scatter; `ΔT=rhs·dt/(areasvol·hnode_new)`. **1.07e-14 vs the dump.**
- [x] **K33** (`fesom_tracer_diff.c:167-246`): `gm_redi.k33_augmentation` → `Kv+K33_aug` (no
  `impl_vert_diff` change — the C's `Ty/Ty1` is one per-interface `K33_aug`). Sanity-checked.
- [x] ⚠️ **AD-safe guards:** `1/(3·areasvol)`, `1/mid`, `1/(areasvol·hnode_new)`, `1/(0.5(h_up+
  h_dn))` all `where`-guarded; `slope²·Ki` no divide; the 3-case masks zero cleanly (no double-count).
- [x] **Gate:** the Redi dump (`T_pre+G7a==T_g7a`, `T_g7a+G7b==T_g7b`) all-node — tight. K33's
  tight gate (it's inside the diffusion) folds into the G.7 assembled post-step T/S.
- [x] **AD:** `d(Σfer_uv²)/d(T)` (G.5) + `d(ΣK33)/d(slope)` finite; full `d/d(T₀)` is the G.7 gate.
- [x] run — **DONE** (test_gm_redi 4 passed). **Lesson:** appended (the 5→3-case collapse, the
  K33-augment-Kv, the valuesold-vs-values threading + the Redi dump).

### Task G.7: GATE 6B — assemble GM/Redi into the CORE2 step + stability + gradient

**Files:** modify `fesom_jax/step.py`, `fesom_jax/integrate.py` (the `gm_cfg` static arg + the
full wiring); create `scripts/core2_gm_stability_run.py` (+ `_gpu.sh`), `scripts/core2_gm_grad_gate.py`
(+ `.sbatch`). `tests/test_gm_step.py`.

- [ ] **Wire GM into the step** behind a static `gm_cfg=None` arg (mirror `ice_cfg`): when given,
  after EOS/`sw_alpha_beta`/smooth, run the GM coefficient block (G.2→G.4) → `fer_uv`/`fer_w`;
  feed the bolus-augmented velocity to tracer advection (G.5); apply the Redi terms (G.6) + K33.
  `gm_cfg=None` ⇒ pi/Phase-5/ice paths **bit-identical** (the dead-branch precedent). Thread
  `gm_cfg` through `step_jit`'s `static_argnames` and the `lax.scan` body.
- [ ] **Assembled gate (step 1):** the full GM-ON CORE2 dump at step 1 — post-step T/S match the
  C (the per-kernel gates G.2-G.6 are the bit-exact ones; the assembled step may be climate-close
  ~1e-9 if any scatter floor propagates, like the ice assembly — gate kernels tight, assembly
  climate-close per locked decision #10). pi + Phase-5 + ice bit-identical with `gm_cfg=None`.
- [ ] **Multi-day stability** (GPU): the assembled **CORE2 + GM/Redi + sea ice** model (the full
  production config) runs multi-day stable — GM should *reduce* spurious convection / smooth
  fronts (a sanity sign the eddy flux is doing physical work), bounded vel/SSH/T/S, no NaN.
  Compare to a matched GM-ON C arbiter trajectory (a few diagnostics to 3 sig figs).
- [ ] **Gradient gate (GATE 6B):** re-run the permanent AD gate with GM live — `d(SST)/d(k_ver)`
  still plateaus; **the new `d(SST)/d(k_gm)` 2nd-hook target** FD↔AD plateau in a smooth regime
  (proves the eddy-flux gradient path, like `k_ver` proved the mixing path); the masked-NaN
  `d(SST)/d(T₀)` finite everywhere + 0 on masked + nonzero on wet (the GM slopes/TDMA/Redi all
  AD-safe). Measure backward memory (GM adds a per-step TDMA + scatters; budget the A100).
- [ ] run — full suite green (ocean + ice + GM tests; ice as a separate group). **Lesson:** append
  (the assembled-GM fidelity class, the 2nd-hook gradient plateau, the GM backward memory, any
  climate-sanity signal).

**GATE 6B (acceptance):** the CORE2 model (PP/linfs/FCT/opt_visc7 + PHC IC + JRA55/SSS/runoff +
sea ice + **GM/Redi**) reproduces the C GM-ON per-kernel dumps (each kernel bit-exact, G.1-G.6;
assembled step-1 T/S tight/climate-close, G.7); runs multi-day numerically stable with GM doing
physical work; the gradient gate passes with GM live **including the new `d/d(k_gm)` 2nd-ML-hook
target** + masked-NaN clean; full suite green. **Phase 6B (GM/Redi) COMPLETE** — the 2nd ML-hook
seam established. Next: Phase 6C (KPP) — own sub-plan (read `fesom_kpp.c` first).

---

## Risks / watch-list

- **The 5 G7b partial-cell branches** (G.6) — the highest-complexity kernel; a level-mask
  off-by-one double-counts at branch boundaries. Mitigation: the masked-per-level-sum collapse +
  an independent numpy ref that exercises a level-mismatch edge (the ocean-upwind-5-zones precedent).
- **`valuesold` vs `values` threading** (G.6) — G7a/G7b build gradients from the **pre-step** T
  but apply to the **post-advection** T ÷`hnode_new`. Getting this wrong silently biases the Redi
  flux. The assembled step-1 dump gate is the check.
- **The `fer_solve_gamma` TDMA conditioning** (G.4) — `b=−a−c−N²` with the `1e-8` `bv` floor; weak
  stratification → near-singular rows. Reuse `ops.tdma`'s padding; gate the residual.
- **GM TDMA backward memory** (G.7) — a per-node 2-component TDMA inside the outer N-step scan
  adds to the (already ice-heavy) backward. Checkpoint; budget the card; may need a short outer N.
- **The bolus constant-tracer property** (G.5) — if `fer_w` ÷area or the per-level scatter is
  wrong, `S=35` drifts. Gate `S` exact under bolus.
- **2nd-hook gradient conditioning** (G.7) — `k_gm` enters through the TDMA `fer_K` RHS + the
  scalings; verify the FD↔AD plateau is clean (not stiff like the EVP `1/delta_min`). If stiff,
  document (trainable gradients still flow; the mixing seam stays the well-conditioned one).
- **GM is ON by default in the C** — double-check no Phase-2..6 gate silently relied on
  `FESOM_NO_GMREDI=1` beyond the dump config (it didn't — GM was off in all JAX runs; `gm_cfg=None`
  preserves that).

## Out of scope (deferred — NOT in the C GM reference, or later phases)

`scaling_LDD97`, `scaling_Rossby`, FESOM14, GINsea, Ferreira/MLD-ref (`K_GM_bvref`),
`K_GM_Ktaper`, `K_GM_rampmax/min`, the Redi-without-GM `K_hor` branch, cavities, partial cells,
mEVP/zstar. **KPP** (Phase 6C) + the spatial-NN eddy flux (Phase 7) get their own sub-plans.

## Revision Log

- **2026-06-07 — created** (Phase-6B GM/Redi sub-plan). Scope **= GM/Redi only** (user-confirmed;
  KPP → 6C). Decisions (user-confirmed): **thread the GM eddy diffusivities through `params.py`
  now** (the 2nd ML-hook seam, default=config → bit-identical, like `k_ver`/`a_ver`); **7-task
  data-flow ladder** (G.1-G.7), each kernel dump-gated. Task ladder + the 6 integration points
  from this session's first-hand reading of `fesom_gm.c`/`.h` + the `fesom_step.c`/`fesom_ale.c`/
  `fesom_tracer_diff.c`/`fesom_eos.c` integration seams. Key findings baked in: GM/Redi is
  **stateless** (no new State fields, unlike ice σ); it threads at **6 points** (not just
  substep 14); it's **mixing-scheme-independent** (dump on PP, GM-ON, ice-OFF); a NEW
  `fesom_gm_dump` all-node/element hook gates the intermediates; the active namelist is a small
  subset (ODM95 taper, GMzexp depth scaling, resolution scaling, Redi=GM sync, Redi_Ktaper); AD
  hazards are all established patterns (safe-sqrt on slope/bv/tapfac, subgradient clips, masked
  safe-divides, the grad-verified `ops.tdma`).
- **2026-06-07 — Task G.1 DONE** (`sw_alpha_beta` + the 2nd ML-hook seam + the GM dump infra).
  `eos.compute_sw_alpha_beta` (verbatim McDougall, term-by-term) — **bit-exact vs the C GM-ON dump**
  (max|Δ|=0 over all 3.7M CORE2 wet lanes, a pure pointwise map like `density`) + an independent
  numpy ref + AD-finite. `params.py` += `k_gm`/`redi_kmax` (2nd ML-hook seam, `default_factory`
  defaults ⇒ the old 2-arg `Params` + the 17-test gradient/integrate seam stay **bit-identical**);
  `config.py` += `K_GM_MAX`/`REDI_KMAX`; `gm.GMConfig` skeleton. The C `fesom_gm_dump` hook
  (`fesom_step.c`, `jax-mesh-export` branch — NEVER port2 main; env-gated, stateless all-node
  snapshot) + `jobs/jax_gm_dump_core2.sh` (GM-ON, ice-OFF, PP; job 25397273, 31 s) →
  `data/gm_dump_core2/` (inputs T/S/bvfreq/hnode + all GM outputs; **seeds G.2-G.4**);
  `io_dump.load_gm_dump` reader. `test_sw_alpha_beta.py` 6 passed; 4 lessons appended. Next: Task
  G.2 (neutral slopes — `compute_sigma_xy` + `compute_neutral_slope`, gated by this dump).
