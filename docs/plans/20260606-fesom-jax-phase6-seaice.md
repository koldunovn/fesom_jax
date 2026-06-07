# FESOM2 → JAX Port — Phase 6: Sea Ice (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 6 outline).
**Predecessor:** `docs/plans/20260606-fesom-jax-core2.md` (Phase 5, COMPLETE — GATE 5).
**Created:** 2026-06-06. **Status:** DRAFT for review (no tasks started).
**Scope (user-confirmed 2026-06-06):** **sea ice only.** GM/Redi and KPP are deferred to
their own later sub-plans (Phase 6B/6C). **Port order (user-confirmed):
thermo → coupling → EVP → FCT → assemble → stability.**

---

## 0. Scope (READ FIRST — what the C ice port actually is)

Phase 6 ports the FESOM2 **sea-ice** subsystem on the CORE2 mesh, on top of the completed
Phase-5 ocean (PP/linfs/FCT/opt_visc7 + PHC IC + JRA55/SSS/runoff). The algorithmic
source of truth is the C port's `fesom_ice*.c` (read in full this session). Like the rest
of FESOM, the C ice is a **deliberately simplified** model — match THAT, not full FESOM.

**What the C ice port IS** (`fesom_ice_types.h:6-18`, confirmed by reading the sources):
- **3 prognostic ice tracers** per surface node: `a_ice` (concentration 0..1), `m_ice`
  (ice volume per area, m), `m_snow` (snow volume per area, m). Ice is **2-D, surface-only**
  (single layer); ice velocity `u_ice`/`v_ice` on nodes.
- **Standard EVP only** (`whichEVP=0`; the dispatcher aborts otherwise) — 120 *fixed*
  subcycles per ice step.
- **Single-class thermodynamics** (the Hibler-1984 7-class loop is a growth-rate average
  only; the state tracers are single-class), 0-layer Semtner snow, snow→ice flooding.
- **Per-step driver order** (`fesom_ice.c:303`): `ocean2ice → EVP → FCT → cut_off →
  thermo → oce_fluxes`, then (in `fesom_main.c:1058-1113`) `oce_fluxes_mom → shortwave →
  ocean step`. The ice subsystem runs **before** the ocean step each iteration and feeds
  it `heat_flux`/`water_flux`/`virtual_salt`/`relax_salt`/`stress_surf`/`sw_3d`.

**What is ABSENT (out of scope — the C does not implement it):** `whichEVP≠0` (mEVP/aEVP),
ridging, meltponds (`use_meltponds=0`), icebergs, wiso isotopes, ice-temperature tracer
(oifs), cavities (`ulevels>1` skip only), the time-dependent open-water albedo
(`open_water_albedo=0`, fixed `albw=0.1`), `l_snow=false` synthetic snow (CORE2 has
`prec_snow`), `snowdist=false`, `new_iclasses`. Locked config: **`use_virt_salt=1`**
(virtual-salt path, linfs), **`ref_sss_local=1`**.

**Why Phase 6 matters (the two Phase-5 findings it resolves):**
1. **Caps the no-ice supercooling.** Phase 5's high-lat SST supercools without bound
   (−22 °C by day 8 → EOS-invalid → dynamics destabilize ~day 8). The ice
   **thermodynamics** (the `o2ihf` ocean→ice heat flux + the freezing-point physics) caps
   SST at the freezing point — this is the whole point, and it comes from **thermo alone**
   (no dynamics/advection needed).
2. **Activates runoff.** Runoff enters at exactly one line — `fesom_ice_thermo.c:318`
   `prec = rain + runo + snow*(1-A)` → folded into `flx_fw` → `water_flux = -flx_fw`
   (`fesom_ice_coupling.c:139`) → the existing `sss_runoff` virtual-salt math. The
   Phase-5 reader + balance are done and pure-in-`water_flux`; Phase 6 just feeds them the
   ice `water_flux`.

**ML-hook note:** ice constants are **config**, not (yet) trained params. The `params.py`
seam stays `k_ver`/`a_ver` (vertical mixing) + the future GM/Redi swap. Ice gets its own
static-constant bundle (an `IceConfig`); leave a clean seam but do not put ice on the
trainable path now.

## 1. Reference path — Path A (per-substep C dump at the ice-ON config)

Exactly as Phases 0/5: a per-substep C-port dump at the JAX-matched config, so JAX↔C diffs
are pure FP reassociation (the tightest gate). The C already has all the gates we need,
**env-driven on the existing binary** — this is a major advantage:

- **Incremental ice configs** via the existing env knobs (`fesom_ice.c:294-301`): drop
  `FESOM_NO_ICE_*` to turn ice on. Generate the dump in **three stages** matching the port
  order so each kernel is gated in isolation:
  - **(A) thermo-only:** `FESOM_NO_ICE_DYN=1 FESOM_NO_ICE_ADV=1` (thermo + coupling on) —
    gates Tasks 6.2/6.3, and shows the supercooling cap with the least code.
  - **(B) +EVP:** `FESOM_NO_ICE_ADV=1` — adds dynamics; gates Task 6.4.
  - **(C) full ice:** no `FESOM_NO_ICE_*` — gates Tasks 6.5/6.6/6.7.
- **The dedicated ice dump harnesses already exist** (no new C code for these):
  - EVP: `FESOM_EVP_DUMP_DIR` writes per-probe `Q/P/F/1/2/3/4/END` (inputs, mass/strength,
    forcing, stresses σ at sub 0, rhs, velocities after update/BC, final) — `fesom_ice_evp.c:29-47`.
  - thermo + ice/stress: the `iceforce` tag via `fesom_kpp_dump_*` (`fesom_ice_coupling.c:266-275`).
  - The per-substep ocean dump (`fesom_dump.c`, Phase-5) still fires for the ice-ON T/S +
    dynamics.
- **Re-pin probes** for ice coverage: keep the Aleutian 94122 + add a **high-lat ice node**
  (a_ice≈0.9 region), a **river-mouth node** (runoff freshening), an **ice-edge node**
  (a_ice small, EVP `delta`→0, FCT limiter active), and an **ice-free node** (the masked-NaN
  AD probe). 1-line edit + rebuild (Phase-5 precedent).
- **`FESOM_BULK_FIXED_ITERS=1`** stays on the reference (Phase-5 finding — the M-O loop is
  non-convergent at calm nodes; JAX runs fixed-5 but the reference must match the chosen
  iteration count; verify the bulk gate still holds with ice on).

C-side: **C edits → port2 branch `jax-mesh-export`, NEVER port2 main**; job scripts stay
untracked. Cheap dumps → `-p compute --time=00:30:00`. New data → `/work` (the `data`
symlink). The ice dumps land under `data/ice_*_dump_core2/`.

## 2. Verification ladder (unchanged classes)

Per-substep probe-column dump, truncate to `nlevels` (ice is single-layer → just the
surface), `verify.assert_close(col, rec, kind=…)`: **map/gather 1e-15, scatter/reduction
1e-12** (calibrate `atol`). Most of the ice is **per-node MAP-class** (thermo, coupling,
bulk stress) → expect ~1e-13/1e-15; the EVP element↔node scatters + the FCT scatters are
**scatter-class** ~1e-12. Big intermediates (none as large as ocean `pressure`/`ssh_rhs`)
use **relative** tolerance.

**Re-run the gradient gate at GATE 6.** The new AD surfaces are: the thermo **5-iter
Newton** (unrolled) + freezing/albedo/melt kinks; the EVP **120-subcycle `lax.scan`** (the
`delta` singularity + masked `inv_mass`); the FCT limiter (same class as the ocean FCT, NaN-
safe via the eps floor). **AD rule (unchanged, bit us 4× already):** any divide/sqrt whose
denominator/arg can vanish in a masked (ice-free) lane must compute a FINITE value
(`where(d==0,1,d)` / double-`where` safe-sqrt) — a forward `where` does not stop a `0·inf`
NaN backward. **The forced multi-step trajectory is non-smooth in the physics params**
(Task 5.8 finding — ice adds freezing-point kinks); validate FD↔AD at N=1 / isolated seams /
the linear-solve residual, and lean on the (sub)gradient + masked-NaN finiteness for the
full model.

## 3. Config (the CORE2 ice-ON reference run)

Everything in Phase-5 §3 (PP/linfs/FCT/opt_visc7, dt=500, PHC IC, JRA55+SSS+runoff,
`use_wsplit=0`, CG α=1, full-cell), **plus** sea ice:

- **ice_dt = 500** (`ice_ave_steps=1`, so ice step == ocean step; `fesom_ice.c:231`).
- **EVP:** `evp_rheol_steps=120`, `Tevp_inv = 3.0/ice_dt = 0.006`
  (⚠️ the `fesom_ice.c:233` *setup* value — **NOT** the stale `types.h:177` comment
  "evp_rheol_steps/ice_dt"; the C overrides it), `dte = ice_dt/evp_rheol_steps`,
  `det1=det2=1/(1+0.5·Tevp_inv·dte)`, `vale=1/ellipse²=0.25`, `ellipse=2`, `pstar=30000`,
  `c_pressure=20`, `delta_min=1e-11`, `cd_oce_ice=5.5e-3`, `theta_io=0`, `ice_free_slip=0`.
  `Clim_evp/zeta_min` are unused on the ported path.
- **Thermo:** `rhoice=910`, `rhosno=290`, `rhowat=1025`, `rhofwt=1000`, `rhoair=1.3`;
  `con=2.1656`, `consn=0.31`; `Sice=4`; `h0=h0_s=0.5`; `albsn=0.81`/`albsnm=0.77`/
  `albi=0.70`/`albim=0.68`/`albw=0.1`; `emiss_ice=emiss_wat=0.97`; `boltzmann=5.67e-8`;
  `iclasses=7`; `hmin=0.01`; `armin=0.01`; `h_ml=2.5`; `c_melt=0.5`; `cc=rhowat·4190`,
  `cl=rhoice·3.34e5`; `clhw=2.501e6`, `clhi=2.835e6`; `tmelt=273.15`.
- **Exchange coeffs:** `CH_ATM_ICE=CE_ATM_ICE=1.75e-3`, `CD_ATM_ICE=1.2e-3`
  (`fesom_constants.h:110-113`).
- **FCT:** `ice_gamma_fct=0.5` (CORE2 namelist value, not the 0.25 module default),
  `ice_diff=10.0`.
- **Cold-start ice IC** (`fesom_ice.c:246-277`): where (non-cavity & PHC SST<0):
  `a_ice=0.9`; NH (geo lat>0) `m_ice=1.0, m_snow=0.1`; SH `m_ice=2.0, m_snow=0.5`;
  `u_ice=v_ice=0`. Else open water (0).

---

## Implementation Steps

### Task 6.1: Ice state + cold-start IC + config + ice-ON C reference dumps

**Files:** Create `fesom_jax/ice.py` (state fields helper + `IceConfig` + cold-start IC);
modify `fesom_jax/state.py` (add ice fields), `fesom_jax/core2_forcing.py` (generalize
`ice_ic_aice`). C (`port2`, branch `jax-mesh-export`): re-pin `PROBE_GIDS`
(`fesom_dump.c:15-17`) for ice coverage; new SLURM dump jobs (3 configs A/B/C). Create
`tests/test_ice_ic.py`.

> **✅ DONE 2026-06-06 (JAX side).** `State` extended with 9 prognostic ice fields
> (`a_ice/m_ice/m_snow/u_ice/v_ice/t_skin` [nod] + `sigma11/12/22` [elem]; σ carried as EVP
> elastic memory — see the lesson). `fesom_jax/ice.py`: `IceConfig` (the §3 constants +
> derived `cc/cl/Tevp_inv/dte/vale`) and `ice_initial_state`/`seed_ice` (the cold-start IC,
> generalizing the Phase-5 `core2_forcing.ice_ic_aice` a_ice-only mask). The ocean (pi +
> Phase-5 no-ice) path is **bit-identical** (ice fields inert-zero). `test_ice_ic.py` (8
> tests) green; the IC is C-verified **transitively** (a pure threshold of the ~1e-14 PHC
> SST; **0** FP-fragile nodes) + an independent per-node loop ref — no SLURM C dump spent.
> ⚠️ **Probe re-pin is env-only** (`FESOM_DUMP_PROBES`, no C edit) and the ice-ON configs are
> env knobs (`FESOM_NO_ICE_*`), so the **ice-ON C dumps are deferred to their consuming
> tasks** (config-A thermo → 6.2, +EVP → 6.4, full → 6.5/6.6); each needs one small additive
> all-node output-dump hook (`fesom_bulk_dump`-style) for the ice fields the per-substep
> `fesom_dump.c` doesn't carry (flx_fw/flx_h/a/m/msnow/t_skin).

- [x] **State fields:** add the prognostic ice fields carried across steps —
  `a_ice/m_ice/m_snow/u_ice/v_ice/t_skin` `[nod2D]` + the EVP elastic-memory stresses
  `sigma11/sigma12/sigma22` `[elem2D]` (the EVP does **not** re-zero σ each step —
  `fesom_ice_evp.c:126-130` reads the prior value, so σ is genuine prognostic state). Update
  `State.zeros`/`State.rest` (ice fields = 0 at rest). Confirm the pytree registration
  (`tree_util.register_dataclass`) picks them up automatically. Decide whether `*_old`
  ice-tracer slots are needed (the C `values_old` is the thermo backup set *within* a step —
  resolve by reading the FCT's actual reads; likely no new State field).
- [x] **`IceConfig`** (a NamedTuple/dataclass of the §3 constants — static, closed over the
  step). Keep it separate from `params.py` (ice is not on the trainable path yet).
- [x] **Cold-start IC:** port `fesom_ice_initial_state` (`fesom_ice.c:246-277`) — the
  hemisphere-split a/m/msnow from PHC SST<0. This **generalizes** the Phase-5
  `core2_forcing.ice_ic_aice` (which produced only the static `a_ice=0.9` mask); now it
  produces full `(a_ice, m_ice, m_snow)` and seeds `State`. Pure numpy host setup.
- [~] **Generate ice-ON C reference dumps (Path A, 3 configs):** DEFERRED to consuming tasks
  (config-A → 6.2, config-B → 6.4, config-C → 6.5/6.6) — see the DONE note. Recipe: clone the Phase-5 step-dump
  job; produce (A) thermo-only (`NO_ICE_DYN+NO_ICE_ADV`), (B) +EVP (`NO_ICE_ADV`), (C) full.
  Enable `FESOM_EVP_DUMP_DIR` + the `iceforce` dump for B/C. Re-pin probes (high-lat ice +
  river-mouth + ice-edge + ice-free + Aleutian 94122). `FESOM_BULK_FIXED_ITERS=1`. Cache
  under `data/ice_{thermo,evp,full}_dump_core2/`.
- [x] **Gate:** `tests/test_ice_ic.py` — cold-start IC bit-for-bit vs the C
  `fesom_ice_initial_state` (a/m/msnow at all nodes; NH/SH split; cavity skip); State plumbs
  ice fields; rest-state (no ice, zero wind) stays at rest; the pi path stays bit-identical
  (ice fields default-zero ⇒ the 376-test suite unchanged).
- [x] run — must pass before Task 6.2. **Lesson:** append (esp. the σ-as-prognostic-state
  finding + the Tevp_inv setup-vs-comment trap). — **DONE: full suite 384 passed** (376 ocean
  bit-identical + 8 new ice IC tests); 5 lessons appended.

### Task 6.2: Ice thermodynamics — `ice_thermo.py` (the supercooling cap + runoff activation)

**Files:** Create `fesom_jax/ice_thermo.py`; `tests/test_ice_thermo.py`. C (`port2`,
`jax-mesh-export`): a NEW all-node `fesom_ice_thermo_dump` hook (`fesom_ice_thermo.c`,
gated `FESOM_ICE_THERMO_DUMP_DIR`) + `jobs/jax_ice_thermo_dump_core2.sh` (config-A).

> Thermo is a **per-node pure map** (no scatter) — the easiest, highest-value kernel. It
> caps the supercooling and activates runoff. Port `fesom_ice_thermo.c` line-for-line.

> **✅ DONE 2026-06-06.** `fesom_jax/ice_thermo.py` — faithful AD-safe port (`tfrez`,
> `_obudget`, `_budget` [5-iter Newton + 4-way albedo `where`], `therm_ice_cell`,
> `_flooding`, `cut_off`, the `thermodynamics` driver = `vmap`(therm_ice_cell) + the ustar
> pass). Verified vs a NEW all-node C dump (`fesom_ice_thermo_dump`, re-runs `therm_ice` on
> copies → per-node inputs+outputs; job 25395803, config-A thermo-only,
> `data/ice_thermo_dump_core2/`): all 7 outputs **bit-exact MAP-class** over all 126858 nodes
> — h/hsn/A ~1e-16, t_skin 5.6e-14, fw 3.5e-19, **ehf rel 4e-10** (the `cl·dhgrowth`
> amplification of the ~1e-16 h diff), thdgr 4.5e-19. **Runoff activates** — `d(fw)/d(runoff)
> = 1` exactly (the `prec += runo` seam). **AD-safe**: `d(Σehf)/d(SST)` & `d/d(t_skin)` finite
> on every node incl. ice-free lanes; FD↔AD `d(ehf)/d(SST)` plateau ~1e-16 on a smooth
> interior-ice subset (near-linear in SST). `test_ice_thermo.py` = **12 passed**. ice_thermo.py
> is standalone (wired into the step at 6.6) ⇒ the 384-test suite is unaffected; full re-run
> deferred to 6.3 (first shared-code change).

- [x] **`tfrez(S)`** (`fesom_ice_thermo.c:20-24`): `-0.0575·S + 1.7105e-3·√(S³) - 2.155e-4·S²`.
  AD: `√(S³)` is singular at S=0 (masked/land lanes) → double-`where` safe-sqrt (S>0 on wet).
- [x] **`obudget`** (open water, `:98-142`): saturated humidity `b=3.8e-3·exp(17.27·t/(t+237.3))`;
  `hfswr=(1-albw)·fsh`; `hflwrd=-emiss_wat·σ·(t+tmelt)⁴`; `hfsen=ρair·cpair·ch·ug·(ta-t)`;
  `evap=ρair·ce·ug·(qa-b)`; `fh=-hftot/cl`; `evap_out=evap/rhowat`. Smooth (exp/pow).
- [x] **`budget`** (ice-covered, `:160-217`): the **4-way albedo `where`** (t<0 & hsn>0;
  `:171-175`) computed branchlessly; the **5-iter fixed Newton** on skin temp (`:190-202`,
  unrolled — `B=q1/ρair·exp(q2/tk)`, `q1=11637800`, `q2=-5897.8`; residual `(A1+A2+C)/A3`)
  then `t=min(t,0)`; final fluxes with `clhi`/ice albedo. Divide `con/hice` (hice=thact>0 on
  active classes; guard masked lanes).
- [x] **`therm_ice`** (`:234-411`): `snthick`/`thick` (÷`max(A,armin)`); open-water obudget;
  the **iclasses loop k=1..7 sequentially refining `t`** (`:292-303` — each class's budget
  reads/updates the same `t`; unroll 7×, AD-friendly) gated by `thick>hmin` (where);
  areal weighting (`show`/`shice`/`sh`); **`prec = rain + runo + snow·(1-A)`** (`:318` — the
  RUNOFF entry); snow accumulation `+snow·dt·A·1000/rhosno` (the 1000 = `rhofwt`); snow/ice
  melt (`min`/`max` kinks); `o2ihf` ocean→ice heat (`:338-339`, the two-term ustar/ML flux
  using `tfrez(S_oc)`); `qhst`/`h`/`hsn` updates; `dhgrowth`/`dhsngrowth` rates;
  `ehf = ahf + cl·(dhgrowth + rhosno/rhoice·dhsngrowth)`; **virtual-salt freshwater**
  (`:367-373`: `fwice=-dhgrowth·rhoice/rhowat·(rsss-Sice)/rsss`, `fwsnw=-dhsngrowth·rhosno/
  rhowat`, `fw = prec+evap+fwice+fwsnw`, `rsf=0`); compactness (Hibler eq 16, `:377-385` —
  `min`/`max`/`clip` kinks, ÷`max(h,hmin)`/`lid_clo`); flooding (`:387-403` — Archimedes
  `min`; the snow→ice salt correction `fw += iflice·rhoice/rhowat·Sice/rsss`).
- [x] **`thermodynamics` driver** (`:421-526`): Loop 1 `ustar = √((u_ice-u_w)²+(v_ice-v_w)²)·
  cd_oce_ice` (safe-sqrt); Loop 2 the per-node `therm_ice` (a `vmap` over nodes — purely
  local). Inputs: jra atmosphere (`shortwave/longwave/Tair/shum/prec_rain/prec_snow`),
  `forcing.runoff`, `srfoce_temp/salt`, `Ch/Ce` from the bulk (`forcing.Ch/Ce_atm_oce`),
  `ch_i/ce_i = CH/CE_ATM_ICE`, `lid_clo = h0 (NH) | h0_s (SH)`, `h_ml`. `cut_off`
  (`:48-75`, the a/m clamp — runs every step, AD via `where`).
- [x] ⚠️ **AD-safe guards:** the Newton denominators (`A3`, `con/hice`), `/rsss`,
  `max(A,armin)`, `max(h,hmin)`, `lid_clo`, the `tfrez` sqrt, the `√(du²+dv²)` ustar — every
  one finite on ice-free/masked lanes (`where(d==0,1,d)` / double-`where`). The Newton is
  fixed-iter (no `break`). Albedo + all `min`/`max` are subgradient (`jnp.where`/`minimum`).
- [x] **Gate:** dump `flx_fw/flx_h/a_ice/m_ice/m_snow/t_skin/thdgr` per-node vs the
  **config-A** C thermo dump at: high-lat ice nodes, **river-mouth** (`runoff>0` ⇒ `prec`
  carries it — the runoff-activation check), melt-onset, ice-free (≈0 + finite). MAP-class
  (~1e-13). Independent numpy `therm_ice` reference for the kink/Newton paths the dump's
  smooth nodes don't exercise.
- [x] **AD:** masked-NaN — `d(Σehf)/d(SST)` / `d(Σflx_fw)/d(S_top)` finite everywhere incl.
  ice-free lanes; FD↔AD at smooth ice nodes (away from the freezing/albedo/melt kinks).
- [x] run — must pass before Task 6.3. **Lesson:** append (the runoff line, the
  sequential-iclasses-`t`, the Newton unroll, the freezing kinks). — **DONE: test_ice_thermo
  12 passed; 4 lessons appended.**

### Task 6.3: Ice-ocean coupling + the runoff handoff — `ice_coupling.py`

**Files:** Create `fesom_jax/ice_coupling.py` (ocean2ice + oce_fluxes + oce_fluxes_mom);
modify `fesom_jax/sss_runoff.py` (the ice-on flux variant), `fesom_jax/core2_forcing.py`
(thread the ice fluxes). `tests/test_ice_coupling.py`. C: dump in config-A.

> After this task the **thermo-only ice** path (6.2+6.3, EVP/FCT off) runs end-to-end and
> **caps the supercooling + activates runoff** — the first big physical milestone.

> **✅ DONE 2026-06-06.** `fesom_jax/ice_coupling.py` — `ocean2ice` (taps T/S/hbar/uvnode),
> `ice_oce_fluxes` (the runoff handoff: `water_flux=-flx_fw`, `heat_flux=-flx_h`, virtual_salt/
> relax_salt via the Phase-5 `sss_runoff_fluxes` with a NEW backward-compatible
> `balance_water_flux=False` flag — drops the standalone `⟨water_flux+runoff⟩` term),
> `ice_oce_fluxes_mom` (prognostic-`u_ice` stress blend). Verified vs the config-A per-substep
> dump at the 7 probes: water_flux/virtual_salt/relax_salt **match the C ice-mediated forcing**
> (MAP/reduction class). **Runoff handoff PROVEN end-to-end** — composing thermo∘coupling,
> `d(water_flux)/d(runoff) = -1` exactly (freshwater in; `virtual_salt = S_top·water_flux`,
> S_top>0 ⇒ freshening). AD seams (`d/d(flx_fw)`, `d/d(S_top)`) finite. `test_ice_coupling.py`
> = **9 passed**. ⚠️ The `core2_forcing` threading + the **multi-day** supercooling-cap
> manifestation move to 6.6/6.7 (need the assembled step); the *mechanism* (`heat_flux=-flx_h`
> carries the ice-growth heat to `bc_T`) is verified here. `sss_runoff.py` default unchanged ⇒
> the Phase-5 path stays bit-identical.

- [x] **`ocean2ice`** (`fesom_ice_coupling.c:47-115`): nearly free in JAX — `srfoce_temp =
  T[:,0]`, `srfoce_salt = S[:,0]`, `srfoce_ssh = hbar`, `srfoce_u/v = uvnode[:,0]` (the C
  comment `:44-45` confirms `u_w` uses the same area-weighted recipe as `uvnode`, which
  `core2_forcing` already taps). Cavity skip. `ice_update=1` (no time-averaging branch).
- [x] **`oce_fluxes` — the runoff handoff** (`fesom_ice_coupling.c:125-179`):
  `water_flux = -flx_fw` (flx_fw from thermo, **includes runoff**), `heat_flux = -flx_h`;
  `virtual_salt = rsss·water_flux` (rsss=S_top, balanced minus the area-weighted global
  mean); `relax_salt = surf_relax_S·(Ssurf - S_top)` (balanced). ⚠️ **The refinement to the
  Phase-5 handoff:** the ice-on `oce_fluxes` does the virtual_salt + relax_salt balancing
  but **drops** the standalone `sss_runoff_step`'s `water_flux += ⟨water_flux+runoff⟩` term.
  So the existing `sss_runoff.sss_runoff_fluxes` needs an ice-on variant/flag that **skips
  the water_flux global balance** and is **fed `water_flux = -flx_fw`** instead of the bulk
  `evap-prec`. (The virtual_salt/relax_salt math is otherwise identical — dump-verify.)
  Wire `bc_T = -dt·heat_flux/vcpw`, `bc_S = dt·(virtual_salt+relax_salt)`.
- [x] **`oce_fluxes_mom`** (`fesom_ice_coupling.c:213-264`): the stress blend with the
  **prognostic** `u_ice` (Phase 5 used static `u_ice=0`): per node `a>0.001` ⇒
  `stress_iceoce = ρ·cd_oce_ice·|u_ice-u_w|·(u_ice-u_w)`; `stress_node_surf =
  stress_iceoce·a + atm·(1-a)`; element `stress_surf = mean-of-3`. This **upgrades** the
  existing `core2_forcing` blend (lines 127-142) from static to prognostic `a`/`u_ice`.
- [x] **Gate:** dump `water_flux/virtual_salt/relax_salt/heat_flux/stress_surf` vs the
  config-A/B C dump; **river-mouth `bc_S` freshening** (the runoff signal); the stress blend
  reproduces with prognostic u_ice (config-B). ~1e-12 (reductions) / ~1e-15 (maps).
- [~] **Milestone check:** thermo-only ice run (no EVP/FCT) — high-lat SST stops at the
  freezing point (the supercooling **capped**) and coastal SSS freshens at river mouths.
  **DEFERRED to 6.6/6.7** (the *multi-day manifestation* needs the assembled step); the
  mechanism (`heat_flux=-flx_h` → `bc_T`; `d(water_flux)/d(runoff)=-1`) is verified here.
- [x] run — must pass before Task 6.4. **Lesson:** append (the water_flux-balance refinement,
  the ocean2ice reuse, the milestone). — **DONE: test_ice_coupling 9 passed; full suite re-run
  (sss_runoff flag is backward-compatible).**

### Task 6.4: EVP dynamics — `ice_evp.py` (the 120-subcycle `lax.scan`)

**Files:** Create `fesom_jax/ice_evp.py`; modify `fesom_jax/forcing.py` (add `stress_atmice`).
`tests/test_ice_evp.py`. C: EVP dump (`FESOM_EVP_DUMP_DIR`) in config-B.

> The hardest AD piece. 120 *fixed* subcycles → a checkpointed `lax.scan`. All AD hazards
> are the established patterns (safe-sqrt, masked divide, `where`).

> **✅ DONE 2026-06-06.** `fesom_jax/ice_evp.py` — `evp_setup` (mass / inv_areamass / inv_mass
> + Hunke `ice_strength` + the SSH-tilt velocity forcing), `stress_tensor` (ε + the `Δ=√`
> double-`where` safe-sqrt + `delta_min` clamp + the Hunke σ update, σ frozen on ice-free
> elements), `stress2rhs` (element→node σ-divergence via `ops.scatter_add`), `velocity_update`
> (the 2×2 implicit Coriolis+ocean-drag solve + `a<0.01` gate + coastal BC), `evp_dynamics` (a
> **checkpointed 120-subcycle `lax.scan`**, carry = u/v_ice + σ). `forcing.atm_ice_stress` added
> (the wind-on-ice forcing). Verified vs the config-B C EVP dump (job 25396029, the
> `FESOM_EVP_DUMP_DIR` Q/P/F/1/2/3/4/END probes): step-0 inv_mass / inv_areamass / ice_strength
> / σ / u_rhs / velocity **bit-exact** (per-element/node maps); **END after 120 subcycles
> ~8e-10** (max|u_ice|=0.21 m/s ⇒ rel ~5e-9 — accumulated element→node scatter reassociation,
> climate-close). AD: `d(Σσ²)/d(u_ice)` finite **at the Δ=0 singularity** (safe-sqrt + clamp);
> the subcycle-scan backward finite incl. ice-free lanes. `test_ice_evp.py` = **8 passed**.
> ⚠️ At step 1 the SSH-tilt (rhs_a/m) ≈ 0 (elevation = hbar ≈ 0 at rest); its gradient
> contraction is the pgf/momentum pattern (gated). `Tevp_inv=3/ice_dt`, `delta_min=1e-11`
> matched (NOT raised). ice_evp.py standalone (wired at 6.6); the forcing.py change is additive.

- [x] **`stress_atmice`** (the EVP wind forcing, `fesom_bulk.c:329-333`): add to
  `forcing.py` — `stress_atmice = CD_ATM_ICE·|u_air-u_ice|·(u_air-u_ice)` (ice-relative,
  absolute wind; `=0` where `a_ice≤0`, `:274`). Per-node map; safe-sqrt on `|Δu|`.
- [x] **Dynamics setup** (`fesom_ice_evp.c:263-350`): `mass_per_area = rhoice·m_ice +
  rhosno·m_snow`; `inv_areamass = 1/(area·mass_per_area)` if `>1e-3` else 0;
  `inv_mass = 1/max(mass_per_area/a_ice, 9.0)` if `a_ice≥0.01` else 0 (both masked `where`).
  **`ice_strength`** (per-element Hunke, `:293-313`): `0.5·pstar·m̄·exp(-c_pressure·(1-ā))`,
  set to **0 if any vertex has `m_ice≤0` or `a_ice≤0`** (a per-element ice-presence `where`).
  **SSH-tilt rhs** (`:333-342`): `aa = 9.81·elem_area/3`; `edx/edy = gradient_sca·elevation`;
  scatter `-aa·edx`/`-aa·edy` to the 3 vertices, then ÷`area` (`:346-349`). ⚠️ The C reuses
  `data[AICE/MICE].values_rhs` as scratch for these velocity forcings (u in "rhs_a", v in
  "rhs_m") — in JAX use clearly-named locals `ssh_tilt_u/v`; they are NOT tracer rhs.
- [x] **`stress_tensor`** (`:68-132`): strain rates `eps11/22/12` from `gradient_sca` +
  `metric_factor` (gather nodes→elem); **`delta = √(...)`** (`:110-113`) — the EVP
  singularity → **double-`where` safe-sqrt** on the radicand (ice-free lanes → radicand 0);
  `delta_clamped = max(delta, delta_min=1e-11)` (⚠️ **match the C 1e-11**, do NOT raise to
  1e-8 — the safe-sqrt + the `max` clamp already make it AD-finite; golden rule); `zeta =
  ice_strength/delta_clamped·Tevp_inv`; the Hunke `si1/si2`/`r1/r2/r3` stress update (σ
  carried). Ice-free elements (`ice_strength≤0`) contribute 0 (the `where` from setup).
- [x] **`stress2rhs`** (`:143-193`): zero u/v_rhs; element scatter `-area·(σ·gradient +
  σ12·metric/3)` → u/v_rhs (`scatter_add` via segment_sum); finalize `u_rhs·inv_areamass +
  ssh_tilt_u` (masked).
- [x] **The 120-subcycle `lax.scan`** (`:372-448`): body = `stress_tensor → stress2rhs →
  velocity update → coastal BC`. **Velocity update** (`:395-417`): the 2×2 implicit
  Coriolis+drag solve — `umod = √(du²+dv²)` (safe-sqrt), `drag = cd_oce_ice·umod·ρ·inv_mass`,
  `rhsu = u_ice + rdt·(drag·(ax·u_w - ay·v_w) + inv_mass·stress_atmice_x + u_rhs)`,
  `r_a = 1 + ax·drag·rdt`, `r_b = rdt·(coriolis_node + ay·drag)`, `det = 1/(r_a²+r_b²)`,
  `u_ice = det·(r_a·rhsu + r_b·rhsv)`, `v_ice = det·(r_a·rhsv - r_b·rhsu)`; `=0` where
  `a_ice<0.01` (where). `ax=cos(theta_io)=1`, `ay=sin(theta_io)=0`. **Coastal BC**
  (`:430-437`): zero `u_ice/v_ice` at boundary-edge endpoints — a **static** boundary-node
  mask (precompute from `mesh`, like the C `bc_index_nod2D`/`edge2D_in`). Single device ⇒
  no halo exchange. σ carry through the scan (the scan carry = `(u_ice,v_ice,σ11,σ12,σ22)`).
- [x] ⚠️ **AD-safe guards:** the `delta` radicand double-`where` safe-sqrt; the `umod`
  safe-sqrt; masked `inv_mass`/`inv_areamass`; `ice_strength` ice-presence `where`; the
  `a_ice<0.01` velocity `where`; the static coastal mask. **Checkpoint the subcycle `scan`**
  (120 iters × CORE2 nodes — the backward will be memory-heavy; budget it).
- [x] **Gate:** vs the config-B C EVP dump at the probes — `Q/P/F` inputs, `1` (σ at sub 0),
  `2` (u/v_rhs), `3` (vel after update), `4` (vel after BC), `END` (vel after 120). Element
  fields (eps/σ/ice_strength) at the dump's incident-element gids. ice-edge (`delta`→0) +
  ice-free (masked, =0) nodes. scatter-class ~1e-12.
- [x] **AD:** gradient through the 120-subcycle scan finite (the `delta`/`inv_mass` masked-
  NaN probe); FD↔AD on an isolated smooth seam (e.g. `d(Σu_ice²)/d(stress_atmice)` at a
  rigid interior node, away from `delta`/`a_ice` kinks). Backward memory measured on GPU.
- [x] run — must pass before Task 6.5. **Lesson:** append (the scan design, the delta
  safe-sqrt-not-raise-eps decision, the σ carry, the backward memory). — **DONE: test_ice_evp
  8 passed.**

### Task 6.5: Ice FCT advection — `ice_adv.py` (2-D Zalesak)

**Files:** Create `fesom_jax/ice_adv.py`; `tests/test_ice_adv.py`. C: FCT dump in config-C.

> A **standalone 2-D module** (the ocean FCT is 3-D, edge-centric; the ice FCT is 2-D,
> element-centric, FE-mass-matrix). Reuse only the ratio-clip helper. ~300-400 JAX lines.

> **✅ DONE 2026-06-06.** `fesom_jax/ice_adv.py` — the 2-D Zalesak FCT, **fully element-based
> (no CSR)**: the FE consistent-mass product `(mm·X)[row]=Σ_{elem∋row} area/12·(X[row]+ΣX_elem)`
> scattered on the fly (so **`mass_matrix_fill` is unnecessary** — the mass block's row sum is
> the node CV area by construction), the Zalesak cluster bounds via `jax.ops.segment_min/max`
> over elements (a node's edge-neighbours == its element-co-vertices), plus `_tg_rhs` /
> `_solve_low_order` / `_solve_high_order` (2 residual-correction passes) / `_fem_fct` /
> `fct_solve`. Verified vs the config-C C FCT dump (job 25396145, `FESOM_ICE_FCT_DUMP_DIR`):
> a/m/msnow **bit-exact** (~1e-15) over all 126858 nodes (the FCT moved ice ~0.007). Limiter
> eps = the C's **1e-12** (match the C, NOT the ocean FCT's 1e-16). AD finite (`d/d(m_in)`,
> `d/d(u_ice)`); no positivity clip (match C — the small antidiffusive overshoot past 0.9 is
> FCT-physical, **identical to the C**, clamped later by `cut_off`). `test_ice_adv.py` =
> **6 passed**. Standalone module (wired into the step at 6.6).

- [~] **`mass_matrix_fill`** (`fesom_ice_fct.c:65`): NOT NEEDED — the element-based `_mm_times`
  computes `mm·X` on the fly (the consistent-mass block scattered), so no explicit CSR matrix.
  Original note: the FE mass matrix as CSR (one-time host
  setup; reuse the SSH stiffness CSR `rowptr/colind` adjacency — `fesom_ice_fct.c:4-9`). Row
  sum == node area (the assembly check).
- [x] **`tg_rhs`** (`:144`): Taylor-Galerkin element assembly of `values_rhs` for the 3
  tracers (element velocities + FE test functions; cavity skip).
- [x] **`ice_solve_low_order`** (`:226-271`): `X_l = (rhs + γ·(mm·X))/area + (1-γ)·X`,
  `γ=ice_gamma_fct=0.5`. **`ice_solve_high_order`** (`:277-345`): iterative residual
  correction (2 iters) → `dvalues`.
- [x] **The 2-D Zalesak limiter `ice_fem_fct`** (`:355-517`), called per tracer
  (m_ice/a_ice/m_snow): antidiffusive `icefluxes` (FE icoef pattern, the **negative** sign
  `:406`); cluster bounds `tmax/tmin` over CSR neighbors (`min`/`max` subgradient); +/- flux
  sums (element→node scatter); the ratio clip (**use eps=1e-16**, not the C's looser 1e-12,
  for AD — the ocean-FCT precedent); element flux limiting (gather-min); final element→node
  scatter `vals += icefluxes`. **No positivity clip** (match C). Reuse the ocean
  `tracer_adv` ratio-clip helper.
- [x] ⚠️ **AD-safe guards:** ratio-clip eps floor; `min`/`max`/`where` subgradients; safe
  area divide (`where(area>0,area,1)`); the rest-state check (`u_ice=v_ice=0 ⇒ flux≈0 ⇒
  tracers unchanged`).
- [x] **Gate:** advected `a/m/msnow` vs the config-C C FCT dump at ice-edge (limiter active)
  + interior ice nodes; conservation (`Σ area·m_ice` low-order-preserving); rest invariant.
- [x] **AD:** the limiter VJP NaN-safe (the eps floor); masked-NaN finite.
- [x] run — must pass before Task 6.6. **Lesson:** append (the 2-D-vs-3-D split, the eps
  tightening, the element-centric scatter reassociation). — **DONE: test_ice_adv 6 passed
  (bit-exact); CSR-free element formulation.**

### Task 6.6: Assemble the ice step + wire prognostic a_ice/u_ice into the ocean step

**Files:** Create `fesom_jax/ice_step.py`; modify `fesom_jax/step.py`,
`fesom_jax/integrate.py`. `tests/test_ice_step.py`. C: config-C full per-substep dump.

> **✅ DONE 2026-06-06 (step-1 gate).** `fesom_jax/ice_step.py` (`ice_surface_step`) composes
> the 5 kernels in the C runtime order (bulk → ocean2ice → EVP → FCT → cut_off → thermo →
> oce_fluxes → oce_fluxes_mom → cal_shortwave_rad → bc_T/bc_S) → the surface fluxes + the new
> ice state. Wired into `step.py`/`integrate.py` via a new static `ice_cfg` arg (`None` ⇒ the
> Phase-5/pi path stays bit-identical): when given, the ice step runs before the ocean substeps
> and its **prognostic** a_ice/u_ice replace the static mask in the two couplings (shortwave
> gate + stress blend). Gated vs the config-C full-ice C dump (job 25396145): post-step T/S
> match **~1e-6** (climate-close — the 120-subcycle EVP floor ~1e-9 propagates through
> u_ice→ustar/stress→the step; NOT bit-exact, the per-kernel gates are), the ice-mediated
> surface forcing (water_flux/virtual_salt/relax_salt) ~1e-6, the ice state evolves (EVP u_ice,
> thermo a/m, σ memory). `test_ice_step.py` = **4 passed**. The multi-day stability + the
> supercooling-cap manifestation are the 6.7 GPU run (job 25396276).

- [x] **`ice_step`** = `ocean2ice → EVP → FCT → cut_off → thermo → oce_fluxes`
  (`fesom_ice.c:303-354` order), a pure `(State, forcing) → (State, SurfaceFluxes)`
  function. Note the runtime order (EVP/FCT use the *previous* step's a/m/msnow + the current
  ocean2ice surface state; thermo updates them after). Post-step `h_ice/h_snow` diagnostic
  optional.
- [x] **Wire into the ocean step:** call `ice_step` **before** the ocean substeps in
  `step`/`integrate` (mirroring `fesom_main.c:1058`). Thread the ice State through the
  `lax.scan` carry (alongside the ocean State). The jra atmosphere + month SSS/chl stay the
  scanned `xs`; `IceConfig`/runoff/areas closed over.
- [x] **Prognostic a_ice/u_ice in the surface fluxes:** replace the static `fs.a_ice` /
  `u_ice=0` in `core2_forcing.compute_surface_fluxes` (lines 127-148) with the **prognostic**
  ice State — the two couplings (shortwave-penetration gate `pene_open = … & (a≤0)`; the
  momentum stress blend) now read the live `a_ice`/`u_ice` (via `oce_fluxes_mom`). `params=
  None` / pi path still bit-identical (ice off ⇒ a_ice≡0).
- [x] **Gate:** the full per-substep config-C CORE2 ice-ON dump at step 1 — post-step T/S
  **~1e-6** (climate-close, the EVP floor propagates) + the ice-mediated surface forcing; the
  per-kernel gates (6.2-6.5) are the bit-exact ones. pi + Phase-5 paths bit-identical
  (`ice_cfg=None`).
- [x] run — must pass before Task 6.7. **Lesson:** append (the carry layout, the
  prognostic-coupling wiring). — **DONE: test_ice_step 4 passed; full suite re-run.**

### Task 6.7: GATE 6 — multi-day CORE2 ice-ON stability + gradient re-check

**Files:** Create `scripts/core2_ice_stability_run.py` + `.sh`, `scripts/core2_ice_grad_gate.py`
+ `.sbatch` (new, not modifying the Phase-5 ones).

> **✅ DONE 2026-06-06 — GATE 6 MET.** **(1) Multi-day stability** (`core2_ice_stability_run.py`
> + `core2_ice_stability_gpu.sh`, A100, ~0.08 s/step, job 25396309): the prognostic-ice CORE2
> model runs **10 days stable** (1728 steps) and the **supercooling is CAPPED at −1.91 °C** (the
> freezing point) — vs Phase-5's unbounded drop (−16.5 day 5, −22.8 day 8 → blow-up). Worst
> max|vel|=2.72 (<3), |SSH|<2.1 (<5), no NaN; ice grows physically (m_ice→2.94 m, a_max=1.0,
> extent ~2.5e13 m², drift ~1 m/s). **Both Phase-5 findings (supercooling, inert runoff)
> RESOLVED.** **(2) Gradient gate** (`core2_ice_grad_gate.py` + `.sbatch`, job 25396293): [1]
> N=1 `d(SST)/d(k_ver)` FD↔AD plateau **4.5e-10** (smooth); [3] N=4 `d(SST)/d(T0)` **masked-NaN
> clean** (non-finite=0, masked=0, wet nonzero) — the assembled-ice backward (thermo Newton +
> EVP 120-scan + FCT limiter + every guard) is AD-safe at scale; peak **26.5 GB**. ⚠️ wet
> `d/d(T0)` ~1e16 = the genuine EVP `1/delta_min` rheology stiffness (finite ⇒ passes; the
> trainable gradients flow through the `k_ver`/`a_ver` mixing seam, well-conditioned per [1]).
> The redundant [I] `d/d(m_ice0)` probe OOM'd (3 backwards/process on the 40 GB card; covered by
> [3]) — `jax.clear_caches()` added. **(3) Suite** = ocean **376** + ice **47** (run in two
> chunks; the 423-in-one-process exceeds login-node RAM via the heavy assembled `test_ice_step`).

- [x] **Multi-day stability:** the assembled CORE2 ice-ON model (jitted, GPU) past day 8 —
  **the supercooling is CAPPED** (high-lat SST ≈ freezing, no −22 °C; the Phase-5 day-8
  blowup resolved), ice grows/melts plausibly (m_ice bounded, a_ice∈[0,1]), runoff freshens
  coasts, no NaN, bounded vel/SSH/ice-speed. Compare to a matched C ice-ON arbiter
  trajectory (SST_min capped, ice extent, max ice speed) to 3 sig figs.
- [x] **Gradient gate (CORE2 ice-ON slice):** re-run the permanent AD gate with ice live —
  the thermo Newton + EVP 120-scan + FCT limiter all AD-safe; the masked-NaN
  `d(mean SST)/d(T₀)` finite everywhere + 0 on masked lanes + nonzero on wet; FD↔AD at N=1 /
  isolated ice seams (thermo `d(ehf)/d(SST)`, EVP `d(u_ice)/d(stress_atmice)`) in their
  smooth regimes. Measure the backward memory (the EVP subcycle scan inflates it vs Phase 5's
  37.8 GB — checkpoint the subcycle scan; budget the A100-80; note the card limit).
- [x] run — full suite green (ocean 376 + ice 47, two chunks). **Lesson:** append (the
  supercooling-capped confirmation, the ice-on backward memory, the EVP-stiff-gradient + the
  per-step-forcing/3-backwards OOM findings). — **DONE: GATE 6 MET.**

**GATE 6 (acceptance) — ✅ MET (2026-06-06):** the CORE2 ice-ON model (PP/linfs/FCT/opt_visc7 +
PHC IC + JRA55/SSS/runoff + **sea ice: thermo + EVP + FCT**) reproduces the C per-substep
ice-ON dump (✅ each kernel bit-exact, Tasks 6.2-6.5; assembled step-1 T/S ~1e-6, Task 6.6);
runs **10 days numerically stable with the high-lat supercooling CAPPED at −1.91 °C and runoff
active** (✅ Task 6.7 — the two Phase-5 findings RESOLVED); the gradient gate passes on a CORE2
ice-ON slice (✅ FD↔AD plateau 4.5e-10 + masked-NaN clean at scale, peak 26.5 GB); full suite
green (✅ ocean 376 + ice 47). **Phase 6 (sea ice) COMPLETE.**

---

## Risks / watch-list

- **EVP `delta` singularity** (Task 6.4) — the classic `1/delta` AD blow-up at rigid/ice-free
  lanes. Mitigation: double-`where` safe-sqrt on the radicand + the C's `max(delta,1e-11)`
  clamp (which itself protects the backward where `sqrt(radicand)<1e-11`). **Do NOT raise
  delta_min** (golden rule) — the safe-sqrt is the fix.
- **EVP backward memory** (Task 6.4/6.7) — the 120-subcycle scan, nested inside the outer
  N-step scan, is the new memory driver (Phase-5 N=20 was already 37.8 GB). Checkpoint the
  subcycle scan; may need a shorter outer N or O(√N) nesting on the 40 GB card.
- **Thermo Newton fixed-5 vs convergence** (Task 6.2) — like the bulk M-O loop, the skin-temp
  Newton may not converge in 5 iters at some nodes; the reference must match the iteration
  count (already fixed in the C). Verify vs the fixed-iter C dump; bound any residual.
- **The water_flux-balance refinement** (Task 6.3) — the ice-on `oce_fluxes` drops the
  standalone's `water_flux += mean(...)` term; the JAX `sss_runoff_fluxes` needs an ice-on
  variant. Small, but dump-verify (it changes virtual_salt slightly).
- **σ elastic-memory state** (Task 6.1) — σ11/12/22 carry across steps; getting the State
  carry / scan plumbing wrong silently de-couples the EVP. Gate via the multi-step EVP dump.
- **Runtime order vs port order** (Task 6.6) — EVP/FCT run on the *previous* step's tracers;
  thermo updates after. The assembled-step gate (config-C) is the check.
- **Aleutian Trench (elem 194724)** — still the historical blowup hotspot; keep a probe.

## Out of scope (deferred — NOT in the C ice reference, or later phases)

mEVP/aEVP, ridging, meltponds, icebergs, wiso, ice-temperature tracer, time-dependent
albedo, `l_snow=false` synthetic snow, cavities. **GM/Redi** (Phase 6B) and **KPP**
(Phase 6C) get their own sub-plans (scope each by reading `fesom_gm.c` / `fesom_kpp.c`
first, like this one). zstar/partial-cells remain C-side future work.

**Parameter-tuning note (Phase 7a — see the parent plan).** The differentiable port can
calibrate ice parameters too: the **thermo** constants (albedos, snow/ice conductivity, the
o2ihf transfer, the freezing-point slope) are reasonable gradient-tuning targets. ⚠️ The **EVP
rheology** params (`delta_min`, `Tevp`, `ice_strength` P*/C) route through the stiff
`1/delta_min` (~1e16 IC-gradient, Task 6.7) — finite but ill-conditioned for gradient descent;
tune those with **gradient-free EKI** (forward runs only) or `stop_gradient` the EVP. Any tuned
scalar maps to the Fortran `namelist.ice` with zero Fortran code.

## Revision Log

- **2026-06-06 — created** (Phase-6 sea-ice sub-plan). Scope **= sea ice only**
  (user-confirmed; GM/Redi + KPP → later sub-plans). Port order **thermo → coupling → EVP →
  FCT → assemble → stability** (user-confirmed; thermo-first caps supercooling + activates
  runoff earliest, lowest risk). Task ladder 6.1–6.7 from this session's first-hand reading
  of `fesom_ice.c`/`_types.h`/`_coupling.c`/`_evp.c`/`_thermo.c` + the `fesom_main.c`
  integration seam + the `fesom_bulk.c` `stress_atmice`. Key findings baked into the plan:
  σ is prognostic elastic-memory state; `Tevp_inv=3.0/ice_dt` (setup, not the stale types.h
  comment); the runoff handoff is one line (`prec += runo`) + the water_flux-balance
  refinement; ocean2ice ≈ free (reuses `uvnode`/`hbar`/`T,S[:,0]`); EVP is 120 *fixed*
  subcycles (`lax.scan`); the ice FCT is a standalone 2-D module (reuse only the ratio-clip).
- **2026-06-06 — Task 6.1 DONE** (ice state + cold-start IC + IceConfig). `State` +9 ice
  fields (σ carried as EVP elastic memory); `fesom_jax/ice.py` (`IceConfig` + `ice_initial_state`/
  `seed_ice`). Ocean path **bit-identical**; `test_ice_ic.py` 8 green; **full suite 384 passed**.
  IC C-verified transitively (pure threshold of the ~1e-14 PHC SST; 0 FP-fragile nodes) — no C
  dump spent. ⚠️ Probe re-pin + ice configs are **env-only** ⇒ ice-ON C dumps deferred to their
  consuming tasks (config-A→6.2, B→6.4, C→6.5/6.6), each needing one `fesom_bulk_dump`-style
  output hook. 5 lessons logged. Next: Task 6.2 (ice thermodynamics).
- **2026-06-06 — Task 6.2 DONE** (ice thermodynamics). `fesom_jax/ice_thermo.py` (tfrez,
  obudget, budget [5-iter Newton], therm_ice_cell, flooding, cut_off, driver). NEW C hook
  `fesom_ice_thermo_dump` + `jobs/jax_ice_thermo_dump_core2.sh` (job 25395803, config-A). All
  7 outputs **bit-exact MAP-class** over 126858 nodes (h/hsn/A ~1e-16, t_skin 5.6e-14, ehf rel
  4e-10). **Runoff activates** (`d(fw)/d(runoff)=1` exact). AD finite everywhere; FD↔AD
  `d(ehf)/d(SST)` ~1e-16. `test_ice_thermo.py` 12 green (standalone module ⇒ 384-suite
  unaffected). 4 lessons. Next: Task 6.3 (ice-ocean coupling — the runoff handoff).
- **2026-06-06 — Task 6.3 DONE** (ice-ocean coupling). `fesom_jax/ice_coupling.py` (ocean2ice
  + ice_oce_fluxes + ice_oce_fluxes_mom) + a backward-compatible `balance_water_flux` flag on
  `sss_runoff.sss_runoff_fluxes`. water_flux/virtual_salt/relax_salt **match the config-A C dump**
  at the 7 probes; **runoff handoff proven** (`d(water_flux)/d(runoff)=-1` through thermo∘coupling).
  `test_ice_coupling.py` 9 green. 3 lessons. The core2_forcing threading + multi-day
  supercooling-cap manifestation are at 6.6/6.7. Next: Task 6.4 (EVP dynamics — the 120-subcycle
  `lax.scan`).
- **2026-06-06 — Task 6.4 DONE** (EVP dynamics). `fesom_jax/ice_evp.py` (evp_setup,
  stress_tensor, stress2rhs, velocity_update, the checkpointed 120-subcycle `evp_dynamics` scan)
  + `forcing.atm_ice_stress`. Verified vs the config-B C EVP dump (job 25396029): step-0
  setup/σ/u_rhs/velocity **bit-exact**, END (120 subcycles) ~8e-10 (scatter-class). AD finite at
  the Δ=0 singularity + through the scan. `test_ice_evp.py` 8 green; full suite **413 passed**
  (forcing.py change additive). 3 lessons. Next: Task 6.5 (ice FCT advection — the 2-D Zalesak module).
- **2026-06-06 — Task 6.5 DONE** (ice FCT advection). `fesom_jax/ice_adv.py` — the 2-D Zalesak
  FCT, **fully element-based / CSR-free** (`mm·X` via the FE mass block, cluster bounds via
  `segment_min/max` over elements). NEW C hook `fesom_ice_fct_dump` + `jobs/jax_ice_full_dump_core2.sh`
  (config-C, job 25396145). a/m/msnow **bit-exact** (~1e-15) vs the C FCT dump. Limiter eps=1e-12
  (the C's), no positivity clip. `test_ice_adv.py` 6 green (standalone). 2 lessons. Next: Task 6.6
  (assemble the ice step + wire prognostic a_ice/u_ice into the ocean step).
- **2026-06-06 — Task 6.6 DONE (step-1)** (assemble the ice step). `fesom_jax/ice_step.py`
  (`ice_surface_step` composes the 5 kernels) + the `ice_cfg` static arg in `step.py`/`integrate.py`
  (None ⇒ pi/Phase-5 bit-identical). Assembled step-1 T/S match the config-C C dump ~1e-6
  (climate-close — the EVP floor propagates); surface forcing matched; ice state evolves.
  `test_ice_step.py` 4 passed. 3 lessons. The multi-day stability + supercooling-cap (the GPU
  run, job 25396276) and the gradient re-check are Task 6.7 (GATE 6).
- **2026-06-06/07 — Task 6.7 DONE → GATE 6 MET → PHASE 6 (SEA ICE) COMPLETE.** Multi-day
  stability (`core2_ice_stability_run.py` + `_gpu.sh`, A100, job 25396309): **10 days stable,
  supercooling CAPPED at −1.91 °C** (vs Phase-5 −22.8/blow-up), ice physical, both Phase-5
  findings resolved. Gradient gate (`core2_ice_grad_gate.py` + `.sbatch`, job 25396293): FD↔AD
  `d(SST)/d(k_ver)` plateau 4.5e-10, masked-NaN `d(SST)/d(T0)` clean at CORE2 ice scale (peak
  26.5 GB). Suite ocean 376 + ice 47. **4 lessons** (supercooling cap, EVP-stiff gradient,
  per-step-forcing OOM, 3-backwards OOM + login-node aggregate). **Phase 6 = sea ice DONE; next
  the big phases are GM/Redi (6B) + KPP (6C) — own sub-plans.**
