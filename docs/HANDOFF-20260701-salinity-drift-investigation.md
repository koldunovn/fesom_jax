# Handoff 2026-07-01 — CORE2 JAX global-mean SALINITY DRIFT (investigation for a separate session)

---

## ✅ RESOLVED 2026-07-01 (same day, follow-up session)

**Root cause: a sublimation term breaks the zstar freshwater-budget closure — the drift is
VOLUME-driven (freshwater leak), not a salt source.** None of the ranked hypotheses below (§5)
was it; the balance code, runoff routing, fixed-area weighting, and surface BC are all faithful
ports. The bug is one mis-surfaced field.

**The diagnostic that cracked it (do this FIRST next time): global-mean SSH.** Area-weighted
⟨SSH⟩: **JAX drifts −0.442 m/63 yr, Fortran flat.** Predicted drop if S̄ rose purely by volume
loss (`−H·ΔS/S̄`, H=3627 m) = −0.444 m → **0.5% match** → salt content is conserved, the ocean is
losing ~7 mm/yr of freshwater, and S̄=salt/V rises. This kills every salt-source hypothesis (floor,
brine/rsf, restoring) at once. (script: area-weighted `ssh` from the monthly zarr via
`common.ushow_to_nodes` + Fortran `ssh.fesom.YYYY.nc` via `common.fortran_iter_months`.)

**The bug.** Under zstar, `oce_fluxes` removes the global-mean freshwater flux via
`flux = evaporation − ice_sublimation + prec_rain + prec_snow·(1−a_old) + runoff − thdgr·ρi/ρw −
thdgrsn·ρs/ρw`. Sublimation is ice→atmosphere (NOT ocean freshwater) so it must cancel — which
requires **`evaporation` to be the BUNDLED `evap_ow + subli`**, so `evaporation − ice_sublimation
= evap_ow` (exactly what `flx_fw` contains). Fortran bundles it at the LAST line of `therm_ice`
(`ice_thermo_oce.F90:651` `evap=evap+subli`) before storing `evaporation(i)` (`:324`); the C-port
bundles it too (`fesom_ice_thermo.c:407` `*evap=_evap+_subli`). **The JAX regressed**:
`ice_thermo.py:254` surfaced `evaporation=evap` = the **open-water** value (from the `evap*(1−A)`
line, before the bundle — the comment "= C's _evap" was the misread), so the balance computed
`evap_ow − subli`, leaving `−⟨subli·A⟩` uncancelled. That injects a net freshwater deficit
**every step**, applied UNIFORMLY by the global-mean correction — which is exactly the near-uniform
+0.031 psu surface offset (§1). ⟨subli·A⟩≈2.2e-10 m/s ≈ 3.6 cm/yr over ice (realistic).

**The fix (applied).** `fesom_jax/ice_thermo.py`: `ThermoOut.evaporation = evap` → `evap + subli`
(1 line + comments). Only consumer is `ice_coupling.fresh_water_balance_zstar` (zstar+ice path);
linfs untouched; no dump/test compares this field. **Verified** by running the real
`fresh_water_balance_zstar`: buggy ⟨water_flux_bal⟩=+1.0e-9 m/s (=−⟨subli⟩), fixed=−2e-25 (machine
zero). Full write-up in `docs/PORTING_LESSONS.md`; memory `salinity-drift-investigation.md`.

**Remaining (end-to-end):** rerun a CORE2 zstar+ice segment with the fix → confirm ⟨SSH⟩/S̄ flat;
regenerate `paper_jax/data/drift.nc` + the F4 drift figures; drop the "one caveat" text.

*(Everything below is the original investigation handoff, kept for the record — its §5 hypotheses
were all wrong, but §3's path trace and the SSH-diagnostic idea in §6/§7 led to the fix.)*

---

**Mandate for the next session:** find the root cause of the small JAX-only global-mean
salinity drift in the CORE2 1958–2020 hindcast, and decide fix-vs-caveat. This is a
**model-code** investigation in `port_jax` (the figures that surfaced it are already done —
see §7). The user's steer: *"we had a very similar problem in the **initial C port**
(`port2/fesom2_port`) — look there for hints"* (§4). It was **not** a zstar-specific issue.

---

## 1. The finding (quantified)

The matched JAX vs Fortran CORE2 hindcast agrees on almost everything (SST RMSE 0.604 vs
0.606 °C; full-depth T̄ gap 0.006 °C). The **one** place the port visibly diverges is the
**global volume-mean salinity**:

- **JAX vol-mean S drifts +0.0044 psu over 1958→2020** (clean ~linear, 34.727→34.7315).
  The matched Fortran run stays **flat** (34.727, seasonal wiggle only). `|JAX−Fortran|₂₀₂₀ ≈ 0.0043 psu`.
- Both start bit-identical at 34.727 from the same PHC IC ⇒ it's a real **salt/freshwater-budget**
  difference that accumulates, not a plotting/reduction artefact.

**Vertical structure** (from `sz_jax` in `drift.nc`, S(z,2020) − S(z,1958); Fortran ΔS(z) ≈ flat):

| depth band | JAX ΔS | reading |
|---|---|---|
| surface, top ~20 m (2.5 m) | **−0.122 psu** | strong **freshening** |
| ~35–45 m | +0.03 | thin subsurface salt gain |
| ~125–375 m (min ~210–250 m) | **−0.028** | intermediate **freshening** |
| **450 m → bottom** | **+0.003 … +0.013** | **broad deep salinification** |

The deep band is tiny per-layer but spans most of the ocean volume ⇒ **it dominates the
+0.0044 psu volume-mean rise.** So the volume mean goes *up* even though the surface freshens:
net **salt gained** / **freshwater lost**, redistributed with the surface fresher and the
interior saltier.

**Surface spatial signature** (from `meanstate.nc`, annual-mean 1980–2009 JAX − Fortran at the surface):

- **SSS: mean +0.031 psu, std 0.009** ⇒ mean/std ≈ 3.6 ⇒ a **near-uniform positive offset**
  (JAX saltier at the surface almost everywhere). This is the classic signature of a *global
  budget* difference, **not** roundoff/chaos (which would be zero-mean).
- **SST: mean +0.005 °C, std 0.053** ⇒ mean ≈ 0, a **zero-mean coherent pattern** (warm tropical
  Atlantic, cool N/W Pacific). Likely *downstream* of the density/circulation response to the
  salinity difference, or an independent numerical divergence — a secondary question.

**Net:** the signal is a slow, systematic **freshwater deficit / salt source** in JAX vs the
matched Fortran, mixed into the interior. Sign + near-uniform surface offset both point at the
surface freshwater/salt **budget closure**, not local physics.

---

## 2. Status / why it matters

- **Paper:** the drift is currently reported **honestly as the one caveat** (GMD model paper).
  User decision 2026-07-01: keep it as a caveat for the paper **and** investigate the cause in a
  separate session (this doc). Not a blocker for the paper.
- **Not a runaway:** +0.0044 psu / 63 yr is small and ~linear; T/OHC/sea-ice all co-track. But
  it's a genuine conservation difference worth understanding (and likely cheap to localize — §6).

---

## 3. The exact model path (ice-ON + zstar — what the hindcast actually runs)

Config: `configs/core2_full.yaml` — **all-on**: zstar ALE + TKE + **mEVP sea ice (`ice.whichEVP=1`)**
+ GM, opt_visc=7, dt=1800, tracer (0,1). Because ice is **on**, `step.py` takes the **ice branch**,
so the surface salt/freshwater BCs come from the ice step, **not** `core2_forcing.compute_surface_fluxes`.

Trace (all in `fesom_jax/`):

1. **`step.py`** — `use_virt_salt = ale_cfg.use_virt_salt` (`:137`); zstar ⇒ **`use_virt_salt=False`**
   (real fresh-/salt-water fluxes, `is_nonlinfs=1`). Ice vs no-ice branch at `:186–221`; zstar live
   geometry (`zbar_3d_n`, per-node/elem) at `:171–176`. The zstar tracer-surface term
   `−dt·sval·water_flux·is_nonlinfs` is added **here** (post-advection) before `impl_vert_diff`.
2. **`ice_step.py:ice_surface_step`** —
   - calls `ice_thermo.fesom_ice_thermodynamics(...)` → produces `flx_fw`, `flx_h`, `thdgr/thdgrsn`,
     `real_salt_flux` (`:114`).
   - `ice_coupling.ice_oce_fluxes(srf.salt, th.flx_fw, th.flx_h, Ssurf_month, runoff_node, …)` (`:133`).
   - under zstar: `water_flux = ice_coupling.fresh_water_balance_zstar(water_flux, evap,
     ice_sublimation, prec_rain, prec_snow, a_ice, runoff_node, thdgr, thdgrsn, areasvol_surf,
     ocean_area, …)` (`:144–146`).
   - **`bc_S = dt·(virtual_salt + relax_salt + real_salt_flux·is_nonlinfs)`** (`:166`). Under zstar
     `virtual_salt≡0`, so `bc_S = dt·(relax_salt + real_salt_flux)` **plus** the step.py zstar term.
3. **`ice_thermo.py:therm_ice_cell`** — **runoff enters ONCE**: `prec = rain + runo + snow·(1−A)`
   (`:190`, Fortran `ice_thermo_oce.F90:318`); `fw = prec + evap + fwice + fwsnw` (`:229`);
   `flx_fw = fw` (`:322`). Under zstar (`:226–227`): `fwice` **unscaled**, `rsf = fwice·Sice` (the
   real-salt producer). ⇒ **runoff is folded into `flx_fw` (non-ICEPACK convention).**
4. **`ice_coupling.py`** —
   - `ice_oce_fluxes` (`:70–98`): `water_flux = −flx_fw`; calls
     `sss_runoff_fluxes(…, balance_water_flux=False)` (`:87–89`) — **drops** the standalone
     `⟨water_flux+runoff⟩` term (runoff already inside `flx_fw`). ✓ (this is the C-port lesson applied)
   - `fresh_water_balance_zstar` (`:101–123`): `flux = evap − ice_sublimation + prec_rain +
     prec_snow·(1−a_ice_old) + runoff − thdgr·ρice/ρwat − thdgrsn·ρsno/ρwat`; `net = ⟨flux⟩`
     (area-weighted **global mean**); `water_flux += net`. Ports `fesom_ice_coupling.c:178–216`.
5. **`sss_runoff.py`** — `sss_runoff_fluxes` (`:254–308`): `relax_salt = surf_relax_S·(Ssurf − S_top)`,
   **minus its area-weighted global mean** (`:294–298`); `virtual_salt` similarly (≡0 under zstar).
   `_area_mean` (`:239–251`) = `Σ(x·areasvol_surf)/ocean_area` — **weights by the FIXED at-rest
   surface area** (`fs.areasvol_surf`, `fs.ocean_area`).

**Key observation:** every global-mean removal (`relax_salt` mean, `fresh_water_balance_zstar`
`net`) uses the **fixed** `areasvol_surf`/`ocean_area`. Under **zstar the surface-layer volume
varies in time**, so "subtract the global mean" does **not** exactly zero the *actual* salt/
freshwater input applied to the (thicker/thinner) surface cells ⇒ a small non-zero residual each
step ⇒ compounds over 63 yr. **This is the leading structural suspect** (matches the near-uniform
+0.031 psu surface offset).

---

## 4. ⭐ THE C-PORT PRECEDENT — look here first (per the user)

The **initial C port** hit a *very similar* freshwater/salinity-budget bug. Primary source
(read these):

- **`port2/fesom2_port/docs/PORT_EXPERIENCE_REPORT.md` §4.5 "Runoff routing depends on which
  thermodynamics path is active"** — the crux. FESOM2 has **two contradictory conventions** for
  getting runoff into `water_flux`:
  - **Non-ICEPACK / standard** (`ice_thermo_oce.F90:therm_ice`): `prec = rain + runo + snow·(1−A)`
    → `fw = prec + evap + fwice + fwsnw`; `oce_fluxes` writes `water_flux = −fresh_wa_flux`,
    **no separate `−runoff`** (runoff already inside `fw`).
  - **ICEPACK** (`ice_oce_coupling.F90:378`): `water_flux = −(fresh_wa_flux·inv_rhowat) − runoff`
    (runoff subtracted *separately*).
  - **The bug:** if the port mirrors one path in the ice thermo and the auxiliary
    `fesom_sss_runoff_step` mirrors the other, **runoff is double-counted (or missing)**. The C
    port *silently mirrored ICEPACK* pre-ice (runoff subtracted in `sss_runoff`, no `therm_ice`),
    then had to **remove that subtraction in lock-step** when sea ice landed
    (`feedback_runoff_fold_when_ice.md`).
- **`port2/fesom2_port/docs/plans/20260425-sea-ice-port.md`** — Tasks **B4b + C3b** (lines 19–25,
  78, 277, 290, 349–353): the paired "fold runoff into `flx_fw` inside `therm_ice`" **and** "remove
  the `−runoff` subtraction in `fesom_sss_runoff`" contract handover. This is the canonical fix.
- **`PORT_EXPERIENCE_REPORT.md` §3.1 (lines 62, 71):** `dt/areasvol` vs **`dt·area/areasvol`** —
  *"Even when `area/areasvol == 1` for linfs, port the ratio so the code survives zstar."*
  **Under zstar `area/areasvol ≠ 1`.** If any surface-BC / global-mean weight silently assumes the
  linfs identity (fixed `areasvol_surf`, §3 above), it imprints mesh geometry / breaks conservation.
- **`PORT_EXPERIENCE_REPORT.md` §3.1 (lines 135–141):** the **conservation diagnostic recipe** that
  cracked their non-conservative-SSH drift — *print the global integral over **owned** nodes after
  assembly; a machine-zero on 1 rank that grows is a leak.* Re-use this shape for the salt/FW budget.
- **`port2/fesom2_port/docs/MPI_PORT_REPORT.md:154–188`** — their *residual* drift lived in the
  `integrate_nod_2D` reductions for **virtual_salt mean / relax_salt mean / water_flux balance**
  (the same three reductions the JAX path has). NB: that particular report is about *multi-rank
  roundoff* drift (a different, smaller effect); the **runoff-routing / area-ratio** lessons above
  are the relevant ones for a *systematic* single-config drift.

**What the JAX port already does right (so don't re-chase these):** the JAX ice thermo **does**
fold runoff into `flx_fw` once (`ice_thermo.py:190`) and the ice path **does** pass
`balance_water_flux=False` (`ice_coupling.py:89`) — i.e., the exact C-port double-count bug looks
**avoided**. ⇒ The JAX drift is most likely a **residual in the global-mean balance under zstar**
(fixed-area weighting, or the `fresh_water_balance_zstar` reconstruction not exactly matching the C),
**not** a gross runoff omission. But **verify the budget closes** (§6) — that's precisely where the
C port's drift hid.

C reference line numbers (for a byte-level re-check against `port2/fesom2_port/src/`):
`fesom_ice_coupling.c:125–179` (ice-on salt balance), `:178–216` (`fresh_water_balance` zstar),
`fesom_sss_runoff_step:382–440` (salt balance), `ice_thermo_oce.F90:318` (prec fold), `:359–372`
(fw + rsf), `ice_oce_coupling.F90:378` (ICEPACK path).

---

## 5. Ranked hypotheses (refined)

1. **Fixed-area global-mean under time-varying zstar volume (leading).** `relax_salt` mean and
   `fresh_water_balance_zstar` `net` both weight by fixed `areasvol_surf`/`ocean_area`
   (`sss_runoff.py:239–251`). Under zstar the actual surface-cell volume changes ⇒ the mean-removal
   leaves a small net salt/FW each step ⇒ slow, near-uniform accumulation. Best matches the +0.031
   near-uniform surface offset **and** the C-port `area/areasvol` lesson (§4).
2. **`fresh_water_balance_zstar` reconstruction ≠ Fortran.** Its `flux` (`ice_coupling.py:117–120`)
   re-assembles the FW flux from `evap/prec/thdgr/thdgrsn/runoff` rather than reusing `flx_fw`. Any
   term/sign/`a_ice_old`-timing mismatch vs `fesom_ice_coupling.c:178–216` → wrong `net` → residual.
3. **Ice-thermo `real_salt_flux` (brine/melt) net budget.** Under zstar `rsf = fwice·Sice`
   (`ice_thermo.py:227`) enters `bc_S`. Net brine rejection (deep salt) vs melt (surface FW) has the
   *exact shape* of the observed drift (deep saltier, surface fresher). Check its global integral and
   the flooding salt-conservation correction (`:243`, Fortran `:645–649`).
4. **`relax_salt` SSS restoring interaction.** Both models restore to PHC (`surf_relax_S=10 m/60 d`);
   but the fixed-area mean-subtraction + restoring toward a target the two models sit differently
   against can imprint the structured surface pattern. Cheap to isolate (toggle restoring, §6).

---

## 6. Concrete diagnostic plan (cheap — the drift is slow but the per-step residual is instant)

The per-step budget residual is measurable in a **short** run (months), no 63-yr rerun needed.

1. **Instrument the global salt/FW budget per step** (add a diagnostic, gated by env like the C port's
   `FESOM_DIAG_SSHRHS`). Print, per step (or per day), over **owned** nodes:
   - `S_in_fixed  = Σ(bc_S · areasvol_surf)`         — net salt tendency, fixed-area weight → should be ~0.
   - `S_in_live   = Σ(bc_S · live_surf_volume)`      — same with the **zstar live** surface thickness.
     `S_in_live − S_in_fixed` **is hypothesis-1's residual** (expect ≠0, ~the drift rate).
   - `FW_in_fixed = Σ(water_flux · areasvol_surf)` after `fresh_water_balance_zstar` → should be ~0.
   - `Σ(real_salt_flux · area)` — the net brine/melt salt budget (hypothesis 3).
   Compare the machine-zero-ness the way §4's recipe does. A term that's zero under fixed-area but
   nonzero under live-volume confirms hypothesis 1.
2. **Cross-check vs Fortran.** The matched Fortran CORE2 run
   (`/work/ab0995/a270088/fesom2_core2/`) can emit the same global salt/FW conservation diagnostic
   (FESOM has `oce_ale`/salt-budget diagnostics). Compare the per-step net salt input directly.
3. **Toggle experiments (short CORE2 segments, cheapest first):**
   - (a) recompute the global-mean subtraction with the **live zstar surface volume** instead of fixed
     `areasvol_surf` → does the drift vanish? (confirms hyp 1; likely the fix).
   - (b) disable SSS restoring / vary `surf_relax_S` → isolates `relax_salt` (hyp 4).
   - (c) freeze ice thermo / zero `real_salt_flux` → isolates brine/melt (hyp 3).
4. **Regional decomposition of the surface offset.** Break `meanstate.nc:sss_jaxfor` by region: uniform
   (⇒ hyp 1/2 global budget) vs polar/ice-covered (⇒ hyp 3 brine) vs high-runoff coasts (⇒ runoff
   routing). Discriminates the hypotheses without a rerun.

Start with **1** (instrumentation) + **3a** — they directly test the leading suspect and are days of
compute at most (short CORE2, or even a small mesh like `farc`/`dars` for the per-step invariant —
per [[model-paper-test-on-small-meshes]] the budget residual should show on any zstar+ice config).

---

## 7. Data & reproduction inventory (nothing to recompute for the diagnosis of §1)

- **Derived data already on disk** (the figures read only these):
  - `paper_jax/data/drift.nc` — `sz_jax`/`sz_fortran` `[time,z]` (per-level S), `tz_*`, `sbar_*`,
    `ohc_*`, `tbar_*`; coords `time` (757 mo, 1958–2021), `z` (47 mid-depths). `has_fortran=1`.
  - `paper_jax/data/meanstate.nc` — `sss_jaxfor`/`sst_jaxfor` `[nod2]` surface JAX−Fortran maps;
    attrs `rms_sss_jaxfor=0.0321`, `rms_sst_jaxfor=0.0580`.
- **Reproduce the §1 depth/spatial numbers** (nereus env
  `/work/ab0995/a270088/mambaforge/envs/nereus/bin/python`):
  ```python
  import numpy as np, xarray as xr
  ds = xr.open_dataset("paper_jax/data/drift.nc")
  sz = ds["sz_jax"].values                     # [time, z]
  dS = sz[-1] - sz[0]                           # 2020 − 1958 by level
  for zk, d in zip(ds["z"].values, dS): print(f"{zk:7.1f} m : {d:+.5f}")
  print("vol-mean S drift:", float(ds["sbar_jax"][-1]-ds["sbar_jax"][0]))
  ```
  (full script at `port_jax/scripts/diag_salinity_drift.py`; also verifies
  `OHC ≡ ρ₀c_p·V·T̄` exactly — the OHC "difference" is **not** a salt signal, it's the 0.006 °C T̄
  gap rescaled; don't chase it.)
- **Raw runs:**
  - JAX CORE2 hindcast monthly: `/work/ab0995/a270088/port_jax/runs/core2_hindcast/monthly/<YYYY>_<MM>`
    (ushow folded zarr; unfold via `paper_jax/scripts/common.ushow_to_nodes`).
  - Fortran CORE2: `/work/ab0995/a270088/fesom2_core2/` (`{temp,salt,…}.fesom.<YYYY>.nc`, native node order).
- **Config:** `port_jax/configs/core2_full.yaml`.
- **Model code:** the §3 files under `port_jax/fesom_jax/` (`ice_step.py`, `ice_thermo.py`,
  `ice_coupling.py`, `sss_runoff.py`, `core2_forcing.py`, `step.py`).

---

## 8. Cross-references

- This session's figure work (what surfaced it): `docs/HANDOFF-20260630-figures-nereus.md` §3;
  figures now show the drift honestly — Fig 2 SSS JAX−Fortran (near-uniform +0.03 offset) + Fig 3
  panels (b)/(e)/(f) (vol-mean S drift, S(z) Hovmöller, ΔS(z) profile). Previews in
  `docs/previews-20260701/`.
- Memory: [[model-paper-plan]] (open salinity finding), [[fortran-forca20-setup]] /
  [[fesom-jax-perchunk-recompile-and-nondeterminism]] (run machinery),
  [[model-paper-test-on-small-meshes]] (test the invariant on farc/dars first).
- C port: `port2/fesom2_port/docs/PORT_EXPERIENCE_REPORT.md` §4.5 + §3.1;
  `.../plans/20260425-sea-ice-port.md` B4b/C3b; memory `feedback_runoff_fold_when_ice.md`.

**One-line summary for the next session:** JAX CORE2 gains ~+0.0044 psu/63 yr (Fortran flat) —
near-uniform surface salt offset + deep salinification ⇒ a **surface freshwater/salt budget residual
under zstar+ice**; the C port hit this class of bug (runoff routing / `area/areasvol` ≠ 1 under
zstar) — instrument the per-step global `Σ(bc_S·area)` fixed-vs-live-volume and toggle the live-volume
weighting first.
