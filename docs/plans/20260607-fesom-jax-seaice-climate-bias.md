# FESOM2 → JAX Port — Sea-ice climate-bias investigation (Phase 6 follow-up)

**Created:** 2026-06-07. **Status:** ✅ RESOLVED + VERIFIED (2026-06-07, both years). The high-lat sea-ice bias was **NOT a kernel port bug**: the climate run
stepped the ocean at `dt=1800` but built `IceConfig()` with the default `ice_dt=500`, so the entire ice
subsystem (thermo rates, FCT transport, EVP `dte`/`Tevp_inv`) integrated **3.6× too slowly**. A full
static re-audit cleared every ice kernel + constant (see §1 UPDATE); the bug was the **`ice_dt`↔`dt`
desync**. Fix: derive `ice_dt = ice_ave_steps·dt` inside `ice_surface_step` (mirrors C
`fesom_ice_setup`). **Masked for all of Phase 6/6B/6C because every prior gate ran at `dt=500`**, where
`ice_dt=500` is coincidentally correct — the `dt=1800` climate run was the first to expose it. See the
Revision Log (2026-06-07 #2) + `[[porting-lessons-log]]` (the `ice_dt` ROOT-CAUSE section).

> **Original framing (kept for the record):** ⏳ OPEN — the first real multi-year climate comparison
> (after KPP landed) revealed a **high-latitude sea-ice climate bias** in the assembled CORE2 model.
> KPP itself is sound; the bias is a **prognostic sea-ice (Phase 6)** issue, surfaced now because every
> prior gate was step-level/per-kernel — this is the first end-to-end climate diff.

**Parent:** `docs/plans/20260605-fesom-jax-port.md`. **Predecessors:** Phase 6 sea ice
(`20260606-fesom-jax-phase6-seaice.md`, GATE 6) + Phase 6C KPP (`20260607-fesom-jax-kpp.md`, GATE 6C —
the KPP work is complete + committed `8b2fcd2`).

---

## 0. THE FINDING — a high-lat marginal-sea-ice climate bias (NOT KPP)

Ran the full production model (**KPP + GM/Redi + prognostic ice**, dt=1800, 1958–1959, monthly means)
and compared annual-mean surface fields to the **C-port-KPP** + **Fortran-KPP** references via
`port_kokkos/scripts/m32_climate_compare.py` (the exact methodology the Kokkos port used).

**Bottom line:** JAX-vs-C-port-KPP **sst RMS 0.49 °C, bias −0.15 °C, corr 0.999** — but the
inter-reference budget (C-port-vs-Fortran, and Kokkos-CUDA-vs-C) is **~0.005–0.014 °C**. So JAX has a
**~35–100× excess**. It is **entirely high-latitude**:

| SST band (JAX−C, 1958) | bias | RMS |
|---|---|---|
| tropics/subtropics [−45,+45] | ~0 | **0.006–0.024 °C** ← *bit-faithful budget* |
| N. subpolar [+45,+66) | −0.324 | 0.770 |
| Arctic [+66,+90) | −0.357 | 0.712 |
| Antarctic [−90,−60) | −0.068 | 0.211 |

**The open ocean matches C to the bit-faithful budget → KPP / dynamics / forcing-in-the-open-ocean are
correct.** The residual is the cold poles, **surface-trapped** (temp Δ: −0.15 @2.5 m → −0.016 @45 m →
~0 below 200 m), co-located with sea ice. Hotspots (|Δ|≈3.5 °C): Sea of Okhotsk (~140°E,52°N),
Bering/Chukchi (~−162°E,66°N) — classic **seasonal-ice seas**.

**The fingerprint — `m_ice` (volume) flips sign by hemisphere:**

| region | m_ice bias | JAX vs C | a_ice bias |
|---|---|---|---|
| Arctic [60,90) | **−0.148 m** | 0.91 vs 1.06 (too THIN) | +0.074 |
| Antarctic [−90,−55) | **+0.266 m** | 1.15 vs 0.89 (too THICK) | +0.030 |

`a_ice` (concentration) is mildly high at **both** poles (thermo-like); `m_ice` (thickness) is
**opposite-sign** (dynamics/redistribution-like). The drift is **bounded & non-growing** (SST bias
−0.150→−0.140 yr1→yr2; RMS even shrank). corr stays 0.999. So: a stable, systematic, hemisphere-
asymmetric ice bias — *not* a runaway, *not* a config error, *not* KPP.

## 1. RULED OUT (don't re-audit — done 2026-06-07)

- **KPP / open ocean** — matches C to 0.006–0.024 °C (the budget). KPP is sound.
- **Config** — JAX run used KPP + GM-on + ice + `ice_gamma_fct=0.5` + dt=1800, **matching** the C ref
  `job_kpp_5yr_fix` (KPP, no `FESOM_NO_GMREDI` ⇒ GM on, γ=0.5, dt=1800). Not a config mismatch.
- **EVP DYNAMICS / METRIC TERMS / CORIOLIS (the first suspect — all bit-faithful to C):**
  - strain rates `eps11 −= mfac·v̄`, `eps12 += mfac·ū` — formula matches `fesom_ice_evp.c` line-for-line;
  - stress divergence `±s·mfac/3` — matches C;
  - **`metric_factor` value bit-exact**: `tan(rot_lat_centroid)/R_earth`, max|Δ|=0.0 vs recompute from
    `coord_nod2D` (centroid = mean-of-3-vertices, R=6367500); 92% geo-N/S sign-aligned (a *wrong* one
    WOULD give this signature — but it's right);
  - `coriolis_node = 2Ω·sin(geo_lat)` — ocean-validated (the CORE2 step gate matched `uv`);
  - velocity-update Coriolis 2×2 solve (`r_b=rdt(coriolis_node+ay·drag)`; `un=det(r_a·rhsu+r_b·rhsv)`;
    `vn=det(r_a·rhsv−r_b·rhsu)`) — sign matches `fesom_ice_evp.c:388-437` exactly;
  - ice advection is scalar (no metric term) — consistent with C.
  ⇒ the per-step ice **dynamics is faithful**. **The metric terms are correct.**

**Why the step-level gates (GATE 6/6C) missed this:** cold-start step 1 has `uv=u_ice=0` ⇒ shear=0,
`dVsq=0`, strain=0 ⇒ every velocity/shear-dependent term is *zero* there, and the multi-step climate
was never compared until now. The bias only emerges as the circulation/ice spins up over months.

## 2. HYPOTHESES (the next session's targets — thermo + forcing + advection)

Since per-step **dynamics is faithful** but a **systematic** bias exists, something else is per-step
systematically different (faithful dynamics + faithful thermo ⇒ pure chaos ⇒ ~0 bias; we see a bias).
The opposite-N/S `m_ice` with same-sign `a_ice` is consistent with **one thermo/forcing offset acting on
two regimes** (Arctic perennial vs Antarctic seasonal ice sit at opposite points of the growth–melt
cycle). Audit, most-likely first:

1. **Ice THERMODYNAMICS** — `ice_thermo.py` ↔ `fesom_ice_thermo.c`. Growth/melt, the ocean→ice heat
   flux (`o2ihf`/freezing), snow→ice conversion, conductive flux, albedo/shortwave over ice, the
   `flx_h`/`flx_fw` balance. A growth/melt-rate offset is the leading candidate for the regime-flip.
2. **Ice FORCING** — `atm_ice_stress` + the bulk over ice (`forcing.py`), the rain/snow partition,
   `srfoce_*` (ocean→ice surface state). Check anything that differs between hemispheres only via the
   atmosphere (it shouldn't, but verify the wind rotation / stress over ice).
3. **Ice ADVECTION / FCT** — `ice_adv.py` ↔ `fesom_ice_fct.c`. The Zalesak limiter
   (`ice_gamma_fct=0.5`) — a transport-asymmetry could redistribute thickness; check the low-order/
   antidiffusive flux + the limiter, and the boundary handling.

**Diagnostic strategy (step-1 gates can't see this — need multi-step / regional):**
- **Spatial maps** (already have the tooling): `scripts/kpp_bias_map.py` writes `bias_map_<yr>.nc`
  (ushow-able). Map `a_ice_diff`/`m_ice_diff`/`temp_diff` + look at WHERE in the growth/melt cycle.
- **A later-step ice dump-gate:** the C KPP dump has 2 steps; step 2 has nonzero shear/velocity — a
  step-2 (or a fresh multi-step) C ice dump would test the velocity-dependent ice paths the step-1
  gate couldn't. Consider a short (e.g. 1–3 month) C ice run dumping thermo growth/melt + `o2ihf`
  per node, then a controlled regional comparison.
- **Single-column / regional isolation:** pick a hotspot node (Okhotsk ~140°E,52°N) and trace the ice
  thermo terms JAX vs C over the first months.
- **Sanity:** confirm the pattern in 1959 (`kpp_bias_map.py --year 1959`) — expected same.

## 3. TOOLS + HOW TO REPRODUCE (all committed)

- **Run:** `scripts/core2_kpp_climate_run.py` (monthly means, C-port format) +
  `scripts/core2_kpp_climate_gpu.sh` (A100, ~1 h for 2 yr). Output → `data/kpp_climate_2yr/
  <var>.fesom.<yr>.monthly.nc` (sst/sss/ssh/a_ice/m_ice 2-D + temp/salt 3-D, 12 rec/yr; ushow-able).
- **Compare:** `port_kokkos/scripts/m32_climate_compare.py <dir> --label JAX --years 1958 1959`
  (defaults: `--fref /scratch/a/a270088/fortran_kpp_5yr_fix --cref /work/ab0995/a270088/port/kpp_5yr_fix`).
  Annual-mean surface corr/bias/RMS + drift. Inter-ref budget: run with `--label C-port` on the cref dir.
- **Bias map:** `scripts/kpp_bias_map.py --year 1958` → `data/kpp_climate_2yr/bias_map_<yr>.nc` +
  lat-band + hotspot + depth printout.
- **ushow a live/finished file:** `HDF5_USE_FILE_LOCKING=FALSE ushow <file.nc>` (HDF5 lock; `--yac-3d`
  for 3-D; `--polar north`). See `[[hpc-job-file-conventions]]`.

## 4. REFERENCE DATA (persisted)

- **C-port-KPP** (canonical): `/work/ab0995/a270088/port/kpp_5yr_fix/<var>.fesom.<yr>.monthly.nc`
  (1958–62, monthly means, C commit `6ecabe8`, KPP/GM-on/γ=0.5/dt=1800).
- **Fortran-KPP:** `/scratch/a/a270088/fortran_kpp_5yr_fix/` (1958–62, 12-rec/yr) and
  `/scratch/a/a270088/fortran_2yr_dt1800/` (2 yr + `fesom.mesh.diag.nc`).
- **PP refs** (only if running PP): `/scratch/a/a270088/fortran_pp_2yr`, `/work/.../port/pp_2yr_rebase`.
- **JAX run output:** `data/kpp_climate_2yr/` (= `/work/ab0995/a270088/port_jax/data/kpp_climate_2yr/`).
- Kokkos precedent (the bar): CUDA-vs-C-port-KPP **sst corr 1.0000, RMS 1.4e-2 °C** (1958) — see
  `port_kokkos/docs/REFERENCE_RUNS.md` (the canonical catalog + `m32_climate_compare.py`).

## 5. KEY PATHS / COMPUTE

- JAX repo: `/home/a/a270088/port_jax` (git `main`). Env python:
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`. GPU: `sbatch -A ab0995_gpu`.
- C SoT: `/home/a/a270088/port2/fesom2_port/src/` (`fesom_ice_thermo.c`/`_fct.c`/`_evp.c`/`_coupling.c`).
  **C edits → port2 `jax-mesh-export`, NEVER main.** Compute conventions: `[[hpc-job-file-conventions]]`.

---

## Revision Log
- **2026-06-07 — Created.** First multi-year climate comparison after KPP. Found the high-lat
  marginal-sea-ice bias (sst RMS 0.49 °C / −0.15 °C, all high-lat; `m_ice` opposite-sign N/S). Ruled
  out KPP/open-ocean (bit-faithful budget), config, and the entire EVP dynamics/metric/Coriolis
  (bit-faithful; `metric_factor` max|Δ|=0). Next: ice thermo → forcing → advection audit.
- **2026-06-07 #2 — ROOT CAUSE FOUND + FIXED.** The audit cleared every remaining ice kernel
  (`ice_thermo`, FCT `ice_adv` incl. the Zalesak limiter, EVP `evp_setup` strength/mass, `ice_coupling`,
  `atm_ice_stress`/bulk) AND all `IceConfig` constants — all bit-faithful to the C, `h0=h0_s=0.5`
  (so `lid_clo` is not the asymmetry). With the kernels clean, the bias localized to the **wiring**:
  the climate run (`core2_kpp_climate_run.py:180`, all the `core2_*` scripts) builds `IceConfig()`
  with the **default `ice_dt=500`** while stepping the ocean at **`dt=1800`** ⇒ the ice integrates
  3.6× too slowly (`ice_dt` feeds thermo `×ice_dt`, FCT `×ice_dt`, EVP `dte`/`Tevp_inv`). **Masked
  because every prior gate ran at `dt=500`** (step tests + all stability/grad scripts), where the
  default is coincidentally right. The opposite-sign `m_ice` is the sluggish ice relaxing toward the IC
  (seeded SH `m_ice=2.0` thicker than NH `1.0`, the reverse of the C equilibrium NH 1.06 > SH 0.89);
  `a_ice` high at both poles = retained concentration ⇒ cold SST. **Fix:**
  `cfg = cfg._replace(ice_dt=cfg.ice_ave_steps*dt)` at the top of `ice_surface_step` (mirrors C
  `fesom_ice_setup`; no-op at dt=500 ⇒ all 55 ice/step tests stay green). **Before** (buggy, year 1958,
  `kpp_bias_map.py`): SST bias −0.150/RMS 0.490; bands [+45,+66)=−0.324/0.770, [+66,+90)=−0.357/0.712,
  [−90,−60)=−0.068/0.211. **AFTER (verified, year 1958, `data/kpp_climate_2yr_fix/`, GPU job 25406976):
  SST bias −0.0004 / RMS 0.0107 °C — a 46× drop, INSIDE the inter-reference budget (0.005–0.014 °C).
  Bands: [+45,+66)=0.0165, [+66,+90)=0.0093, [−90,−60)=0.0031 (was 0.77/0.71/0.21 → 45–80× better);
  tropics unchanged (0.0064→0.0057). m_ice RMS 0.196→0.0030 (opposite-sign fingerprint erased);
  a_ice RMS 0.072→0.0027. Hotspots are now ordinary coastal SST-gradient nodes (Korea Strait, Kola)
  at ~0.3 °C, not the Okhotsk/Bering ice seas.** ⇒ ROOT CAUSE CONFIRMED; the full ice climate now
  matches C to the bit-faithful budget everywhere incl. the poles. **1959 corroborates (even tighter):
  SST RMS 0.0071, m_ice RMS 0.0012, a_ice RMS 0.0006; all SST bands < 0.011 °C (Arctic 0.0029,
  Antarctic 0.0056) — no growing drift, the ice has spun into agreement with C.** Run = GPU job
  25406976, `exit 0`, 24/24 months. The `srfoce_u/v` 1-step lag is sub-dominant and clearly not
  material at this residual. **ISSUE CLOSED → Phase 7a (param tuning) unblocked.**
