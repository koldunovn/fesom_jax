# Next-session prompt — FESOM2 → JAX port (investigate the high-lat SEA-ICE climate bias)

> **⚡ STATE (2026-06-07): the full model (KPP + GM/Redi + prognostic ice) runs 2 yr stable at dt=1800,
> and the FIRST real climate comparison vs the C-port-KPP + Fortran-KPP references surfaced a
> HIGH-LATITUDE SEA-ICE BIAS.** JAX-vs-C-port-KPP SST = **0.49 °C RMS / −0.15 °C bias / corr 0.999**,
> but the inter-reference budget (C-vs-Fortran, Kokkos-CUDA-vs-C) is **~0.005–0.014 °C** → a ~35–100×
> excess, **entirely in the high latitudes** (tropics/subtropics match C to the bit-faithful 0.006–0.024 °C).
> The fingerprint: **`m_ice` flips sign by hemisphere** (Arctic too THIN −0.15 m, Antarctic too THICK
> +0.27 m), surface-trapped, in the marginal-ice seas (Okhotsk/Bering). **KPP is sound; this is a Phase-6
> sea-ice issue** that only now showed up (every prior gate was step-level — this is the first climate diff).

## START HERE
1. **The investigation plan (source of truth — READ FULLY):**
   `docs/plans/20260607-fesom-jax-seaice-climate-bias.md` — §0 the finding (numbers + spatial map), §1
   what's **RULED OUT** (KPP/open-ocean, config, and the **entire EVP dynamics / metric terms /
   Coriolis — all bit-faithful to C; `metric_factor` max|Δ|=0**, so DON'T re-audit those), §2 the
   hypotheses + diagnostic strategy, §3 tools, §4 reference data, §5 paths.
2. **Why step-1 gates missed it:** cold-start `uv=u_ice=0` ⇒ all shear/velocity/strain-dependent terms
   are zero at step 1, and the multi-year climate was never compared until now.
3. **Lessons:** `docs/PORTING_LESSONS.md` (the K.8/K.9 lru_cache tracer-leak trap + the climate-vs-step
   finding). **STANDING RULE: append a lesson per task.** **Memory:** `[[fesom-jax-port]]`,
   `[[hpc-job-file-conventions]]` (incl. the `HDF5_USE_FILE_LOCKING=FALSE ushow` live-file trick).

## IMMEDIATE WORK — audit ice THERMO → FORCING → ADVECTION for the systematic offset
The per-step ice **dynamics is faithful** (ruled out), so a **systematic** bias means thermo/forcing is
per-step systematically different. The opposite-N/S `m_ice` (with same-sign `a_ice`) ⇒ likely **one
thermo/forcing offset acting on two regimes** (Arctic perennial vs Antarctic seasonal ice). Order:
1. **`ice_thermo.py` ↔ `fesom_ice_thermo.c`** (leading): growth/melt, ocean→ice heat flux
   (`o2ihf`/freezing), snow→ice conversion, conductive flux, albedo/shortwave-over-ice, `flx_h`/`flx_fw`.
2. **Ice forcing**: `atm_ice_stress` + bulk-over-ice (`forcing.py`), rain/snow partition, `srfoce_*`.
3. **`ice_adv.py` ↔ `fesom_ice_fct.c`**: the Zalesak limiter (γ=0.5), low-order/antidiffusive flux, BCs.

**Diagnostics (step-1 gates can't see this — go multi-step/regional):** the spatial bias map
(`scripts/kpp_bias_map.py`, ushow `bias_map_1958.nc`); a **later-step or short multi-month C ice dump**
(thermo growth/melt + `o2ihf` per node) for a controlled regional compare (the step-1 dump had ice
velocity = 0); single-column trace at a hotspot (Okhotsk ~140°E,52°N). Confirm the pattern in 1959.

## REPRODUCE / TOOLS (all committed)
- Re-run climate: `sbatch scripts/core2_kpp_climate_gpu.sh` (A100 ~1 h) → `data/kpp_climate_2yr/<var>.fesom.<yr>.monthly.nc`.
- Compare: `<env-py> port_kokkos/scripts/m32_climate_compare.py data/kpp_climate_2yr --label JAX --years 1958 1959` (KPP refs are the defaults).
- Bias map: `<env-py> scripts/kpp_bias_map.py --year 1958` → `data/kpp_climate_2yr/bias_map_1958.nc`.
- View: `HDF5_USE_FILE_LOCKING=FALSE ushow data/kpp_climate_2yr/bias_map_1958.nc [--yac-3d|--polar north]`.

## REFERENCE DATA (persisted)
- C-port-KPP (canonical): `/work/ab0995/a270088/port/kpp_5yr_fix/<var>.fesom.<yr>.monthly.nc` (KPP/GM-on/γ=0.5/dt=1800).
- Fortran-KPP: `/scratch/a/a270088/fortran_kpp_5yr_fix/` + `/scratch/a/a270088/fortran_2yr_dt1800/` (has `fesom.mesh.diag.nc`).
- Bar: Kokkos CUDA-vs-C-port-KPP sst RMS **1.4e-2 °C, corr 1.0** (`port_kokkos/docs/REFERENCE_RUNS.md`).

## KEY PATHS / COMPUTE
- JAX repo `/home/a/a270088/port_jax` (git `main`). **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python` → `JAX_PLATFORMS=cpu … -m pytest`.
  Heavy/full-suite/CORE2-backward → `sbatch scripts/run_suite.sbatch`; GPU → `-A ab0995_gpu -p gpu --gres=gpu:1`.
- C SoT: `/home/a/a270088/port2/fesom2_port/src/` (`fesom_ice_thermo.c`/`_fct.c`/`_evp.c`/`_coupling.c`).
  **C edits → port2 `jax-mesh-export`, NEVER main.** Large files → `/work` (`data/`→`/work/.../port_jax/data`).
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu).

## STATUS OF THE BROADER PORT
- **Phases 0–6 + 6B + 6C ALL COMPLETE + committed** (`8b2fcd2` = KPP K.8–K.11, GATE 6C). Suite 483 green.
  The KPP forward is bit-faithful per-step; the per-kernel/AD/stability gates all pass. **The one open
  item is this sea-ice climate bias** (a Phase-6 issue surfaced by the first climate comparison).
- **Phase 7a (differentiable parameter tuning, `docs/plans/20260607-fesom-jax-paramtune.md`) is the
  eventual next phase — DEFERRED until the ice bias is understood/accepted** (you want the forward
  climate to match C before calibrating). `optax 0.2.8` installed; the `calibrate.py` seam is designed.

## WORKFLOW
- Tick/log in the investigation plan; **commit per-task on `main` when asked**; append a lesson per task.
- See memory `[[fesom-jax-port]]`, `[[porting-lessons-log]]`, `[[hpc-job-file-conventions]]`.

Confirm you've absorbed this; then start the **ice-thermo audit** (`ice_thermo.py` ↔ `fesom_ice_thermo.c`)
— the leading candidate for a growth/melt offset that thins Arctic + thickens Antarctic ice. Read the
investigation plan §1 first so you don't re-audit the (already-cleared) metric/dynamics.
