# CORE2 single precision (Tier B, B3): sub-plan

Sub-plan of `docs/plans/20261020-core2-gpu-speed.md`, Task 13. Branch `perf/core2-sp` (worktree
`/work/ab0995/a270088/fesom_jax_dev/wt_sp`), from `perf/core2-gpu` 65b528f. Nothing here is measured unless a job id
or a cited source says so.

## Overview

**Goal.** An opt-in single-precision (SP) mode for the CORE2 production physics on GPU (zstar + cvmix-TKE + mEVP +
GM, dt 1800, `dist_4`) that **repeats upstream FESOM2's single-precision build** (the stack ending in PR #1207) and is
faster than double precision (DP). DP stays the default everywhere and stays bit-for-bit what it is today.

**Scope.** `precision: single` in a run config (default `double`) and `--precision single` on `run_from_config.py` and
`scripts/perf/gpu_profile.py`. SP is accepted only for the CORE2 production physics (zstar, TKE, mEVP, GM, no
w_split, no KPP); any other combination raises. The adjoint stays DP and refuses SP. SP gradients are out of scope
(a later Tier C plan).

**User decisions (2026-10-20):**
- **Policy = FESOM2 #1207, nothing more** (Q2): fesom-jax is the shadow of FESOM2. Whatever upstream keeps in
  double (`WP_full`, `real(kind=8)`, `cvmix_r8`, `real64`) is f64 in JAX; everything upstream runs at `WP=4` is f32.
  No extra islands. Where the JAX structure forces a difference it is listed under "Deviations" for the user to
  decide; this plan does not decide them.
- The XLA excess-precision guard stays: it is needed to reproduce #1207's arithmetic.
- Fortran anchor on zstar (Q3, "everything on z*"): a FESOM2 CORE2 zstar DP/SP pair from a fresh clone of upstream
  main (Task 1b).
- The tracer carry is saved in SP restarts (Q4, yes) — upstream does not save it (Deviation D6).
- The 10-year DP reference is shared with the parent plan's dt 2700 twin and runs at dt 1800 (Q5); this plan does
  not depend on dt 2700 (the user doubts 45 min is stable for FESOM2).
- Task 1 approved up to ⛔ STOP S1; the precision-policy tasks (2-10) wait for the user.

**Design in five lines.**
1. **Mechanism: fesom3-jax's** (`~/fesom3_jax/fesom3_jax/precision.py`): an SP chunk is traced and run with
   `jax_enable_x64` off and `jax_explicit_x64_dtypes = "allow"`, so every dtype-less array is float32 (= `WP`) and
   only explicit `.astype(float64)` sites (= `WP_full`) stay float64. Mixed f32/f64 arithmetic promotes as Fortran's
   mixed-kind expressions do. All helpers are the identity in DP, so **the DP jaxpr does not change** (fesom3's gate
   A). Our JAX 0.10.1 has both switches (checked: `jax.enable_x64`, `jax._src.config.explicit_x64_dtypes`).
2. **The f64 list is read off the upstream code** (table below, with file:line in the clone).
3. **XLA must not fold f64→f32→f64 convert pairs**: SP jits get `compiler_options={"xla_allow_excess_precision":
   False}`. Task 1 showed on A100 that without it the carry is destroyed (below).
4. **Conservation gate from day one:** a port of upstream's `fesom_conservation` budgets (volume, heat, salt against
   the applied surface fluxes, carry included) in every DP and SP run; SP − DP judged against the Fortran zstar
   pair's SP − DP.
5. **DP and the adjoint locked:** StableHLO hashes of the DP programs (forward, sharded, adjoint reverse) recorded at
   the base commit and re-checked after every task; `core2_adj.py` refuses SP.

## Results so far

### Task 1 — the convert-pair experiment (2026-10-20, f87b3c5, snapshot `wt_sp_f87b3c5`)

`scripts/perf/sp_convert_pairs.py`: three f64 kernels on 4096 points with exact-product inputs (so FMA cannot change
a bit and numpy is an exact reference) — `carry` (FESOM2's `acc = T·h + d; T' = f32(acc/h_new); carry =
f32(acc − T'·h_new)`), `feed` (`f64(f32(x))·2 − x·2`, a rounded value read by later f64 code) and `scan` (the carry
8 times in a `lax.scan`). Variants: (a) default flags; (b) per-jit `compiler_options={"xla_allow_excess_precision":
False}`; (c) the same flag process-wide in `XLA_FLAGS`; (d) default flags, rounding by `lax.reduce_precision(x, 8, 23)`
in f64; (e) default flags, `lax.optimization_barrier` on the f32 value. Results in
`/work/ab0995/a270088/fesom_jax_dev/runs/perf/sp_convert_pairs/` (CPU job 28288483, A100 job 28288484).

| variant | CPU: all fields bitwise numpy | A100: carry | A100: feed | A100: scan T / carry | A100 optimised HLO: convert (f32→f64) |
|---|---|---|---|---|---|
| a default | yes | **3748/4096 carries 0, max err 2.3e-4** | **all 0** (rounding folded) | **T off by 1.9e-6 (one f32 ulp), carry 3732 zeros** | 2 (0) — pairs removed |
| b per-jit option | yes | bitwise | bitwise | bitwise | 3 (1) |
| c process flag | yes | bitwise | bitwise | bitwise | 3 (1) |
| d reduce_precision | yes | bitwise | bitwise | bitwise | 2 (0), 2 reduce-precision kept |
| e optimization_barrier | yes | bitwise | bitwise | bitwise | 3 (1) |

A float64-only program compiles to the identical optimised HLO with and without option (b), on CPU and A100.
XLA:CPU keeps every pair, so a CPU test can never see the failure. This reproduces fesom3's GH200 finding on A100
and our JAX 0.10.1. **Recommendation: (b), the per-jit option on SP programs only** — it covers every rounding in the
program at once (fesom3: the fold also removed the rounding of the tracer value between the two carry sites, which a
per-site fix misses), and DP programs are compiled exactly as today. (d) and (e) also survive the default flags but
are per-site; they are not planned. A GPU test with the negative control (variant a) stays in the suite (Task 7).
Cost: A100 35 s (~0.01 GPU-h), CPU 22 s.

### Tasks 2-9 status (2026-10-21, perf/core2-sp 358e1b9)

- Implementation done (2cd3fd4 … 1e4f8c4): precision switch, the #1207 float64 sites, carry, point-slope forcing,
  SP preconditioner variant, budget diagnostic, triplet tooling; merged with perf/core2-gpu up to 2db760b (D7 in the
  shared COO/ELL `_spmv`, D1 in the explicit increment). DP lock green (baseline 2db760b = 1e4f8c4 on all three
  programs; single host thread, one process per program).
- Speed, 4 A100, full GPU set, same node alternating (job 28295120): DP 35.85 / 36.00, SP 25.11 / 24.72 ms/step —
  **SP −31 %** (sp/dp 0.69). The first measurement (470 ms) was a per-chunk recompile (fixed e8f1e77).
- 30-day triplet (job 28292762): T1 volume means pass (S +5.8e-9 psu, T −8.5e-8 °C, volume 9.4e6 m³); the
  budget-closure numbers are reported as a diagnostic only (see "Budget criterion" below).
- **Budget criterion — for the user.** The `--budget` diagnostic tracks total salt (ocean + ice) from the State; it
  does not integrate the fluxes the model applies, as upstream's `fesom_conservation` does. It does not close in DP
  (+7.2e10 psu m³ in 28 days; likely the zstar surface advective flux S·water_flux, which has no counterpart term in
  `bc_S` and is not counted), so the SP − DP difference of it (1.05e10 psu m³ / 28 d) mixes different surface fluxes
  with any leak and is not comparable with the Fortran CONS residual. Options: (a) port `fesom_conservation`
  (accumulate the applied volume/salt/heat fluxes per step in the scan); (b) drop the budget part of T1/T2 and judge
  on volume means, the f32-IC envelope and the Fortran zstar pair. Until decided it is printed, not judged.
- 1-year triplet (job 28295391, d5e7b13, 3 × 38 min on 4 A100): **T2 pass** — S̄ +6.2e-8 psu (≤1.5e-7), T̄ −3.8e-7 °C
  (≤2.5e-6), global-mean SSH +3.4e-8 m (≤5e-7), pointwise rms SST/SSS/SSH/a_ice/m_ice 2.5e-2/1.7e-2/1.1e-3/6.7e-3/
  6.2e-3 against 3× the envelope's 2.0e-2/1.3e-2/6.9e-4/6.1e-3/5.7e-3. Fortran zstar pair: S̄ +3.1e-8, T̄ −2.6e-7,
  SSH +9.6e-8. Budget diagnostic: SP volume +1.18e7 m³/yr (Fortran SP 3.7e7), total salt SP 2.18e11 vs DP 8.47e10
  psu m³ (DP itself does not close; diagnostic only).
- GH200 (coordinator, job 28295508): DP 20.4, SP 12.9 (−37 %), SP + ordered sums 10.7 ms/step.
- CG iterations: the first SP and DP solves both take 42 iterations with the same residual and tolerance; no SP excess
  is confirmed. The comparison of later steps did not finish: with FESOM_CG_ITERS=1 the CPU runs on e6beb67 segfault
  (jax.debug.print; the diagnostic only, production runs are unaffected) — open.
- ⛔ STOP 4 reached (2026-10-21): results to the user; no long SP run started.

### Task 1b — FESOM2 zstar Fortran anchor: one CORE2 year, SP − DP (2026-10-20, jobs 28288582 / 28288583)

Upstream e96a2b13, R2 namelists (zstar, cvmix_TKE, mEVP, GM/Redi, JRA55-do, dt 1800, PHC cold start 1958-01-01),
dist_512 on 4 nodes, both runs finite. State on 1959-01-01 (`/work/ab0995/a270088/fesom2_sp_zstar_runs/
compare_year_sp_minus_dp.txt`, script `jobs/compare_restarts.py`):

| quantity | DP | SP − DP | FESOM_F-SP (linfs) SP − DP |
|---|---|---|---|
| volume-mean S [psu] | 34.72713386 | **+3.07e-8** | +2.26e-8 |
| volume-mean T [°C] | 3.621834185 | **−2.58e-7** | +4.08e-6 |
| area-mean SST / SSS | 18.0826 / 34.5881 | −3.0e-4 / −9.4e-5 | −4.5e-5 / +2.8e-5 |
| global-mean SSH [m] | 1.4e-16 | **+9.6e-8** | −9.1e-10 |
| ice area NH / SH [10¹² m²] | 14.483 / 7.444 | +1.5e-3 / +2.2e-2 | −1.2e-5 / +5.7e-3 |
| ice volume NH / SH [10¹² m³] | 24.014 / 9.377 | −1.1e-3 / −1.5e-3 | −1.2e-4 / +1.6e-2 |
| pointwise SST / SSS / SSH area-rms | | 2.2e-2 / 1.4e-2 / 7.4e-4 | 1.4e-2 / 7.8e-3 / 7.5e-4 |

Upstream budgets (`conservation_freq = 1440`, last CONS line, step 17,280 = day 360; change since start, integrated
surface flux, residual):

| budget | DP residual (ratio to change) | SP residual (ratio to change) |
|---|---|---|
| volume [m³] | 2.6e2 (change 2.6e2) | **3.7e7** (change 3.7e7, flux 1.7e5) ≈ 1.0e-7 m of global-mean SSH |
| heat [J] | −3.49e18 (−4.2e-5) | −3.48e18 (−4.2e-5): the scheme's own residual, the same in both |
| salt [psu m³] | 1.29e4 (5.6e-10) | **1.28e9 (5.6e-5)** ≈ 1e-9 psu of volume-mean S per year |

Reading: in SP the salt budget closes to ~1e-9 psu a year, so the +3.1e-8 psu SP − DP difference of S̄ is mostly
different surface fluxes on a diverging trajectory, not a leak; the SP volume does not close to f64 (+3.7e7 m³ a
year, the SP global-mean SSH rising ~1e-7 m a year — upstream's open item, the f32 SSH/hbar path). Speed on 512 CPU
ranks: 60.8 → 46.0 ms/step (SP/DP **0.76**), CG iterations mean 88.5 → 59.1 (the #995 preconditioner). Cost: builds
0.06, smokes ~0.1, year pair 2.15 node-h.

## Context (from discovery)

### Upstream FESOM2 SP (fresh clone `/work/ab0995/a270088/fesom2_sp_zstar`, e96a2b13, 2026-10-20)

Contains #1054 (0ef286b6), #1056 (9971594c), #1206 (d15df321), #1207 (50467fa0). `~/fesom2_sp` (one-commit clone at
#1038, without #1206/#1207, with someone's uncommitted instruments) is not used. Kinds: `WP` real32 under
`USE_SINGLE_PRECISION`, `WP_full = real64` always, `use_tracer_carry = (WP /= WP_full)` (`src/oce_modules.F90:17-31`);
`MPI_WP_FULL` (`src/MOD_PARTIT.F90:15-29`). Upstream's own summary: `docs/getting_started/single_precision.rst`.
**#1056 (FCT low order in flux form) and #940 (forcing point-slope form) are unconditional upstream, in DP too**
(`src/oce_adv_tra_driver.F90:89-219`, `src/gen_surface_forcing.F90:200-210`); the carry is the only SP-only branch in
the tracer code (`src/oce_ale_tracer.F90:864`, `:1615`).

Upstream `fesom_conservation` (`src/fesom_conservation.F90:1-30`): with `conservation_freq=N` in `&run_config`, rank 0
prints every N steps volume, heat and salt integrals, their change, the integrated surface flux and the residual,
in `WP_full`, carry included (`:197`, `:228`).

### fesom3-jax (`~/fesom3_jax`, read-only)

- Mechanism and FESOM_F-SP policy port: `fesom3_jax/precision.py`, handoff `docs/plans/20261019-single-precision-jax.md`
  (merged into `main`). Its SP step is bitwise against a FESOM_F-SP gfortran oracle on the oracle's inputs; its DP
  programs are unchanged (identical StableHLO for 10 DP programs).
- Speed (`docs/PERFORMANCE.md:313-339`, CORE2 dt 2700, split-explicit): sp/dp 0.65-0.69 on 1 GH200, 0.74-0.78 on 4
  GH200, 0.81 on 8; CG + gathered SSH at dt 1800 in the year run 0.83. FESOM_F-SP on 512 CPU ranks 0.837.
- **The salt leak** (`docs/PERFORMANCE.md:341-392`, fixed in 4fc80c5, merged 9145681): XLA:GPU's default
  `xla_allow_excess_precision=true` folded the convert pairs; one CORE2 year SP − DP was −1.71e-5 psu / −4.83e-5 °C
  (job 28261473), 10 years +2.2e-4 psu. With the per-jit option: +6.9e-9 psu / +5.1e-6 °C (job 28274495; FESOM_F-SP
  +2.26e-8 / +4.08e-6), no speed change. A DP chunk holds no float32 at all (job 28283210), so the flag cannot
  change DP.
- Merge lessons: JAX's x64 switch is per thread (a forcing thread that places device arrays must enter the scope);
  batching helpers must not concatenate f32 with f64; a host f64 placed with x64 off is narrowed (`keep_f64`).

### fesom-jax today (this worktree)

- x64 on globally at import (`fesom_jax/config.py:31`); explicit float64 in constructors (`state.py:110, 149-159`,
  `params.py:86-131`, `gm.py:333`, `ice_evp.py:66`, `ice.py:247-249`, `ssh.py:203-233, 448-450`).
- `State` (`state.py:34-97`, registered at `:191-192`); `del_ttf` is a State field (`state.py:41`).
- Reductions: one choke point `reductions.global_sum` (`reductions.py:25-38`); dense CG dot `jnp.sum` (`ssh.py:649`).
- SSH: CSR `segment_sum` matvec/precond (`ssh.py:455-483`), PCG (`ssh.py:621-701`), zstar stiffness increment rebuilt
  matrix-free from `hbar` each step (`ssh.py:704-751`), `rtol` (`ssh.py:920-922`), `hbar` update (`ssh.py:993`).
- zstar: `hnode_new = hnode + Δzbar·dd` (`ale.py:301`), live geometry by reverse cumsum (`ale.py:222-231`).
- Tracers: FCT low order (`tracer_adv.py:300-316`), FCT `del_ttf` via `−T·hnode + LO·hnode_new`
  (`tracer_adv.py:493`, the pre-#1056 form), ALE reconstruction (`tracer_adv.py:144-153`); **Redi increments are
  added after the reconstruction** (`step.py:506-515`, upstream adds them into `del_ttf` before it); implicit
  vertical diffusion (`tracer_diff.py:68-129`).
- On-device forcing `rdate·coef_a + coef_b` (`surface_forcing.py:164-165`; the pre-#940 form, `rdate` ~2.44e6).
- Driver jit sites `integrate.py:223`, `integrate_sharded.py:648, 667, 917, 954, 958`; executable cache key
  `integrate_sharded.py:948-951`; in-scan output sums `:835-852`; restart `zarr_output.py:246, 306`;
  `exchange_pair` (`halo.py:127-139`).
- Adjoint: `scripts/adjoint/core2_adj.py` reads `configs/core2_full.yaml`, `assert_physics` at `:354-373`.

### Where a CORE2 step goes, and what SP can touch

Ledger `docs/PERFORMANCE_CORE2_GPU.md` (4 A100, dt 1800, options on, forcing prefetch; job 28284314): 73.92 ms/step,
kernels 70.0 ms per GPU. About half is NCCL (3-D halo all_gathers are bandwidth-bound, ~300 us each; CG psums and the
120 mEVP subcycle exchanges are mostly latency); SSH, mEVP and TKE kernels are launch-bound (475 / 518 / 1075
launches per step). **Estimate: 74 → ~52-57 ms/step on 4 A100 (sp/dp 0.70-0.77, ~67 → ~90 SYPD)**; 4 GH200
0.74-0.80 by fesom3's measurement. SP does not buy: launch-bound kernels, collective latency, host and compile time.

## Precision design

### The policy: what #1207 keeps in double (upstream file:line in the clone, `src/`)

| quantity / computation | upstream | JAX SP |
|---|---|---|
| layer thickness `hnode`, `hnode_new` | `WP_full` storage, `MOD_MESH.F90:140-145` | f64 State fields |
| its halo exchange | hi/lo split through `MPI_WP`, `oce_ale.F90:4429-4443` | f64 exchange (D2) |
| restart of `hnode` | staged through WP (`hnode_wp`), `io_restart.F90:172-184` | widened f64 on disk (see D6) |
| tracer tendencies `del_ttf`, `del_ttf_advhoriz/advvert` | `WP_full`, `MOD_TRACER.F90:36-41` | f64 |
| FCT low-order tendency `fct_LO_tend` (flux form, #1054/#1056) | `WP_full`, `MOD_TRACER.F90:63`; edge scatter `oce_adv_tra_driver.F90:124`, vertical `:219` | f64, SP path only (D4) |
| flux·dt/areasvol products added to the tendencies | `real(flux, WP_full)…`, `oce_adv_tra_driver.F90:514, 549, 589-593` | f64 |
| explicit vertical diffusion and Redi increments | f32 increment added to `WP_full` `del_ttf`, `oce_ale_tracer.F90:1052-1054, 1715-1720, 1904` | same, into the accumulator before the rounding (D3) |
| ALE reconstruction + carry | `acc` `WP_full`, carry `WP`, `oce_ale_tracer.F90:864-869` | same |
| implicit vertical diffusion column (a, b, c, rhs, Thomas, `zbar_n`, `z_n`, `zinv`) + carry | `WP_full`, `oce_ale_tracer.F90:1074-1082`, carry `:1615-1622` | f64 compute, T/S rounded once, carry updated |
| carry `carry(nl-1, nod, ntr)` | `WP`, SP only, `MOD_TRACER.F90:42`; zeroed at setup `oce_setup_step.F90:482` | f32 State fields `carry_T`, `carry_S` (`None` in DP) |
| SSH CG scalars, dots, stopping test, row sums of A·x, M⁻¹r, A·p | `WP_full`, rounded once into `WP` vectors, `solver.F90:112-120, 146-157, 168-169, 193-194, 201-250` | f64 dots/scalars, f64 row sums rounded once (`segment_sum` in f64); vectors and matrix f32 |
| SSH preconditioner formula (#995) | `precond_variant = -1` = auto: 1 (textbook symmetric Jacobi `M⁻¹ = 2D⁻¹ − D⁻¹AD⁻¹`) in SP, 0 (as coded since 60b46bdc, = ours) in DP, `oce_setup_step.F90:723-727`, `solver.F90:65-80` | variant 1 in SP (built on the host), SP path only (D4); with Chebyshev on (`core2_gpu.yaml`) see D7 |
| zstar stiffness matrix accumulation | real64 shadow `values_full` rounded to WP once per step, `MOD_MESH.F90:29`, `oce_ale.F90:1556-1640, 1706-1763` | no accumulated matrix in JAX (D1) |
| global integrals (`integrate_nod_2D/3D`) and ocean areas | `WP_full` + `MPI_WP_FULL`, `gen_support.F90:326-377`, `oce_mesh.F90:2218-2244` | f64 inside `reductions.global_sum` |
| forcing time axis and interpolation | `time_t0`, `nc_time`, `rdate` real64, point-slope `coef_b + (rdate − time_t0)·coef_a` with `coef_b` the WP value, `gen_surface_forcing.F90:200-210, 963, 1129-1133` | host f64 `elapsed`, f32 `coef_a`/`coef_b`, SP path only (D4) |
| CVMix TKE column | `cvmix_r8`, WP↔r8 copies at the call, `cvmix_driver/gen_modules_cvmix_tke.F90:290-298` | f64 compute, outputs rounded to f32 |
| viscosity filter `visc_filt_bidiff` (opt_visc 7, ours) | locals `real(kind=8)` `u1, v1, len, vi, viLapl, update_u/v`, `oce_dyn.F90:745-747` | f64 compute, rounded into `uv_rhs` |
| output running means | real64 (`local_values_r8`), `io_meandata.F90:74, 109` | f64 in-scan sums |
| conservation budgets | `WP_full`, `fesom_conservation.F90` | f64 (Task 6) |
| split-explicit SSH (opt-in `ssh.split_explicit`, plan `20261021-core2-split-explicit.md` D5): forcing column sums, the M substeps (eta, U rings, <<U>>), the trim | upstream's SE file (a different scheme) has no `WP_full`: `oce_ale_ssh_splitexpl_subcycl.F90` at e96a2b13 (grep 0); `hnode`/`hnode_new` stay real64 in the zstar tail | f32 throughout (`ssh_se.py`; the host tables cast to WP), rings f32 State fields (f64 on disk); `hnode`/`hnode_new` f64 as everywhere |
| **everything else** — T, S, uv, w, eta, d_eta, hbar, ssh_rhs(_old), CG vectors, matrix and preconditioner, all sea ice incl. mEVP stresses, EOS/density/N²/pressure, PGF, GM/Redi slopes, momentum incl. its vertical solve, TKE state and Kv/Av, mesh geometry, bulk formulae, forcing values, shortwave `exp(z/sc)`, Zalesak limiter | `WP` | f32 |

Not used by the CORE2 configuration: the w-split implicit advection column (`WP_full`, `oce_adv_tra_ver.F90:27-33`),
KPP's SP `epsln` (`oce_ale_mixing_kpp.F90:59`), `init_ref_density`'s real(8) locals
(`oce_ale_pressure_bv.F90:3425`; `use_density_ref=.false.`, `eos.py:6-7`), the salt anomaly option (#986, off by
default). SP refuses w_split and KPP (YAGNI).

### Deviations the JAX structure forces (for the user; not decided here)

- **D1 zstar stiffness.** Upstream accumulates the per-step stiffness increments in a real64 shadow and rounds the
  matrix to WP once per step. JAX never accumulates a matrix: it rebuilds the increment from `hbar` each step
  (`ssh.py:704-751`), and `hbar` is WP (f32) upstream. Options: (i) rebuild from the f32 `hbar` (no extra state;
  the increment then carries `hbar`'s f32 rounding), (ii) keep an f64 copy of the telescoped `hbar` used only for the
  increment (closest to upstream's real64 shadow; one 2-D f64 field). Recommendation if asked: (ii).
- **D2 hnode halo.** Upstream sends f64 `hnode_new` as two f32 halves through `MPI_WP` (exact to 2⁻⁴⁸); JAX
  exchanges f64 directly. The halo values differ by at most 2⁻⁴⁸ relative; an MPI detail (fesom3 did the same).
- **D3 Redi order.** JAX DP adds the Redi increments after the reconstruction (`step.py:506-515`); upstream adds them
  into `del_ttf` before it. To repeat #1207's single rounding, the SP path adds them into the accumulator before the
  rounding; the DP path keeps its order (DP-bitwise rule). So DP and SP differ in where Redi enters.
- **D4 unconditional upstream changes.** #1056 (FCT low order in flux form) and #940 (point-slope forcing) apply in
  upstream DP too (and #995's preconditioner variant 1 is selectable in DP by namelist). Our DP is pinned to the C/Fortran reference it was validated against (and the adjoint), so they
  enter JAX as SP-only branches. Consequence: JAX DP and JAX SP use different arithmetic in these two places.
- **D5 mesh geometry.** Upstream SP computes the geometry in real32; JAX computes it on the host in f64 and rounds once
  (fesom3 the same; the real32 path is the less accurate one). A real32 host path is possible if the user wants
  bit-closeness to upstream from the cold start.
- **D6 carry in the restart** (user-approved Q4): upstream drops the carry (`io_restart.F90` has none); JAX writes it so
  an SP resume is bitwise on CPU. Under the "no departure" rule the user may want to revisit this.
- **D7 GPU-only options** with no upstream SP counterpart: the Chebyshev CG preconditioner of `core2_gpu.yaml`
  (`cheb_degree 3`) and the gathered SSH solve. Proposed: apply #1207's CG rule (each operator row summed in f64,
  rounded once) to every polynomial application; the gathered solve runs the same CG.

- **D8 rounding order within one f32 rounding — provisional, user to confirm.** JAX does not repeat Fortran's
  rounding expression by expression: readers of the float64 thickness take one rounded copy `wp(hnode)` where Fortran
  rounds each mixed-kind expression on assignment; per-edge accumulations into WP arrays (e.g. `U_c` of the bidiff
  viscosity) are summed in float64 and rounded once instead of rounded after every edge; float64 host geometry
  (D5). Each differs from upstream by at most one float32 rounding of the quantity. Provisionally accepted
  (coordinator, 2026-10-20) so the work can proceed to STOP 4.

### The switch (mechanism)

- `fesom_jax/precision.py` (new, after fesom3's): `scope(p)`, `sp()`, `full(x)` (= `real(x, WP_full)`), `wp(x)`
  (= `real(x, WP)`), `wp_dtype()`, `cast_mesh/cast_state/cast_tree/keep_f64`, `compiler_options()` (SP:
  `{"xla_allow_excess_precision": False}`, DP: `None`), and the policy lists.
- `RunConfig.precision: str = "double"` (`"double" | "single"`); CLI `--precision`. Chunk calls run inside
  `PR.scope`; the cache key at `integrate_sharded.py:948-951` gains the precision; every jit site passes
  `compiler_options=PR.compiler_options()` (`None` in DP ⇒ today's call).
- Host inputs cast once where placed: State f32 except `hnode`, `hnode_new` (f64) and the carry (f32); mesh f32;
  SSH operator f32; params f32; forcing `elapsed` from host f64. The forcing prefetch thread (1c82624) enters
  `PR.scope` and places f64 leaves with `keep_f64`; `exchange_pair` asserts equal dtypes.
- Guards: SP with linfs, KPP, w_split, EVP (whichEVP 0) or `ice_ave_steps != 1` raises; `core2_adj.assert_physics`
  rejects `precision != "double"`.
- Nets for silent promotion (an f64 value reaching an f32 field): `lax.scan` refuses a carry whose dtype changes; the
  Task 7 HLO allow-list fails on any materialised f64 3-D array that is not in the table.

### Why DP and the adjoint stay unchanged

Every helper is the identity in DP and `PR.sp()` is False, so a DP trace emits the same jaxpr; DP jits get no compiler
option; `core2_full.yaml` is untouched; `core2_adj.py` refuses SP. Checked by the StableHLO hashes of Task 2 after
every task, the CPU replay gates, and the parent plan's adjoint gate at Task 10.

## Development Approach

As in the parent plan (gate-first; each task complete with all tests green before the next; small commits on
`perf/core2-sp`; plan updated when scope changes; tests on compute nodes only, CPU via `scripts/adjoint/job_cpu.sbatch`,
GPU via `scripts/adjoint/job_gpu.sbatch`; new test files in `scripts/runs/suite_groups.txt`; GPU jobs from a frozen
snapshot worktree `wt_sp_<sha>` of a committed HEAD; each GPU job announced with its cost; Slurm waits by a
background `sacct` loop; a lesson per task in `docs/PORTING_LESSONS.md`, a ledger row per measurement; nothing deleted,
no permission changes, English commit messages without attribution, files staged explicitly). A100 only for this
plan's GPU work unless the user frees a dolpung node.

## Testing Strategy

- **DP lock** (every task): StableHLO hashes of the DP programs identical to Task 2's baseline; CPU gates green.
- **SP unit tests** (CPU, x64-off scope): helpers; casts per the policy table; an SP step returns every State field in
  its policy dtype; each upstream-f64 computation is f64 inside (lowered HLO).
- **Conservation unit tests** (CPU): with surface fluxes off, SP content (Σ T·hnode·area + carry) changes by ≤1e-12
  relative over 20 steps, like DP; negative control with the carry zeroed drifts ≥100× more.
- **GPU tests** (`-m gpu`): Task 1's carry kernel exact with the production option; negative control (default flags)
  folds; a 48-step CORE2 SP chunk on 1 A100 finite and conserving.
- **Sharded smoke**: SP step on 4 fake CPU devices finite, dtypes per policy, within f32 round-off of the dense SP step.
- **Fidelity**: thresholds T1-T3 (Technical Details) against the Fortran zstar pair.

## Progress Tracking

`[x]` when done, ➕ added tasks, ⚠️ blockers, ⛔ user stops.

## Implementation Steps

### Task 1: The decisive convert-pair experiment (no model code)

**Files:**
- Create: `scripts/perf/sp_convert_pairs.py` (jobs through `scripts/adjoint/job_cpu.sbatch` / `job_gpu.sbatch` from a
  snapshot worktree; variant c in a child process with `XLA_FLAGS` set)

- [x] script: carry / feed / scan kernels, variants a-e, bitwise vs numpy, zero counts, optimised-HLO convert counts,
  f64-only control (f87b3c5)
- [x] CPU (job 28288483): every variant bitwise
- [x] 1 A100 (job 28288484): (a) folds — carry zeroed, rounded state off by one ulp; (b)-(e) bitwise; the option leaves
  a f64-only program's HLO identical. GH200 not run (dolpung reserved by the coordinator)
- [x] result and recommendation in "Results so far"; ledger row and `docs/PORTING_LESSONS.md` entry go in with
  Task 2's first commit (the ledger lives on `perf/core2-gpu` too; written once, merged)
- [x] ⛔ STOP S1: reported; user 2026-10-20: go ahead with (b), D1-D7 as recommended, thresholds as proposed

### Task 1b: FESOM2 zstar Fortran anchor (CPU, outside this repo)

**Files (all new):**
- `/work/ab0995/a270088/fesom2_sp_zstar` (fresh clone of upstream FESOM2 main, e96a2b13)
- `/work/ab0995/a270088/fesom2_sp_zstar_runs/` (`build_{dp,sp}`, `jobs/{build,run}.sbatch`, `logs/`,
  `smoke_{dp,sp}/`, `year_{dp,sp}/`)

- [x] build DP and SP on a compute node (job 28288391, 3.5 min, 0.06 node-h): `configure.sh levante.dkrz.de
  -DUSE_SINGLE_PRECISION=OFF|ON`; binaries `build_dp/bin/fesom.x`, `build_sp/bin/fesom.x`
- [x] run dirs with the R2 namelists copied from `/work/ab0995/a270088/fesom2_core2_SUPERSEDED_experimental_icefct/`
  (zstar, cvmix_TKE, mEVP, GM/Redi, JRA55-do, dt 1800, PHC IC, dist_512 on 4 nodes); changes: `ResultPath`,
  `run_length` (2 d smoke, 1 y year), and in the year dirs `conservation_freq = 1440` (upstream's budgets every 30
  days)
- [x] smoke test, 2 model days = 96 steps on 512 ranks (jobs 28288534 DP / 28288535 SP, 44 s / 42 s): both finite, no
  NaN; banners `WP=8 bytes (DOUBLE PRECISION MODE)` / `WP=4 bytes (SINGLE PRECISION MODE)`; ocean+ice time per step
  52.4 / 38.3 ms (SP/DP 0.73); CG iterations mean 122.1 (DP) vs 78.2 (SP), max 142 / 86, no non-convergence — the
  difference is #995: in SP upstream switches the preconditioner to the textbook symmetric Jacobi
  (`oce_setup_step.F90:723-727`), which needs ~36 % fewer iterations on CORE2 (now in the policy table)
- [x] estimate: one year ≈ 17,520 × (~0.06 s DP, ~0.045 s SP) + start-up ≈ 19 + 14 min on 4 nodes ≈ 2.2 node-h
  (within the user's ~3 node-h); submitted as jobs 28288582 (DP) / 28288583 (SP), 50 min limit each
- [x] year pair ran (DP 18.3 min, SP 14.0 min wall on 4 nodes); compared, results in "Results so far"; thresholds
  T1-T3 re-derived from it (Technical Details), for the user to confirm at STOP S1

### Task 2: DP lock

**Files:**
- Create: `scripts/perf/dp_hlo_hash.py`, `fesom_jax/tests/test_dp_lock.py`, `fesom_jax/tests/data/dp_hlo_hashes.json`

- [x] `dp_hlo_hash.py`: lower (not compile) the DP chunk programs — dense step chunk with the CORE2 production physics
  on the test mesh, sharded chunk on 4 fake CPU devices (all_gather; gathered SSH), the `core2_adj.py` reverse program
  on a 2-step window — strip locations, hash, count f32/f64 ops
- [x] generate the JSON at the branch base (fb8841d = perf/core2-gpu 8980bf8 merged; job 28291307, 2 min: 0 float32
  in all three programs) on a compute node; `test_dp_lock.py` recomputes and compares (the
  JSON is regenerated only in a commit that says it changes DP, e.g. a merge from `perf/core2-gpu`)
- [x] run tests on a compute node - must pass before Task 3 (job 28291881 on the SP commit 2cd3fd4: DP programs
  identical)

### Task 3: The precision switch (plumbing)

**Files:**
- Create: `fesom_jax/precision.py`, `fesom_jax/tests/test_precision.py`
- Modify: `fesom_jax/run_config.py`, `scripts/run_from_config.py`, `scripts/perf/gpu_profile.py` (`--precision`)
- Modify: `fesom_jax/run.py`, `fesom_jax/integrate.py:223`, `fesom_jax/integrate_sharded.py` (scope, casts, compiler
  options, cache key), `fesom_jax/state.py` (`carry_T`, `carry_S`, default `None`), `fesom_jax/halo.py`
  (`exchange_pair` dtype assert), the forcing prefetch thread, `scripts/adjoint/core2_adj.py` (`assert_physics`)

- [ ] tests first: helpers are the identity in DP; SP dtypes of `full`/`wp`; casts per the policy table; `keep_f64`;
  `RunConfig` parses `precision`, default `double`; `core2_full.yaml`/`core2_gpu.yaml` load as `double`; SP with
  linfs / KPP / w_split / whichEVP 0 raises; `core2_adj.assert_physics` rejects `single`; `compiler_options()` is
  `None` in DP
- [ ] implement; replace explicit `jnp.float64` in physics constructors by `PR.wp_dtype()` where the table says WP
  (identical in DP); audit device-side int64 (x64 off makes them int32) and record the result
- [ ] the SP step must trace: `PR.wp` at every State assignment the scan's carry check flags
- [ ] DP lock + CPU gates green; run tests - must pass before Task 4

### Task 4: Upstream f64 sites outside the tracers

**Files:**
- Modify: `fesom_jax/reductions.py`, `fesom_jax/ssh.py` (CG dots/scalars/row sums; D1 per the user's answer),
  `fesom_jax/tke.py` / `fesom_jax/cvmix_tke.py`, `fesom_jax/momentum.py` (bidiff viscosity), `fesom_jax/surface_forcing.py`
  + `fesom_jax/jra55.py` (point-slope in SP, D4)
- Modify: `fesom_jax/tests/test_precision.py`

- [ ] tests first (CPU, SP scope): `global_sum` of f32 returns the f64 sum of the widened values; the SP CG on a small
  system reproduces upstream's formulas (f64 dots, row sums rounded once) against a numpy transcription of
  `solver.F90:146-250`; SP `mixing_tke` = round(DP `mixing_tke` of widened inputs) bitwise on CPU; SP bidiff
  viscosity = round(f64 computation); SP forcing value = `coef_b + f32(elapsed)·coef_a` with host-f64 `elapsed`
- [ ] implement under `if PR.sp():` / `PR.full` (identity in DP)
- [ ] DP lock + CPU gates green; run tests - must pass before Task 5

### Task 5: Tracer conservation (#1054/#1056/#1206/#1207)

**Files:**
- Modify: `fesom_jax/tracer_adv.py` (SP: f64 `fct_LO_tend` in flux form, f64 flux products, `LO = f32((T·hnode +
  fct_LO_tend)/hnode_new)`; reconstruction with the carry)
- Modify: `fesom_jax/gm_redi.py`, `fesom_jax/step.py` (SP: Redi content into the accumulator before the rounding, D3)
- Modify: `fesom_jax/tracer_diff.py` (f64 column, carry update)
- Create: `fesom_jax/tests/test_precision_conservation.py`

- [ ] tests first (CPU): closed-basin conservation (≤1e-12 relative over 20 steps, carry included) and the
  negative control; reconstruction and implicit step against a numpy transcription of `oce_ale_tracer.F90:864-869`
  and `:1615-1622` on hand-made columns; the salinity floor after the rounding
- [ ] implement under `if PR.sp():` branches only
- [ ] DP lock + CPU gates green; run tests - must pass before Task 6

### Task 6: I/O and the conservation diagnostic

**Files:**
- Modify: `fesom_jax/zarr_output.py` (restart: widen on write, carry written when present; DP restarts readable by SP
  and back), `fesom_jax/integrate_sharded.py:835-852` (f64 in-scan sums in SP), `fesom_jax/run.py`
- Create: `fesom_jax/conservation.py` (port of upstream `fesom_conservation`: volume, heat, salt integrals, integrated
  surface fluxes, residual, carry included; f64), `scripts/run_from_config.py` (`--conservation-freq N`, CONS lines in
  upstream's format)
- Create: `scripts/perf/compare_precision.py` (two restarts → the Task 1b comparison table)
- Modify: `fesom_jax/tests/test_restart.py`, `test_stream_output.py`; create `fesom_jax/tests/test_conservation.py`,
  `scripts/tests/test_compare_precision.py`

- [ ] tests first: SP restart round trip bitwise on CPU (carry included); DP restart bytes unchanged; SP output mean =
  f64 mean of the samples; budgets equal a numpy reference (dense and 4 fake devices) and close for DP to round-off
- [ ] implement (diagnostic off by default: DP lock unaffected)
- [ ] run tests - must pass before Task 7

### Task 7: Sharded SP smoke, HLO allow-list, GPU tests

**Files:**
- Create: `fesom_jax/tests/test_precision_shard.py`, `fesom_jax/tests/test_precision_gpu.py`

- [ ] sharded smoke (4 fake CPU devices, 4 steps): finite, dtypes per policy, within f32 round-off of the dense SP step
- [ ] HLO allow-list: every materialised f64 array with a level axis is in the policy table
- [ ] `@pytest.mark.gpu`: Task 1's carry kernel exact with the production option, negative control folds; a 48-step
  CORE2 SP chunk on 1 A100 finite, conservation residual ≤ 10× the CPU one
- [ ] CPU tests and GPU tests (1 A100, ~0.3 GPU-h) - must pass before Task 8

### Task 8: Speed (A100)

- [ ] `gpu_profile.py time --precision single|double`, `core2_gpu.yaml` options, at the Tier A HEAD of the day (merge
  `perf/core2-gpu` first; regenerate the DP hashes in that commit): 1 and 4 A100, 2 reps (~1.5 GPU-h)
- [ ] nsys + `hlo_kernel_map.py` on 4 A100 SP: per-area table next to DP; f64 3-D traffic beyond the table is a bug
  (~0.5 GPU-h)
- [ ] GH200 only if the user frees a dolpung node (~0.5 GPU-h)
- [ ] ledger rows; run tests - must pass before Task 9

### Task 9: Conservation gate, 30 days and one year (4 A100)

**Files:**
- Create: `scripts/perf/sp_triplet.sbatch` (DP, DP from the f32-rounded IC, SP; same commit; CONS budgets on)

- [x] 30-day triplet from the cold start 1958-01-01: T1 pass on volume means (job 28292762)
- [x] one-year triplet: T2 pass (job 28295391)
- [x] ledger section "Single precision"
- [x] ⛔ STOP 4 (parent plan): results to the user before any further ≥1-year SP run and before SP enters
  `core2_fast.yaml` — user 2026-10-21: SP accepted (after the 10-year pair of Task 10)

### Task 10: After STOP 4 — ten years, adjoint gate, suite

- [x] 10-year SP run and the shared dt 1800 DP reference (parent Task 11) at the same commit and options, monthly output,
  CONS budgets (~11 + ~15 GPU-h); T3; Tier B judgement with the parent plan's `compare_twins.py` — run as SP + ice-2
  vs DP at dt 2700, 1990-1999 on 4 GH200 (jobs 28304114/5, ledger "STOP 4 pair"): salt +2e-8 psu a year, mean SSH
  -1.5e-7 m in 10 years, T chaotic, all 120 months finite
- [ ] adjoint gate of the parent plan vs its Task 2 baseline (~1 GPU-h)
- [ ] full CPU suite + GPU tests green
- [ ] `docs/USER_GUIDE.md` (`precision`, supported physics, guards), `docs/PORTING_LESSONS.md`, ledger
- [x] ⛔ user decision: SP into `core2_fast.yaml` or not — yes (2026-10-21); it is `configs/core2_gpu_fast.yaml`

## Technical Details

**Fidelity thresholds** (proposed from the Fortran zstar pair, Task 1b; the user confirms them at STOP S1). Anchors:
Fortran zstar one year SP − DP S̄ +3.07e-8 psu, T̄ −2.58e-7 °C, global-mean SSH +9.6e-8 m, CONS residuals salt
1.28e9 psu m³ and volume 3.7e7 m³ per year; fesom3-jax's leak was −1.7e-5 psu / −4.8e-5 °C. DP-f32IC = JAX DP from
the f32-rounded initial T and S (the envelope of an f32-sized perturbation; T̄ is chaos-dominated, so its bound uses
the larger of the two).
- **T1, 30 days:** |S̄_SP − S̄_DP| ≤ 2e-8 psu (fesom3's leak would show −1.4e-6); |T̄_SP − T̄_DP| ≤ max(3× the DP-f32IC
  envelope, 1e-6 °C); JAX SP CONS salt and volume residuals ≤ 10× the Fortran SP rate scaled to 30 days.
- **T2, one year:** |ΔS̄| ≤ 1.5e-7 psu (5× Fortran); |ΔT̄| ≤ max(1.3e-6 °C (5× Fortran), 3× the DP-f32IC envelope);
  |Δ global-mean SSH| ≤ 5e-7 m (5× Fortran); CONS residuals salt ≤ 1.3e10 psu m³ and volume ≤ 3.7e8 m³ (10×
  Fortran SP); pointwise rms of SST, SSS, SSH, a_ice, m_ice ≤ 3× the DP-f32IC envelope or ≤ 2× the Fortran pair's
  rms, whichever is larger.
- **T3, ten years:** |ΔS̄| ≤ 1e-6 psu with no growth (mean yearly increment of years 8-10 ≤ 2× years 1-3), CONS
  residual rates as T2; maps and ice series judged against the DP run's interannual standard deviation with the
  criterion the user approves for the parent plan's twins.

**Cost** (A100: DP 74 ms/step, SP ~55 ms/step estimated; 17,520 steps/yr; 4 GPUs):

| step | GPU-h | CPU node-h | wall |
|---|---|---|---|
| Task 1 (done) | 0.01 | 0.01 | — |
| Task 1b builds + smokes + one-year pair (done) | — | 2.3 (builds 0.06, smokes 0.1, pair 2.15) | ~40 min |
| Tasks 2-6 tests | — | ~4 | — |
| Task 7 GPU tests | 0.3 | 0.5 | <1 h |
| Task 8 speed + profile (A100) | 2.0 | — | ~2 h |
| Task 9 30-day + one-year triplets | 5.0 | — | ~2.5 h |
| **before STOP 4** | **~7.5** | **~7.5** | |
| Task 10 ten years SP + shared DP reference | ~26 | — | ~4 h each |
| Task 10 adjoint gate + full suite | 1 | ~3 | 6 h |
| **total** | **~35** | **~10** | ~3-4 working days plus queue waits |

## Risks and open questions

- **XLA folds something else.** The per-jit option covers convert pairs; the HLO allow-list and the GPU carry test
  catch other surprises.
- **Silent f64 promotion** costs speed, not correctness: the scan carry check and the allow-list catch it.
- **N² in f32** (upstream policy): an f32 in-situ density near 1030 resolves 1.2e-4 kg m⁻³, so N² at the mixed-layer
  base carries large relative rounding; upstream accepts it, and so do we. The Fortran zstar pair shows its effect on
  the same physics.
- **Integer width** under x64-off (device-side int64 becomes int32): audited in Task 3.
- **GPU run-to-run noise** in SP is larger (f32 atomics); envelopes are measured, not assumed.
- **Tier A interaction**: Pallas tridiagonal kernels (parent Task 4) need an f32 variant (momentum) and an f64 one (the
  tracer column); the DP-hash JSON is regenerated on every DP-changing merge from `perf/core2-gpu`.

**Open questions for the user:** D1 (stiffness: rebuild from f32 hbar, or an f64 hbar copy for the increment), D3/D4
(SP-only branches where upstream DP also changed — accept that JAX DP and SP differ there), D5 (f64-then-round host
geometry vs a real32 host path), D6 (keep the carry in the restart although upstream drops it), D7 (CG rule for the
Chebyshev preconditioner); and the proposed thresholds T1-T3.

## Post-Completion

*Outside this repo or for later plans — informational only*

- SP in the adjoint's forward recompute and SP gradients: a Tier C plan.
- Results worth telling the fesom3-jax session go through the user.
- `~/fesom2_sp` is someone else's working tree with uncommitted instruments: never built or edited from here.
