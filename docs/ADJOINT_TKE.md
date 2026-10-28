# The classical-TKE adjoint on CORE2: where it grows and what carries it (stage 1)

Status: stage 1 (diagnosis) of the TKE-adjoint work, 2026-09-27. Stage 2 (derivative-only candidate settings,
calibration against finite differences, evaluation) builds on §10. Driver `scripts/adjoint/tke_diag.py`, tables
`scripts/adjoint/tke_diag_summary.py`, figures `scripts/adjoint/plot_tke_diag.py` (in `docs/figures/adjoint_tke/`).
Every number is a measurement; each row, with its provenance (git state, Slurm job, device, window, chunk, setting),
is in `/work/ab0995/a270088/fesom_jax_dev/runs/tke/a5/<run>/results.jsonl`.

## 1. Question

The exact reverse pass of the CORE2 all-on model (zstar, classical TKE, GM/Redi, mEVP with prescribed sea ice)
grows by orders of magnitude within days, and `AdjointConfig.long_window()` (`freeze_mixing` + `freeze_gm_slope`)
makes it usable by cutting every derivative of the vertical mixing closure — which also sets every TKE-constant
gradient to exactly zero. Before building narrower derivative-only settings: where and when does the cotangent
grow, which terms of the closure carry the growth, is the derivative correct (with a small linear range) or wrong,
and which fix does the evidence support?

## 2. Set-up and method

* **Model.** `configs/core2_full.yaml`, `IceConfig(whichEVP=1, adjoint_mode="prescribed")`, float64, one A100-80
  (GPU) or one CPU node, 30-min steps. Windows start from the shared 2013 States (one forward of 2013; monthly
  States), at 2013-01-01 and 2013-07-01.
* **Screens** (`tke_diag.py screen`): the reverse sweep of a terminal cost (volume-weighted box-mean T, 50-60N,
  40-20W, 0-500 m), the norm of the whole State cotangent at every chunk boundary (3-h chunks) and daily
  column-norm maps of the cotangent of tke, T, S, `uv`, `eta_n`. Growth is the OVERALL per-step rate
  `(end/seed)^(1/n_steps)` with the worst 4-chunk stretch, the worst lag-day and the largest chunk rate — the chunk
  median alone (≈ 1.02 here while the norm grows by 1e3-1e6 in bursts) hides the growth. Settings are compared on
  the same window with repeats; numbers are "median [min, max]" over the repeats. On a GPU two runs of the same
  compiled program give different forwards (scatter-add order; the convective switch differs at 600-2600 points
  after 6 steps), so exact-mode repeats diverge once the loop amplifies that difference (§7).
* **Power iteration** (`locate`, `sites`, `svd`): σ₁ of the linearised window map, tangent and adjoint on the SAME
  pinned per-step States (1-step chunks; with longer chunks the two maps linearise about different recomputed
  forwards), seeded in a box, in the `tke` norm (wet interfaces), the `ts` norm (volume-weighted T and S, 1 K and
  0.1 psu) or the `prog` norm (`ts` plus `uv` and `eta_n`, 0.1 m/s and 0.1 m; never `uvnode`). The leading vectors
  are logged with the closure regime of their cells (`cvmix_tke.tke_regimes`: mixing-length regime, tke − tke_min,
  floor contact, N², Ri, Pr, stiffness F = dt·dKm/dtke·|S² − N²/Pr|).
* **Cuts**: the diagnostic derivative cuts `tke.*` inside the column core, `mix.kv_out` / `mix.av_out` after the
  closure, and the freeze flags. Every cut leaves the forward bitwise unchanged. A term is in the loop if its cut
  removes the growth.
* **Column zoom and census** (CPU): the one-step tke → tke block of the full model at chosen columns against
  `jax.jacfwd` of the column core on the step's own inputs (`step_tke_inputs`), and the same Jacobian
  `∂tke_new/∂tke_old` for EVERY column (`colcensus`): the TKE-only recursion (S², N², the geometry and the surface
  flux held fixed), its spectral radius ρ per column, with and without each cut.
* **fdlin** (CPU): central differences of the nonlinear window map along the leading vector against the
  tangent-linear response, h = 1e-1 … 1e-12, with the linear range predicted from the forward State (distance to
  the tke floor, the KappaM cap, the mixing-length switch and the convective switch, per unit tangent). The CPU
  forward is deterministic; `--ssh-tol 1e-12` tightens the SSH solve of every forward (FD forwards and the tangent's
  primal) from the production relative residual 1e-5, whose early stop otherwise makes the forward non-smooth at
  small h. The noise floor is the part of a central difference at h/100 that the tangent does not explain.

## 3. Where and when: screens

**Two-day windows** (3-h chunks; exact: 7 repeats in three jobs from 2013-01-01, 3 from 2013-07-01; the flags 3
repeats each):

| setting | 2013-01-01 overall /step | total growth | worst 4-chunk | 2013-07-01 overall /step | total growth |
|---|---|---|---|---|---|
| exact | 1.112 [1.083, 1.165] | 2.8e4 [2.2e3, 2.4e6] | 1.37 [1.32, 1.70] | 1.110 [1.108, 1.137] | 2.2e4 [1.9e4, 2.2e5] |
| `freeze_gm_slope` | 1.121 [1.103, 1.127] | 5.8e4 | 1.31 | 1.125 [1.086, 1.131] | 8.2e4 |
| `freeze_tke_state` | 1.100 [1.084, 1.155] | 9.4e3 | 1.27 | 1.132 [1.111, 1.140] | 1.5e5 |
| `freeze_stratification` | 1.005 [1.005, 1.005] | 1.67 | 1.012 | 1.006 [1.006, 1.006] | 1.73 |
| `freeze_mixing` | 1.005 [1.005, 1.005] | 1.66 | 1.012 | 1.006 [1.006, 1.006] | 1.72 |
| `long_window()` | 1.005 (1 run) | 1.66 | 1.012 | – | – |

Jobs 27738864 (3 exact + `long_window()`), 27740109 (2013-01-01), 27740110 (2013-07-01), 27743266 (1 exact).
The first day of lag grows at 1.01-1.08 per step; nearly all the growth sits in the second day, in bursts of
1.2-2.3 per step (largest chunk rate), and the repeats separate there (by more than 1 % after 0.4-1.0 d of lag).

**Ten-day windows** (2013-01-01, job 27741381; 1 day of stride): exact 1.282 [1.276, 1.287] per step, ×2e52 [6e50,
5e52] over 10 days, worst day 1.59, every lag-day between 1.07 and 1.44 per step; `freeze_convection` 1.274 [1.273,
1.275]; `long_window()` 1.004 (×5.5). No overflow (the trace stays below 1e250). The July 10-day window was not run
(GPU queue; see §13).

**Where.** The tke-cotangent maps concentrate in a handful of nodes: the top 0.1 % of the nodes hold ≥ 99 % of the
squared norm at every lag beyond one day. Over two days the burst sites differ between repeats — (10.5W, 62.3N)
Faroe-Shetland, (80.7W, 73.7N) Barrow Strait, (28.7W, 65.6N) north of Iceland, (8.4W, 58.5N) Hebrides shelf,
(14.6E, 39.8N) Tyrrhenian Sea in January; (26.7E, 70.9N) North Cape, (8.9W, 46.6N) Bay of Biscay, (14.8E, 38.4N)
Tyrrhenian, (50.1E, 41.7S), (26.5W, 75.4S) Weddell shelf in July. Over ten days one site dominates every lag from
day 3 on in all four January runs: **the Tyrrhenian Sea at (15.1E, 39.6N)**, growing at 1.2-1.44 per step, day
after day. Growth is not pervasive: it lives at a few weakly stratified, sheared sites, and which of them bursts
first in a given repeat depends on the forward's round-off.

**Which fields.** All prognostic fields carry the same overall rate (2 days, January: tke 1.13, T 1.10, `uv` 1.09
per step; 10 days: tke 1.28, T 1.28, `uv` 1.27): the loop couples the closure to the resolved state.

## 4. The loop terms

### 4.1 Attribution on the screens

Each single cut on the 2-day window from 2013-01-01, 2 repeats each (job 27743266), against the exact spread
1.112 [1.083, 1.165] per step (7 repeats):

| cut | overall /step | total growth | resolved? |
|---|---|---|---|
| `tke.pr` (the Prandtl number's state dependence) | 1.005 [1.005, 1.005] | 1.67 | **removes all growth** |
| `tke.n2` (N² in the buoyancy term and in Ri) | 1.005 [1.005, 1.005] | 1.67 | **removes all growth** |
| `mix.kv_out` (the tracer diffusivity Kv) | 1.005 [1.005, 1.005] | 1.66 | **removes all growth** |
| `mix.av_out` (the viscosity Av) | 1.101 [1.080, 1.121] | 3.1e4 | no effect |
| `tke.shear` (S² in production and in Ri) | 1.135 [1.071, 1.199] | 1.9e7 | no effect |
| `tke.ml_n2` (N² in the free mixing length) | 1.126 [1.101, 1.151] | 3.8e5 | no effect |
| `tke.km_e` (√tke in KappaM) | 1.119 [1.113, 1.124] | 5.3e4 | no effect |
| `tke.ml_e` (√tke in the free mixing length) | 1.092 [1.091, 1.093] | 4.9e3 | inside the spread |
| `tke.prod`, `tke.buoy`, `tke.diss`, `tke.ke_diff`, `tke.surf` | 1.145, 1.094, 1.118, 1.128, 1.123 | – | no effect |
| `tke.deep` (closure outputs below 500 m: negative control) | 1.166 [1.117, 1.215] | 6.5e7 | no effect |
| `freeze_convection` (the `mo_convect` increment) | 1.085 [1.065, 1.104] | 7.1e3 | at the lower edge: not resolved |

The loop that destroys the gradient is therefore **N² → Ri → Pr → KappaH (= Kv) → T, S → N²**: the tracer
diffusivity's dependence on the stratification through the Prandtl number. It needs neither the TKE memory
(`freeze_tke_state` has no effect) nor the shear (`tke.shear`), and the viscosity plays no part (`mix.av_out`).
An interpretation, not yet a measurement: in the active branch `Pr = 6.6·Ri` (0.15 < Ri < 1.5; clipped to 1 and
10 outside) `KappaH = KappaM·S²/(6.6·N²)`, so `∂KappaH/∂N² = −KappaH/N²`, and the linearised tracer flux
`δ(Kv ∂zT) = Kv ∂z δT + δKv ∂zT` loses its diffusive part — for a temperature-stratified column the two terms
cancel exactly, and with salinity (or the free mixing length ∝ 1/N) the T-S cross terms can make it
anti-diffusive: the linearised form of the layering instability of a closure whose buoyancy flux does not grow
with the stratification. The forward stays bounded (implicit diffusion, clips); the tangent and the adjoint do
not. The fastest site is local and deep: the Tyrrhenian Sea (§4.4), near-neutral below 500 m.
At the Tyrrhenian site the same cuts act locally (§4.4). Ten-day and July confirmation of the cut: §13.

### 4.2 The TKE-only recursion (column census)

`∂tke_new/∂tke_old` of the column core in every column (`colcensus`, CPU, jobs 27744190, 27744191, 27744192):

| time | columns with ρ > 1 | ρ > 1.1 | ρ > 1.5 | ρ max | where (by latitude band) |
|---|---|---|---|---|---|
| 2013-01-01 00:00 | 3300 (2.6 %) | 0.9 % | 0.16 % | 373 | all bands; 1097 in 30S-0, 923 in 0-30N, 534 south of 60S |
| 2013-01-02 12:00 | 3491 (2.8 %) | 0.9 % | 0.06 % | 7.2 | 1382 in 30S-0, 1098 in 0-30N, 508 south of 60S |
| 2013-07-01 00:00 | 5427 (4.3 %) | 1.4 % | 0.13 % | 4.0 | 2532 north of 60N (Arctic summer) |

The unstable columns are unstable at their **first interior interface** (5 m): its tke is near the floor (51-69 % of
them with tke − tke_min < 1e-5), its mixing length free (78-88 %; the rest the surface wall bound), below a
surface interface with far more tke (median 3e-4 to 2e-3 m²/s², against ≈ 1e-6 at the first interface). The recursion's gain there is the TKE diffusivity
`ke = α·½(KappaM[k]+KappaM[k+1])`, which grows with tke at the first interface (free length: `KappaM ∝ tke/N`,
`dKm/dtke = √2 c_k/N`): more tke there lets more of the surface tke diffuse down, which raises the tke there
(self-gain `J11` 1.5-2.0 at typical cells, 374 at the extreme). The single cuts inside the core, share of the
no-cut unstable columns they stabilise (Jan 1 / Jan 2 12:00 / Jul 1): `tke.km_e` 93 / 99 / 99 %, `tke.ml_e`
94 / 99 / 99 %, `tke.km_e+tke.ml_e` 91 / 98 / 99 %, `tke.ke_diff` 53 / 52 / 72 %, `tke.prod` 61 / 65 / 40 %,
`tke.diss` 2 / 0 / 0 %, `tke.buoy` 0 % (it raises the unstable count 1.6-2.1-fold). The S²/N² cuts do not act on this
Jacobian (it holds S² and N² fixed).

### 4.3 The tke-norm leading mode is this recursion, not the screens' loop

The scan (power iteration of the 6-h map in the tke norm, seeded everywhere, every 6 h over the first two days;
jobs 27741379, 27741380) finds the recursion at every time: σ₁(6 h) = 1565-2317 (January; ≈ 1.85-1.91 per step)
and 4130-9136 (July), the input vector at the first interior interface (free length, tke at the floor, N² 2.7e-3 to
1.3e-2, Ri 0.5-560), the output at the surface interface, at the Antarctic coast (93-99E, 64-66S; 94-97W, 71.5S),
the Gulf of Guinea (10E, 3S), the southern Brazil shelf (46W, 25S) in January, and Hudson and James Bay, the
Mackenzie shelf and Amundsen Gulf in July: fresh, strongly stratified surface layers under a wind- or ice-forced surface. At the
Antarctic site the one-step map (CPU, job 27743077) has σ₁ = 153.5; the full model's one-step tke block equals
`jax.jacfwd` of the column core on the step's own inputs to 2.2e-16 (six columns, job 27743080); its columns have
`∂tke0/∂tke1` = −129 to −154 (the non-normal transient behind σ₁) and spectral radius 1.23-1.91.
It is transient: over one day the same windows give σ₁ = 92 (Brazil shelf, 2013-01-02 00:00) and 580 (Amundsen
coast, 2013-01-02 12:00), below their six-hour 2124 and 2316 (job 27741379; the tke-floor clip and the changing
regime end each episode), and 7.5-7.7 in the prog norm (quiet control: 15.6). At the Amundsen site the six-hour
cuts repeat the Hudson Bay picture (`tke.ml_e` ×0.015, `tke.km_e` ×0.030, all others within 5 %).
This mode is real linearised growth, but the screens do not excite it within ten days: `freeze_tke_state` (which
removes the recursion) leaves the screens' growth unchanged, and `tke.pr` (which keeps it) removes it. Its tke
cotangent enters the resolved state only through KappaM at the first interface.

### 4.4 The Tyrrhenian site: the same loop, locally

Power iteration seeded in a box around (15.1E, 39.6N), 0-4000 m, from the State of 2013-01-03 00:00 (CPU; the
two-day spin-up from the monthly State in the same job; jobs 27745245-47, 27745547):

* one step: σ₁ = 24.2 in the `ts` norm (production solve; 49.2 with the 1e-12 solve, a slightly different
  forward), and 1.01 in the `tke` norm — the tke → tke map does not grow here; the leading T/S vector lives at
  interfaces 21-26 (580-1180 m), 83 % of it where 0 < N² < 1e-6, 17 % where N² ≤ 0; 25 % in the active Prandtl
  branch, 58 % at Pr = 10, 17 % at Pr = 1;
* six hours: σ₁ = 5390 in `ts` (tight solve; ≈ 2.0 per step), the leading vector at 1800-2000 m, 73 % of it where
  N² ≤ 0;
* single cuts, one step, `ts` norm (CPU, deterministic): `tke.pr` and `tke.n2` 24.2 → 3.85 (×0.16),
  `mix.kv_out` and `freeze_stratification` → 1.59 (×0.066), `tke.deep` (closure outputs below 500 m) → 16.0
  (×0.66: the loop here is partly below 500 m, although the box's median mixed layer is 39 m); `tke.ml_n2`,
  `tke.shear`, `tke.km_e`, `tke.ml_e`, `tke.ke_diff`, `mix.av_out`, `freeze_convection` ×1.000. The same terms as
  on the screens. Pairs: every pair with `mix.kv_out` or `freeze_stratification` → 1.59, `tke.pr+tke.n2` → 3.85
  (the two cut the same path);
* single cuts, six hours, `ts` norm (CPU, production solve; jobs 27746523, 27747619, 27747620): σ₁ = 1.20e5
  (≈ 2.65 per step); `tke.pr` → 12.3 (×1.0e-4), `tke.n2` → 10.9 (×9e-5), `mix.kv_out` → 1.95 and
  `freeze_stratification` → 1.95 (×1.6e-5), `tke.deep` → 81 (×7e-4: the loop lives below 500 m); `tke.ml_n2`,
  `tke.shear`, `tke.km_e`, `tke.ml_e`, `tke.ke_diff`, `mix.av_out`, `freeze_convection` within 0.4 %. Over six
  hours the Prandtl cut removes 99.99 % of the amplification.

### 4.5 The lens recursion at Hudson Bay (July, tke norm)

Scan top site, 2013-07-01 00:00 (GPU job 27741380): σ₁ = 263 (one step), 9137 (6 h) in the tke norm, 10.1 and
10.5 in the prog norm (as at the quiet control: the mode stays in the TKE). Single cuts (one step / six hours):
`tke.ke_diff` → 0.98 / 1.16 (×0.0037 / ×1.3e-4), `tke.ml_e` ×0.51 / ×0.0057, `tke.km_e` ×0.51 / ×0.012,
`tke.prod` ×1.0 / ×0.78, `tke.buoy` ×1.0 / ×1.36 (worse, as in the census), `freeze_tke_state` → 0 (the map is
cut); `tke.ml_n2`, `tke.shear`, `tke.n2`, `tke.pr`, `tke.diss`, `tke.surf`, `mix.kv_out`, `mix.av_out`,
`freeze_convection`, `freeze_stratification` ×1.01 / ×1.08 (the spread of the warm-started iteration: these cuts
cannot act on a tke → tke map), and the negative control `tke.deep` ×1.08 (box median mixed layer 10 m). The
recursion is tke at the first interface → KappaM (√tke in both KappaM and the free length) → the TKE diffusivity
→ the flux of surface tke into the first interface: cutting the diffusivity's tke dependence alone removes it.
Pairs at six hours: `tke.km_e+tke.ml_e` → 0.66 (×7e-5, both halves of dKm/dtke: the map now decays),
`tke.ke_diff+tke.ml_e` 0.88, `tke.ke_diff+tke.km_e` 0.89; `freeze_stratification`, `freeze_gm_slope` ×1.08.

## 5. Bug or linear range (fdlin)

CPU, one step, along the leading vector of the window (power iteration in the same run), central differences;
"noise" is the part of a difference at h/100 the tangent does not explain.

| site, window, norm | SSH tol | ‖FD − TL‖/‖TL‖ at h = 1e-5 / 1e-6 / 1e-7 / 1e-8 / 1e-9 / 1e-10 / 1e-11 / 1e-12 | predicted range |
|---|---|---|---|
| Antarctic lens (95.9E, 65.9S), 2013-01-01, 1 step, tke (σ₁ 153.5) | 1e-5 and 1e-12 (identical) | 0.32 / 7.1e-5 / 7.1e-7 / 8.9e-9 / 5.4e-9 / 5.4e-9 / 6.7e-9 / 4.4e-8 (noise) | 2.0e-6 (tke floor) |
| Tyrrhenian (15.1E, 39.6N), 2013-01-03, 1 step, ts (σ₁ 24.2) | 1e-5 | 1.0 / 1.0 / 0.99 / 0.88 / 0.17 / 1.4e-2 / 1.4e-4 / 8.6e-5 (noise) | – (see below) |
| same, tight solve (σ₁ 49.2) | 1e-12 | 1.0 / 1.0 / 0.99 / 0.93 / 0.47 / 0.12 / 1.0e-3 / 3.0e-5 (noise) | – |

Jobs 27743075, 27743077, 27745245, 27745246 (the two Tyrrhenian rows run from different CPU forwards: with the
tight solve the two-day spin-up from the monthly State gives a slightly different State, hence another σ₁). **The
derivative is correct**: at both sites FD converges to the tangent-linear response once h is small enough, to 5e-9
at the lens site. **The linear range is tiny.** At the lens site it is the distance to the tke floor, as
predicted (2.0e-6, transition between 1e-6 and 1e-5). At the Tyrrhenian site it ends near h = 1e-10: the
vector is normalised in the volume-mean norm, so h = 1e-10 is a temperature change of ≈ 4e-8 K in the few deep
cells where it lives — cells with N² between −1e-8 and 1e-8 s⁻² (83 % of the vector; interfaces 21-26, 580-1180 m),
where such a change already moves N² across zero or across the Prandtl branch limits. The predicted range of the
driver covers only the tke switches (floor, cap, mixing length) and the convective switch through the State's
own `bvfreq` tangent, which a T/S seed reaches only one step later; it therefore reports no bound for this
one-step T/S map. The production solve (1e-5) adds a roughness of 8e-5 relative at h = 1e-12 at the Tyrrhenian
site and none at the lens site over one step (the tke → tke map does not pass through the SSH solve within a step).

Over **six hours** at the Tyrrhenian site (CPU, tight solve, job 27745247) the tangent-linear map grows by
σ₁ = 5390 (≈ 2.0 per step; the leading vector sits at interfaces 30-31, ≈ 1800-2000 m, where N² ≤ 0 in 73 % and
0 < N² < 1e-8 in 27 % of it), and **no h verifies it**: from h = 1e-2 down to 1e-8 the finite difference stays
far below the tangent (the difference quotient is 0.28 at h = 1e-2, 175 at 1e-8 and 1960 at 1e-9 against the
tangent's 5390), and below 1e-9 it is dominated by the forward's own roughness (2e-6 in the ts norm at
h₀ = 1e-14, even with the tight SSH solve). The predicted range agrees: carried along the window, the tangent
reaches the convective switch (`bvfreq` = 0) within h = 7e-12 after one step and 5e-14 after nine (the tke
switches: 2.4e-9). The forward is not smooth at round-off there: in cells with N² within round-off of zero any
perturbation flips the convective switch. The exact adjoint's growth at this site is the derivative of a map that no resolvable perturbation
follows. One-day fdlin (CPU jobs 27743081, 27743083, 27745248) had not started at the report (§13).

## 6. Hypotheses

* **H1 — wall-bound first interface with small tke (dKm/dtke = c_k·mxl/(2√tke)).** Partly confirmed, as a
  different mode: the TKE-only recursion is unstable at the first interior interface with near-floor tke in
  2.6-4.3 % of the columns (§4.2), but 78-88 % of those have a FREE mixing length (dKm/dtke = √2 c_k/N), and the
  gain runs through the TKE diffusivity to the surface interface; `tke.km_e` stabilises 93-99 % of them. It is
  NOT the loop that destroys the gradient: `tke.km_e` and `freeze_tke_state` leave the screens unchanged.
* **H2 — wall-bound length in neutral/convecting columns (large mxl).** Rejected: no unstable recursion column has
  a `wall_other` first interface; `freeze_convection` does not remove the screen growth.
* **H3 — the free-length path ∂Km/∂N² ∝ N⁻³ in weakly stratified cells.** Rejected as the driver: `tke.ml_n2`
  leaves the 2-day growth unchanged (1.126 [1.101, 1.151]) and the one-step Tyrrhenian σ₁ unchanged (×1.000).
* **H4 — the Prandtl/Ri switch.** **Confirmed as the driver**: `tke.pr` alone removes all 2-day growth (1.005 per
  step, as `freeze_mixing`), and so do `tke.n2` (N² in Ri) and `mix.kv_out`; `tke.shear` (S² in Ri) does not. The
  loop is N² → Pr → KappaH → Kv → T, S → N² (§4.1); at the Tyrrhenian site the same cuts take the one-step
  T/S σ₁ from 24.2 to 3.85 (`tke.pr`, `tke.n2`) and 1.59 (`mix.kv_out`) (§4.4).
* **H5 — interplay with `mo_convect`.** Not the driver: `freeze_convection` 1.085 [1.065, 1.104] (2 days, at the
  lower edge of the exact spread) and 1.274 [1.273, 1.275] (10 days, exact 1.282 [1.276, 1.287]) — at most a small
  effect; at the Tyrrhenian site it changes σ₁ by < 0.03 % over one step and six hours. Its switch (`bvfreq < 0`) is, with the tke floor, where two runs of the same GPU program part (§7), and
  near-neutral cells (N² ≈ 0) bound the linear range at the Tyrrhenian site (§5).

## 7. The linearisation point is not reproducible on the GPU; the derivative is

Two GPU forwards of the same program from the same State (Hudson Bay, 2013-07-01, 12 steps; job 27741380)
differ by 3 convective switches (at step 5), 3 tke-floor contacts, 14 sea-ice skin temperatures by more than
0.1 K (up to 1.17 K) and 1.7e-4 K in T at most; the site's σ₁ is the same on both (9137, ratio 1.0000).

Repeats of the same power iteration on the same pinned States agree exactly on the A100 with 1-step chunks
(Hudson Bay: 1-step σ₁ 263 and 6-h σ₁ 9137 twice, ratio 1.0000); the screens' repeats differ because each
reverse sweep recomputes a slightly different forward. On the CPU every repeat is bitwise. So "the adjoint is
not reproducible" means "the linearisation point is not reproducible", and the loop of §4.1 amplifies that
difference; the tangent/adjoint pair itself is consistent (P0's dot test on 1-step chunks, ≤ 6e-9).

## 8. The old attribution table, re-measured

The 1960 two-day table (`AdjointConfig` docstring) had no resolving power (unfrozen scatter 1.21-1.55 with sign
flips). Re-measured on the 2013 windows with ≥ 3 repeats (2 for the 10-day and cut rows):

| flag | 1960 table | 2013-01-01, 2 days | 2013-07-01, 2 days | verdict |
|---|---|---|---|---|
| (none) | 1.21-1.55 | 1.112 [1.083, 1.165] | 1.110 [1.108, 1.137] | reference spread |
| `freeze_mixing` | 1.0070 | 1.005 (bitwise repeats) | 1.006 | **resolved**: removes the growth (cuts Kv, which carries the loop) |
| `freeze_stratification` | 1.0322 | 1.005 | 1.006 | **resolved**: removes the growth, as fully as `freeze_mixing` (it cuts the N² in Pr) |
| `freeze_gm_slope` | 1.4298 | 1.121 [1.103, 1.127] | 1.125 [1.086, 1.131] | not resolved (inside the spread): no effect at 2 days |
| `freeze_tke_state` | 1.2913 | 1.100 [1.084, 1.155] | 1.132 [1.111, 1.140] | not resolved: the loop needs no TKE memory |
| `freeze_convection` | 1.2550 | 1.085 [1.065, 1.104] (2 repeats) | – | not resolved (lower edge); 10 days: 1.274 [1.273, 1.275] vs 1.282 [1.276, 1.287] |
| `freeze_advection` | 1.3238 | not re-measured | – | – |

The difference between `freeze_stratification` (1.032 in 1960) and `freeze_mixing` is gone in 2013: both remove
the whole 2-day growth because both cut the Prandtl loop.

## 9. The TKE-constant gradients under partial cuts (what a parametric-only adjoint would lose)

The 2-day screens return the gradient of the box-T cost with respect to the log TKE constants in every run. Where
the reverse pass is clean the values repeat to 3-4 digits; the exact values scatter over three orders of
magnitude with sign changes (c_k: 0.0206, −0.0033, 0.052, 0.043, −0.0103, 0.0017, 1.72 in seven runs) and carry
no information.

| setting (2013-01-01, 2 days) | dJ/d ln c_k | dJ/d ln c_eps | dJ/d ln cd | dJ/d ln α |
|---|---|---|---|---|
| `tke.pr` (only the Prandtl number's state dependence cut) | −2.43e-5 | −1.09e-5 | −1.58e-6 | 9.04e-6 |
| `tke.n2` (N² in Ri and in the buoyancy term cut) | −3.63e-5 | −2.63e-6 | −3.78e-6 | 9.00e-6 |
| `freeze_stratification` (every N² derivative cut) | −3.64e-5 | −1.80e-6 | −3.86e-6 | 8.31e-6 |
| `mix.kv_out` (Kv cut; Av, hence the constants, still differentiated) | −3.62e-5 | +2.18e-6 | −4.33e-6 | 7.65e-6 |
| `freeze_mixing`, `long_window()` | 0 | 0 | 0 | 0 |

The three settings that cut more of the state feedback than `tke.pr` agree with each other on c_k but differ from
`tke.pr` by 50 % in c_k, by a factor 2.4-2.7 in cd and by a factor 4-6 (and a sign) in c_eps; only α agrees
within 15 %. None of these is the truth (stage 2 computes finite differences); the spread says that the
ocean-state feedback on mixing carries an O(1) share of the c_k, c_eps and cd sensitivities over two days.

Ten days, `tke.pr` against the parametric-only proxy (`tke.shear+tke.n2+tke.ml_n2+freeze_convection`): CPU jobs
27745920, 27745921, running at the report (§13).

## 10. Fix options for stage 2 and the recommendation

Stage 2 builds derivative-only settings (forward bitwise, off ⇒ identical program; options, never defaults):

1. **km_e cut** — no dKm/dtke in near-floor cells (tke < tke*). As planned it applies to wall-bound cells; the
   evidence says it must cover the first interior interface in ANY mixing-length regime (78-88 % of the unstable
   recursion columns have a free length there). It stabilises 93-99 % of the unstable columns of the TKE-only
   recursion (§4.2) and cuts the surface-lens mode's 6-h σ₁ by 85-175× (Hudson Bay: `tke.km_e` 9137 → 108,
   `tke.ml_e` → 52). It does nothing for the loop that destroys the gradient (the 2-day screen is unchanged).
2. **calibrated stiffness** (freeze where F > F*) — not supported: F does not identify the unstable recursion
   (F = 0.08 at the most unstable column, ρ = 373; F 0.2-25 at the lens sites) and it does not describe the
   Prandtl loop.
3. **regularised N² derivative** — the family that targets the loop, in its NARROWEST form: the Prandtl number's
   state dependence cut in the backward (`tke.pr` as a derivative-only setting), optionally only where
   N² < N²* or with Pr differentiated at max(N², N²*) (`differentiate_at`). `tke.pr` alone removes the whole 2-day
   growth in January (1.005/step, as `freeze_mixing`) while keeping every other derivative of the closure,
   including every path of the four TKE constants (Pr does not depend on them); at the Tyrrhenian site it takes
   the one-step T/S σ₁ from 24.2 to 3.85.
4. **whole-column variants** — needed for family 1 at the first interface (the recursion couples the surface and
   first interfaces through the implicit TKE diffusion; a cell mask would cut that coupling on one side only);
   not needed for the Prandtl cut, which acts point-wise on Pr.
5. **targeted GM-slope cut** — no evidence for it: `freeze_gm_slope` has no effect on the 2-day screens in either
   season (the 10-day check with the Prandtl cut, §13, would show any residual growth).
6. **parametric-only adjoint** (always built) — cut the closure's ocean-state derivatives, keep the constants and
   the TKE recursion. Stage-1 prediction: (a) it removes the Prandtl loop (it cuts N² in Pr; `tke.n2` and
   `freeze_stratification` alone already do); (b) where that loop grows, the recursion it keeps is stable (the
   Tyrrhenian box: ρ ≤ 1.00 in January, one column at 1.03 in July), but elsewhere it is unstable in 2.6-4.3 % of
   the columns at any time (ρ up to 4-373) — the Antarctic lens (63 columns, ρ ≤ 1.91 on 1 January), Hudson Bay
   (267, ρ ≤ 1.95 on 1 July) and the cost box itself (54 columns, ρ ≤ 1.55 on 1 January). The episodes are
   short (a lens site's σ₁ over one day, 92-580, is below its six-hour value), so the kept recursion is expected
   to amplify by up to ~1e3 at such sites rather than without bound; it still needs family 1 (extended as above)
   wherever tke is near the floor, the combination the plan anticipates; (c) its TKE-constant gradients will be biased at O(1) for c_k, c_eps and cd (the state
   feedback carries 50-500 % of them over two days, §9) and to ≈ 15 % for α.

**Recommendation.** Build **family 3 in its narrowest form, the Prandtl-number cut**, as the primary fix: the
Prandtl number's state dependence omitted in the backward (a derivative-only `tke.pr`), with the variants "only
where N² < N²*" and "Pr differentiated at max(N², N²*)" for the threshold calibration. It is the one term whose cut
removes the gradient-destroying growth on its own (2 days: 1.112 [1.083, 1.165] → 1.005 per step, as
`freeze_mixing`; Tyrrhenian σ₁ 24.2 → 3.85 over one step and 1.2e5 → 12 over six hours), and it keeps every other
closure derivative, including all paths of the TKE constants. Build **family 6** (mandatory) **combined with family 1 extended to near-floor first
interfaces of any mixing-length regime** (tke < tke*, column-wise as family 4): family 6 already cuts the Prandtl
loop, and the extended km_e cut stabilises 93-99 % of the columns in which its kept TKE recursion is unstable. A
narrower variant worth calibrating beside it cuts only the TKE diffusivity's tke dependence (`tke.ke_diff`): it
removes the lens mode at Hudson Bay (×1.3e-4 in six hours) but stabilises only 52-72 % of the unstable columns.
Do not build families 2 and 5 unless the 10-day checks in §13 show residual growth that points to them.

## 11. Figures

* `figures/adjoint_tke/setting_rates_2013-01-01_2d.png` — overall growth per step of every setting on the 2-day
  January window (each dot one run; grey band: the seven exact runs). Only the settings that cut the Prandtl
  number's N² dependence (`tke.pr`, `tke.n2`, `freeze_stratification`) or Kv (`mix.kv_out`, `freeze_mixing`,
  `long_window()`) leave the band.
* `figures/adjoint_tke/screen_traces_2013-01-01_2d.png` — the reverse-pass norm against the lag for the exact
  repeats and the old-table flags.
* `figures/adjoint_tke/screen_maps_2013-01-01_10d_exact.png` — tke-cotangent column norm per model day of a 10-day
  exact screen: from day 7 back the Tyrrhenian Sea holds the whole norm (panels normalised by their maximum;
  days 0-1 exceed the float32 range of that run's maps).
* `figures/adjoint_tke/rho_map_2013-01-01.png` — spectral radius of every column's TKE-only recursion (columns
  with ρ > 1 in colour).
* `figures/adjoint_tke/fdlin_lens_1step.png`, `fdlin_tyrrhenian_1step.png`, `fdlin_tyrrhenian_6h.png` — FD against
  the tangent-linear response, with the noise floor and the predicted range.
* `figures/adjoint_tke/cuts_tyrrhenian_1step_ts.png`, `cuts_hudson_6h_tke.png` — σ₁ under each single cut
  relative to no cut: the Prandtl loop at the Tyrrhenian (T/S norm, one step) and the lens recursion at Hudson
  Bay (tke norm, six hours).

## 12. Reproduction

All from a checkout with `PYTHONPATH=<checkout>` through `scripts/adjoint/job_gpu.sbatch` (A100, `--mem` ≥ 120G)
or `job_cpu.sbatch` (`--mem=200G`); rows land in `--out`/`results.jsonl`.

```bash
D=scripts/adjoint/tke_diag.py
# screens: 2 days, 3 repeats, the old flags 3 repeats each; single cuts 2 repeats each ('=' = on top of exact)
$D screen --start 2013-01-01 --days 2 --chunk-steps 6 --repeats 3 \
   --also freeze_stratification,freeze_gm_slope,freeze_tke_state,freeze_mixing --also-repeats 3
$D screen --start 2013-01-01 --days 2 --chunk-steps 6 --repeats 1 --also "tke.pr*2,tke.n2*2,mix.kv_out*2,..."
$D screen --start 2013-01-01 --days 10 --chunk-steps 6 --stride 8 --repeats 2 --also "freeze_convection*2,long_window*1"
# scan + sites + cuts (1-step chunks: tangent and adjoint on the same pinned States)
$D locate --start 2013-07-01 --steps 1 --chunk-steps 1 --spec locate.json
# CPU: column census, column zoom, fdlin (tight SSH solve)
$D colcensus --start 2013-01-01 --steps 1 --chunk-steps 1 --singles tke.km_e,tke.ml_e,tke.ke_diff,...
$D column --start 2013-01-01 --steps 1 --chunk-steps 1 --nodes svd --box ant --box-bounds=-67.9,-63.9,91,100.8,0,1000 --power-iters 20
$D fdlin --spinup-from 2013-01-01 --start 2013-01-03T00:00:00 --steps 12 --chunk-steps 1 --ssh-tol 1e-12 \
   --vector svd --weights ts --box tyr --box-bounds=37.6,41.6,12.5,17.7,0,4000
python scripts/adjoint/tke_diag_summary.py <run dirs> --maps --md summary.md
python scripts/adjoint/plot_tke_diag.py <run dirs> --out figures/
```

The `locate` specs used are in `/work/ab0995/a270088/fesom_jax_dev/runs/tke/a5/specs/`.

## 13. Open at STOP 1

The GPU account's fairshare ran out in the afternoon (priority 1 against ≈ 500), and the CORE2 step's reverse
program does not fit the 40-GB A100s of the development partition (a single-step chunk asks for 25.6 GB beyond
the rest), so the last experiments moved to CPU nodes, where a run is deterministic and one run replaces GPU
repeats (CPU and GPU numbers are never compared as repeats of each other). At the time of this report:

* the 10-day screens from 2013-01-01 with `tke.pr` alone and with the parametric-only proxy
  (`tke.shear+tke.n2+tke.ml_n2+freeze_convection`) — which decide whether the Prandtl cut suffices beyond two
  days and what family 6's kept recursion does over ten — ran into their 5.5-h CPU limit (jobs 27745920, 27745921;
  a screen writes its rows when a setting finishes, so nothing was kept) and were resubmitted with 8 h (27751746,
  27751747);
* on CPU at the report: a 10-day exact reference (27747614, running), one-day fdlin at the lens site and the
  Tyrrhenian (27743081, 27743083, 27745248, queued);
* not run: the July 10-day screens, the July 2-day `tke.pr`/`tke.n2` confirmation, the one-day power iteration at
  the Tyrrhenian in both norms.


## 14. Stage 2: candidate settings, their thresholds and the preset

**Settings** (`AdjointConfig`, derivative-only: forward bitwise, off traces the identical program; tests in
`test_tke_candidates{,_pi,_core2}.py`):

| field | what the backward omits | variants |
|---|---|---|
| `tke_pr_cut` = N²* | dPr/dRi (hence dPr/dN², dPr/dS²) where the column-input N² < N²* | `inf` = everywhere (= `tke.pr`); `tke_pr_mode="at"`: Pr differentiated at max(N², N²*) |
| `tke_kme_tke` = tke* | d KappaM / d √tke in columns whose first interior interface has tke < tke* (any mixing-length regime) | `tke_kme_scope` `column` / `iface1`; `tke_kme_term` `km_e` (every use) / `ke_diff` (TKE diffusivity only) |
| `freeze_mixing_state` | every derivative of the closure w.r.t. S², N², the layer and interface thicknesses and the surface flux (the ocean state); the TKE recursion and the four constants stay | combine with `tke_kme_tke` |
| `tke_pr_ratio` = R* | dPr/dRi where the smoothed N² the closure reads is a cancellation residual of the raw one: \|raw N²\| > R*·\|smoothed N²\| | joined with `tke_pr_cut` (a cut where either holds); `"cut"` mode only |
| `tke_pr_raw_cut` = N²_raw* | dPr/dRi where the unsmoothed N² < N²_raw* (inverted and neutral cells, whatever the smoothed N²) | joined with the other two Prandtl masks |
| `tke_ml_cut` = N²* | dmxl0/dN² of the free mixing length mxl0 = √2·√tke/√max(N², 1e-12) where the smoothed N² < N²*; its √tke dependence, the walls and every constant path stay | `inf` = everywhere (= `tke.ml_n2`) |
| `tke_ml_ratio` = R* | dmxl0/dN² where \|raw N²\| > R*·\|smoothed N²\| (the Prandtl ratio mask on the mixing length) | joined with `tke_ml_cut` |
| `tke_ml_raw_cut` = N²_raw* | dmxl0/dN² where the unsmoothed N² < N²_raw* | joined with the other two mixing-length masks |

**N²\*** (CPU, 2013 States). The Prandtl derivative is live on the interior interfaces with 1 < 6.6·Ri < 10 (178-202
thousand of 3.58 million); its KappaH sensitivity G = |∂KappaH/∂N²| = KappaM·6.6/(Pr²·S²) is concentrated where N² and
S² are both tiny: half of ΣG lies below N² ≈ 1e-12 s⁻², 98.4-99.6 % below 1e-8 (0.5-0.7 % of those cells). At the
Tyrrhenian growth site (2013-01-03) the T/S σ₁ of the one-step map is 24.2 exact and, with the cut below N²* = 1e-11,
1e-10, 1e-9, 1e-8, 1e-7, 1e-6: 24.2, 16.0, 16.0, 3.85, 3.85, 3.85 (3.85 = the cut everywhere); over six hours
1.2e5 exact, 81 at 1e-9, 14.0 at 1e-8 (12.3 everywhere). **N²\* = 1e-8**, the smallest decade value that reproduces the
everywhere cut. The `"at"` variant is worse than the exact adjoint at N²* = 1e-10 and 1e-9 (one-step σ₁ 839 and 118):
raising N² moves cells whose Ri lies below the clip band (dPr/dRi = 0) into it, where the derivative is 6.6/S² with a
tiny S². A regularised linearisation point can create derivatives the model does not have.

**tke\*** (column census of the TKE-only recursion, one and two steps, 1 January, 2 January 12:00, 1 July). The first
interior interface of the unstable columns is not near the floor (median tke 4e-6 … 1e-5, 90 % quantile 1.3-1.8e-4
m²/s²). The column km_e cut stabilises 29-43 % of them at tke* = 3e-6, 75-87 % at 1e-4 and 92-99 % at 1e-3, which is
within one point of the km_e cut everywhere (and flags 93-97 % of the columns). **tke\* = 1e-3**, whole column, all uses
of KappaM. The first-interface-only scope stops at 50-73 % and the ke_diff term at 49-71 % while making up to 333
stable columns unstable.

**R\*** (steps C1-C3, docs/PORTING_LESSONS.md). The 5-day box-T burst left under `tke_pr_cut = 1e-8` (×2.7e4 in one chunk) was one
ice-covered East Siberian shelf column whose smoothed N² (+3.3e-7) is the residual of a raw −8.0e-5 (ratio 240). R* = 3, 10 and 100
all remove it (5-day reverse traces agree to 3 digits); **R\* = 100**, the narrowest, keeps J_MLD at 5 days within
3.1 %.

**N²_raw\*** (steps C3 at 10 days and C4). At 10 days a second class bursts: Norwegian-shelf inversions with
\|raw\| / \|smoothed\| of only 1-30, which R* = 100 misses. A closure that reads the local N² (MITgcm's GGL90) has Pr = 1
and no Prandtl derivative there. With the other two cuts on (January): raw* = 0, −1e-7, −1e-6, −3e-6 improve monotonically,
−1e-5 bursts (×500 in one chunk). **N²_raw\* = −1e-6**, a factor 10 from the cliff.

**Mixing-length N²\*** (step C5). Beyond 30 days the July sweeps burst (×1e3) in deep Tyrrhenian columns below ~400 m
(smoothed N² ~ ±1e-10, raw −1e-8 … −4e-7) through dmxl0/dN² = −mxl0/(2N²), which the Prandtl cuts do not touch. Growth
of the reverse trace over chunks 72-89 of the July sweep (2013-08-06 to 08-14), with the three Prandtl cuts on: ×2690
without the cut; ×8.9 at N²* = 1e-9 and 1e-8, ×7.2 at 1e-7, ×1.22 at 1e-6. Over 30, 60 and 90 days, 1e-8 passes the
July bar for both costs and adds no January bias; 1e-6 has cleaner traces but raises the January box-T error (30 days,
c_k 62 → 84 %). **N²\* = 1e-8**, the value of `tke_pr_cut`.

**The preset** `AdjointConfig.long_window_tke()` = `tke_pr_cut=1e-8, tke_pr_ratio=100, tke_pr_raw_cut=-1e-6,
tke_ml_cut=1e-8`, measured with `IceConfig.adjoint_mode="prescribed"`, GM on and not frozen. Its docstring lists the
30/60/90-day results from 2013-01-01 and 2013-07-01; docs/PORTING_LESSONS.md, steps C3-C5, has the detail.

**Mixing-length masks** (combined adjoint, 2026-10-11). With the sea-ice adjoint live and GM/Redi differentiated, two of
four January 30-day ice runs burst in one inverted Northeast Greenland shelf cell (raw N² −3.7e-5 … −6.9e-5, smoothed
1.5e-7 … 4.7e-6, above the mixing-length N²* = 1e-8): dmxl0/dN² is live there, the Prandtl derivative already cut. The
Prandtl masks applied to dmxl0/dN² remove it: R* = 100 with N²_raw* = −1e-6 (the preset
`AdjointConfig.long_window_combined()`) is clean in three runs, as are R* = 100 with −1e-5 (three runs) and R* = 100
alone (two runs). The raw mask costs January box-T accuracy at 30 and 60 days (vector error 26 / 28 % against 15 / 18 %
unmasked; R* alone 14.5 / 18.2 %) and gains it at 90 days (4.6 % against 8.8 %).
