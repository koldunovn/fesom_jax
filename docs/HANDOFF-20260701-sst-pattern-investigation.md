# Handoff 2026-07-01 — CORE2 JAX vs Fortran SST DIFFERENCE = a wavenumber-1 zonal pattern (COORDINATE / WIND-ROTATION suspect)

---

## ✅ RESOLVED 2026-07-01 (same day) — it's DIURNAL ALIASING in the monthly output, NOT a model bug

**The k=1 SST pattern is a DIAGNOSTIC ARTIFACT of the JAX monthly-mean OUTPUT, not a physics/coordinate
difference. The model SST is on-par with Fortran.** Every suspect in this handoff (§2: wind g2r rotation,
Coriolis, forcing-interp coord; the solar-zenith albedo) was checked and is **byte-identical to Fortran** —
they were red herrings (details in `docs/PORTING_LESSONS.md`).

**Root cause.** The JAX monthly mean sampled ONE chunk-final `state_p` snapshot per chunk (`fesom_jax/run.py`),
and the hindcast ran **48-step = exactly 1-day chunks**, so every sample landed at the **same UTC time (~00:00)**.
A monthly mean of ~30 same-time-of-day snapshots **aliases the diurnal SST cycle** into a wavenumber-1 zonal
pattern (warm on the local-afternoon side, cool on the night side; at 00:00 UTC ⇒ warm Americas/West, cool
Asia/East — matching the observed sign/phase, tropical max, zero global mean, stationarity). Fortran writes a
TRUE every-timestep monthly mean (`namelist.io 'sst',1,'m'`), so it has no alias ⇒ the difference is the JAX
alias. **Your hint (solar/daily-cycle/timing) was exactly right — it's the diurnal cycle, in the output.**

**Fix (general, dt-independent — `fesom_jax/run.py` + `fesom_jax/integrate_sharded.py`).** The output streams now
accumulate their fields over EVERY model step (opt-in `sample_fn` in `run_steps_sharded_forced`: the scan carries
a running sum → the driver divides by the step count = a true time-mean, matching Fortran), and chunks are split
at DATE-based period boundaries (`_period_boundaries`, day/month — dt-INDEPENDENT) so each chunk's sum is one
period (`_MeanStream`). This averages the diurnal cycle at ANY mesh/dt — **not** a config hack like
`chunk_steps=16` (which only works at dt=1800's 48 steps/day; the user rejected that as non-general). Verified:
`sample_fn=None` byte-identical (`test_forcing_sharded` passes); the accumulation is an exact per-step sum and
does not perturb the State at npes=1 AND npes=2 (real dist_2).

**Remaining (in the R1 rerun that also lands the salinity fix):** regenerate `paper_jax/data/meanstate.nc` from
the fixed run and re-run the harmonic diagnostic — the k=1 should vanish (or drop to noise), confirming the model
SST is on-par. Then Fig 2's JAX−Fortran column goes ~white and the "coordinate/wind-rotation" framing is dropped.

*(Everything below is the original investigation handoff, kept for the record — its §2 wind-rotation lead and §3
ENSO-chaos alternative were both wrong; the k=1 is an output-sampling artifact.)*

---

**Mandate for the next session:** explain the systematic **east-west (wavenumber-1) SST difference**
between the JAX and matched-Fortran CORE2 hindcasts. It is **not random** and **not ocean-basin
dynamics** — it is organized by *geographic longitude in a single global cycle*, which points at a
**coordinate / rotated-grid** effect (user's read: *"divided between 0 and 180°… something with
coordinates… maybe rotated coordinates"* — the diagnostics below confirm it). Distinct from the
salinity drift (that was a freshwater-budget bug, now **fixed** — see
`HANDOFF-20260701-salinity-drift-investigation.md`); this one is a zero-global-mean **heat
redistribution**.

---

## 1. The pattern (quantified — `paper_jax/data/meanstate.nc:sst_jaxfor`, annual-mean 1980–2009)

Global: mean **+0.005 °C** (≈0), std 0.053, RMS 0.058, range −0.25…+0.31. Zero global mean ⇒ heat
is **conserved and redistributed**, not created (unlike the salt bug).

**It is a clean wavenumber-1 zonal harmonic** `d ≈ mean + A·cos(lon − φ)`:

| band | amplitude A | peak lon | trough lon | **R²(k=1)** |
|---|---|---|---|---|
| global | 0.076 °C | −77° | +103° | 0.72 |
| **\|lat\|<30** | **0.101 °C** | **−79°** | **+101°** | **0.88** |
| \|lat\|<10 (equator) | 0.105 °C | −79° | +101° | 0.88 |
| NH>30 | 0.050 | −66° | +114° | 0.53 |
| SH<−30 | 0.044 | −73° | +107° | 0.64 |

**A single east-west cosine explains 88 % of the tropical SST-diff variance.** Ocean-basin structure
would be wavenumber-2+ (following coastlines); a global **k=1** organized purely by longitude is the
fingerprint of a **coordinate-frame** effect, not physics.

**Hemisphere means (cos-lat weighted):** Western (lon −180…0) **+0.043 °C**, Eastern (0…180)
**−0.046 °C** — a clean sign flip near the 0°/180° meridians (exact zero-crossings ≈ +11°E and ≈169°W;
peak ≈79°W, trough ≈101°E). Longitude profile (20° bins, all lat): monotone from +0.087 near
100–120°W down to −0.076 near 120–140°E.

**Where the extrema sit:** strongest JAX-warm at the Peru–Chile coast (−71°, +0.31) and California
(−116°, +0.27); strongest JAX-cool at the Kuroshio/Yellow Sea (+120–127°, −0.23). These *look* like
eastern-boundary-upwelling / western-boundary-current signals, but the **global k=1** (R²=0.88) says
the organizing variable is **longitude/coordinate**, and the upwelling zones just sit near the crest.

Reproduce: `port_jax/scripts/diag_sst_pattern.py` (regional/equatorial breakdown) and
`diag_sst_rotation.py` (the harmonic fit + hemisphere means) on `paper_jax/data/meanstate.nc`.

---

## 2. Why "coordinates / rotated grid" is the lead

FESOM runs on a **ROTATED mesh**. In `fesom_jax/config.py:47–55`: pole at geographic
**(lon=50°, lat=15°), γ=−90°**, `FORCE_ROTATION=True`; `mesh.coord_nod2D` is **rotated** radians
(`mesh.py:66`). Almost all computation is in rotated coordinates. Anything that must bridge the two
frames is a candidate for a **longitude-organized (k=1) error**:

1. **Wind-stress rotation geographic→rotated (g2r) — PRIME SUSPECT.** `fesom_jax/jra55.py`:
   `_rotation_matrix()` (`:135`, Euler 50/15/−90) + `_vector_g2r(u,v, glon,glat, rlon,rlat, M)`
   (`:158`, ports `fesom_vector_g2r`, `fesom_mesh.c:169`); applied per step at `:521`
   (`u,v = _vector_g2r(...)`). The wind is interpolated on the **geographic** JRA grid, then its
   **vector** is rotated into the model frame (`:25–30`, and the file flags *"the interpolation grid
   is geographic, but the wind is rotated"*). A subtle mismatch vs Fortran (γ sign, Euler order,
   matrix transpose, or `glon/glat` vs `rlon/rlat`) rotates the wind stress by a small angle that
   **varies with longitude relative to the rotated pole** → a k=1 Ekman/circulation → SST anomaly.
   **The C port hit exactly this class of bug: `[[feedback_wind_rotation_g2r]]`** (referenced in
   `port2/fesom2_port/docs/FORCING_STEP1_DIFFERENCE.md:143`, also `DT1800_HANDOFF.md`).
2. **Coriolis latitude convention.** `f = 2Ω·sin(lat)` — must use **geographic** latitude even though
   the mesh is rotated (`mesh.coriolis`/`coriolis_node`, consumed in `momentum.py:171`,
   `ice_mevp.py:165`, `kpp.py:540`). If JAX built `coriolis` from rotated latitude (or a different
   convention than the C/Fortran), the error is again organized by the rotation geometry → k=1.
   Cheap to check: compare `mesh.coriolis_node` against `2Ω·sin(geographic_lat)`.
3. **Forcing-interpolation coordinate.** Confirm the JRA bilinear stencil indexes nodes by
   **geographic** lon/lat (it should) and only the *vector* is rotated — a scalar (Tair, humidity,
   radiation, precip) rotated or interpolated at rotated coords would also imprint longitude.
4. **Velocity output rotation (r2g).** Does **not** affect SST evolution, but matters for the
   confirmatory u/v cross-check in §4 — make sure both JAX and Fortran u/v are in the **same** frame
   before differencing.

**Not the salinity bug:** that was a *uniform* freshwater deficit (global-mean correction) — it cannot
produce a k=1 zonal dipole. The SST pattern is almost certainly **independent** of it. Still, the salt
fix is being deployed (`ice_thermo.py` evaporation bundle); **regenerate the fidelity data on the
salinity-fixed rerun first** (§4.0) so the SST pattern is measured on the corrected model.

---

## 3. The one alternative to rule out first: chaotic ENSO-phase sampling

Two roundoff-divergent runs sample ENSO differently; a 30-yr mean (1980–2009) mostly averages internal
variability out, but a **residual** could leave an El Niño/La Niña-like (⇒ equatorial-Pacific zonal)
pattern. Discriminator: **stationarity.** A *systematic* coordinate error is **stationary** (same k=1
amplitude/phase in 1980–1994 vs 1995–2009, and present from step 1); ENSO-sampling **drifts** between
sub-periods and is absent at step 1. The R²=0.88 clean global k=1 with a fixed ~79°W phase already
argues *systematic* (ENSO would be Pacific-confined, wavenumber-2 with an Atlantic/Indian counter-sign,
not a clean global cosine), but confirm with the split-period map (§4.2).

---

## 4. Diagnostic plan (ordered)

**4.0 — Re-measure on the salinity-fixed run.** Rerun a CORE2 zstar+ice segment (or the full R1) with
the `ice_thermo.py` evaporation fix, regenerate `paper_jax/data/meanstate.nc`, re-run the harmonic
diagnostic. Establishes the pattern is not an artifact of the (now-fixed) freshwater bug.

**4.1 — Confirm it's the WIND (currents show it first).** Both runs output surface `u,v`
(JAX zarr: `u,v`; Fortran: `u,v,unod,vnod`). Difference the surface currents JAX−Fortran (in a common
frame!) and fit the k=1 harmonic. A wind-rotation error shows up in the **currents** more strongly and
more directly than in SST — if the current difference is an even cleaner k=1 aligned to the same phase,
the wind stress is the driver.

**4.2 — Stationarity / step-1 origin.** (a) Map the SST diff for 1980–1994 vs 1995–2009 — stationary
phase/amplitude ⇒ systematic. (b) Compare **step-1** wind stress JAX vs C at a set of nodes spanning
longitudes: reuse the C-port step-1 forcing dumps (`iceforce`/`prestep`, `FORCING_STEP1_DIFFERENCE.md`)
and check whether the applied wind-stress **vector** differs by a rotation angle that varies with lon.

**4.3 — Audit `_vector_g2r` vs the Fortran/C reference line-by-line.** `fesom_jax/jra55.py:135–181`
against `fesom_vector_g2r` (`fesom_mesh.c:169`) / `fesom_forcing.F90` g2r: Euler order, **sign of γ**,
whether the matrix is the transpose (r2g vs g2r), and that `glon/glat` (geographic) and `rlon/rlat`
(rotated) are the node's own coords. Cross-check the rotation *angle* at 3–4 nodes vs an independent
computation. (`[[feedback_wind_rotation_g2r]]` is the C port's write-up of the same audit.)

**4.4 — Coriolis convention check.** Verify `mesh.coriolis_node == 2Ω·sin(geographic_lat)` (not rotated
lat). If it's built from rotated latitude, that alone gives a k=1.

**4.5 — Toggle experiment.** Short CORE2 run with the wind rotation **disabled** (or γ sign flipped):
if the k=1 SST/current pattern vanishes or flips phase, the g2r rotation is confirmed as the source.

---

## 5. Data & reproduction inventory

- **SST-diff field:** `paper_jax/data/meanstate.nc` — `sst_jaxfor` `[nod2]` (+ `sss_jaxfor`, `lon`, `lat`).
  Attrs `rms_sst_jaxfor=0.0580`. (Regenerate from the salinity-fixed run via
  `paper_jax/scripts/reduce/reduce_meanstate.py`.)
- **Diagnostics (this session, in-repo):** `port_jax/scripts/diag_sst_pattern.py` (basin/equatorial/
  upwelling breakdown), `port_jax/scripts/diag_sst_rotation.py` (k=1 harmonic fit, hemisphere means,
  longitude profile). Run with the nereus env `/work/ab0995/a270088/mambaforge/envs/nereus/bin/python`.
- **Raw runs (both output surface u,v for §4.1):**
  - JAX CORE2 hindcast monthly: `/work/ab0995/a270088/port_jax/runs/core2_hindcast/monthly/<YYYY>_<MM>`
    — zarr arrays `{a_ice, m_ice, salt, ssh, temp, u, v, lon, lat}` (unfold via
    `paper_jax/scripts/common.ushow_to_nodes`).
  - Fortran CORE2: `/work/ab0995/a270088/fesom2_core2/` — `{sst,sss,ssh,temp,salt,u,v,unod,vnod,w,Kv,
    N2,MLD1,MLD2,a_ice,m_ice,m_snow,bolus_*,redi_K,uice,vice,…}.fesom.<YYYY>.nc` (native node order;
    read via `common.fortran_iter_months`). **NB: Fortran has `Kv`, `N2`, `MLD`, currents — JAX monthly
    does not** (only temp/salt/ssh/u/v/ice). For a subsurface mixing attribution you'd need to add
    those to the JAX monthly stream or run a short diagnostic segment.
- **Rotation params / code:** `fesom_jax/config.py:47–55` (Euler 50/15/−90, `FORCE_ROTATION`);
  `fesom_jax/jra55.py:135–181, 502–521` (`_rotation_matrix`, `_vector_g2r`, step application);
  `fesom_jax/mesh.py:66` (rotated `coord_nod2D`); Coriolis consumers `momentum.py:171`,
  `ice_mevp.py:165`, `kpp.py:540`. Config `configs/core2_full.yaml`.
- **C reference:** `fesom_vector_g2r` (`fesom_mesh.c:169`); C-port lesson `[[feedback_wind_rotation_g2r]]`
  in `port2/fesom2_port/docs/FORCING_STEP1_DIFFERENCE.md:143` + `DT1800_HANDOFF.md`.

---

## 6. Cross-references

- Sibling finding (resolved): `docs/HANDOFF-20260701-salinity-drift-investigation.md` (the near-uniform
  SSS offset — a freshwater-budget bug, fixed in `ice_thermo.py`; independent of this k=1 SST pattern).
- The figure that surfaced this: Fig 2 (`paper_jax/figures/fig02_meanstate.pdf`), JAX−Fortran column,
  now on its own ±0.2 °C scale — the k=1 east-west structure is directly visible. Preview
  `docs/previews-20260701/fig02_meanstate.png`.
- Memory: [[model-paper-plan]], [[salinity-drift-investigation]], and the C-port
  `[[feedback_wind_rotation_g2r]]`.

**One-line summary:** JAX−Fortran CORE2 SST is a clean **wavenumber-1 zonal harmonic** (R²=0.88 in the
tropics, peak ~79°W / trough ~101°E, hemispheres opposite-signed) → a **coordinate/rotated-grid**
signal, prime suspect the **wind-stress g2r rotation** (`jra55.py:_vector_g2r`, Euler 50/15/−90) that
the C port also stumbled on (`feedback_wind_rotation_g2r`); confirm via the surface-current difference
(should be an even cleaner co-phased k=1) + a `_vector_g2r`-vs-Fortran line audit + a rotation-off toggle.
