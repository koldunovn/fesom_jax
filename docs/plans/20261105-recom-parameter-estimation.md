# REcoM spectral parameter estimation in fesom-jax: weak-constraint variational state and parameter estimation

Written 2026-11-05 on Albedo. **Draft for discussion with the user: nothing here has been started or measured.**

## Goal

Estimate REcoM spectral parameters, and the state and model error alongside them, from observations, using the adjoint
that JAX gives. Two outputs matter equally:

1. **A better, objectively chosen parameter set** with uncertainties. The current reference `kdind15wco2sl1` came from
   sequential single-change adoption by expert judgement, with no objective cost.
2. **A diagnosis of model structure**: where and when the data ask for what fixed parameters cannot give.

## Why weak constraint (the lessons this builds on)

From the full papers (`/albedo/scratch/user/sloza/parameter_reference`, read 2026-11-05).

- **Losa, Kivman & Ryabchenko 2004** (J. Mar. Syst. 45, 1-20, doi:10.1016/j.jmarsys.2003.08.005) is the direct ancestor: a
  generalized inverse, `F(c, p) = int |dc/dt - M_p(c)|^2 dt + beta sum |L_m(c) - d_m|^2` (omega = 1), minimised over the whole
  trajectory and six parameters by conjugate gradients, with a periodicity condition `c(0) = c(T)`, for a 0-D four-component
  model in independent 5 x 5 degree North Atlantic boxes, against monthly CZCS chlorophyll.
- **Losa, Kivman, Schröter & Wenzel 2003** (J. Mar. Syst. 43, 31-49, doi:10.1016/j.jmarsys.2003.06.001): a sequential (SIR)
  counterpart for a nine-compartment model at BATS, with chlorophyll, nitrate, PON and POC; 1000 members; bacteria held out.

What the plan takes from them:

1. **Secondary inversion (2004, Section 7.2).** The weak solution breaks the model equations, so fluxes cannot be read from
   it. Mass balance is then imposed strongly and the process formulas weakly: the residuals are redistributed among the
   processes (primary production, grazing, mortality, ...), staying closest to what each formula gives. The processes that
   move most are the parameterizations to improve (at Station S: phytoplankton mortality, zooplankton losses, detritus
   breakdown; primary production and grazing held up). **With constant parameters this is the route from model error
   to named processes, and it restores the budgets.**
2. **Weights are chosen a posteriori.** Model error cannot be estimated a priori and data error barely. 2004 uses the maximum
   data cost criterion (Kivman et al. 2001): the data cost tends to 0 for very small and very large beta, so take beta at its
   maximum; well above it the solution goes negative. 2003 chose noise by trial and error, about 20 % of the characteristic flux
   of each equation: a first guess for Q.
3. **The credibility ratio.** `r` = data cost / model cost (2004 Eq. 17), mapped: r < 1 marks where the data are worse than the
   model (coastal and shelf waters, the cloudy north-east), r < 10 over most of the basin meant model error cannot be neglected.
4. **Model error has a systematic part.** The residuals were biased, markedly so along the Gulf Stream and North Atlantic
   Current, although the formulation assumed white noise.
5. **Parameters must be in the control.** With parameters fixed at a wrong set, the state-only inverse failed: "errors in
   specification of the model parameters cannot be compensated by accounting for the model noise" (2004 Section 6; also 2003
   Fig. 6 and Kagan et al. 1997).
6. **Do the twin after the real inversion too.** Following Schartau et al. (2001), 2004 argues for a twin around the real
   solution: the problem is nonlinear, so a twin elsewhere in control space says little about the error bars.
7. **An apparent process signal can be an operator error.** The seasonal mortality of 2003 may be an artifact of a constant
   Chl:C ratio in the observation operator. The discussion calls it "fact or artifact".
8. **Not everything is determined.** 2004 tuned 6 parameters from 12 monthly data with no parameter penalty. In 2003, 8 of 15
   reached stable estimates. `g` and `k3` enter one term and could not be separated. Chlorophyll alone "may not suffice",
   and combination with in situ data is "vital". Holding out a compartment (bacteria) was a strong test.

**A tension to keep in view.** Both papers conclude that parameters vary: 2004 found significant, stable spatial variation
(larger than the twin's errors), found it "unlikely to find a unified set" for the North Atlantic "or even more for the global
ocean", and answered "no" to whether estimates apply elsewhere. Constant global parameters are the user's deliberate start. Those
were local 0-D models with one phytoplankton group, and part of that variation may be what REcoM's four PFTs and 3-D transport
now represent explicitly, which is an open question this study can test. Where it does not, real parameter variation will appear
in `eta` and must not be mistaken for a wrong process.

The structural errors in the user's notes make the case for this model. Hand tuning did not remove them:

- Southern Ocean Jun-Oct Chl 4-6x high, surviving all mortality forms tried.
- HNLC lost: Station P DIN 0.04 against 9.9, surviving Fe removal.
- Preformed N 45-60S 7.3/8.8 against WOA 13.6/16.8, with a mode layer too thin (physics, not biology).
- Gyre NPP 2.8x CbPM (kdind15w).

A strong-constraint fit would bend parameters to absorb these, and the parameters would then be wrong for the right
biology. This is the main scientific argument for weak constraint.

## What decides the design

**1. Physics never reads a BGC tracer (`step.py`, `recom/wiring.py`), so the BGC adjoint is one-way.** With respect to
REcoM parameters or BGC state the physics is a constant. `ADJOINT_HORIZON.md`'s day-scale limit came from the turbulence
closure's response to the ocean state, and chaos needs a flow that responds to the control: neither applies here. What
remains is the biology's own growth, the FCT limiter (subgradient, `LIMITER_GRADIENTS.md`) and the floors. **So the
horizon may be months rather than days. That is an expectation from the code, not a measurement; only 1-day gradients
exist (G6, `test_recom_grad*.py`). Phase 1 measures it, and everything after depends on it.**

**2. The forcing is repeat-year `clim61`.** The model has no real 1998 weather, so compare monthly climatologies, with
interannual variability inside R. `obs_table.txt` section A already gives that yardstick (Rrs log10 sd 0.012-0.13).
Dated matchups would be wrong for this model.

**3. Rrs alone cannot constrain the biology.** `obs_table.txt`: about two separable directions (backscatter settings
share a cosine of 0.84-0.96, cdomcoeff against QYmax -0.90), and quantum yield of coccolithophores, photoinhibition,
deg_Chl, k_din and dark respiration are undetectable (< 0.5 interannual sd). Rrs must be combined with nutrients and
carbon data.

**4. The Rrs misfit is operator bias as much as biology.** Clear-water misfit of r2bdoc is 0.05-0.18 in log10 against an
interannual sd of 0.012-0.05, and NEXT_STEPS 21e/21k note that Rrs there asks for more carbon while carbon is already
too high. Fitting raw Rrs would drive carbon the wrong way, so operator bias must be estimated, not ignored.

**5. Equifinality is known.** NMinSlope ~ NCmin ~ photoinhibition; agg_PP ~ cdomcoeff; pMax cancels between N_assim and
Cphot; grazing compensates growth; PFTs replace each other. Priors and Hessian analysis must expect this.

**6. Some parameters are dead or need care.** Under the spectral 4-PFT build `P_cm`, `P_cm_d`, `P_cm_c` and the light
slopes `alfa*` have exactly zero derivative (`test_recom_grad_fork.py`). `graz_max` is numerically zero over 1 day.
`max(tiny, ...)` floors stop gradients through a collapsed group, so a dead population cannot be revived by the adjoint.

**7. Cost is the practical limit.** One model year is 11680 steps, about 7.4 h on one PCIe A100 (jobs 48698174-75). If a
gradient costs about 5 forward runs (an assumption, from the reverse/forward ratio in `ADJOINT_API.md`), 30 iterations of
a one-year window take about 46 days on one Albedo GPU. The port's notes give 66 ms/step on 4 GH200 (job 28458068), about
13 min a year. **Global one-year windows need faster hardware or shorter windows** (decision D3).

## The formulation

**Decision (user, 2026-11-05): parameters are constant in time and space, at least to start.** The system is far more
complicated than the 0-D models of 2003-04. Time- or space-varying parameters remain a possible later extension.

Minimise over `z = (theta, b, x0, eta)`:

```
J = 1/2 |theta - theta_b|^2_{B_theta}            parameters: constant, log space (positive by construction)
  + 1/2 |b - b_b|^2_{B_b}                        observation-operator bias (Q, background backscatter)
  + 1/2 |x0 - x_b|^2_{B_x}                       initial state, low-dimensional basis
  + 1/2 sum_k eta_k^T Q^-1 eta_k                 model error
  + 1/2 sum_i |y_i - H_i(x; b)|^2_{R_i}          observations, log space for Rrs and Chl

with  x_{k+1} = M(x_k; theta) + xbar_k ⊙ G eta_k        (all controls normalised by the reference, below)
```

**Model error as a low-dimensional additive term in the BGC tendencies (recommended).** `eta` is a smooth correction by
month (AR(1) through Q) and region (about 10 biomes), with a time-mean part because the residuals of 2004 were biased; `G`
spreads it over the surface layers. It covers **each compartment class**: nutrients, phytoplankton, zooplankton, detritus,
so that the secondary inversion has a residual in each equation to redistribute. That is a few thousand controls.
First-guess Q: about 20 % of each equation's characteristic flux (2003), the ratio then set a posteriori. `eta = 0` recovers
strong constraint exactly, so both are one code path. It enters only BGC tracers, so the one-way adjoint stays one-way.

**How this differs from 2004.** There the whole trajectory was the control, the cost was analytic in all variables, and no model
integration was needed; that formulation is better posed than strong constraint. Ours keeps a forward model with a reduced
`eta`, because 37 tracers on 3-D CORE2 rule out the full form, so it keeps more of strong constraint's nonlinearity and kinks.
**To test on pi:** state controls at monthly nodes with the jumps as model error (multiple shooting), which is closer to 2004
and gives short, independent windows.

**Periodicity.** With repeat-year forcing, 2004's `c(0) = c(T)` applies to the upper ocean as a penalty, an alternative to
free `x0` control. The bisect's surface Alk drift would violate it. Not for the interior, which is not periodic on a year.

**Secondary inversion (after the weak solution).** Per region and month, mass balance strong, process formulas weak, using REcoM's
flux diagnostics: the output is budgets that close, and a ranking of the parameterizations that carry the error.

**Rejected: gridded model error at every step** (37 tracers x about 6M points x 11680 steps). Unaffordable, and far beyond
what the data determine.

**Normalisation by a reference solution.** The user's formulation, used in the 2004 study though not stated in the paper
([note](https://claude.ai/artifact/VHJ3qAaMBfMJD1azBiC1am)): `x(t) = xbar(t) ⊙ xt(t)`, `p = pbar ⊙ pt`, with the
reference kept fixed for the whole inversion, so `xt = 1, pt = 1` is the reference. All controls are nondimensional:

- Gradients become `xbar ⊙ dJ/dx` and `pbar ⊙ dJ/dp`, and the Hessian `D^T H D`, so no compartment or season dominates the search.
- Unit-variance penalties mean a diagonal prior of relative errors, sigma = xbar: the preconditioning of Courtier et al. (1994),
  with relative rather than absolute errors. A chosen relative error scales it.
- Model error is relative too: `x_{k+1} = M(x_k; theta) + xbar_k ⊙ G eta_k`, the residual `r = (dx/dt - M) / xbar`.
- `x = xbar ⊙ exp(xi)` adds positivity and a lognormal prior, of which the normalised form is the first-order limit. Use it for
  `theta` (already log space) and `x0`; test it for `eta`.

Choices for this model:

- **Reference (decided 2026-11-05):** time-varying, not one constant per tracer: seasonal ranges are large. For CORE2 the source must be
  a run on the chosen physics: a FESOM 2.8 run with the reference's REcoM settings (Phase 0.1), not `kdind15wco2sl1bf`, which
  has the 2.6 production physics. Monthly means to start, as the output is monthly. With biome-month
  `eta`, `xbar` is only needed on that basis, so daily biome means are cheap; multiple shooting uses the fields directly.
- **A floor on `xbar` per tracer.** Sigma = xbar means a group near `tiny` gets almost no freedom, so a collapsed group stays
  collapsed. Near-zero water also makes the division ill-defined, as do dry cells.
- **It conditions the problem, not the dynamics.** It removes magnitude effects but not a true instability of the adjoint,
  which is why the scaled sensitivity is the right Phase 1 diagnostic.

**Solver.** Control-variable transform, then L-BFGS (`calibrate.py`, optax). Posterior from Gauss-Newton Hessian-vector
products (`jvp` of `vjp`) and Lanczos: leading eigenvectors show which parameter combinations the data determine, which
replaces intuition about equifinality with a computed answer.

**Weights and a posteriori checks.** Choose the data/model weight by the maximum data cost criterion (2004), one sweep in the
twin and in Phase 4. Map the credibility ratio `r` by region. Add `J_min` against its expected value, Desroziers-type
statistics for B, Q and R, and held-out data, including a held-out compartment as with bacteria in 2003.

## Observations

Variational `R` = instrument/algorithm error + interannual variability + representation error, on monthly climatologies.

| Priority | Data | Constrains | On disk | Note |
|---|---|---|---|---|
| **Core** | OC-CCI v6 Rrs, 6 bands, 2003-12 | optics, CDOM, biomass, clear-water N:C | `obs_data/oceancolour/occci_rrs` | R from `obs_table.txt`; the 2007-08 daily product carries `Rrs_*_rmsd`, `_bias` (not used so far) |
| **Core** | WOA nitrate, silicate (+ phosphate, O2 if wanted) | drawdown, HNLC, preformed N | WOA13 annual; WOA18 in `pool/recom/eval` | monthly surface fields needed for the seasonal drawdown |
| **Core** | GLODAPv2 DIC, Alk, nutrients | carbon, remineralisation, interior | `pool/recom/eval` | unused so far |
| **Add** | SOCAT / MPI-SOM-FFN surface pCO2 | air-sea carbon | `pool/recom/eval` | unused so far; the bisect shows carbon is sensitive to code, see D1 |
| **Add** | HPLC (Soppa, Xi) | PFTs, in situ | `obs_data/HPLC_PFT` | sparse but direct; the strongest PFT constraint |
| **Add** | Huang dFe | iron, HNLC | `pool/recom/eval` | sparse; needed for k_Fe, kScavFe |
| **Add, low weight** | CMEMS PFT | PFT patterns | `obs_data/oceancolour/cmems_pft` | stated uncertainty Dia ~149 %, Hap ~70 % |
| **Validate only** | CbPM and Arrigo NPP, MAREDAT, Burt 2018 | independent checks | on disk | NPP products are themselves models |
| **Add** | BGC-Argo nitrate, Chl, bbp, O2 (Sprof, 1595 floats) | SO winter, vertical structure | `p_specbic/sloza/obs_bgc/bgc_argo` | fetched 2026-11-05; 125k nitrate, 198k Chl, 195k bbp profiles; the best single addition. Mostly 2012 onwards, so climatological use |
| **Add** | POC in situ: GO-POPCORN v2, 2664 samples | carbon, separates Chl from carbon | `p_specbic/sloza/obs_bgc/poc_insitu` | fetched 2026-11-05; surface, Bio-GO-SHIP 2011-20. bbp from Argo adds depth, through a bbp-to-POC conversion |
| **Validate only** | BICEP satellite POC (OC-CCI v5, monthly, 4 km, with bias and rmsd) | POC patterns | not fetched (CEDA, open, 247 GB) | built from Rrs490/Rrs560, so not independent of Rrs |
| **Get** | in-situ aCDOM | CDOM | not found | |
| **Maybe** | BATS, HOT, Station P time series | seasonal cycle, 1-D checks | not found | the setting of the 2003-04 papers |

**The satellite POC algorithm (read 2026-11-05).** BICEP v5.0 is exactly `POC = 292 (Rrs490/Rrs560)^-1.49` mg m-3: on January 2003 the
fit returns 292.0 and -1.490, and the product matches the formula to 2e-7, so it holds no information beyond that ratio. It
is Stramski et al. (2008) (a power law, exponent -1.639 for this ratio, from 12-270 mg m-3 in the tropical and subtropical Pacific and Atlantic, over
50 pairs) recalibrated on in situ matchups by Type II regression, chosen over Loisel 2002, Le 2018 and Tran 2019. Stated error
`POC_rmsd`, median 0.17 in log10 (a factor of 1.5). Limits: the same ratio drives Chl, so POC and Chl are "unrealistically
forced to always covary" (Stramski 2008), and POC:Chl from this product carries no independent information; the high end is
underestimated in turbid water (Li et al. 2023); the 2008 paper warns of *Phaeocystis* bloom data off the Rrs-bb relation.
For validation, two comparisons: the algorithm applied to model Rrs (like with like, tests the optics), and model particulate carbon
tracers against the product (tests the carbon, inherits the algorithm's error). Their difference shows the operator's share.

**Do not double count.** OC-CCI Chl, adg_443, CMEMS PFT and CbPM all derive from the same radiances. If Rrs is assimilated,
these are not independent data: keep them for validation, or use them with an honest shared error.

**Assimilating Rrs directly rather than Chl** is the right call: it avoids the band-ratio algorithm and uses the model's
own spectral optics, which is what the spectral version is for.

## Parameter set

Log-space, from the user's sensitivity brackets, about 15-20 to start. Trim with Phase 2.

- **Optics:** cdomcoeff, bb_to_b, bb_bg443, and the BMASS/APART/ACDOM choices (table-based; see Phase 0).
- **Quota and carbon:** NCmin, NMinSlope, res_ovf / f_ovf.
- **Light and pigment:** quantum yield per PFT, deg_Chl.
- **Nutrients:** k_din_d, k_Fe, kScavFe, reminSi.
- **Losses:** lossC, graz_max, agg_PP, rho_N, reminN.

Fix at first: insensitive or dead ones (`P_cm*`, `alfa*`, res_phy_const, totalligand, SiCUptakeRatio). Expect pairs in one term
to stay inseparable, like `g` and `k3` in 2004: `graz_max` with its half-saturation.

## Phases, each with an exit test

### Phase 0: foundations (no assimilation)

- 0.1 A new reference on the chosen physics: FESOM 2.8 (`fesom2.8_recom_spectral`, which carries the ballasting fix) with
  `kdind15wco2sl1`'s REcoM settings (L1 OASIM, `RECOM_CO2LIM_SPECTRAL`, the Phaeocystis CO2 set), its fesom-jax twin, and a
  check that the pair still agrees. Source of `xbar` and of the first guess. `kdind15wco2sl1bf` (2.6 production physics, job
  48871413) answers a separate question: what the fix does to the production reference.
- 0.2 fesom-jax runs the reference: map `RECOM_CO2LIM_SPECTRAL` to `spectral_co2lim` (the only active key it rejects in
  `kdind15wco2sl1`'s namelist), use the L1 OASIM set, add a parity gate against the Fortran.
- 0.3 Port the Rrs operator: `RECOM_CALC_REFLEC` (out of scope today, `params.py`), then `rrs_compare.py`'s conversion
  (R -> rrs -> Rrs, band interpolation, ice and light masks) with Q as a control. Monthly means accumulated in the scan.
  Gate: against `rrs_compare.py` on the same fields.
- 0.4 Make the optics differentiable: `load_tables` is host numpy, so table-derived quantities are constants today.
- 0.5 Add the model-error input: a per-step additive BGC tendency from `eta` by region and month, with `None`-means-off
  keeping the default jaxpr unchanged (the project rule; add the test). `RecomParams` stay scalar.

**Exit:** reference reproduced, Rrs matches offline, defaults byte-identical.

### Phase 1: how far can the BGC gradient reach? (the go / no-go)

On pi, then CORE2: gradients of a surface cost against finite differences over 1, 5, 30, 90, 365 days, for about 6
parameters spanning fast and slow processes, with reverse-field growth per step measured as in `ADJOINT_HORIZON.md`.

Also check that JAX really keeps physics out of the linearisation, and measure time and memory. If physics replay
dominates, consider stored transport (a stated deviation, so measured against the full path first).

**Exit:** a usable window length and an honest cost per gradient. If gradients die early, windows shrink and multi-window
or EKI (`eki.py`) takes over for the slow targets.

### Phase 2: identifiability on twins (pi mesh)

Synthetic observations with the real sampling and R from a perturbed truth; pi needs no new driver.

- 2a Strong constraint: recover known parameters. Compute the Hessian spectrum; report which combinations are determined by
  Rrs alone, by Rrs + nutrients, and by all data. This turns the observability table into a formal result and tests whether
  the data set above is enough.
- 2b Weak constraint: a truth with a process the fitted model lacks (a seasonal mortality), then fit constant parameters
  with `eta` on and off. Strong constraint should bias the parameters; weak constraint should keep them closer to the truth
  and show the missing process in `eta`. This is the 2003 lesson as a controlled test.

**Exit:** known parameters and model error recovered within posterior uncertainty; final parameter list and Q scales.

### Phase 3: real data, strong constraint (the baseline)

CORE2, windowed as Phase 1 allows, core data, with operator bias `b`. Needed as the comparator for Phase 4, and it should
show where parameters are forced to compensate.

Also compute the gradient of the misfit per biome: if biomes pull a parameter in opposite directions, constant parameters
are under strain there. This tests the 2004 conclusion at the cost of a few gradients, with no varying parameters.

**Exit:** lower misfit on held-out data than the reference, or a documented reason why not.

### Phase 4: real data, weak constraint

Add `eta` and the x0 basis, with the weight by maximum data cost. Then, in this order:

- rule out operator error (`b`, the Chl:C lesson of 2003) and real parameter variation (the per-biome gradients) as causes;
- run the secondary inversion: closed budgets and the ranking of parameterizations;
- map `r`, to see where Rrs and the model are each the less credible;
- repeat a twin around this solution, for honest error bars (2004).

**Exit:** checks consistent; parameters credible with error bars; budgets closed; a stated structural diagnosis.

### Phase 5: back to the Fortran, and the science

Run the estimated set in the Fortran model over several years to see that the improvement holds outside the window and
in production. Multi-year quantities (preformed nutrients, interior carbon) get checked here, or via EKI as a
cross-check. Then write up.

### Phase 1 record

- 2026-11-05, 1-day smoke on pi, chl cost, CPU (job 48832211): adjoint against central FD at h = 1e-4: `expon_phy` 1.9e-7,
  `rho_N` 1.1e-6, `deg_Chl` 7.2e-8, `cdomcoeff` 8.6e-7, `k_Fe_d` 4.4e-7; `graz_max` numerically zero. Gradient 18x forward.
- GPU, one A100 (job 48870956): gradient 6.3 s against forward 1.4 s per day, **4.5x**; the 18x was the CPU. Gradients
  agree with the CPU run to 7 digits. **Physics is out of the linearisation:** linearising one step with respect to
  (parameters, BGC state), every physics field's tangent is exactly zero (T, S, uv, w, hnode, Kv, Av, ice, ...); only
  `bgc`, `bgc_old`, `bgc_benthos`, `bgc_carb`, `bgc_diag` are nonzero. The tangent costs 0.030 s against 0.043 s for the
  forward step. This confirms the structure in point 1 above for one step; the horizon itself is still unmeasured.

- 2026-11-06, first sweep to 330 days from the 30-day restart (jobs 48871169-70): `|dJ/dx0|` Chl 0.14, 3.6e6, 3.9e11, 2.1e15, 3.7e27 at
  5, 30, 90, 180, 330 days; DIN 0.12, 2.7e4, 2.9e11, 1.2e11, 1.4e22. Parameter gradients stay O(1) to 90 days, leave the FD range at 180
  days for Chl (dJ/dlog p up to 25) and reach 1e13 at 330 days (DIN 1e8). Not usable beyond about 90 days from this young state.
- Mature start (spin-up 48871811, test 48871812), Chl: noise of J 1.3e-7 at 1 day, 6.4e-7 at 5 days, about 50 times below the young state
  at 5 days. Adjoint against FD (h = 1e-2) within 0.01-0.6 % at 1 day, 0.01-1.1 % at 5 days for five parameters; `graz_max` now live, within 0.4 %
  at 1 day and 3 % at 5 days (slightly outside the noise bar). `|dJ/dx0|` 0.032 to 0.057, scaled 0.0034 to 0.0039: no growth so far. Zoo2 is
  near its floor after two years (surface mean 1.9e-8). Windows beyond 5 days from the mature state are not yet measured.
- FESOM 2.8 reference pair, first 22 months (Fortran 48872095, fesom-jax 48872096-98): NPP 1998 106.90 against 106.90 Pg C/yr, groups within
  0.2 %, inventories within 1e-5 relative (DFe 3e-4), surface patterns corr 1.000. The first test of the spectral CO2 limitation against the Fortran: passed.
- Ballast freeze test, 2026-11-07 (jobs 48931248-49, `FESOM_ADJ_FREEZE_BALLAST=1`, stop-gradient on the particle densities, backward pass only;
  backward scan 48931037 before it): forward unchanged. `|dJ/dx0|` at 330 days Chl 1.0e116 (full 6.4e123), DIN 2.5e103 (full 2.4e127), so ballasting is
  not the whole cause. At 180 days the frozen gradient is worse than the full one: Chl 442 against 0.75, `expon_phy` -38 against -0.027 (FD -0.003 to -0.10).
  The ballast path damps growth in the tangent there, so freezing it is not a remedy. Largest values move to PhyN and DIN (node 1455 level 6 at 180 days,
  nodes 1823, 1882 at 330 days). FD stays normal, so the growth is in the local derivative, not in the response. (180 days holds from a January start only; see the entry on the cap test.)
- Column Jacobian, 2026-11-07 (`--column-jacobian`): one-step Jacobian of single columns, raw and normalised by the column, spectral radius,
  leading singular vector, tracer-pair blocks, FD along the leading vector. Smoke 48931649 (day 2, nodes 1455, 1823): FD agrees with the
  tangent at h = 1e-2 to 1e-6. Full run 48931721: nodes 1455, 1763, 1822, 1823, 1882 on days 150-270, midnight and noon.
- Column Jacobian result (48931721, 50 Jacobians, nodes 1455/1763/1822/1823/1882, days 151-271): FD along the leading singular vector
  agrees with the tangent almost everywhere, so the one-step derivatives are genuine. Spectral radius up to 1.92 per step in about 20 of 50.
  Almost every expanding mode is detritus, mostly the second class (DetZ2C/N/Si/Calc), in one cell. Most have a negative self-derivative
  -1.2 to -1.57 at levels 0-1 (the 5 m layers), at values near zero and alternating in sign, with +0.78 into the cell below: an explicit
  sinking overshoot. Det2 sinks at up to 250 m/d (w_ref2 200, max_sinking_velocity 250), 7.8 m a step against 5 m (CFL 1.56), and the fork's
  sinking has no CFL cap (sinking.py, fork K:574-611; upstream REcoM caps at 0.9 dz/dt). A few deeper cases have a positive self-term
  +1.8 to +1.9 (node 1823 level 16, 1455 level 21, 1882 level 25), not yet explained. No coupling to the other Det2 tracers at the same
  level, so this is not the ballast composition derivative, which is why the freeze test did not cure it.
- CFL test, 2026-11-07: `FESOM_RECOM_FORK_SINK_CFL=1` (sinking.py, read at trace time, default off, test
  `test_fork_cfl_cap_experiment_switch`) gives the fork the upstream cap 0.9 dz/dt. Mature sweep with the cap, 180 and 330 days,
  FD at h = 1e-2, 3e-3: jobs 48932194 (chl), 48932195 (din) -> bgc_gradient_horizon/mature_cfl. Finite and FD-consistent at 330 days
  would confirm the Det2 sinking overshoot as the cause. The cap changes the forward, so it is a test, not yet a model decision.
- CFL test result, 2026-11-08: the cap does not cure it. `|dJ/dx0|` at 330 days: Chl 2e134 (uncapped 6e123), DIN 1e136 (2e127); 180-day gradients unchanged
  (48932194-95). Capped column Jacobians (48936843, 8 of 12 before the time limit) show the switch works: self-derivatives at levels 0-1 go from -1.50, -1.22, -1.53 to
  -1.28, -1.05, -1.17, still beyond -1 although CFL <= 0.9, so a pure CFL overshoot is too simple. Open: the flux limiter at near-zero alternating values, local sources.
  **Usable window depends on season:** from the backward scan (48931037, cost on day 330), 60 days from day 270 is normal, 90 days from day 240 is 1e34. Next proposed: a scan of
  15, 30, 60, 90-day windows every 30 days. Additive eta does not shorten the adjoint path; only segment start states as controls (multiple shooting) cut the chain.
  Explainer page: https://claude.ai/artifact/LhXbn9DHxHxKBz2xyRQ5do
- Decisions 2026-11-08: seasonal window scan approved (`--window-scan`, lengths 15, 30, 60, 90 days, starts every 30 days, smoke 48936964 passed, FD and adjoint within 1e-4 relative at 1-2 days; full scan 48936979 (chl), 48936980 (din) -> bgc_gradient_horizon/window_scan. Segment start states: a low-dimensional
  increment on the forward run for CORE2, compared with the full state plus Q on pi. Length of the outer-loop rerun: not fixed in advance, chosen from the cost in the pi twin, against a baseline rerun with unchanged parameters.
- Daily against interpolated monthly x-bar, 2026-11-08 (48937105, `scripts/recom/xbar_daily_vs_monthly.py`, run kdind15w28sl1d 2001, REcoM floors `max(tiny_i, x)` applied on read; summary in
  fesom_jax_runs/xbar_daily_vs_monthly/summary.txt), relative error, volume-weighted, 0-50 m, median / p90: Alk, DIC, O2 <= 0.2 % / 0.8 %; DIN, DFe, DON, DOC, CDOM, PhyN, PhyC 1 % / 6-10 %; Cocco, Phaeo, Dia, Det, DSi 1-8 % / 20-63 %;
  HetN/C, Zoo3N/C, idetz2* 13-25 % / 250-2000 %, 20-39 % of the volume above 50 %. Below 1000 m nearly zero except detritus. Latitude bands stand in for biomes. Largest errors at low concentrations, 15S-45N; the
  relative metric is sensitive there, and `tiny` (2.23e-16) is too small as a scaling floor, to be decided.
- Local scaling floor, 2026-11-08 (48937173, `--epsilon`): xs = max(x, eps * max_t x, tiny_i) per grid point and level, each product from its own data. At eps = 0.01 the 0-50 m p90 of the monthly-against-daily error falls from 20x to
  100 % for HetC, 3x to 63 % for Zoo3C, 8-10x to 79-100 % for idetz2*; other tracers barely change; 0-11 % of cell-days lifted (Het, Zoo3, idetz2 16-22 %). The remaining differences in Het and idetz2 (20-33 % of the volume above 50 % even at eps = 0.1) look like real intermittency.
  **Decision (user, 2026-11-08):** reference state `max(tiny_i, x)`; scaling floor eps = 0.01 local; monthly x-bar for all tracers to start; daily x-bar per segment for HetN/C, Zoo3N/C, idetz2* as a refinement (about 7 GB per 60-day segment).
- Seasonal window scan (48936979-80, `bgc_gradient_horizon/window_scan`, 45 windows per cost, pi, mature start, last-day cost): **the failure is set by the calendar, not by window length.** Windows ending by about day 195 or starting at day 270 or later are
  good at 15-90 days (`|dJ/dx|` 0.05-0.3, adjoint/FD along the most sensitive direction 0.96-1.06, one 1.28). Any window touching about days 195-270 fails, even at 15 days (start 210: 2e33; start 240: 3e21; start 180: 15 days good, 30 days 2e4).
  Affected columns: mostly western subtropical North Pacific (nodes 1763, 1767, 1822, 1823, 1826, 1883, 1887: 17-28 N, 117-143 E), node 1455 in the tropical Atlantic, coastal node 1882. Multiple shooting alone does not avoid this: a segment inside the band fails. CORE2 not yet measured.
- Daily backward scan inside the band (48938275, `backward_daily`, cost: chl on the last day before day 225, starts 210-224, pi, mature start): scaled `‖x ⊙ λ‖` flat at 3.6-4.2e-3 for starts 224-221, then 0.20 (220), 14.6 (219), 7.1e5 (218), 9.1e27 (210). `dJ/dlog p` stays smooth to start 219 and is corrupted from 218. From start 221 back the raw peak stays at nodes 1763 (28.2 N, 138.7 E, 3373 m) and 1822 (25.7 N, 134.8 E, 3987 m), levels 10-11 (95-108 m), in DetN, DetC, DetCalc, DetSi (first detritus class, ordinary values, DetC about 3.5e-4). Blow-up amplitude from 210 is not repeatable (2e33, 1e35, 2e31 over three runs). The step-by-step `--reverse-trace` failed its `jax.grad` check (48938247) and must not be used.
- Column Jacobians at the onset (48938318, `jacobian_onset`, nodes 1763/1822, printed days 219-222 = scan starts 218-221, steps 0/8/16/24): node 1763 has spectral radius 1.69-2.47 in 15 of 16 samples (1.12 in the other); leading left eigenvectors are negative and real: -1.82 DetN at levels 9-10 (x about 3.2e-5, a local minimum of the profile with a 6x rise to level 11), -1.1 DetC at levels 9-10, and the same values on other tracers (-1.83 DiaN level 2, -1.69 DetZ2Calc levels 3-4, -1.69 PhaeoChl level 2). Node 1822 has radius 1.00 in 14 of 16 samples, so its reverse-field peak is probably fed from neighbours. The DetN block has entries that a sinking scheme at Det1 speeds cannot give (level 9 <- level 10 = +1.32, diagonal -0.44 to -0.83); a sinking-only Jacobian of the DST3 formula on the same profile gives super-diagonal <= 0.06-0.19 at 20-60 m/d. Identical eigenvalues on unrelated tracers point to a tracer-independent operator (transport, FCT limiter a candidate, untested). Earlier sparse dates also have radius 1.5-1.8 at this node (days 151, 181, 241, 271), so a radius above 1 alone does not explain the calendar dependence. The scan's "onset" is when the growing mode emerges above the background; the amplification may begin closer to the cost time. Open: term-split Jacobian (sources and sinks, sinking, advection, vertical mixing).
- Term split (48938360, `--term-split`, `term_split/`, node 1763, printed days 219, 220, 222, steps 0 and 8; state dependence of one term cut by `stop_gradient`, patched calls logged): **cutting the sinking removes every expanding mode** (radius 0.9999-1.008, none above 1.02, in all 6 samples). Cutting mixing or Redi changes nothing (radius as full); cutting the FCT limiter or the whole antidiffusive flux leaves radius 1.69-1.85 in all 6 samples, and removes only the DetC L9-10 mode (-1.72, day 220 step 8; that mode needs both the limiter, which gives new L9 = 0.99 old L10, and the sinking); cutting the sources and sinks makes it worse (2.8, DetZ2 at level 5). This corrects the previous entry: the sinking-only estimate there assumed Det1 at 20-60 m/d, but the blocks give c = 0.39 for Det1 at levels 9-10 (about 125 m/d) and c = 0.78 for Det2 at 250 m/d in 10 m layers.
  The sinking part of the blocks matches the two branches of the DST3 limiter exactly. With psi = 1 (saturated) the flux is `w (2 t_up - t_dn)`, so row k has -2c on itself and +c on the cell below (DetZ2Calc L3: -1.60, +0.78), and the pair L3-L4 has eigenvalue -1.69 for c = 0.78. With `psi = (1-c)/c theta` the correction no longer depends on w: entries +-(1-c) and eigenvalue 2 - c (DetZ2N L18: 1.846, c = 0.15). `recom_sinking.F90` (fork `:665-666`, awiesm-2.6 `:479-480`) has `0.5*wM*(trarr(nz-1) + psiP*Rj)` with `Rj = trarr(nz-1) - trarr(nz)`: for downward w the upwind cell is nz-1, so the correction moves the face value AWAY from the downstream cell, whereas the upward branch `trarr(nz) + psiM*Rj` moves it towards the downstream cell. The JAX port reproduces the Fortran. Not yet tested: the same Jacobian and the 15-day window with the opposite sign; what sets the calendar dependence.
- Audit of the DST3 sign, 2026-11-11 (no job; 1-D column test on CPU in the session scratchpad): the REcoM limiter is MITgcm `pkg/generic_advdiff/gad_dst3fl_adv_r.F` with the same `Rj = t(k-1) - t(k)`. MITgcm has `t(km1) - psiP*Rj` for downward flow, REcoM `+ psiP*Rj`: a sign error (with theta = 1, psi = (1-c)/2 and the face value must be Lax-Wendroff, `t_up + (1-c)/2 (t_dn - t_up)`). Present in the 2.8 spectral branch (`src/int_recom/recom_sinking.F90:665-666`), the 2.6 fork, and the standalone REcoM and FESOM2 copies on Albedo. Second slip: the psiM bound divides by `(1.d-20 - cfl)` (`:663`), inactive because sinking has no upward flux (the JAX port has `+cfl` there, also inactive). 1-D test, 20 cells of travel, relative L2 error, as coded / MITgcm sign / upwind: Gaussian c = 0.15 0.55/0.06/0.42, c = 0.39 0.49/0.05/0.35, c = 0.78 0.29/0.02/0.18; mass conserved and no negatives in all three, so the coded scheme is more diffusive than upwind. Largest one-step Jacobian radius over 201 random profiles, coded / MITgcm sign / upwind: c = 0.15 1.70/1.15/1.00, 0.39 1.22/1.39/1.00, 0.78 1.69/1.78/1.00, 1.56 6.12/2.56/1.00. The coded 1.69 at c = 0.78 is the node 1763 eigenvalue. **The flip fixes the forward scheme but is unlikely to cure the adjoint alone**: limiter linearisations expand with either sign. Candidates for the gradient: a smooth limiter or upwind sinking in the adjoint. Changing the model sign (Fortran too) is the user's decision.
- Sinking variants, 2026-11-11 (pi, mature start, `--start-day 1`; first submission 48963345-50 failed at setup without it; `sink_variants/<v>/`): (a) `FESOM_RECOM_SINK_GRAD=upwind`, (b) `FESOM_RECOM_SINK_DST3_SIGN=mitgcm`, (c) both. Column Jacobians at node 1763, printed days 219-222, steps 0/8/16/24 (48964422, 48964424, 48964426): radius (baseline 1.69-2.47) (a) 1.00-1.04, FD along the leading direction within 0.1 %; (b) 1.00-1.18, FD agrees in 15 of 16 (0.65 once, a limiter branch switch); (c) 1.01-1.11 but FD/tangent 0.05-0.14 in 13 of 16 (the upwind tangent is a poor model of a scheme whose correction is really third order; with the coded sign the correction is mostly near zero, so (a) is nearly exact). Window scans (48964423, 48964425, 48964427; 15 and 30 days from starts 120, 180, 210, 240, 270; chl and din; usable = `|dJ/dx0|` < 1e3 and adjoint/FD along p sign(g) within 10 %): baseline 10 of 20, (a) 2, (b) 2, (c) 8. **No sinking treatment cures the failure; the usable windows move.** (a) fixes 210/15 only partly (6e3, adjoint/FD 1.00) and breaks 120 and 180 (1e17-1e45) with the same forward run; (b) all 30-day windows fail, 15-day parameter gradients right at 120-240 but state gradients 1e5-1e8; (c) 240 good at 15 and 30 days, 120 and 210 fail. The sign changes pi surface Chl and DIN costs by only 0.1-0.5 %. Backward scan (a) (48965082, cost day 135, chl): flat from start 134 to 129, 1.3e5 at 126, 1.4e15 at 123, 1.3e22 at 120; peak in DetZ2N with HetN at nodes 1919, 1831, 1891 (levels 5-10) where x = 1e-16 to 1e-12. Term split with (a) at nodes 1919 and 1831 (48965342, printed days 122/124/126, steps 0/16): full radius 1.00-1.63, the few modes above 1.02 (CoccoN, PhaeoC level 37; PhaeoN level 2) vanish when the FCT limiter, the antidiffusive flux or the sinking is cut; mixing and Redi change nothing; cutting the sources and sinks raises it to 2.36 (DetZ2 modes). The single-column Jacobians do not reproduce the backward-scan growth, so that growth is spread over neighbouring columns or many steps. Reading (hypothesis, untested): the adjoint linearises round-off noise in the Zoo2 compartment, at sign changes where the flux limiters are not smooth. **Correction (user, 2026-11-11): Zoo2 is not extinct but not yet established.** The fork's cold start puts Zoo2 and DetZ2 at exactly 0 when the coccolithophores are on too (`tracers.init_value`), the sms clip floors them at tiny. Restarts: surface Zoo2N mean 2.2e-16 at day 30, 1.9e-8 at day 730, about 9.5 e-folds a year and still exponential; HetN 4e-16 to 5.6e-3 (above 1e-3 at 12 % of surface nodes); MicZooN 2e-12 to 0.18 (90 %). At that rate Zoo2 reaches the mesozooplankton level after roughly 3.5 years, consistent with macrozooplankton crossing the mesozooplankton plateau at month 41 on CORE2 (page Spectral Light in REcoM3). The two-year pi spin-up (48871811, 2.4 h on one A100) is therefore not a mature state for Zoo2 and only partly for HetN. FESOM_RECOM_GRAD_FREEZE_EXTINCT (the name kept) is then a diagnostic of that compartment's role, not a remedy; the remedy is a spin-up long enough for the zooplankton to establish. Proposed next: no gradient through Zoo2*/DetZ2* below a threshold (forward unchanged); horizon on CORE2 where Zoo2 may live; later, a linear BGC advection in the adjoint.
- Freeze test, 2026-11-11 (48966043 thr 1e-6, 48966044 thr 1e-10, `FESOM_RECOM_GRAD_FREEZE_EXTINCT`, same 20 windows, 2-year start): usable 10 of 20 at both thresholds, the same windows as the baseline, blow-up sizes of the same order. **The Zoo2 compartment is not the cause of the baseline failure**; the DetZ2 growth of the backward scan (48965082) belongs to the upwind-gradient variant. The baseline failure stays in the first detritus class near 100 m at node 1763 and the sinking limiter (term split 48938360).
- Spin-up to 5 years, 2026-11-11 (48966128, 3 h 25 on one A100, from the 2-year restart, yearly restarts in `pi_recom_spectral_spin5/archive`): surface means (share of nodes above 1e-3) Zoo2N 1.9e-8 (0 %) at 2 years, 1.2e-3 (7 %) at 3, 8.5e-2 (21 %) at 4, 8.3e-2 (32 %) at 5; MicZooN 0.175, 0.152, 0.058, 0.041 (grazed down by Zoo2); HetN 5.6e-3, 4.9e-3, 4.7e-3, 4.7e-3 (12-16 %); PhyN 0.110 to 0.127; DIN 5.56 to 5.62. Zoo2 establishes in year 3, as the CORE2 macrozooplankton at month 41; MicZoo is still falling at year 5. Scan from the 5-year state: 48968290 (`yr5/window_scan`).

## Decisions for the user

- **D1. Physics (decided 2026-11-05):** the FESOM 2.8 pair, validated against fesom-jax, GM/Redi on. It differs from the 2.6
  production physics in seven of the eight bisect factors (cvmix TKE, no momix, advection weights, mEVP, ice_diff, opt_visc,
  no partial cells; only GM/Redi agree), and physics carries 0.86 of the diatom gap and all of the sea-ice gap in stage 1.
  So the estimated parameters belong to the 2.8 physics, and Phase 5 validates in the 2.8 Fortran, not in 2.6.
- **D2. First aim (decided 2026-11-05):** production parameters first, then structural diagnosis.
- **D3. Hardware (decided 2026-11-05):** start on Albedo, then GH200 on Levante. Albedo (one A100): Phases 0-2 and the CORE2
  gradient check. GH200 (dolpung, project `mh1571`; 0.060 s per CORE2 + REcoM step on 4 GH200, `RUNNING_ALBEDO.md`, about 12
  minutes a model year, against about 7.4 h here): Phases 3-4. To prepare: the project membership, moving data and restarts, and
  repeating the gates there, because the noise of the forward differs by machine.
- **D4. Region (decided 2026-11-05):** global.
- **D5. New data (decided 2026-11-05):** acquire BGC-Argo and POC.
- **D6. Ensemble methods (decided 2026-11-05):** adjoint first, then a comparison with ensemble methods (EKI, PDAF, SIR).

## Risks

- **Fitting physics errors with biology.** The thin mode layer cannot be repaired by REcoM parameters. Hence `eta`, and care
  with preformed N as a target.
- **Reading `eta` wrongly.** It mixes wrong processes, operator error and real parameter variation: see the order in Phase 4.
- **Spin-up.** Windows of a year do not reach deep equilibrium; nutrients in the interior respond over decades. Windows
  constrain fast processes, so interior targets need x0 control or a low weight.
- **Non-smoothness.** Limiters, floors and `min()` limitation laws give kinks; L-BFGS may stall. Log-space misfits and warm
  starts help; Phase 1 tests it.
- **Collapsed PFTs.** Floors give zero gradient: keep optimisation inside regions where groups exist, and watch for
  replacement between PFTs.
- **The surface alkalinity drift** seen in the bisect (Alk down 17-21 in a year) is unexplained, and affects carbon targets.
