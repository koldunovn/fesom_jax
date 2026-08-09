# Adjoint horizon — lessons learned (NN-GM study, 2026-08-07)

**Headline.** The adjoint of the full CORE2 configuration (zstar + TKE + mEVP ice + GM) was
unusable beyond about a day, and the cause was not chaos. It was the prognostic turbulence
closure's response to the ocean state. Freezing that one feedback in the backward pass — the
forward is untouched — drops the growth of the reverse-pass field from 1.21–1.55x per 1800 s
step to **1.0070x**, and turns a gradient that flipped sign between identical runs into one
that repeats to six significant figures.

This is not a novel fix. It is what MITgcm/ECCO documented twenty years ago and what lets
them run a stable adjoint over a 26-year integration. We rediscovered it the slow way.

---

## 1. The measurements

### 1.1 Attribution: which term amplifies

Two-day window, perfect-model twin (truth at eddy-diffusivity multiplier 1.25, student at
1.0), one term's state-dependence cut from the backward at a time, forward bit-identical
throughout. The column that matters is the growth per timestep, because that is what
compounds; the gradient value is a single noisy sample.

| frozen in the backward | growth/step | over 2 days | dL/dm |
|---|---|---|---|
| nothing | 1.2118 | 1.0e8 | +5.841e+02 |
| **mixing coefficients Kv, Av** | **1.0070** | **1.96** | **−1.236e−02** |
| stratification N² at source | 1.0322 | 21 | −1.247e−02 |
| convective adjustment only | 1.2550 | 2.9e9 | −2.351e+02 |
| N² inside the GM block | 1.4298 | 8.0e14 | −3.612e+07 |
| prognostic TKE state | 1.2913 | 4.5e10 | +3.679e+03 |
| limited antidiffusive tracer flux | 1.3238 | — | +6.383e+05 |

Four independent measurements of the *unfrozen* growth rate — 1.2118, 1.3476, 1.4088, 1.5514
— put its own scatter at roughly 1.21 to 1.55. So `convection`, `gm_slope`, `tke_state` and
`advection` are **indistinguishable from doing nothing**; do not read them as harmful. Only
`mixing` (1.0070) and `stratification` (1.0322) sit outside that band, and they sit far
outside it.

Two independent cuts agreeing is the strongest evidence here: freezing the mixing
coefficients gives −1.236e−02 and freezing stratification at its source gives −1.247e−02,
**agreeing to 1%**, both with the correct sign, both the same order as the finite-difference
estimate. Nothing else in the table is close.

### 1.2 Reproducibility: the gradient was not a function of its inputs

Five passes in one process, same compiled executable, same inputs, nothing varied.

| window | frozen | dL/dm across passes | verdict |
|---|---|---|---|
| 1 day | nothing | +8.54e−02, +5.45e−01, −1.46e−01, +1.99e−01, −1.81e+00 | 21x spread, **sign flips** |
| 2 days | nothing | +9.59e+06, −2.16e+06, −1.50e+06, +2.88e+05, +9.65e+05 | 33x spread, **sign flips** |
| 2 days | mixing | −1.235851e−02, −1.235936e−02, −1.235954e−02 | **6 significant figures** |

The forward is nondeterministic on GPU (loss scatters 2.6e−04 to 7e−03 relative) because the
finite-element assembly scatters element contributions onto nodes with atomic adds, whose
completion order varies. That is normal. What made it fatal was the amplification: with the
closure differentiated, last-bit differences injected every step and amplified by every
remaining step dominate the answer. **Freezing the closure fixed reproducibility without
making the forward deterministic** — the forward still scatters by 2.6e−04 and the gradient
no longer cares.

### 1.3 The horizon question, restated

The per-step growth rate is what decides reach, and it extrapolates cleanly:

| window | steps | growth, unfrozen | growth, mixing frozen |
|---|---|---|---|
| 1 day | 48 | 1e8 – 1e9 | 1.4 |
| 5 days | 240 | — | 5.3 |
| 10 days | 480 | — | 29 |

A factor of 29 over ten days is a workable gradient. Measurement at 5 and 10 days was running
when this was written; check `scripts/nn_eddy/adjoint_horizon_results.jsonl`.

---

## 2. What the ocean-adjoint community does

This should not have been happening. **ECCO integrates a stable, useful adjoint over 26
years** (V4r4, 1992–2017) on a global ~1-degree MITgcm — the same class of configuration as
our CORE2 student, coarse and non-eddying. MITgcm's own AD verification suite runs 30-, 50-
and 360-day adjoint integrations agreeing with finite differences to better than 1%. A
two-day ceiling was four orders of magnitude off what the field considers normal, and that
gap was the clue that this was a term rather than the ocean.

Their remedies are all backward-only:

1. **Omit the state-dependence of the mixing coefficients** — ECCO uses the diffusivities and
   viscosities the closure computes but omits their parametric dependency on the ocean state.
   Exactly `stop_gradient(Kv), stop_gradient(Av)`. This is the one that worked.
2. **Damp the modes that destabilize the backward** — ECCO filters grid-scale vertical
   velocity specifically because large `w` harms adjoint stability.
3. **Never differentiate iterative solvers** — already honoured here; the SSH conjugate
   gradient uses a proper transpose solve with a `stop_gradient`-ed warm start.
4. **Run more diffusive numerics in the adjoint than in the forward** — standard in NWP
   incremental 4D-Var. Implemented here as `AdjointConfig.freeze_advection`; measured, not the
   amplifier for us.
5. **Configure the forward to be tame** — ECCO's decadal adjoint owes something to a coarse,
   viscous configuration. We inherit this: the student is CORE2, not eddy-resolving.

### 2b. FOR THE PAPER — three freezes, and every one of them is documented ECCO practice

By 2026-08-09 the study carries exactly **three** backward-only freezes. None is an invention
of ours; each has a direct antecedent in the ocean state-estimation literature, and the two
GM/Redi-relevant ones come from the SAME paragraph of the ECCO v4 description. This is the
framing for the methods section: we did not stabilise the adjoint with ad-hoc tricks, we
adopted the field's standing practice, and we then MEASURED what each cut costs the gradient.

| # | what is frozen | our switch | antecedent |
|---|---|---|---|
| 1 | sea-ice **dynamics** — the mEVP momentum solve; thermodynamics, advection, cut-off and the ocean→ice coupling stay differentiated | `IceConfig.adjoint_mode="rheology"` (`--ice rheology`) | Toyoda et al. (2019) approximation, used by Lyu et al. (2023) to build a usable viscous-plastic sea-ice adjoint |
| 2 | the **mixing coefficients'** dependence on the ocean state — the closure's `Kv`, `Av` are used, their state-derivative is not | `AdjointConfig.freeze_mixing` (`--freeze mix`) | Forget et al. (2015): *"It is only the parametric dependency of these diffusivities and viscosities on the ocean state that is omitted."* |
| 3 | the **neutral/isopycnal slope's** dependence on the density field — the `1/N²` factor inside the GM block | `AdjointConfig.freeze_gm_slope` (`--freeze slope`) | Forget et al. (2015), same paragraph: *"...by omitting only the parametric dependency of isopycnal slopes on the ocean density field in the adjoint, following a reasoning similar to that of Jiang et al. (2002)."* |

**The quotation that carries (3) is worth giving in full, because it also states the scale of
the problem and the property that makes the whole NN-GM idea possible:**

> "Until 2008, applications of the MITgcm adjoint were also omitting the Redi (1982) and Gent
> and Mcwilliams (1990) components, which precluded optimal control of their parameters. This
> situation was resolved by using a simple clipping scheme for large isopycnal slopes, and by
> omitting only the parametric dependency of isopycnal slopes on the ocean density field in the
> adjoint, following a reasoning similar to that of Jiang et al. (2002). Thus, the parametric
> dependency of turbulent transports on Kgm, Kσ and Kd is **retained** in the adjoint, so that
> these parameters can be optimally controlled."
> — Forget et al. (2015), GMD 8, 3071, §"Adjoint modeling"

Two things to draw out in the paper:

* **This blocked GM/Redi in the MITgcm adjoint for two decades**, so an eddy-parameterization
  adjoint failing exactly here is not a curiosity of our port — it is the known obstacle, and
  the cure is the known cure. Our contribution is not the cure but the *diagnosis at window
  scale*: a seven-way bisection localising it to one term on one model day.
* **The dependency on `Kgm` is RETAINED.** That is the sentence that makes online learning of a
  GM closure possible at all: the slope's dependence on density is cut, the transport's
  dependence on the diffusivity is not — and the NN's output IS that diffusivity. Freeze (3)
  therefore costs us nothing on the path we are training, which is why it is the right cut and
  not merely a working one.

**And it is cheap, measured, not assumed.** On the 60-day window ending 1961-02-14, freeze (3)
takes cumulative reverse-pass growth from ×4.363e+07 to ×1.591 and returns dL/dm = −9.41e−01
against the full model's finite difference of −9.80e−01 — a **3.98 %** residual, ordinary for
this protocol. Freeze (1) costs ~2 % of backward time and moves the 60-day gradient by 2.98 %.
Freeze (2) is the one without which nothing beyond two days runs at all.

The forward half of ECCO's recipe — slope clipping, after Cox (1987) — we already have as the
ODM95 taper, which is what bounds the slope's VALUE. The freeze bounds its DERIVATIVE, and the
distinction matters: our forward is healthy through the event (its finite difference is a
perfectly ordinary −0.98) and only the linearisation diverges. That is consistent with the
well-posedness analysis of the GM–Redi primitive equations, which needs a density
regularisation precisely because `N` — the denominator of the density slopes — is non-smooth
(arXiv:2304.03242).

⚠️ Provenance caveat to keep with these numbers: the older per-flag attribution table in
`AdjointConfig`'s docstring was measured on a 2-day window with no other knob on, where the
unfrozen scatter is 1.21–1.55/step. It recorded `freeze_gm_slope` as "no effect". It had no
resolving power there; three of its four "no effect" verdicts are overturned on a window where
the baseline does not scatter. Quote the 60-day numbers, not the 2-day ones.

### 2c. What else ECCO does that we have NOT taken — three open options

Read from the same §"Adjoint modeling" of Forget et al. (2015). Listed because each is a live
choice for us, not because any is obviously right.

**(a) ECCO has NO sea-ice adjoint at all — and we are currently doing MORE than they do.**

> "Despite successful adjoint simulations with particular versions of the sea-ice model
> (Heimbach et al., 2010; Fenty and Heimbach, 2013), the sea-ice adjoint is omitted in ECCO v4
> due to persisting issues. A pseudo-sea-ice adjoint is introduced instead to account at least
> for the most basic effect of sea ice — the shielding of seawater from the atmosphere. […] The
> forward pseudo-component merely tapers air–sea fluxes to zero according to (1−a), where a is
> the sea-ice fraction computed by the actual forward sea-ice model. This gross, local
> approximation omits the thermodynamics and dynamics of sea ice, and is never used in forward
> mode. In the adjoint, it masks out open ocean adjoint sensitivities that do not apply where
> ice cover is present. A fraction of open ocean sensitivity is preserved at the ice edge, which
> is physically reasonable and avoids a discontinuity in adjoint fields."

⚠️ **The ice DYNAMICS is already dark on our side too — "just cut the dynamics" is not an
available further step, it is what we already do.** `ice_step.py:102`:
`_evp_kw = {"adjoint_subcycles": 0} if cfg.adjoint_mode == "rheology" else {}` — the forward
runs all 120 mEVP subcycles, the backward unrolls none, so the ice velocity has no gradient path
at all. (The forward is full mEVP, `whichEVP=1`, not free drift.) The one published alternative
that keeps EVP rheology in the adjoint — Newtonian damping of the subcycle carry, the
Kauker-line `eps_N·sigma` term — was implemented, measured and rejected here: amplification
1.1795 per sweep means that over 120 sweeps the result swings from annihilated at `T_N=6`
(1e-44) to blown at `T_N=7` (1e+27), cancelling exactly at 6.57, so no single global constant
is right anywhere.

So the gap to ECCO is NARROWER than "we differentiate the rheology and they do not": dynamics is
dark in both. What we keep beyond them is ice **thermodynamics, advection, cut-off and the
ocean→ice coupling**; their pseudo-ice keeps only the flux shielding `(1−a)`.

**Consequence worth stating in the paper:** the two ice-only events on the February window
(×84.5 on 02-11, ×955 on 02-07) therefore come from the THERMODYNAMICS / advection / coupling
path, not from the rheology — the rheology has no gradient path in those runs. Our own
measurement says why ECCO stopped where it did: on
the 1961-02-14 window ice adds two events of its own (×84.5 on 02-11, ×955 on 02-07) and takes
the final gradient from +5.69e+06 to −3.26e+14. On CLEAN windows it is worth 0.21–2.98 % of the
gradient, FD-confirmed, for ~2 % of backward time — so it is not free to drop either.
**Decision to put to the user, not to settle here:** full ice adjoint (current), ECCO's
pseudo-ice mask, or no ice (the old default). The pseudo-ice option does not exist in this
codebase yet; it is a small forward-only component that is never run forward.

**(a2) Free drift in the backward — CONSIDERED AND DECLINED, with the reason.** MITgcm can run
free-drift ice dynamics (`SEAICE_ALLOW_FREEDRIFT` at compile time, `SEAICEuseFREEDRIFT` at run
time, with a run-time switch to enable it in adjoint mode only, added in checkpoint 63p, 2012;
`seaice_dynsolver.F` also uses the free-drift velocity as the LSR initial guess). It looks like
the obvious way to give our ice velocity a gradient path, since ours currently has none.

**The literature says it is the LESS accurate option, not a stabilisation.** Toyoda et al.
(2019), quoted by Lyu et al. (2023): including the adjoint of the rheology yields a sensitivity
to sea-ice velocity *"weaker by O(10²) in the central Arctic Ocean than the adjoint of
free-drift sea ice dynamics"*. Free drift therefore OVERSTATES the ice-velocity sensitivity by
about two orders of magnitude — there is no internal stress resisting the motion — and Lyu et
al. add that *"adjoint-FD shows much stronger sensitivities to wind than adjoint-VP does … in
the central Arctic Ocean"*. Even their VP adjoint needed extra taming on top: viscosities made
independent of the strain rate, plus zeroing any local ice-velocity sensitivity more than 50×
the global mean.

**Cost to build here.** The free-drift solve itself is easy — the ice momentum balance without
the internal stress term, quadratic ocean drag, a small non-linear 2×2 system per node; ~70
lines of JAX, differentiable, no 120-sweep scan. The hard part is having free drift in the
BACKWARD while mEVP runs in the FORWARD: that needs `jax.custom_vjp` around the dynamics, with
the primal from mEVP and the cotangent from the free-drift Jacobian, which is a day or two plus
its own correctness story because the value and the gradient then belong to different models.
Running free drift in both directions instead would change the forward and forfeit the
bit-identity gate against the Fortran model and the C dumps.

**Verdict: not worth it here.** Our loss is T, S and SSH with no velocities, so the GM
coefficient's influence on ice drift is remote; the entire ice contribution to the gradient is
measured at 0.21–2.98 % on clean windows, of which the dynamics share is currently zero and
would be a fraction; and the pathway it would add is one the literature measures as ~100× too
strong in the central Arctic in winter — precisely where the February event lives. If it is ever
wanted, the honest cheap version is free drift in BOTH forward and backward as a standalone
sensitivity experiment, never as the production configuration.

**(b) Raise the viscosity in the adjoint run.** *"it is common practice to increase viscosity
parameters to add stability to MITgcm adjoint simulations (Hoteit et al., 2005)"*. We have
`freeze_vert_visc` — which omits Av's state-dependence — but no knob that raises Av's VALUE for
a gradient run. Different lever, not implemented.

**(c) Choose forward numerics that are gentler on the adjoint.** ECCO v4 added AB-3 time
stepping and implicit vertical advection, which *"have a minor impact on the forward model
solution but provide additional stability also in adjoint mode."* We run AB-2.

⚠️ **(b) and (c) change the FORWARD**, and this project's entire validation rests on the forward
being bit-for-bit the trajectory gated against the Fortran model and the C dumps. Adopting
either means re-gating everything and giving up "the forward is untouched", which is what makes
the three freezes cheap. (a) does not: ECCO's pseudo-component is never used in forward mode.

Also noted, not actionable for us: ECCO's checkpointing is 1458 hand-placed storage directives,
where we have chunked reverse accumulation with a flat device peak; and they regard including
the non-linear free surface with a real freshwater flux in the adjoint as a major improvement in
physical realism, which we already have.

Two results outside the ocean literature are worth keeping:

- **Added viscosity in the adjoint is a closure, not a hack.** In a chaotic flow every
  individual adjoint realization grows at the Lyapunov rate while the *ensemble mean* decays;
  an eddy-viscosity closure in the ensemble-averaged adjoint equations is the principled way
  to model that decay (Wang & Zaki 2026). Implemented as `make_nodal_smoother`; not needed
  once the closure was frozen.
- **Gradients explode iff the maximal Lyapunov exponent is positive**, with truncation near
  the Lyapunov time as the remedy (Mikhaeil et al., NeurIPS 2022). The prediction this makes
  — that no term-freezing can move the horizon if the growth is genuinely chaos — is exactly
  what failed here, which is how we know it was not chaos.
- **NeuralGCM** never differentiates past 72 h (120 h for its ensemble model), uses no
  gradient clipping and no truncation, and instead grows the rollout on a curriculum and
  makes the loss progressively ignore high wavenumbers at longer lead times. Its target
  horizon and its adjoint horizon coincide, so it never faces our problem. The transferable
  idea is the loss-side one: score progressively coarser structure as the window grows.

---

## 3. Method lessons (the expensive ones)

**A single sample of a gradient is not a measurement.** The whole first half of this study
was built on one draw each. The `--repeat` flag exists because repeating a computation with
nothing changed cost seconds and changed every conclusion. Do it before reporting any
gradient number from this model.

**Check the finite-difference reference against its own noise floor.** With the loss
scattering 7e−3 relative, a central difference at step size 1e−3 has a signal-to-noise ratio
below one. Choosing the step size that best agrees with the AD value then *selects* whichever
noise draw agrees. The horizon scan's `best_fd` does exactly this; it is fine as a sign check
and misleading as a magnitude check.

**Sign agreement alone is too weak a gate.** The scan's `clean` verdict required a finite
gradient, magnitude below 1e4, and finite-difference sign agreement. At the window that
passed, its own recorded `plateau` was 4.11 — a 411% AD/FD disagreement. Report the plateau
next to the verdict; a horizon should be the longest window with a *converged* plateau.

**The growth rate is a far more robust statistic than the gradient.** The scalar gradient
scatters by 20–30x between identical runs; the per-step growth of the reverse-pass field
scatters by about ±0.15. Attribution experiments should be read off the growth rate. This is
why the chunked reverse pass reports the cotangent norm at every chunk boundary.

**Verify that a diagnostic knob actually fired.** A knob wired to a call site that is not on
the active code path silently reports "this term is not the amplifier" for a term never
tested. `scripts/nn_eddy/adjoint_horizon.py` counts wrapper invocations and aborts if a
requested knob never fired; `test_adjoint_config.py` asserts every flag that is LIVE ON THE PI
PATH moves the gradient. ⚠️ That is not every flag. `freeze_pgf` is read only under zstar
(`step.py:273`, inside `if ale_cfg is not None`) and `freeze_gm_slope`/`freeze_k33` only when a
`gm_cfg` is passed — and neither the pi fixture nor the CORE2 fixture supplies either, while
production runs `zstar+TKE+mEVP+GM`. So those three have no effect-test anywhere; they are
covered for "does not break anything", not for "does something". Closing that gap needs a CORE2
fixture built with `ale_cfg`+`gm_cfg` (open, 2026-08-07).

**Import path: `python scripts/foo.py` does not put the cwd on `sys.path`.** It puts the
*script's* directory there. From this worktree, `import fesom_jax` therefore resolved through
the editable install to the **main checkout** `~/port_jax`, so worktree-only library changes
were silently absent while runs still produced plausible numbers:

```
$ python scripts/nn_eddy/_probe.py          # run from the worktree root
sys.path[0] = /home/a/a270088/nn_eddy/port_jax/scripts/nn_eddy
fesom_jax   = /home/a/a270088/port_jax/fesom_jax/__init__.py     # <- MAIN
```

This invalidated one experiment (below). Scripts under `scripts/` should put the checkout
root on `sys.path` **and assert the import came from it**, as `adjoint_horizon.py` now does.
`pytest` is unaffected — it inserts the rootdir itself.

**The earlier "FCT limiter stop-gradient is harmful" result is void.** That A/B set
`FESOM_FCT_STOPGRAD_AE=1`, but the constant it controls exists only in the worktree copy of
`tracer_adv.py` and the imported library was the main checkout's. The variable was inert, so
the experiment compared the configuration against itself, and the −2.7 → −1.9e5 difference it
reported was the gradient's own run-to-run scatter. The hatch was never tested.

It is probably still the wrong idea, for a reason that A/B never reached. Writing the limited
flux as `ae·adf`, the tangent-linear increment is `d(ae)·adf + ae·d(adf)`, and the first term
is a **negative feedback**: as the tracer field steepens, `ae` falls. Freezing `ae` deletes
the stabilizing term and keeps the destabilizing one. `AdjointConfig.freeze_advection` tests the
other direction — drop the whole antidiffusive correction — and measures it as not the
amplifier.

**Memory was never the real constraint on window length.** The old scan ran out of GPU memory
at 7 and 10 days because it built the entire window's reverse pass as one computation.
Accumulating it chunk by chunk, with chunk-boundary states parked on the host, caps peak
device memory at one chunk regardless of window length: measured 43.1 GB at one day and
43.7 GB at two, against ~1.9 GB of host memory per saved day. The gradient is exact — this is
manual checkpointing at a coarser granularity, not an approximation.

---

## 4. What is implemented

### `fesom_jax.adjoint_stab.AdjointConfig` — per-term, in the model

One flag per term, passed to `step`/`integrate` as `adjoint_cfg`. All default off, and an
all-off config is a dead branch: the traced graph is identical to `adjoint_cfg=None`. Because
`stop_gradient` is the identity in the forward, **any flag leaves the trajectory bit-for-bit
the validated model** — asserted per flag in the tests.

```python
from fesom_jax.adjoint_stab import AdjointConfig

adj = AdjointConfig(freeze_mixing=True)        # the measured fix
final = integrate(state, mesh, op, stress, n_steps=n, params=p, dt=1800.0,
                  gm_cfg=gm_cfg, tke_cfg=tke_cfg, ale_cfg=ale_cfg,
                  ice_cfg=ice_cfg, adjoint_cfg=adj)
```

`freeze_mixing`, `freeze_stratification`, `freeze_convection`, `freeze_tke_state`,
`freeze_vert_visc`, `freeze_gm_slope`, `freeze_k33`, `freeze_advection`, `freeze_pgf` —
each named for what it FREEZES, not for what it enables. Two carry warnings: `k33` cuts part of the eddy coefficients' own gradient
path (it is how they reach the vertical diffusion), and `pgf` removes the tracer-to-momentum
coupling entirely and is diagnostic only.

**Why freezing the mixing is a cheap trade for this study specifically:** the eddy
parameterization's vertical term is added to the diffusivity *after* the closure runs, so
freezing the closure leaves the trained parameter's own gradient path fully intact. What is
cut is a separate feedback loop contributing amplification and little signal. Tuning the
turbulence closure itself would be the opposite case and this flag would be wrong.

### `fesom_jax.adjoint_stab` — backward-pass operators

`freeze` / `freeze_increment` (the ECCO move), `make_nodal_smoother` (adjoint diffusion via
the exact transpose of the model's mesh-Laplacian filter), `make_discounter`,
`make_clipper`, and the primitive `cotangent_op` they are built from.

Two properties of the smoother are asserted because getting them wrong is invisible: it
applies **`Fᵀ`, not `F`** (the FE Laplacian is self-adjoint in the *area-weighted* inner
product, so on this mesh they differ by a large structured factor), and `Fᵀ` **conserves the
total cotangent exactly** — it can redistribute sensitivity in space but never inflate it.

### `scripts/nn_eddy/adjoint_horizon.py` — the diagnostic

Chunked reverse accumulation (removes the memory ceiling; reports the cotangent norm at every
chunk boundary), `--repeat` for reproducibility, `--freeze` for the nine knobs applied by
monkeypatch (so a research knob cannot leak into a production run), and the stabilizers
`--gamma` / `--filter-factor` / `--clip`.

The chunk-boundary norm trace is the chaos-versus-switch discriminator: growth spread evenly
across chunks is a genuine Lyapunov exponent; growth concentrated in a few chunks is a switch
firing. Needs many chunks to mean anything — use `--chunk-steps 6` or smaller, not the
default 48, when reading it.

### Tests and example

`fesom_jax/tests/test_adjoint_stab.py` (operators), `test_adjoint_config.py` (flags on the pi
mesh), `test_adjoint_config_tke.py` (**the TKE closure and a CORE2 step — the pi mesh runs PP
mixing and cannot reach TKE at all**, so the flag that matters would otherwise be untested),
`test_ice_adjoint_mode.py` (the three ice modes, on both EVP solvers).

`examples/04_adjoint_stability.ipynb` walks through reading the growth rate and switching terms
off on a small mesh that runs in seconds. `examples/05_five_day_adjoint.ipynb` is the real
thing: a 5-day adjoint of the global 1-degree ocean with both switches on, using only the
public data release (`scripts/fetch_data.py`,
[doi:10.5281/zenodo.21324319](https://doi.org/10.5281/zenodo.21324319)), including the chunked
reverse accumulation that keeps it inside one GPU and a finite-difference sanity check.

Run the tests with `scripts/nn_eddy/adjoint_tests.sbatch` (CPU group) and
`scripts/nn_eddy/adjoint_tests_gpu.sbatch` (the CORE2 gates).

---

## 5. Open items

### Sea ice in the adjoint — SOLVED, and how to switch it on

**Use `IceConfig(adjoint_mode="rheology")`.** It differentiates ice thermodynamics, advection,
the concentration cut-off and the ocean-to-ice coupling, and unrolls **zero** momentum-solver
subcycles in the backward, so only the ice velocity is dark. Available on both the mEVP and the
standard-EVP path. Measured at a 5-day window: clean, reproducible, growth 1.0004 per timestep,
and it changes an eddy-diffusivity gradient by 0.21% against freezing the whole ice — so it
costs nothing in stability and recovers the pathway that matters (eddy transport changes SST,
which changes ice growth, which changes the surface heat and salt fluxes back into the ocean).

```python
adj = AdjointConfig(freeze_mixing=True)                        # ocean
ice = IceConfig(whichEVP=1, adjoint_mode="rheology")    # ice
integrate(state, mesh, op, stress, n_steps=n, params=p, dt=1800.0,
          gm_cfg=gm_cfg, tke_cfg=tke_cfg, ale_cfg=ale_cfg,
          ice_cfg=ice, adjoint_cfg=adj)
```

The three modes: `"rheology"` (recommended), `"frozen"` (freeze the whole ice update — the
conservative fallback, and what every published adjoint sea-ice study fell back to),
`"exact"` (faithful, and returns NaN on realistic configurations). An unknown mode raises.
Tests: `fesom_jax/tests/test_ice_adjoint_mode.py`. Worked example on the public data release:
`examples/05_five_day_adjoint.ipynb`.

#### How it was found, and what does not work

The component attribution over an 8-step window, freezing one piece of an ice step at a time:

| frozen | dL/dm |
|---|---|
| nothing | −1.96e+39 |
| ocean-to-ice coupling | +1.26e+33 |
| **momentum solve** | **+3.39e−05** |
| ice advection | −1.97e+37 |
| cut-off clipping | −3.69e+40 |
| thermodynamics | +5.08e+40 |

The momentum solve is solely responsible; the two dominant cotangent components are `u_ice` and
`v_ice`, three orders above everything else.

**It is the unrolling, not a term.** The reverse pass amplifies by a steady ~4e8 per model step
from the very first backward step — 1.18 per subcycle over 120 sweeps. It is not a threshold:
the forward deformation rate, ice concentration near the 1% branch, and nodes at full
concentration are all flat across the apparent "cliff". (The cliff in the gradient at 4 steps
was an artefact: `dL/dm` sums per-step contributions of alternating sign, and for short windows
they cancel while the cotangent underneath is already growing.)

**Two published fixes were tested and neither beats it.**

- *Constant viscosities in the adjoint* (Toyoda et al. 2019, used by Lyu et al. 2023) — removes
  one term's contribution to the per-sweep Jacobian but leaves the 120-sweep unrolling
  untouched. Measured: still NaN.
- *Newtonian damping of the adjoint* (the only published route that keeps EVP rather than
  removing it; damping timescale 7 sweeps generally, 3 in thick converging ice) — measured at a
  damping timescale of 4 sweeps it returns −3.753715e−03 against the rheology-frozen
  −3.751338e−03, i.e. **the same answer as freezing, while still paying to unroll all 120
  sweeps.** Its premise is also falsified here: uniform per-sweep damping is the right shape
  only if the amplification is uniform per sweep, and it is not — truncating to the last 32
  sweeps stays clean where uniform growth would predict ×197. Dead idea for this model.

**Truncated unrolling is the option if ice-velocity sensitivity is ever wanted.** Differentiate
only the last K subcycles (`adjoint_subcycles=K`); K=0 is `"rheology"`, K=120 is `"exact"`.
Measured over 8 steps, K = 0, 1, 2, 4, 8, 16 and 32 all stay at O(3e−05) and only 120 blows up
— so the amplification is concentrated in the early sweeps, not spread evenly. Untested beyond
8 steps; check it over a full day before relying on it.

#### Background: the published route forward

The ice is currently excluded entirely (`IceConfig.adjoint_mode="frozen"`): the forward runs
full mEVP, the backward skips the whole ice update, so no gradient flows through ice at all.
Changes in eddy diffusivity that alter SST, hence ice formation, hence surface fluxes, are
invisible to the gradient.

**Exact ice was tested and returns NaN**, immediately — at the first chunk of the backward,
with the forward healthy and matching the frozen-ice forward to 2.6e−04 (the usual GPU
scatter). This is arithmetic, not chaos, and the cause is identifiable by inspection. The
mEVP viscosity is `pressure = P/(Delta + delta_min)` with `delta_min = 1e-11`
(`ice_mevp.py:137-141`), so its derivative with respect to the deformation rate goes as
`1/(Delta+delta_min)^2`:

| ice state | d(viscosity)/d(strain) per subcycle | float64 overflows after |
|---|---|---|
| rigid, `Delta` at the floor | 1.0e+22 | **14 subcycles** |
| ordinary deforming ice, `Delta ~ 1e-8` | 1.0e+16 | **19 subcycles** |

There are **120 subcycles in one ice timestep** (`IceConfig.evp_rheol_steps`). The reverse
pass therefore reaches infinity inside the first model step, and `inf - inf` gives NaN. Note
the second row: this is generic deforming pack ice, not a landfast corner case. The obvious
suspects are already guarded — `ice_mevp.py` uses the codebase's AD-safe sqrt for `Delta` and
masks the divides by concentration and mass — so this term is what is left.

**The published fix removes exactly that term.** Toyoda et al. (2019), as used by Lyu et al.
(2023, *Ocean Sci.* 19, 305) to build the first usable adjoint of viscous-plastic sea-ice
rheology, "eliminate the dependence of bulk and shear viscosities on the strain rate
(eps_ij) in the adjoint" — i.e. the viscosities are held constant when differentiating. That
is the same move as freezing the ocean mixing coefficients, applied to the ice. Implemented
here as `mevp_iterate(..., freeze_viscosity=True)`, a backward-only `stop_gradient` on the
`Delta` in the viscosity denominator (`AdjointConfig` flag pending the measurement;
`--freeze icevisc` in `adjoint_horizon.py` today).

Their full recipe, worth having on record because we already have most of the machinery:

| Lyu et al. (2023) measure | our equivalent |
|---|---|
| viscosities held constant w.r.t. strain rate in the adjoint | `freeze_viscosity` (implemented) |
| zero any local adjoint ice-velocity sensitivity > 50x the global mean absolute value | a **local relative** clip; we have only a global-norm `make_clipper` — small addition |
| disable KPP mixing | consistent with our own KPP-vs-TKE measurement and with Losch's experience |
| horizontal diffusivity to 500 m^2/s, lateral viscosity to 10 000 m^2/s | a *forward* change, unlike everything else here |
| spatial filter on the ice-thermodynamics adjoint sensitivities | `make_nodal_smoother` (implemented) |

Payoff for scale: over a **one-year** assimilation window their approximated viscous-plastic
adjoint reduced the cost function by 40.2% (gradient norm 89.3%) against 32.3% (59.2%) for a
free-drift adjoint, at 1.2x the cost of free drift.

**Free drift** remains the simpler intermediate rung and is what the MITgcm community used for
years — it drops the internal stress entirely, which is precisely the term producing the NaN,
and keeps ice advection and thermodynamics. The Amundsen/Bellingshausen ECCO setup does not
even go that far: "the sea-ice adjoint is not used in this study due to persistent instability
issues". So freezing the ice, as we do, is the established default rather than a shortcut.
- **Seasonal dependence.** Every number here is from a 1960-02-20 seed — peak Northern
  Hemisphere deep convection, when the turbulence closure is most active and its switches
  fire most. This is plausibly the worst case of the year. A summer seed exists
  (`snap_step044640.pkl`, 1960-07-19); the contrast is one cheap job.
- **Spinup dependence.** The seed is 2.14 years into the run, not a cold start, but 36
  snapshots only reach 2.96 years. The upper ocean and mixed layer are developed; the deep
  ocean is nowhere near equilibrium. Whether the growth rate depends on spinup length is
  untested and needs either a longer JAX run or a converted Fortran restart.
- **Bit-reproducibility.** `--xla_gpu_deterministic_ops=true` compiled for over 90 minutes on
  a 2-day window without producing a result, so it is not practical as used.
  `scripts/nn_eddy/determinism_probe.py` tests the narrower
  `--xla_gpu_deterministic_reductions` and a CPU reference on a 6-step forward, which
  compiles fast. Worth having for regression testing even though the gradient no longer needs
  it.

---

## References

- Forget et al. (2015), *ECCO version 4*, GMD 8, 3071 — the adjoint's documented
  simplifications, including the omitted parametric dependency of the mixing coefficients.
- Gaillard, Hill et al. (2024), *MITgcm-AD v2 ... Tapenade*, arXiv:2401.11952 — gradient
  verification protocol; "do not differentiate iterative solvers"; 30–360 day integrations.
- Lea, Allen & Haine (2000/2002), *Sensitivity analysis of the climate of a chaotic ocean
  circulation model*, QJRMS — adjoint converges in the non-chaotic regime, diverges in the
  chaotic one.
- Wang & Zaki (2026), *Mitigating adjoint chaos in wall turbulence*, arXiv:2606.25399 — each
  adjoint realization grows at the Lyapunov rate while the ensemble mean decays.
- Mikhaeil, Monfared & Durstewitz (2022), *On the difficulty of learning chaotic dynamics with
  RNNs*, NeurIPS — gradients explode iff the maximal Lyapunov exponent is positive.
- Kochkov et al. (2024), *Neural general circulation models*, Nature 632, 1060 — 72/120 h
  rollout ceiling, lead-time-dependent spectral loss filtering, no gradient clipping.
- Lyu et al. (2023), *Effects of including the adjoint sea ice rheology on estimating Arctic
  Ocean-sea ice state*, Ocean Sci. 19, 305 — the approximated viscous-plastic sea-ice adjoint:
  viscosities constant w.r.t. strain rate, local sensitivity thresholding, KPP off, added
  horizontal diffusivity, spatial filtering of the thermodynamics adjoint.
- Toyoda et al. (2019) — the source of the constant-viscosity adjoint approximation.
- Nakayama et al. (2021), *Adjoint-based ocean state estimation for the Amundsen and
  Bellingshausen seas*, GMD 14, 4909 — "the sea-ice adjoint is not used in this study due to
  persistent instability issues".
