"""Backward-pass stabilizers — buying adjoint horizon in a stiff, switch-laden model.

The NN-GM gradient-horizon scan (Task 2) measured this model's adjoint clean at a 1-day
window and blown by 2 days, with |dL/dm| growing ~5 orders of magnitude per extra day.
That is ~1.3x amplification per 1800 s step, an e-folding time of ~80 minutes. **No ocean
process on a 1-degree mesh does that.** Resolved baroclinic instability e-folds in days;
the model's own convective adjustment removes the fast static instabilities. A growth rate
three orders of magnitude above anything physical is the signature of the tangent-linear
model picking up stiff, state-dependent parameterization feedbacks — not of chaos.

This is a re-run of a problem the MITgcm/ECCO community solved twenty years ago, and their
answer is the one implemented here. From the ECCO v4 description of the adjoint: the
diffusivities and viscosities computed by the turbulent mixing schemes are used, *but their
parametric dependency on the ocean state is omitted* in the adjoint. In other words the
transport is differentiated with the mixing coefficients held fixed. ECCO additionally
suppresses grid-scale noise in vertical velocity specifically because large w harms adjoint
stability. Both are backward-only interventions: the forward model is untouched.

Three families of intervention live here, in increasing order of how much they cost you:

1. **Freeze** (:func:`freeze`, :func:`freeze_increment`) — drop a term's state-dependence
   from the backward while the forward runs unchanged. `stop_gradient` is the identity in
   the forward, so the trajectory stays bit-for-bit the validated model. This is the ECCO
   move, and it is also already used in this codebase for the mEVP rheology
   (``IceConfig.adjoint_mode="frozen"``) and for KPP's discrete level indices. It yields a
   *different* gradient, not an approximate one: the gradient of the model with that
   feedback loop cut. Whether that gradient is still useful is an empirical question, which
   is what ``adjoint_horizon.py`` (nn_eddy research repository) measures.

2. **Diffuse** (:func:`make_nodal_smoother`) — apply a spatial low-pass to the cotangent
   each step. This is the classical "add viscosity to the adjoint" fix, and it has a
   cleaner justification than it looks: in a chaotic system every individual adjoint
   realization grows at the Lyapunov rate while the *ensemble mean* adjoint decays, and an
   eddy-viscosity closure in the ensemble-averaged adjoint equations is precisely how you
   model that decay (Wang & Zaki, "Mitigating adjoint chaos in wall turbulence", 2026).
   The operator used here is the exact transpose of the model's own mesh-Laplacian filter
   (:mod:`fesom_jax.coarsen`), so it is a genuine linear operator on the cotangent, and it
   **conserves the total sensitivity exactly** (see :func:`make_nodal_smoother`): it can
   only redistribute the gradient in space, never inflate or deflate its sum.

3. **Discount / clip** (:func:`make_discounter`, :func:`make_clipper`) — scale the cotangent
   down each step. Discounting by a constant ``gamma`` weights the contribution of a step
   ``k`` steps before the loss mark by ``gamma**k``: it deliberately forgets the distant
   past, which is exactly the part whose sensitivity chaos has destroyed. It stays linear,
   so the result is still an exact gradient of a reweighted objective. Norm clipping is
   nonlinear in the cotangent and therefore is **not** the gradient of anything; it
   preserves direction but biases toward whichever steps happen to saturate the cap. Use it
   as a diagnostic, not as a training gradient.

Nothing in this module is on by default anywhere. It is opt-in, assembled by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from . import coarsen
from .mesh import Mesh

__all__ = [
    "cotangent_op", "freeze", "freeze_increment", "cotangent_norm",
    "make_nodal_smoother", "make_discounter", "make_clipper", "make_state_stabilizer",
    "StabConfig", "AdjointConfig", "FREEZE_FLAGS", "KNOWN_CUTS", "dcut", "dcut_where",
    "differentiate_at", "cut_where", "scale_tangent", "TKE_CANDIDATE_FIELDS", "TkeAdjoint",
    "Amplification", "amplification", "amplification_by_field", "field_norms", "AMP_CLEAN_MAX", "AMP_SPREAD_MAX", "AMP_SUSTAINED_MAX", "AMP_SUSTAINED_W",
]


# ---------------------------------------------------------------------------
# which terms' state-dependence to omit from the backward

class AdjointConfig(NamedTuple):
    """Per-term backward-only simplifications. **The forward is never affected.**

    Every flag switches one term from "differentiate it as written" to "use the value it
    computes but omit its dependence on the ocean state". Because ``stop_gradient`` is the
    identity in the forward, turning any of these on leaves the trajectory bit-for-bit the
    validated model; only the gradient changes, and it becomes the exact gradient of a model
    with that feedback loop cut. All default to ``False``, and an all-``False`` config is a
    dead branch — the traced graph is identical to passing ``adjoint_cfg=None``.

    **Every flag is named for what it FREEZES, not for what it enables.** ``freeze_mixing=True``
    does not switch mixing on — the closure runs exactly as before and computes exactly the same
    diffusivities. It tells the reverse pass to treat those diffusivities as given constants
    instead of tracing how they depended on the ocean state.

    Pass it to :func:`fesom_jax.step.step` as ``adjoint_cfg``. Static and hashable, like
    ``GMConfig`` / ``TkeConfig`` / ``KppConfig`` / ``AleConfig``. Sea ice has its own
    equivalent switch, ``IceConfig.adjoint_mode``, which predates this object.

    ⚠️ **THE TABLE BELOW HAS NO RESOLVING POWER AND ITS "no effect" VERDICTS ARE ARTEFACTS.**
    Every row is a 2-day window with NO other knob on, where the unfrozen scatter is 1.21-1.55 —
    so any flag landing inside that band was recorded as "no effect" when the truth is that the
    experiment could not tell. Re-measured on a 60-day window ending 1961-02-14 with
    ``freeze_mixing`` already on (so the baseline does not scatter), **three of the four
    "no effect" rows have effects**, one of them decisive:

    =========================  =====================  ==========================================
    flag                       2-day table says       60-day, over freeze_mixing (cumulative
                                                      cotangent growth vs a x4.363e+07 baseline)
    =========================  =====================  ==========================================
    ``freeze_gm_slope``        1.4298, no effect      **x1.591 — the whole cure** (3.65e-08)
    ``freeze_stratification``  1.0322, nearly as good x1.590 — same, it subsumes the above
    ``freeze_convection``      1.2550, no effect      x9.254e+07 — **2.12x WORSE**
    ``freeze_advection``       1.3238, no effect      x3.463e+07 — 0.79, a real 21 %
    ``freeze_k33``             not measured           x8.092e+07 — 1.85x WORSE
    ``freeze_pgf``             not measured           x1.145e+09 — 26x WORSE
    ``freeze_tke_state``       1.2913, no effect      x1.275e+08 — 2.92x WORSE
    ``freeze_vert_visc``       not in the table       x3.541e+07 — 0.81, still BLOWN
    =========================  =====================  ==========================================

    The table is complete: **of the eight flags, only ``freeze_gm_slope`` and the wider
    ``freeze_stratification`` that contains it produce a usable long-window gradient.** Every
    other cut leaves the reverse pass blown by five to seven orders of magnitude, and four of
    them make it worse than not cutting at all. ``freeze_vert_visc`` shaves 19 % off the growth
    and is the only near-miss, which is worth stating precisely because 0.81 looks like progress
    and is not: it still ends at ``dL/dm = -2.007e+05`` against a true finite difference of
    ``-9.80e-01`` (jobs 26828150 / 26828151, 2026-08-09).

    ⚠️ **Both of those last two rows pass a median-only amplification screen.** They record a
    median of exactly 1.0000/step — the bar is 1.010 — and are caught only by the log-spread
    (0.024 against a 0.020 bar) and the worst-3-chunk statistic (1.0900 against 1.030). A
    single-statistic screen would have called a gradient that is wrong by five orders of
    magnitude clean. This is the concrete case the third bar was added for.

    **Read the 2-day table only as "this flag did not rescue an otherwise unusable 2-day
    gradient".** It is not evidence about any flag's effect in a configuration that already
    works. The four that make things WORSE all sit in stabilising loops, which a 2-day scatter
    band also could not have shown.

    Original 2-day measurements (docs/ADJOINT_HORIZON.md), growth of the reverse-pass field per
    1800 s step:

    =========================  ============  ====================================
    flag                       growth/step   verdict
    =========================  ============  ====================================
    (none)                     1.21 - 1.55   unusable; sign flips between repeats
    ``freeze_mixing``          1.0070        reproducible to 6 significant figures
    ``freeze_stratification``  1.0322        nearly as good; a blunter cut
    ``freeze_convection``      1.2550        inside the unfrozen scatter: no effect
    ``freeze_gm_slope``        1.4298        see the WARNING below — this row misleads
    ``freeze_tke_state``       1.2913        inside the unfrozen scatter: no effect
    ``freeze_advection``       1.3238        inside the unfrozen scatter: no effect
    =========================  ============  ====================================

    ``freeze_mixing`` is the one that matters at 2 days, and it is the documented ECCO practice
    (Forget et al. 2015): use the diffusivities and viscosities the turbulence closure computes,
    but omit their parametric dependency on the ocean state. The eddy-parameterization term that
    augments the vertical diffusivity is added *after* the closure runs, so freezing the
    closure does not touch the GM/Redi gradient path.

    ⚠️ **``freeze_gm_slope`` IS NOT "no effect" — that row was measured in a regime where it
    could not show.** The whole table is a 2-day window with NO other knob on, where the
    unfrozen scatter is 1.21-1.55 and everything short of ``freeze_mixing`` disappears into it.
    On a 60-day window ending 1961-02-14, WITH ``freeze_mixing`` already on, this flag is the
    only thing that works: cumulative cotangent growth over the first nine reverse chunks falls
    from **x4.363e+07 to x1.591**, and the resulting gradient is **-9.41e-01 against the full
    model's finite difference of -9.80e-01, a 3.98 % residual** — ordinary for this protocol.
    ``freeze_stratification`` gives x1.590, i.e. the wide N^2 cut's entire cure IS this narrow
    one. Freezing convection, the PGF or K33 instead makes the same window 2.1x, 26x and 1.85x
    WORSE — they sit in stabilising loops. Measured 2026-08-09, jobs 26822977/26827064; the
    bisection is in `docs/HANDOFF-20260809c-M3B-AND-THE-FEBRUARY-EVENT.md`.

    **This is also the ECCO recipe's other half, and it is 20 years old.** Forget et al. (2015),
    §"Adjoint modeling": *"Until 2008, applications of the MITgcm adjoint were also omitting the
    Redi (1982) and Gent and Mcwilliams (1990) components, which precluded optimal control of
    their parameters. This situation was resolved by using a simple clipping scheme for large
    isopycnal slopes, and by omitting only the parametric dependency of isopycnal slopes on the
    ocean density field in the adjoint, following a reasoning similar to that of Jiang et al.
    (2002). Thus, the parametric dependency of turbulent transports on Kgm, K_sigma and K_d is
    retained in the adjoint, so that these parameters can be optimally controlled."*
    That is this flag, sentence for sentence — including the part that matters for NN-GM: the
    dependency on ``k_gm`` STAYS live, so the parameter under optimisation is still
    differentiated. The forward half of their recipe (slope clipping, after Cox 1987) is already
    here as the ODM95 taper.

    The mechanism is a near-singular denominator, not an instability: the neutral slope is
    ``-grad_h(sigma) / d(sigma)/dz``, so its state-derivative carries ``1/N^4`` and diverges as a
    winter column destratifies — with ``N^2`` still POSITIVE. The forward is untouched by this
    (its finite difference is healthy); only the linearisation blows up. Consistent with the
    well-posedness analysis of the GM-Redi primitive equations, which needs a density
    regularisation precisely because ``N`` — the denominator of the density slopes — is
    non-smooth (arXiv:2304.03242).
    """

    freeze_mixing: bool = False
    """Kv and Av as the turbulence closure returns them (post-convection). The ECCO move,
    and the one measured to work."""

    freeze_nn_features: bool = False
    """The ``gm_nn`` network's INPUTS as given constants: use the multipliers the network
    computes, but omit their dependence on the ocean state.

    **This is ``freeze_mixing`` for the learned closure, and it exists because the learned
    closure is a second instance of the same problem.** ``freeze_mixing`` covers the
    *vertical* turbulence closure. ``gm_nn`` (added 2026-08-08, after the table above was
    measured) is a NEW state-dependent diffusivity: it reads stratification, horizontal
    buoyancy gradient and depth off the live state and returns a GM/Redi multiplier, so
    ``d(kappa)/d(state)`` re-enters the backward through a path no existing flag touches.
    The symptom is the one in docs/ADJOINT_HORIZON.md 1.2 — the gradient stops being a
    function of its inputs. Measured on the 5-day seam probe: two identical jobs returned
    dL/dalpha = +5.93e-02 and +2.30e-03, a factor of **26** apart and of opposite sign to
    finite differences, while the loss itself agreed to three digits (jobs 26800197/8).

    ⚠️ The horizon result does NOT cover this. There the diffusivity was a constant times a
    scalar, with zero state dependence, so there was nothing to inject noise; "clean to 30
    days" is a statement about that model, not about a network reading the state.

    **What it preserves is the derivative training needs.** The weights multiply the (now
    constant) features directly, so ``dL/d(weights)`` is untouched; what is cut is the
    indirect loop weight -> multiplier -> state -> feature -> multiplier. Per this module's
    own framing that is a DIFFERENT gradient, not an approximate one, and whether a network
    trained on it recovers the right closure is an empirical question — the twin-recovery
    experiment is what answers it."""

    freeze_stratification: bool = False
    """N^2 at its source, which cuts its feedback into mixing, convection and the eddy
    diffusivity in one move. Blunter than ``freeze_mixing`` and slightly less effective."""

    freeze_convection: bool = False
    """Only the convective-adjustment bump (the ``N^2 < 0`` switch), leaving the rest of the
    closure differentiated."""

    freeze_tke_state: bool = False
    """The prognostic turbulent kinetic energy carried between steps."""

    freeze_vert_visc: bool = False
    """Av where it enters the implicit vertical momentum viscosity. Subsumed by ``freeze_mixing``;
    useful on its own to separate the tracer and momentum halves of the closure's feedback."""

    freeze_gm_slope: bool = False
    """N^2 inside the eddy-parameterization block — the 1/N^2 neutral-slope amplifier. The
    eddy diffusivity itself and T, S stay differentiated."""

    freeze_k33: bool = False
    """The GM/Redi vertical isoneutral augmentation of the diffusivity. **Cuts part of the
    gradient path of the eddy coefficients themselves** — unlike the others, this one is not
    free with respect to the parameter being trained."""

    freeze_advection: bool = False
    """The limited antidiffusive tracer flux, so the backward sees plain monotone upwind
    advection while the forward keeps the full flux-corrected scheme. This is the NWP
    incremental-4D-Var practice of running more diffusive numerics in the adjoint than in
    the nonlinear forward. Measured: not the amplifier in the 60-day table above. On the
    sea-ice drift gradient it removes the FCT burst (PORTING_LESSONS I17e), but
    ``freeze_fct_limiter`` is the narrower cut that does."""

    freeze_fct_limiter: bool = False
    """The Zalesak limiter factors of the ocean tracer FCT (``tracer_adv.zalesak_limit``: the
    fraction ``ae`` of each antidiffusive flux the limiter lets through) as constants. The
    backward keeps the high-order antidiffusive flux and the limiting the forward applied,
    but not how the limiting would change with the state.

    The exact derivative of the factors is a selection (the per-node max/min bounds) divided by
    sums of the same small flux differences, so it is O(1) per step, not O(CFL). Where the
    forward ties two values to round-off, it hands the cotangent back and forth between them:
    a growing mode, not a sensitivity.

    Measured (PORTING_LESSONS I17e): on the 30-day March 2013 sea-ice drift gradient, a
    salinity-cotangent burst on near-tied deep cells (Bismarck Sea, 1.4-1.9 km, ×600 in
    12 h) disappears when only the factors are frozen. The sustained growth goes from 1.054
    to 1.002 per step, and the ice gradients move by ≤ 0.05 %.

    Narrower than ``freeze_advection``, which also drops the antidiffusive flux. The same
    operation as the older process-wide switch ``FESOM_FCT_STOPGRAD_AE=1``
    (``tracer_adv.FCT_STOP_GRADIENT_AE``). Unlike that switch, this flag is part of the static
    config, so :meth:`describe` records it and it keys the compiled chunk."""

    freeze_pgf: bool = False
    """The baroclinic pressure-gradient force. Cuts the tracer-to-momentum coupling
    entirely. **Diagnostic only** — the resulting gradient is not usable for training."""

    diag_cuts: frozenset = frozenset()
    """Named DIAGNOSTIC derivative cuts: a set of cut-point names from :data:`KNOWN_CUTS`.
    At each named point of :func:`fesom_jax.step.step` the value is passed through
    :func:`dcut`, i.e. ``stop_gradient`` on every floating leaf — the forward is unchanged,
    the backward stops there. Unlike the ``freeze_*`` flags, which are recommended
    simplifications with a measured effect, a cut is an instrument for locating where a
    reverse-pass amplification enters; the set of points grows with the diagnosis. The
    existing flags are not duplicated as cut names.

    **Why a static field and not a trace-time context.** Each cut set is part of the
    config, so it is part of whatever closure (``chunk_fn``) a caller builds around
    ``step``; the lru-cached executables of :func:`fesom_jax.adjoint.chunked_grad` are keyed
    by that closure and can never silently reuse a program traced WITHOUT the cut. A
    context manager active only around a primal trace would have that failure mode, and it
    would also miss custom-derivative rules, which JAX traces lazily (later than the primal).

    A NamedTuple has no constructor hook (and ``_replace`` bypasses constructors), so the
    set is checked by :meth:`validate`, which ``step`` calls: a plain ``set``/``list``/
    ``tuple`` is coerced to a ``frozenset`` (hashable), and an unknown name raises
    ``ValueError``."""

    # --- classical-TKE candidate settings (derivative-only; ``None`` = off) --------------
    # The TKE adjoint diagnosis (docs/ADJOINT_TKE.md) found the loop that destroys the exact
    # gradient beyond about a day: N^2 -> Richardson number -> Prandtl number -> KappaH (Kv)
    # -> T, S -> N^2, and, separately, a transient TKE-only recursion at near-floor first
    # interior interfaces below fresh surface lenses. The fields below are the candidate
    # cures. Each keeps the forward bitwise and each, when ``None``, traces nothing.

    freeze_mixing_state: bool = False
    """The parametric-only TKE adjoint: every derivative of the classical-TKE closure with
    respect to the OCEAN STATE is omitted, every derivative with respect to its four constants
    (``tke_c_k``, ``tke_c_eps``, ``tke_cd``, ``tke_alpha``) is kept, and so is the TKE's own
    recursion ``tke_old -> tke_new`` that carries ``c_eps``, ``cd`` and ``alpha`` into the
    next step's diffusivities (they reach KappaM only through the next step's tke, so a cut
    of ``tke_old`` would zero those three gradients).

    Cut in :func:`fesom_jax.tke.mixing_tke`: the column inputs ``Ssqr`` (shear from the
    velocity and the live layer depths), ``Nsqr`` (also inside the mixing length), the layer
    and interface thicknesses ``dzw``/``dzt`` (zstar geometry) and the surface flux
    ``forc_tke_surf`` (the ice-blended stress, which depends on the surface current). The
    surface flux value is kept, so ``d/d cd`` survives; the wind stress no longer reaches the
    TKE in the backward, so a surface-forcing control loses its mixing pathway under this
    flag. The convective adjustment is left exact (its ``N^2 < 0`` switch has no derivative;
    add ``freeze_convection`` explicitly for the straight-through bump derivative).

    The gradient is BIASED (the ocean-state feedback on mixing is neglected), not approximate
    in the sense of noise: it is the exact gradient of a model whose closure reads a
    prescribed ocean state. Only the TKE closure reads it (no effect under PP or KPP)."""

    tke_pr_cut: float | None = None
    """N^2* of the Prandtl-number cut (``None`` = off). The derivative of the Prandtl number
    ``clip(6.6 Ri, 1, 10)`` with respect to Ri (hence N^2 and S^2) is omitted where the
    column-input N^2 < N^2* (``math.inf`` = everywhere, the same derivative as the diagnostic
    cut ``tke.pr``); every other closure derivative, including every path of the TKE
    constants (Pr does not depend on them), is kept. With ``tke_pr_mode="at"`` the Pr
    derivative is instead taken at ``max(N^2, N^2*)`` (see there)."""

    tke_pr_mode: str | None = None
    """How :attr:`tke_pr_cut` acts: ``None`` or ``"cut"`` — no Pr derivative where N^2 < N^2*;
    ``"at"`` — the Pr derivative evaluated at the regularised point ``(max(N^2, N^2*), S^2)``
    (:func:`differentiate_at`, straight-through form): exact where N^2 >= N^2*, zero where the
    regularised Ri leaves the clip band, finite everywhere. ``"at"`` needs a finite N^2*."""

    tke_pr_ratio: float | None = None
    """R* of the smoothing-cancellation Prandtl cut (``None`` = off). The N^2 the closure reads
    is the horizontally smoothed one (``eos.smooth_nod3D``, one sweep). Where a locally
    inverted column sits among stable neighbours, the smoothed value is the small residual of
    a cancellation, and ``dKappaH/dN^2 = -KappaH/N^2`` taken at that residual turns the
    N^2 -> Pr -> KappaH -> T, S -> N^2 loop unstable even where the residual is well above
    any N^2 threshold (measured: one ice-covered East Siberian shelf column, raw N^2 -8.0e-5,
    smoothed +3.3e-7, docs/PORTING_LESSONS.md 2026-10-06). The derivative of the Prandtl number
    is omitted where ``|raw N^2| > R* * |smoothed N^2|`` (forward values; the raw N^2 is the
    unsmoothed ``eos.pressure_bv`` field the step computes when this is set). Combined with
    :attr:`tke_pr_cut` the two masks are joined (a cut where either holds). Needs the
    ``"cut"`` mode; a finite R* > 0."""

    tke_pr_raw_cut: float | None = None
    """N^2_raw* of the unsmoothed-N^2 Prandtl cut (``None`` = off). The derivative of the
    Prandtl number is omitted where the UNSMOOTHED N^2 (the ``eos.pressure_bv`` field the step
    computes when this is set; forward values) is below N^2_raw*: with 0 or a small positive
    value, every locally inverted or neutral cell, whatever the smoothed N^2 the closure reads
    there. A TKE closure that reads the local N^2 (MITgcm's GGL90) has Ri < 0 in such a cell,
    Pr clipped at 1 and no Prandtl derivative; the horizontal smoother hands FESOM's closure a
    positive residual instead, with Pr unclipped, also where |raw| / |smoothed| is only 1-30
    (Norwegian-shelf columns in the 10-day 2013-01 sweep, where :attr:`tke_pr_ratio` = 100
    misses them). Joined with :attr:`tke_pr_cut` / :attr:`tke_pr_ratio` (a cut where any
    holds). Needs the ``"cut"`` mode; finite."""

    tke_kme_tke: float | None = None
    """tke* of the near-floor km_e cut (``None`` = off). In columns whose first interior
    interface has ``tke_old < tke*`` (forward values, any mixing-length regime) the derivative
    of KappaM = c_k·mxl·sqrt(tke) with respect to the ``sqrt(tke)`` factor is omitted (the
    ``tke.km_e`` term, masked); the mixing length's own tke dependence and every constant
    path are kept. Where and on which uses: :attr:`tke_kme_scope`, :attr:`tke_kme_term`.
    ``tke* -> 0`` is exact; ``math.inf`` with scope ``"column"`` and term ``"km_e"`` equals
    the diagnostic cut ``tke.km_e``."""

    tke_kme_scope: str | None = None
    """Where the near-floor km_e cut acts in a flagged column: ``None`` or ``"column"`` — every
    interface of the column (no mask edge inside the implicit TKE diffusion, which couples the
    surface and first interfaces); ``"iface1"`` — the first interior interface only."""

    tke_kme_term: str | None = None
    """Which uses of KappaM the near-floor km_e cut acts on: ``None`` or ``"km_e"`` — every use
    (KappaH, production, buoyancy, the TKE diffusivity, Kv/Av); ``"ke_diff"`` — only the TKE
    self-diffusivity ``ke = alpha·(KappaM[k+1]+KappaM[k])/2`` (the narrower variant)."""

    tke_ml_cut: float | None = None
    """N^2* of the mixing-length cut (``None`` = off). The derivative of the free Blanke-
    Delecluse length ``mxl0 = sqrt(2)·sqrt(tke)/sqrt(max(N^2, 1e-12))`` with respect to N^2 is
    omitted where the column-input (smoothed) N^2 < N^2* (forward values); its sqrt(tke)
    dependence, the walls / min-scans and every constant path are kept. ``dmxl0/dN^2 =
    -mxl0/(2 N^2)`` grows without bound as N^2 -> 0, and in near-neutral deep cells (smoothed
    N^2 ~ 1e-10, mxl0 of hundreds of metres) the N^2 -> mxl -> KappaH -> T, S -> N^2 loop is
    unstable: the 60-day July 2013 sweep bursts x1e3 in deep Tyrrhenian columns, which the
    Prandtl cuts do not touch and the diagnostic cut ``tke.ml_n2`` removes (docs/
    PORTING_LESSONS.md, step C5). ``math.inf`` equals ``tke.ml_n2``."""

    tke_ml_ratio: float | None = None
    """R* of the smoothing-cancellation mask on the mixing-length cut (``None`` = off): the
    :attr:`tke_pr_ratio` mask, ``|raw N^2| > R* * |smoothed N^2|`` (forward values), applied
    to the N^2 derivative of the free length as :attr:`tke_ml_cut` does below N^2*. In a
    locally inverted cell the smoothed N^2 can be a small positive residual (raw -7e-5,
    smoothed 1.5e-7 on the Northeast Greenland shelf in the January 2013 window), above any useful
    N^2* but with ``dmxl0/dN^2 = -mxl0/(2 N^2)`` as large as in a neutral cell. Joined with
    :attr:`tke_ml_cut` / :attr:`tke_ml_raw_cut` (a cut where any holds). A finite R* > 0."""

    tke_ml_raw_cut: float | None = None
    """N^2_raw* of the unsmoothed-N^2 mask on the mixing-length cut (``None`` = off): the
    :attr:`tke_pr_raw_cut` mask, raw N^2 < N^2_raw* (forward values), with the same effect as
    :attr:`tke_ml_ratio`. Finite."""

    tke_pr_ice_cut: float | None = None
    """a* of the sea-ice mask on the Prandtl-number cut (``None`` = off): the whole Pr
    derivative (what ``tke.pr`` cuts everywhere) is omitted in the columns whose sea-ice
    concentration at the start of the step exceeds a* (forward value). Joined with the N^2
    masks above (a cut where any holds). Under ice the ML-base Richardson number can sit in
    the live band 1 < 6.6 Ri < 10 with small shear, where dPr/dN^2 = 6.6/S^2 drives a slow
    two-level tke/w_e mode (Canadian Archipelago, December 2013, in the July-2013 year
    window); a global N^2* that removes it (1e-5) costs 11-47 % of J_MLD on the ice-free
    2013 30-day windows. a* in [0, 1)."""

    tke_pr_lat_cut: float | None = None
    """phi* (degrees) of the latitude mask on the Prandtl-number cut (``None`` = off): the
    whole Pr derivative is omitted in the columns poleward of phi* (geographic
    ``|lat| > phi*``), forward bitwise. Joined with the masks above. The same two-level
    tke/w_e mode at the mixed-layer base also grows in ice-free polar columns (West
    Spitsbergen, 79N, December 2013: one 12-h step x26), which the sea-ice mask misses, while
    the default cost box (50-60N) lies well equatorward of it. A geographic stand-in for a
    state criterion: it drops a real derivative path wherever it applies, so it is valid only
    for costs far from phi* (the 50-60N box: year windows from January and July 2013 within
    0.5-7 % of finite differences), never for an Arctic or Southern Ocean cost or one that
    depends on polar mixing. phi* in (0, 90)."""

    tke_pr_tu_cut: float | None = None
    """Tu* (degrees) of the diffusive-regime mask on the Prandtl-number cut (``None`` = off):
    the Pr derivative is omitted at the interfaces whose Turner angle lies in
    ``-90 < Tu < Tu*`` (:func:`fesom_jax.eos.turner_angle`, forward values), i.e. stably
    stratified with the temperature increasing downward and salinity holding the
    stratification: cold fresh water over warm salty water. Both polar mixed-layer-base
    events (Canadian Archipelago under ice and ice-free West Spitsbergen, December 2013) sit
    at Tu ~ -60, where the shear is weak and the Richardson number in the live Pr band; the
    thermally stratified North Atlantic summer interfaces (Tu > -45) keep the derivative.
    Joined with the masks above. Tu* = -45 is the regime boundary; Tu* in (-90, 0)."""

    gm_n2_ratio: float | None = None
    """R* of the smoothing-cancellation cut on the N^2 the eddy-parameterization block reads
    (``None`` = off): the :attr:`tke_pr_ratio` mask, ``|raw N^2| > R* * |smoothed N^2|``
    (forward values), applied to the GM/Redi input instead of the Prandtl number. Where it
    holds, the derivative of every N^2 use in the block (neutral slope and taper, wave speed,
    the streamfunction solve) is omitted, as :attr:`freeze_gm_slope` does everywhere; the
    rest of the ocean keeps the live slope. Joined with :attr:`gm_n2_raw_cut` (a cut where
    either holds). Not combinable with ``freeze_gm_slope``; a finite R* > 0."""

    gm_n2_raw_cut: float | None = None
    """N^2_raw* of the unsmoothed-N^2 cut on the GM/Redi input (``None`` = off): the
    :attr:`tke_pr_raw_cut` mask, raw N^2 < N^2_raw* (forward values), with the same effect as
    :attr:`gm_n2_ratio`. Not combinable with ``freeze_gm_slope``; finite."""

    gm_slope_cut: float | None = None
    """S* of the steep-slope cut (``None`` = off): the N^2 derivative of the neutral slope, and
    so of the ODM95 taper and the Redi diffusivity built from it, is omitted where the forward
    slope ``|s| > S*`` (``gm.compute_neutral_slope``). The adjoint-only counterpart of the
    slope clipping MITgcm's ECCO configurations use for a stable GM adjoint, where the clipped
    slope has no magnitude derivative beyond the clip. Not combinable with
    ``freeze_gm_slope``; a finite S* > 0."""

    conv_surrogate: float | None = None
    """N^2 half-width (s^-2) of the convective-adjustment surrogate (``None`` = off). Where the
    smoothed N^2 < 0 the convective adjustment raises the node Kv to ``INSTABMIX_KV``
    (:func:`fesom_jax.pp.mo_convect`); the switch itself has no derivative, so the adjoint
    cannot see that a perturbation moves where and when a column convects. With this set the
    forward is unchanged and the backward differentiates the ramp-smoothed switch
    ``Kv_in + w·(max(Kv_in, INSTABMIX_KV) − Kv_in)``, ``w = clip(−N^2/d, 0, 1)``: an N^2
    derivative ``−(max(Kv_in, INSTABMIX_KV) − Kv_in)/d`` in the band ``−d < N^2 < 0``, the
    exact derivative outside it. The ramp is one-sided: on the stable side the linearised flux
    ``Kv·N^2`` would fall as N^2 rises (anti-diffusive), and a centred ramp
    (``|N^2| < d``) blew the 30-day January reverse sweep up ×1e30 in its first chunk.
    ``Av`` keeps the exact switch. Not combinable with ``freeze_convection``; a finite d > 0."""

    @property
    def active(self) -> bool:
        return (any(bool(getattr(self, f)) for f in FREEZE_FLAGS) or bool(self.diag_cuts)
                or any(getattr(self, f) is not None for f in TKE_CANDIDATE_FIELDS))

    @property
    def gm_n2_cut(self) -> bool:
        """True when the GM/Redi input N^2 has a masked cut (:attr:`gm_n2_ratio` /
        :attr:`gm_n2_raw_cut`)."""
        return self.gm_n2_ratio is not None or self.gm_n2_raw_cut is not None

    def tke_options(self) -> "TkeAdjoint | None":
        """The candidate settings the TKE column core reads (``None`` when all are off, so the
        core traces nothing extra). Call on a :meth:`validate`-d config."""
        if (self.tke_pr_cut is None and self.tke_pr_ratio is None
                and self.tke_pr_raw_cut is None and self.tke_kme_tke is None
                and self.tke_ml_cut is None and self.tke_ml_ratio is None
                and self.tke_ml_raw_cut is None and self.tke_pr_ice_cut is None
                and self.tke_pr_lat_cut is None and self.tke_pr_tu_cut is None):
            return None
        return TkeAdjoint(
            pr_n2_star=None if self.tke_pr_cut is None else float(self.tke_pr_cut),
            pr_mode=self.tke_pr_mode or "cut",
            kme_tke=None if self.tke_kme_tke is None else float(self.tke_kme_tke),
            kme_scope=self.tke_kme_scope or "column",
            kme_term=self.tke_kme_term or "km_e",
            pr_ratio=None if self.tke_pr_ratio is None else float(self.tke_pr_ratio),
            pr_raw_cut=None if self.tke_pr_raw_cut is None else float(self.tke_pr_raw_cut),
            ml_n2_star=None if self.tke_ml_cut is None else float(self.tke_ml_cut),
            ml_ratio=None if self.tke_ml_ratio is None else float(self.tke_ml_ratio),
            ml_raw_cut=None if self.tke_ml_raw_cut is None else float(self.tke_ml_raw_cut),
            pr_ice=None if self.tke_pr_ice_cut is None else float(self.tke_pr_ice_cut),
            pr_lat=None if self.tke_pr_lat_cut is None else float(self.tke_pr_lat_cut),
            pr_tu=None if self.tke_pr_tu_cut is None else float(self.tke_pr_tu_cut))

    def describe(self) -> str:
        on = [f.removeprefix("freeze_") for f in FREEZE_FLAGS if getattr(self, f)]
        if self.diag_cuts:
            on.append("cuts=" + ",".join(sorted(self.diag_cuts)))
        on += [f"{f}={getattr(self, f)!r}" for f in TKE_CANDIDATE_FIELDS
               if getattr(self, f) is not None]
        return "+".join(on) if on else "none"

    def validate(self) -> "AdjointConfig":
        """Return this config with ``diag_cuts`` as a ``frozenset`` of known names.

        Raises ``TypeError`` for a bare string or a non-collection (a string would be read
        as a set of characters) and ``ValueError`` for a name not in :data:`KNOWN_CUTS`.
        Returns ``self`` unchanged when there is nothing to coerce, so the default config
        stays the same object."""
        cuts = self.diag_cuts
        if isinstance(cuts, (str, bytes)) or not isinstance(cuts, (set, frozenset, list, tuple)):
            raise TypeError(f"AdjointConfig.diag_cuts must be a set of cut names, got "
                            f"{type(cuts).__name__} {cuts!r}")
        names = frozenset(cuts)
        bad = sorted(repr(n) for n in names if not isinstance(n, str))
        if bad:
            raise TypeError(f"AdjointConfig.diag_cuts names must be strings: {', '.join(bad)}")
        unknown = sorted(names - KNOWN_CUTS.keys())
        if unknown:
            raise ValueError(f"unknown AdjointConfig.diag_cuts {unknown}; known cut points: "
                             f"{sorted(KNOWN_CUTS)}")
        self._validate_tke_candidates()
        if type(cuts) is frozenset:
            return self
        return self._replace(diag_cuts=names)

    def _validate_tke_candidates(self) -> None:
        def number(name, *, finite=False):
            v = getattr(self, name)
            if v is None:
                return None
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"AdjointConfig.{name} must be a float or None, got "
                                f"{type(v).__name__} {v!r}")
            if math.isnan(v) or (finite and not math.isfinite(v)):
                raise ValueError(f"AdjointConfig.{name} must be {'finite' if finite else 'a number'}"
                                 f", got {v!r}")
            return float(v)

        def choice(name, allowed, needs):
            v = getattr(self, name)
            if v is None:
                return
            if v not in allowed:
                raise ValueError(f"AdjointConfig.{name} must be one of {allowed} or None, got {v!r}")
            if getattr(self, needs) is None:
                raise ValueError(f"AdjointConfig.{name}={v!r} has no effect without {needs}")

        number("tke_pr_cut")
        choice("tke_pr_mode", TKE_PR_MODES, "tke_pr_cut")
        if self.tke_pr_mode == "at":
            number("tke_pr_cut", finite=True)
        ratio = number("tke_pr_ratio", finite=True)
        if ratio is not None and ratio <= 0.0:
            raise ValueError(f"AdjointConfig.tke_pr_ratio must be > 0, got {ratio!r}")
        if ratio is not None and self.tke_pr_mode == "at":
            raise ValueError("AdjointConfig.tke_pr_ratio is a cut; it cannot be combined with "
                             "tke_pr_mode='at'")
        raw = number("tke_pr_raw_cut", finite=True)
        if raw is not None and self.tke_pr_mode == "at":
            raise ValueError("AdjointConfig.tke_pr_raw_cut is a cut; it cannot be combined with "
                             "tke_pr_mode='at'")
        kme = number("tke_kme_tke")
        if kme is not None and kme < 0.0:
            raise ValueError(f"AdjointConfig.tke_kme_tke must be >= 0, got {kme!r}")
        choice("tke_kme_scope", TKE_KME_SCOPES, "tke_kme_tke")
        choice("tke_kme_term", TKE_KME_TERMS, "tke_kme_tke")
        number("tke_ml_cut")
        ml_ratio = number("tke_ml_ratio", finite=True)
        if ml_ratio is not None and ml_ratio <= 0.0:
            raise ValueError(f"AdjointConfig.tke_ml_ratio must be > 0, got {ml_ratio!r}")
        number("tke_ml_raw_cut", finite=True)
        ice = number("tke_pr_ice_cut", finite=True)
        if ice is not None and not 0.0 <= ice < 1.0:
            raise ValueError(f"AdjointConfig.tke_pr_ice_cut must be in [0, 1), got {ice!r}")
        lat = number("tke_pr_lat_cut", finite=True)
        if lat is not None and not 0.0 < lat < 90.0:
            raise ValueError(f"AdjointConfig.tke_pr_lat_cut must be in (0, 90), got {lat!r}")
        tu = number("tke_pr_tu_cut", finite=True)
        if tu is not None and not -90.0 < tu < 0.0:
            raise ValueError(f"AdjointConfig.tke_pr_tu_cut must be in (-90, 0), got {tu!r}")
        gm_ratio = number("gm_n2_ratio", finite=True)
        if gm_ratio is not None and gm_ratio <= 0.0:
            raise ValueError(f"AdjointConfig.gm_n2_ratio must be > 0, got {gm_ratio!r}")
        number("gm_n2_raw_cut", finite=True)
        sc = number("gm_slope_cut", finite=True)
        if sc is not None and sc <= 0.0:
            raise ValueError(f"AdjointConfig.gm_slope_cut must be > 0, got {sc!r}")
        if sc is not None and self.freeze_gm_slope:
            raise ValueError("AdjointConfig.gm_slope_cut cuts part of what freeze_gm_slope "
                             "freezes everywhere; set one or the other")
        if self.gm_n2_cut and self.freeze_gm_slope:
            raise ValueError("AdjointConfig.gm_n2_ratio / gm_n2_raw_cut cut part of what "
                             "freeze_gm_slope freezes everywhere; set one or the other")
        cs = number("conv_surrogate", finite=True)
        if cs is not None and cs <= 0.0:
            raise ValueError(f"AdjointConfig.conv_surrogate must be > 0, got {cs!r}")
        if cs is not None and self.freeze_convection:
            raise ValueError("AdjointConfig.conv_surrogate replaces the convective-adjustment "
                             "derivative that freeze_convection removes; set one or the other")

    @classmethod
    def exact(cls) -> "AdjointConfig":
        """Preset: every term differentiated as written (all flags off, no cuts)."""
        return cls()

    @classmethod
    def long_window(cls) -> "AdjointConfig":
        """Preset: ``freeze_mixing`` + ``freeze_gm_slope``, the combination measured to give a
        usable 60-day gradient (see the table above and ``examples/06_long_window_adjoint``).
        It is today's recommendation for windows beyond about a day. ``freeze_mixing`` zeroes
        the gradients with respect to the TKE constants; :meth:`long_window_tke` keeps them."""
        return cls(freeze_mixing=True, freeze_gm_slope=True)

    @classmethod
    def long_window_tke(cls) -> "AdjointConfig":
        """Preset (opt-in): the classical-TKE closure differentiated in full, ocean-state
        feedback included, except two derivatives with respect to N^2 at near-neutral and
        inverted cells. The Prandtl-number derivative is omitted at N^2 < 1e-8
        (:attr:`tke_pr_cut`), where the smoothed N^2 is a cancellation residual of the raw one
        (|raw| > 100 |smoothed|, :attr:`tke_pr_ratio`) and where the raw N^2 < -1e-6
        (:attr:`tke_pr_raw_cut`); the derivative of the free mixing length is omitted at
        N^2 < 1e-8 (:attr:`tke_ml_cut`). The gradients with respect to the TKE constants are
        kept, which :meth:`long_window` cannot do.

        Measured 2026-10-09 on CORE2 (prescribed ice, GM on and not frozen, 24-step chunks; box
        ``natl_subpolar``; central FDs in ln c_k, ln c_eps, ln cd, ln alpha at h = 3e-3, 1e-2,
        3e-2) over 30, 60 and 90 days from 2013-01-01 and 2013-07-01: window-mean MLD within
        7.7 % for all four constants at every window in both seasons; July window-mean box T
        within 2.2 %. A July cotangent transient (x11 in half a day, deep Tyrrhenian Sea,
        2013-08-08) decays without spoiling the gradient; ``tke_ml_cut=1e-6`` removes it too but
        adds bias to the January box T. The January window-mean box T misses the 10 % bar at
        every window (c_k 63 % and cd 88 % at 30 days, alpha 517 % at 60 days, 11-22 % at 90
        days), with or without the mixing-length cut and with ``freeze_gm_slope``: a separate,
        open error. Without :attr:`tke_ml_cut` the July sweeps burst from 60 days on (errors
        x10 to x1e6). With ``IceConfig.adjoint_mode="rheology"`` the January box-T vector error
        falls from 19.6 to 14.7 % at 60 days and from 13.4 to 12.8 % at 90 days.

        Reverse-sweep screen 2026-10-10 (window-mean MLD, prescribed ice, no FDs): 90 days from
        2013-04-01 and 2013-10-01 grow smoothly (largest one-chunk growth x2). 180 days from
        2013-01-01 and 2013-07-01 each show one isolated spike that does not run away: the
        salinity cotangent x43 over one day on 2013-03-16 (afterwards x2.2 above its level
        before), and the TKE cotangent x187 in half a day on 2013-07-08 (gone a day later). The
        180-day gradients are not checked against FDs.
        docs/PORTING_LESSONS.md, steps C3-C7."""
        return cls(tke_pr_cut=1e-8, tke_pr_ratio=100.0, tke_pr_raw_cut=-1e-6, tke_ml_cut=1e-8)

    @classmethod
    def long_window_combined(cls) -> "AdjointConfig":
        """Preset (opt-in): :meth:`long_window_tke` plus ``tke_ml_ratio=100``,
        ``tke_ml_raw_cut=-1e-6`` and ``freeze_fct_limiter``, for the all-on model with the
        sea-ice adjoint live: ``IceConfig(adjoint_mode="exact", adjoint_damping="frozen_sweep")``
        with ``adjoint_alpha_kappa`` 1 and ``adjoint_theta`` 0.97 (the defaults). GM/Redi is
        differentiated in full.

        Measured 2026-10-11 on CORE2 (24-step chunks; central FDs at h = 3e-3, 1e-2, 3e-2).
        Ice drift cost (Arctic box; ln P*, ln C_d): adjoint within 3.1 % of the FD at 30 days
        from January (three runs, equal to four digits), March and July and at 60 and 90 days
        from January; in the Antarctic from July P* is 8.2 % small in magnitude (C_d 0.8 %),
        as with the GM slope frozen. TKE costs (box ``natl_subpolar``; ln c_k, ln c_eps, ln cd, ln alpha):
        window-mean MLD within 8.4 % at 30, 60 and 90 days from January and July; July box T
        within 2.3 %; January box T within 5.1 % at 90 days, and missing at 30 days (c_k 134 %,
        cd 188 %) and 60 days (alpha 780 %, c_eps 57 %), as every setting tried does there.

        The two mixing-length masks are what allows GM live. Without them two of four January
        30-day ice runs burst (P* 0.96 and 1.53 of the FD), the run decided by GPU round-off:
        one inverted cell on the Northeast Greenland shelf (raw N^2 -7e-5, smoothed 1.5e-7, above
        :attr:`tke_ml_cut`) where the free mixing length's N^2 derivative is large. Freezing the
        GM slope hid it at the price of the January accuracy (box T at 90 days 33 % off).
        ``freeze_fct_limiter`` removes the salinity cotangent spikes of the FCT limiter (x10 at
        30 days from March, x12 at 90 days from January) and moves the ice gradients by less
        than 1e-3 relative. docs/PORTING_LESSONS.md, combined adjoint, 2026-10-10 and
        2026-10-11."""
        return cls.long_window_tke()._replace(tke_ml_ratio=100.0, tke_ml_raw_cut=-1e-6,
                                              freeze_fct_limiter=True)

    @classmethod
    def year_window_combined(cls) -> "AdjointConfig":
        """Preset (opt-in): :meth:`long_window_combined` for one-year windows, with the
        Prandtl-number derivative also omitted at N^2 < 1e-6 (:attr:`tke_pr_cut`, not 1e-8) and
        poleward of 70 degrees (:attr:`tke_pr_lat_cut`); the mixing-length cut stays at 1e-8.
        The latitude cut is valid only for costs far from 70 degrees. Use it with
        ``IceConfig(adjoint_mode="exact", adjoint_damping="frozen_sweep")`` at
        ``adjoint_alpha_kappa`` 1 and ``adjoint_theta`` 0.97, plus ``adjoint_sweep_guard=10``,
        ``adjoint_thin_mass=100`` and ``adjoint_cuts=("fct_flux", "strength_conc")``. As a
        ``tke_diag.parse_setting`` spec: ``tke_pr_cut=1e-6+tke_pr_ratio=100+
        tke_pr_raw_cut=-1e-6+tke_ml_cut=1e-8+freeze_fct_limiter+tke_ml_ratio=100+
        tke_ml_raw_cut=-1e-6+tke_pr_lat_cut=70``.

        Measured 2026-10-17 on CORE2 (24-step chunks; one-year window-mean MLD in box
        ``natl_subpolar``; central FDs at h = 1e-2 and, where given, 3e-2) for ln c_k, ln c_eps,
        ln cd, ln alpha and ln k_gm. Window from 2013-01-01: within 2.9-5.8 % for all five.
        From 2013-07-01: c_k 6 %, c_eps 8 %, k_gm 2 %, cd 15 % and alpha 14 % too large. From
        2007-07-01: c_k, c_eps and cd within 2.1 %, alpha 9.5 %, k_gm 12 % too small.

        FDs perturbing a control only from a given date to the window end show where the
        July-start errors come from. Perturbing from late January to mid-April, the adjoint is
        within 1 % of the FD (cd and k_gm, July 2013). The excess of cd builds up between July
        and January. The k_gm shortfall on the 2007 window comes from the same months.
        With ``tke_ml_cut=1e-6``, k_gm is 18 % small on both July windows, because 33-46 % of the
        box's interfaces lose the mixing-length N^2 derivative in winter. The other four
        constants then pass, but on July 2013 only because a winter shortfall cancels the
        July-January excess.

        A year reverse sweep on a GPU can, from round-off alone, contain a cotangent burst that
        a repeat of the same run does not. Read the per-field cotangent trace, not just the
        total. docs/PORTING_LESSONS.md, year-window combined adjoint, 2026-10-13 to
        2026-10-17."""
        return cls.long_window_combined()._replace(tke_pr_cut=1e-6, tke_pr_lat_cut=70.0)


FREEZE_FLAGS: tuple = tuple(f for f in AdjointConfig._fields if f.startswith("freeze_"))
"""The boolean ``freeze_*`` fields of :class:`AdjointConfig`, in declaration order. Enumerate
these (not ``AdjointConfig._fields``) when iterating over flags: ``diag_cuts`` is a set."""

TKE_CANDIDATE_FIELDS: tuple = ("tke_pr_cut", "tke_pr_mode", "tke_pr_ratio", "tke_pr_raw_cut",
                               "tke_kme_tke", "tke_kme_scope", "tke_kme_term", "tke_ml_cut",
                               "tke_ml_ratio", "tke_ml_raw_cut", "tke_pr_ice_cut", "tke_pr_lat_cut",
                               "tke_pr_tu_cut",
                               "gm_n2_ratio", "gm_n2_raw_cut", "gm_slope_cut",
                               "conv_surrogate")
"""The non-boolean candidate settings of :class:`AdjointConfig` (``None`` = off): the
classical-TKE ones, the two masked GM/Redi input cuts, the steep-slope cut and the
convective-adjustment surrogate."""
TKE_PR_MODES = ("cut", "at")
TKE_KME_SCOPES = ("column", "iface1")
TKE_KME_TERMS = ("km_e", "ke_diff")


class TkeAdjoint(NamedTuple):
    """The static candidate settings the TKE column core reads (:meth:`AdjointConfig.
    tke_options`); ``None`` entries are off."""
    pr_n2_star: float | None
    pr_mode: str
    kme_tke: float | None
    kme_scope: str
    kme_term: str
    pr_ratio: float | None = None
    pr_raw_cut: float | None = None
    ml_n2_star: float | None = None
    ml_ratio: float | None = None
    ml_raw_cut: float | None = None
    pr_ice: float | None = None
    pr_lat: float | None = None
    pr_tu: float | None = None

KNOWN_CUTS: dict = {
    "mix.kv_out": "Kv after the mixing closure and its halo exchange, before freeze_mixing "
                  "and before the GM/Redi K33 augmentation (step.py)",
    "mix.av_out": "Av after the mixing closure and its halo exchange, before freeze_mixing "
                  "(step.py)",
    # --- the classical-TKE column core (cvmix_tke.integrate_tke_column / mixing_length) ---
    "tke.km_e": "sqrt(tke) in KappaM = c_k*mxl*sqrt(tke) (every use of KappaM: KappaH, "
                "production, buoyancy, TKE diffusivity, Kv/Av) (cvmix_tke.py)",
    "tke.ml_e": "sqrt(tke) in the unconstrained mixing length mxl0 = sqrt(2)*sqrt(tke)/N "
                "(cvmix_tke.mixing_length)",
    "tke.ml_n2": "N^2 in the unconstrained mixing length mxl0 (cvmix_tke.mixing_length)",
    "tke.ml_wall": "the mixing length where the walls / Blanke-Delecluse min-scans set it below "
                   "mxl0: treated as a fixed length (cvmix_tke.mixing_length)",
    "tke.shear": "S^2 in shear production and in the Richardson number (cvmix_tke.py)",
    "tke.n2": "N^2 in the buoyancy term and in the Richardson number (not in mxl0: that is "
              "tke.ml_n2) (cvmix_tke.py)",
    "tke.pr": "the Prandtl number clip(6.6*Ri, 1, 10) (cvmix_tke.py)",
    "tke.prod": "the shear production S^2*KappaM (cvmix_tke.py)",
    "tke.buoy": "the buoyancy term N^2*KappaH (cvmix_tke.py)",
    "tke.diss": "the Patankar dissipation dt*c_eps*sqrt(tke)/mxl on the diagonal of the "
                "implicit TKE solve (cvmix_tke.py)",
    "tke.ke_diff": "the TKE self-diffusivity ke = alpha*(KappaM[k+1]+KappaM[k])/2 "
                   "(cvmix_tke.py)",
    "tke.surf": "the surface Neumann flux cd*forc_tke_surf^1.5/dzt[0] (cvmix_tke.py)",
    "tke.kh_floor": "KappaH where it leaves the closure for Kv, at interfaces with KappaH < "
                    "TKE_KH_FLOOR = 1e-5 m^2/s (the derivative a background-diffusivity floor "
                    "max(KappaH, 1e-5) would have; cvmix_tke.py)",
    "tke.deep": "every closure output (tke_new, KappaM, KappaH) at interfaces deeper than "
                "TKE_DEEP_CUT_Z = 500 m (depth from the layer thicknesses; a negative "
                "control) (cvmix_tke.py)",
    # --- the eddy-parameterization block (gm.gm_diagnostics): the three uses of N^2 ---
    "gm.slope_n2": "N^2 in the neutral slope sigma_xy/N^2, the ODM95 taper and the Redi "
                   "diffusivity built from it (gm.compute_neutral_slope)",
    "gm.cm_n2": "N^2 in the baroclinic wave speed cm = int sqrt(N^2) dz / 3 pi "
                "(gm.init_redi_gm)",
    "gm.bvp_n2": "N^2 on the diagonal of the streamfunction solve (gm.fer_solve_gamma)",
}
"""Registry of the derivative cut points that :func:`dcut` call sites in the model use,
``{name: where it sits}``. A new cut point is added here and at its call site together;
``{"mix.kv_out", "mix.av_out"}`` together equal ``freeze_mixing``.

Two TKE-path cuts are NOT registered here because a flag already performs exactly that
operation (one operation, one name): the convective-adjustment increment of ``mo_convect``
is cut by ``freeze_convection`` (``pp.mo_convect(freeze_increment=True)``, used by the TKE,
KPP and PP drivers alike), and the prognostic TKE handed to the next step is cut by
``freeze_tke_state`` (``stop_gradient`` on ``tke_new`` in ``step.py``). Use those flags
together with ``tke.*`` cuts to cut the whole closure piece by piece."""


def _cut_names(cfg):
    """The active cut names of ``cfg``: ``None`` -> none; an :class:`AdjointConfig` -> its
    ``diag_cuts``; a set/frozenset of names (what the TKE column core receives) -> itself."""
    if cfg is None:
        return ()
    if isinstance(cfg, (set, frozenset)):
        return cfg
    return cfg.diag_cuts


def dcut(cfg, name: str, x):
    """Named derivative cut: ``stop_gradient`` on every floating leaf of ``x`` when ``name``
    is in ``cfg.diag_cuts``, else ``x`` itself.

    The decision is made in Python at trace time, so an inactive cut adds nothing to the
    traced program. ``cfg`` may be ``None``, an :class:`AdjointConfig`, or a set of cut
    names (the form passed down to the column cores). ``name`` must be registered in
    :data:`KNOWN_CUTS` (a misspelled call site would otherwise be a cut that can never
    fire)."""
    if name not in KNOWN_CUTS:
        raise ValueError(f"cut point {name!r} is not registered in adjoint_stab.KNOWN_CUTS")
    if name not in _cut_names(cfg):
        return x

    def one(leaf):
        dt = getattr(leaf, "dtype", None)
        if dt is not None and jnp.issubdtype(dt, jnp.floating):
            return lax.stop_gradient(leaf)
        return leaf
    return jax.tree.map(one, x)


def dcut_where(cfg, name: str, mask_fn, x):
    """Named derivative cut restricted to a mask: where ``mask`` is True the derivative of
    ``x`` (one floating array) is cut, elsewhere it passes; the value is ``x`` everywhere.

    ``mask_fn`` is a zero-argument callable returning the boolean mask (broadcastable to
    ``x``); it is called only when the cut is active, so an inactive cut traces nothing.
    Build the mask from forward values (``stop_gradient``-ed inputs). The cut is a
    ``custom_jvp`` whose primal output is ``x`` itself and whose tangent is
    ``where(mask, 0, dx)``, so the forward is bitwise the uncut one (no ``select`` on the
    value) and reverse mode transposes the masked tangent."""
    if name not in KNOWN_CUTS:
        raise ValueError(f"cut point {name!r} is not registered in adjoint_stab.KNOWN_CUTS")
    if name not in _cut_names(cfg):
        return x
    return _masked_cut(x, jnp.broadcast_to(mask_fn(), jnp.shape(x)))


def n2_cancel_mask(n2, n2_raw, ratio=None, raw_cut=None):
    """Forward-value mask of the cells whose smoothed ``n2`` is a cancellation residual
    (``|n2_raw| > ratio * |n2|``) or whose unsmoothed ``n2_raw`` is below ``raw_cut``, joined;
    ``None`` turns a test off. The two masks of :attr:`AdjointConfig.tke_pr_ratio` /
    :attr:`~AdjointConfig.tke_pr_raw_cut`, here for the GM/Redi input cuts."""
    n2 = jax.lax.stop_gradient(n2)
    n2_raw = jax.lax.stop_gradient(n2_raw)
    mask = jnp.zeros(jnp.shape(n2), bool)
    if ratio is not None:
        mask = mask | (jnp.abs(n2_raw) > ratio * jnp.abs(n2))
    if raw_cut is not None:
        mask = mask | (n2_raw < raw_cut)
    return mask


def cut_where(x, mask):
    """``x`` in the forward; its derivative omitted where ``mask`` (boolean, built from forward
    values, broadcastable to ``x``) is True. The primal output is ``x`` itself (a
    ``custom_jvp``), so the forward is bitwise the uncut one."""
    return _masked_cut(x, jnp.broadcast_to(mask, jnp.shape(x)))


@jax.custom_jvp
def _masked_cut(x, mask):
    return x


@_masked_cut.defjvp
def _masked_cut_jvp(primals, tangents):
    x, mask = primals
    dx, _dmask = tangents                    # mask is boolean: no tangent of its own
    return x, jnp.where(mask, jnp.zeros_like(dx), dx)


def scale_tangent(x, f):
    """``x`` in the forward; its tangent multiplied by ``f`` (Newtonian damping of a tangent).

    Panteleev, Yaremchuk, Stroh, Francis and Allard (2020), The Cryosphere 14, 4427-4451,
    doi:10.5194/tc-14-4427-2020, stabilise the tangent-linear and adjoint models of an
    elastic-viscoplastic sea-ice rheology by adding a friction term ``eps_N * delta_sigma`` to
    the TANGENT stress equations only, the forward model being left as it is. One sub-cycle of
    such a term multiplies the stress tangent by a factor ``f < 1``; this primitive is that
    multiplication, applied wherever the caller places it.

    Contract:
      * the primal output is ``x`` itself (a ``custom_jvp`` whose primal rule returns its
        input), so the forward is bitwise unchanged for every ``f``;
      * the tangent is ``f * dx`` with ``f`` cast to ``dx``'s dtype; ``f`` is a primal
        argument whose own tangent is ignored (build it from forward values); reverse mode
        transposes this linear rule, so the cotangent is ``f * g``;
      * ``f`` (scalar or array) must broadcast to ``x``'s shape WITHOUT widening it: a
        per-element ``f`` on a per-element field is fine, an ``f`` that would make the tangent
        larger than ``x`` raises ``ValueError`` at trace time;
      * ``f = 1`` is plain autodiff (``1 * dx == dx`` bitwise); ``f = 0`` acts as a derivative
        cut; integer/bool ``x`` is returned unchanged (its tangent is float0 anyway).
    Inside a ``lax.scan`` the factors compound step by step, so a per-step ``f`` gives the
    product of the per-step factors, not a closed form."""
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(dtype, jnp.inexact):
        return x
    xs, fs = tuple(jnp.shape(x)), tuple(jnp.shape(f))
    try:
        bs = tuple(jnp.broadcast_shapes(fs, xs))
    except ValueError:
        bs = None
    if bs != xs:
        raise ValueError(f"scale_tangent: factor shape {fs} does not broadcast to the value "
                         f"shape {xs} without widening it")
    return _scaled_tangent(x, f)


@jax.custom_jvp
def _scaled_tangent(x, f):
    return x


@_scaled_tangent.defjvp
def _scaled_tangent_jvp(primals, tangents):
    x, f = primals
    dx, _df = tangents                       # f's own tangent is deliberately ignored
    return x, jnp.asarray(f, dtype=dx.dtype) * dx


# ---------------------------------------------------------------------------
# linearise at another point

def differentiate_at(fn, args, alt):
    """``fn(*args)`` in the forward, with the derivative of ``fn`` taken at ``alt``.

    The value is exactly ``fn(*args)``. The tangent (and, by transposition, the cotangent)
    is ``d fn(alt) · d args``: the Jacobian of ``fn`` evaluated at ``alt`` applied to the
    tangent of ``args``. ``alt`` is a pytree with the same structure, shapes and dtypes as
    ``args`` and gets no tangent of its own (its derivative is zero). With ``alt`` equal to
    ``args`` this is plain autodiff.

    Use it to linearise a term about a smoother or reference state (for example the previous
    step's state, a time mean, or a clipped value) while the forward keeps the model's own
    value — the backward-only analogue of evaluating a coefficient's derivative elsewhere.

    Built on ``jax.custom_jvp`` (not ``custom_vjp``) so that forward mode keeps working;
    reverse mode transposes the linear tangent rule. ``fn`` is a static (hashable) callable,
    ideally a module-level function: every array it depends on must come in through
    ``args`` — an array captured by a closure is invisible to the rule and would be treated
    as a constant. The rule evaluates ``fn`` at both ``args`` (value) and ``alt`` (tangent),
    so a derivative costs one extra primal evaluation."""
    args, alt = tuple(args), tuple(alt)
    s_args, s_alt = jax.tree.structure(args), jax.tree.structure(alt)
    if s_args != s_alt:
        raise ValueError(f"differentiate_at: alt has structure {s_alt}, args has {s_args}")
    for a, b in zip(jax.tree.leaves(args), jax.tree.leaves(alt)):
        if jnp.shape(a) != jnp.shape(b) or jnp.result_type(a) != jnp.result_type(b):
            raise ValueError(f"differentiate_at: alt leaf {jnp.shape(b)} {jnp.result_type(b)} "
                             f"does not match args leaf {jnp.shape(a)} {jnp.result_type(a)}")
    return _differentiate_at(fn, args, alt)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _differentiate_at(fn, args, alt):
    return fn(*args)


@_differentiate_at.defjvp
def _differentiate_at_jvp(fn, primals, tangents):
    args, alt = primals
    dargs, _dalt = tangents                  # alt has no tangent of its own
    y = fn(*args)
    _, dy = jax.jvp(fn, alt, dargs)
    return y, dy


# ---------------------------------------------------------------------------
# the primitive: identity forward, a chosen map applied to the cotangent

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def cotangent_op(fn, x):
    """Identity in the forward pass; ``fn`` is applied to the cotangent in the backward.

    ``x`` may be any pytree; ``fn`` must map a cotangent of that structure to the same
    structure. ``fn`` is a static argument, so build it **once** outside any traced loop —
    a fresh closure per call forces a retrace every step.

    This is the single hook every stabilizer below is built from. Because the forward is
    the identity, inserting it anywhere leaves the trajectory bit-for-bit unchanged; only
    the reverse pass sees it.
    """
    return x


def _cot_fwd(fn, x):
    return x, None


def _cot_bwd(fn, _res, g):
    return (fn(g),)


cotangent_op.defvjp(_cot_fwd, _cot_bwd)


# ---------------------------------------------------------------------------
# freezing

def freeze(x):
    """``stop_gradient`` over a whole pytree — the ECCO "omit the parametric dependency"
    operator. Forward value is unchanged (``stop_gradient`` is the identity); the backward
    simply does not propagate through whatever produced ``x``."""
    return jax.tree.map(lax.stop_gradient, x)


def freeze_increment(base, out):
    """Keep ``out``'s value but differentiate only through ``base``.

    Use it to freeze *one stage's contribution* rather than the whole quantity — e.g. the
    convective-adjustment bump on top of a mixing coefficient, where you want the closure's own
    state-dependence kept and only the ``N^2 < 0`` switch removed.

    Written in the straight-through form ``stop_gradient(out) + (base - stop_gradient(base))``:
    the forward is ``out + 0.0``, which is ``out`` exactly for every finite ``base`` (only the
    sign of a ``-0.0`` in ``out`` becomes ``+0.0``), and the derivative is ``d base``. The
    increment form ``base + stop_gradient(out - base)`` has the same derivative but is NOT exact
    in floating point: ``out - base`` rounds, and adding ``base`` back need not return ``out``
    (e.g. ``base = 0.03046625406612638``, ``out = 0.3``).
    """
    sg = lax.stop_gradient
    return jax.tree.map(lambda b, o: sg(o) + (b - sg(b)), base, out)


# ---------------------------------------------------------------------------
# measuring

def _real_leaves(tree):
    """Leaves that carry a real cotangent (skip ``float0``, which JAX uses for the
    cotangent of integer/unused leaves and which supports no arithmetic)."""
    out = []
    for leaf in jax.tree.leaves(tree):
        dt = getattr(leaf, "dtype", None)
        if dt is not None and dt == jax.dtypes.float0:
            continue
        out.append(leaf)
    return out


def cotangent_norm(tree):
    """Euclidean norm of a cotangent pytree (``float0`` leaves skipped)."""
    leaves = _real_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def _map_real(fn, tree):
    """Apply ``fn`` to every real leaf, passing ``float0`` leaves through untouched."""
    def one(leaf):
        dt = getattr(leaf, "dtype", None)
        if dt is not None and dt == jax.dtypes.float0:
            return leaf
        return fn(leaf)
    return jax.tree.map(one, tree)


def field_norms(tree) -> dict:
    """``{leaf path: Euclidean norm}`` of a cotangent pytree, floating leaves only.

    Paths are :func:`jax.tree_util.keystr` with ``simple=True, separator="."`` (``"T"``,
    ``"uv"``, ``"tke"`` for a :class:`~fesom_jax.state.State`; ``"a.b"`` for nested dicts).
    ``float0`` and integer leaves are skipped. The sum of the squares equals
    ``cotangent_norm(tree)**2`` up to round-off. One device→host transfer for all leaves."""
    names, vals = [], []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        dt = getattr(leaf, "dtype", None)
        if dt is None or dt == jax.dtypes.float0 or not jnp.issubdtype(dt, jnp.floating):
            continue
        names.append(jax.tree_util.keystr(path, simple=True, separator="."))
        vals.append(jnp.sqrt(jnp.sum(jnp.square(leaf))))
    got = jax.device_get(vals)
    return {n: float(v) for n, v in zip(names, got)}


# ---------------------------------------------------------------------------
# 2. adjoint diffusion — the exact transpose of the model's own mesh filter

def make_nodal_smoother(mesh: Mesh, factor: float, *, safety: float = 0.5):
    """Build ``g -> F^T g``, the exact transpose of :func:`fesom_jax.coarsen.filter_field`
    at a scale of ``factor`` x the local node spacing.

    Returns ``(apply, info)``. ``apply`` acts on a single array whose leading axis is
    ``nod2D`` (rank 1, 2, or 3 — a rank-3 field is smoothed component by component along
    its last axis); arrays that are not nodal are returned untouched, so it is safe to map
    over a whole ``State`` cotangent. ``info`` is the ``(n_steps, kappa_dt)`` filter plan.

    **Why the transpose and not the filter itself.** The cotangent is a linear functional
    on state perturbations, so the operator that belongs on it is the adjoint of the one
    you would apply to the state. Getting this wrong is not cosmetic: the FE Laplacian is
    self-adjoint in the *area-weighted* inner product, not the Euclidean one, so ``F`` and
    ``F^T`` differ by the area metric — and on a mesh whose cell areas span two orders of
    magnitude that is a large, spatially structured error. ``jax.linear_transpose`` gives
    the exact operator with no hand-derived algebra, which is the point.

    **What it guarantees.** ``F`` preserves constants (the mesh Laplacian annihilates them,
    in floating point too), so ``sum(F^T g) == sum(g)`` exactly: the smoother redistributes
    sensitivity across neighbouring nodes but cannot change the total. A stabilizer that
    could silently rescale the gradient would be indistinguishable from a bug; this one
    cannot.
    """
    n_steps, kappa_dt = coarsen.plan_filter(mesh, factor, safety=safety)
    mask2 = jnp.asarray(mesh.node_layer_mask, jnp.float64)      # (nod2D, nl)
    mask1 = mask2[:, 0]                                          # (nod2D,)
    nod2D = int(mesh.nod2D)
    cache: dict = {}

    def _forward(v, mask):
        return coarsen.filter_field(mesh, v, n_steps=n_steps, kappa_dt=kappa_dt,
                                    node_mask=mask, check_stability=False)

    def _transpose_for(shape, dtype):
        key = (shape, jnp.dtype(dtype).name)
        if key not in cache:
            mask = mask1 if len(shape) == 1 else mask2[:, :shape[1]]
            fT = jax.linear_transpose(lambda v: _forward(v, mask),
                                      jax.ShapeDtypeStruct(shape, dtype))
            cache[key] = lambda g: fT(g)[0]
        return cache[key]

    def apply(x):
        if not hasattr(x, "shape") or x.ndim == 0 or x.shape[0] != nod2D:
            return x                                             # element field, scalar, ...
        if x.ndim <= 2:
            return _transpose_for(x.shape, x.dtype)(x)
        if x.ndim == 3:                                          # e.g. uvnode (N, nl, 2)
            t = _transpose_for(x.shape[:2], x.dtype)
            return jnp.stack([t(x[..., k]) for k in range(x.shape[2])], axis=-1)
        return x

    return apply, (n_steps, kappa_dt)


# ---------------------------------------------------------------------------
# 3. discount and clip

def make_discounter(gamma: float):
    """Scale the cotangent by ``gamma`` once per application.

    Inserted once per model step, this weights a step ``k`` steps upstream of the loss mark
    by ``gamma**k``. It is linear, so the result remains an exact gradient — of an
    objective whose per-step sensitivities have been geometrically reweighted toward the
    recent past. The bias is transparent and in the right direction: it discards precisely
    the long-lag sensitivity that the adjoint cannot represent anyway. Pick ``gamma`` from
    the measured per-step amplification ``a``: ``gamma = 1/a`` cancels the growth exactly
    and leaves an effective memory of ``1/ln(a)`` steps.
    """
    g = float(gamma)

    def apply(tree):
        return _map_real(lambda x: x * g, tree)
    return apply


def make_clipper(max_norm: float):
    """Rescale the cotangent to ``max_norm`` whenever it exceeds it.

    **Not a gradient.** This map is nonlinear in the cotangent, so the composed backward
    pass is not the adjoint of anything; it preserves the direction of the cotangent at the
    moment of clipping but reweights steps by which of them saturated the cap. It is the
    BPTT gradient-clipping trick (Pascanu et al. 2013) and is included because it answers a
    specific diagnostic question — "is there a usable direction underneath the blow-up, or
    is the direction itself garbage?" — cheaply. Do not train on it without checking the
    answer against a linear stabilizer.
    """
    c = float(max_norm)

    def apply(tree):
        nrm = cotangent_norm(tree)
        scale = jnp.minimum(1.0, c / jnp.maximum(nrm, 1e-300))
        return _map_real(lambda x: x * scale, tree)
    return apply


# ---------------------------------------------------------------------------
# assembling one state-level stabilizer

@dataclass(frozen=True)
class StabConfig:
    """Which backward-pass stabilizers to apply to the state carry, and how hard.

    All default to off, and with everything off :func:`make_state_stabilizer` returns
    ``None`` rather than an identity — so the caller can skip the wrapper entirely and keep
    the traced graph identical to the unstabilized model.

    ``gamma``          per-step cotangent discount (1.0 = off).
    ``filter_factor``  adjoint-diffusion scale, in multiples of local node spacing (0 = off).
    ``clip``           per-step cotangent norm cap (0 = off). Nonlinear — see :func:`make_clipper`.
    ``filter_safety``  passed to :func:`fesom_jax.coarsen.plan_filter`.
    """
    gamma: float = 1.0
    filter_factor: float = 0.0
    clip: float = 0.0
    filter_safety: float = 0.5

    @property
    def active(self) -> bool:
        return self.gamma != 1.0 or self.filter_factor > 0.0 or self.clip > 0.0

    def describe(self) -> str:
        bits = []
        if self.gamma != 1.0:
            bits.append(f"gamma={self.gamma:g}")
        if self.filter_factor > 0.0:
            bits.append(f"filter={self.filter_factor:g}dx")
        if self.clip > 0.0:
            bits.append(f"clip={self.clip:g}")
        return "+".join(bits) if bits else "none"


def make_state_stabilizer(mesh: Mesh, cfg: StabConfig):
    """Build ``state -> state``: identity forward, stabilized cotangent backward.

    Drop the returned function into a scan body right after ``step(...)`` and the forward
    trajectory is untouched while every reverse-mode traversal of that step sees the
    cotangent passed through, in order, the adjoint diffusion, the discount, and the clip.
    Returns ``None`` when ``cfg`` asks for nothing.

    The stabilizers compose in that order on purpose: diffusion first, because it is the
    only one that acts on the *structure* of the cotangent and it should see the field
    before any scalar rescaling; the scalar maps commute with it anyway, so the order is a
    readability choice rather than a numerical one.
    """
    if not cfg.active:
        return None

    ops = []
    plan = None
    if cfg.filter_factor > 0.0:
        smooth, plan = make_nodal_smoother(mesh, cfg.filter_factor, safety=cfg.filter_safety)
        ops.append(lambda tree, _s=smooth: _map_real(_s, tree))
    if cfg.gamma != 1.0:
        ops.append(make_discounter(cfg.gamma))
    if cfg.clip > 0.0:
        ops.append(make_clipper(cfg.clip))

    def backward_map(g):
        for op in ops:
            g = op(g)
        return g

    def stabilize(state):
        return cotangent_op(backward_map, state)

    stabilize.plan = plan          # (n_steps, kappa_dt) of the diffusion, or None
    stabilize.cfg = cfg
    return stabilize


# ---------------------------------------------------------------------------
# 5. the amplification screen — does this seed state have a usable adjoint at all?

# Per-step growth of the cotangent norm, above which a window's gradient is not to be
# believed or trained on. The two populations that set it were measured with the network
# ENTIRELY ABSENT, so this is a property of the SEED STATE and not of any closure:
#
#   foundation_snaps/snap_step037440.pkl   30 d   1.0008/step   log-spread 0.001   CLEAN
#   student_free_1958/snap_day150.pkl       5 d   1.0539/step   log-spread 0.071   BLOWN
#
# Over a 240-step window those are 1.2x against 3e5. Clean 30- and 60-day runs sit in
# 1.0008-1.0013 with log-spread <= 0.001, so the cut below is nowhere near either
# population and does not need tuning. Two statistics, not one, because they fail
# differently: a raised MEDIAN is smooth exponential growth (chaos, the honest end of the
# horizon), while a raised LOG-SPREAD with a normal median means the growth is concentrated
# in a few chunks, which is a switch firing and is a defect.
#
# ⚠️ BOTH OF THOSE DILUTE WITH WINDOW LENGTH, and a measured case proved it. The 60-day
# window ending 1961-02-14 (job 26819415) blows up in a SINGLE day — the cotangent goes x1406
# on 1961-02-06 and then sits on a plateau of ~1.0 ratios for the remaining fifty chunks. Its
# finite difference says the true gradient is -9.80e-01 while the adjoint returns +5.69e+06,
# so it is as blown as a window gets. Yet over the full trace the MEDIAN is 1.0001/step — it
# passes the first bar outright, because the plateau outnumbers the event — and the log-spread
# clears the second by only 1.2x (0.024 against 0.020). On the first seventeen chunks both had
# fired (median 1.007, spread 0.040); the screen got WEAKER as more evidence arrived, which is
# the wrong direction for a screen. Holding one x1200 event fixed and lengthening the plateau:
#
#   flat chunks after the event        2        50       100       400
#   median                          1.000413  1.000413  1.000413  1.000413   never fires
#   log-spread                      0.0696    0.0205    0.0146    0.0074     stops firing at ~100
#
# i.e. the two-bar screen stopped catching a single-event blow-up on windows past ~100 days,
# and the campaign's target is a YEAR.
#
# ⚠️ THE OBVIOUS FIX — bar the worst per-chunk rate — IS UNSAFE, and a counterexample from
# this project's own records says so. The 30-day seam-gate trace contains one chunk at x6.02
# followed immediately by x0.184: the norm spikes and comes straight back. That run is clean by
# every independent check (finite difference agrees to 7.9 %, the gradient reproduces to
# 6.5e-4), yet its worst per-chunk rate is 1.03811 — close enough to the blown population to
# leave no usable gap. A spike that reverses is not a blow-up.
#
# So the third statistic is the worst rate over THREE consecutive chunks. A transient cancels
# against its own recovery; a real jump does not, because the norm never comes back. Measured
# over every trace on record, and both populations are exactly the ones above:
#
#                                  worst 1 chunk     worst 3 chunks
#   clean (FD-confirmed), 5-180 d  1.00123..1.03811  1.00123..1.01332
#   5 d, no freeze knobs                   1.67623           1.40121
#   5 d seam gradient, blown               1.22949           1.07642
#   60 d ending Feb (26819415)             1.16302           1.09016
#   180 d ending Feb (26818134)            1.16238             (same shape)
#
# The one-chunk gap is 0.0381 -> 0.0711 in excess over 1, a factor of 4.3; the three-chunk gap
# is 0.0133 -> 0.0711, a factor of 5.3. 1.030 is its geometric middle: 2.3x above the worst
# clean and 2.3x below the least blown. Unlike the other two bars it does NOT dilute, because a
# max over fixed-width windows does not care how much quiet surrounds the event.
AMP_CLEAN_MAX = 1.010     # per step, median over chunks
AMP_SPREAD_MAX = 0.020    # std of log(per-chunk rate) across chunks
AMP_SUSTAINED_MAX = 1.030  # per step, worst 3-chunk stretch — the bar that does not dilute
AMP_SUSTAINED_W = 3       # chunks per stretch; short traces fall back to a single chunk


class Amplification(NamedTuple):
    """Verdict of :func:`amplification`. ``clean`` is the only field a caller should gate on."""
    median: float          # per-step growth rate, median over chunks
    max: float             # per-step growth rate, worst SINGLE chunk (diagnostic only —
    #                        a transient spike that reverses lands here and is not a defect)
    sustained: float       # per-step growth rate, worst AMP_SUSTAINED_W consecutive chunks
    log_spread: float      # std of log(per-step rate) across chunks
    worst_chunk: int       # forward-time index of the worst chunk (-1 if undefined)
    per_step: tuple        # per-chunk rates, in REVERSE-pass order (same order as `trace`)
    clean: bool
    reason: str            # "" when clean; else which bound failed, with the number


def amplification(trace, steps_per_chunk) -> Amplification:
    """Per-step cotangent amplification from a reverse-pass chunk-boundary norm trace.

    ``trace`` is the sequence of ``cotangent_norm`` values recorded by a chunked reverse
    accumulation: ``trace[0]`` is the loss seed at the END of the window and each further
    entry is one chunk further back in time, so ``trace[i+1]/trace[i]`` is the growth over
    one chunk of ``steps_per_chunk`` steps travelling backwards.

    This is the screen the training driver runs on every window before the window's gradient
    is allowed into a minibatch. A blown window does not announce itself — it averages in,
    poisons the update, and leaves a train-loss curve that looks like slow learning rather
    than a fault. The trace is a by-product of the reverse sweep, so the screen is free.

    Callers must not re-derive this. There were once two implementations of "freeze mixing"
    in this project with nothing asserting they agreed; that cost a day of chasing a gap
    that was not there.
    """
    import numpy as _np

    tr = [float(x) for x in trace]
    if len(tr) < 2:
        return Amplification(float("nan"), float("nan"), float("nan"), 0.0, -1, (), False,
                             "trace too short to measure (need >= 2 boundaries)")
    k = max(1, int(steps_per_chunk))
    rates = [(tr[i + 1] / tr[i]) ** (1.0 / k) if tr[i] > 0 else float("nan")
             for i in range(len(tr) - 1)]
    fin = [r for r in rates if _np.isfinite(r) and r > 0]
    if not fin:
        return Amplification(float("nan"), float("nan"), float("nan"), float("nan"), -1,
                             tuple(rates), False,
                             "no finite per-chunk rate — the reverse pass produced a zero or "
                             "non-finite cotangent norm")
    med = float(_np.median(fin))
    mx = float(_np.max(fin))
    # Worst stretch of W consecutive chunks. A spike followed by its own recovery cancels
    # inside the window; a jump the norm never returns from does not. W shrinks on a trace too
    # short to hold it, where the statistic degenerates to the single-chunk max.
    w = max(1, min(int(AMP_SUSTAINED_W), len(tr) - 1))
    sus = [(tr[i + w] / tr[i]) ** (1.0 / (k * w)) for i in range(len(tr) - w) if tr[i] > 0]
    sus = [r for r in sus if _np.isfinite(r) and r > 0]
    sustained = float(_np.max(sus)) if sus else float("nan")
    spread = float(_np.std(_np.log(fin))) if len(fin) > 1 else 0.0
    n_chunks = len(tr) - 1
    worst = int(n_chunks - 1 - int(_np.nanargmax(_np.asarray(rates, dtype=float))))
    why = []
    if not med <= AMP_CLEAN_MAX:
        why.append(f"median {med:.4f}/step exceeds {AMP_CLEAN_MAX:.3f}")
    if not spread <= AMP_SPREAD_MAX:
        why.append(f"log-spread {spread:.3f} exceeds {AMP_SPREAD_MAX:.3f} "
                   f"(growth concentrated in chunk {worst} — a switch firing, not chaos)")
    if not sustained <= AMP_SUSTAINED_MAX:
        why.append(f"worst {w}-chunk stretch {sustained:.4f}/step exceeds "
                   f"{AMP_SUSTAINED_MAX:.3f} (around chunk {worst}) — growth the norm never "
                   f"comes back from, which the median and the log-spread both dilute away as "
                   f"the window grows")
    return Amplification(med, mx, sustained, spread, worst, tuple(rates), not why,
                         "; ".join(why))


def amplification_by_field(field_trace, steps_per_chunk) -> dict:
    """Per-field :class:`Amplification` from :attr:`fesom_jax.adjoint.ChunkedGrad.field_trace`.

    ``field_trace`` is a list of ``{field: norm}`` dicts, one per chunk boundary in the order of
    ``trace`` (seed first). Entries that are ``None`` (boundaries reported by an earlier leg of
    a resumed sweep) are skipped from the front; the result covers the recorded boundaries.
    Returns ``{field: amplification(trace_of_that_field, steps_per_chunk)}``. Which field grows
    fastest in the reverse pass is where a blow-up lives; the total norm alone mixes fields
    whose magnitudes differ by orders of magnitude."""
    rows = list(field_trace or [])
    while rows and rows[0] is None:
        rows.pop(0)
    if any(r is None for r in rows):
        raise ValueError("field_trace has a missing entry after the first recorded boundary")
    if not rows:
        return {}
    return {name: amplification([r[name] for r in rows], steps_per_chunk) for name in rows[0]}
