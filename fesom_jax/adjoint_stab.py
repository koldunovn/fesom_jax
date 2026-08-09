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
   is what ``scripts/nn_eddy/adjoint_horizon.py`` measures.

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
    "StabConfig", "AdjointConfig",
    "Amplification", "amplification", "AMP_CLEAN_MAX", "AMP_SPREAD_MAX", "AMP_SUSTAINED_MAX", "AMP_SUSTAINED_W",
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
    the nonlinear forward. Measured: not the amplifier here."""

    freeze_pgf: bool = False
    """The baroclinic pressure-gradient force. Cuts the tracer-to-momentum coupling
    entirely. **Diagnostic only** — the resulting gradient is not usable for training."""

    @property
    def active(self) -> bool:
        return any(self)

    def describe(self) -> str:
        on = [f.removeprefix("freeze_") for f, v in zip(self._fields, self) if v]
        return "+".join(on) if on else "none"


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

    ``base + stop_gradient(out - base)`` equals ``out`` exactly in the forward while the
    backward sees only ``base``. Use it to freeze *one stage's contribution* rather than
    the whole quantity — e.g. the convective-adjustment bump on top of a mixing
    coefficient, where you want the closure's own state-dependence kept and only the
    ``N^2 < 0`` switch removed.
    """
    return jax.tree.map(lambda b, o: b + lax.stop_gradient(o - b), base, out)


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
