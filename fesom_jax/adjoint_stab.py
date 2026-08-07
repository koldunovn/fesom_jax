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

    Measured effect on the CORE2 twin over a 2-day window (docs/ADJOINT_HORIZON.md), as
    growth of the reverse-pass field per 1800 s step — the quantity that decides whether a
    long window is usable at all:

    =========================  ============  ====================================
    flag                       growth/step   verdict
    =========================  ============  ====================================
    (none)                     1.21 - 1.55   unusable; sign flips between repeats
    ``freeze_mixing``          1.0070        reproducible to 6 significant figures
    ``freeze_stratification``  1.0322        nearly as good; a blunter cut
    ``freeze_convection``      1.2550        inside the unfrozen scatter: no effect
    ``freeze_gm_slope``        1.4298        inside the unfrozen scatter: no effect
    ``freeze_tke_state``       1.2913        inside the unfrozen scatter: no effect
    ``freeze_advection``       1.3238        inside the unfrozen scatter: no effect
    =========================  ============  ====================================

    ``freeze_mixing`` is the one that matters, and it is the documented ECCO practice (Forget et
    al. 2015): use the diffusivities and viscosities the turbulence closure computes, but
    omit their parametric dependency on the ocean state. The eddy-parameterization term that
    augments the vertical diffusivity is added *after* the closure runs, so freezing the
    closure does not touch the GM/Redi gradient path.
    """

    freeze_mixing: bool = False
    """Kv and Av as the turbulence closure returns them (post-convection). The ECCO move,
    and the one measured to work."""

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
