"""NN-GM child-plan M3 — TRAIN the Arm B closure: a learned bolus streamfunction.

Arm B's ceiling has been measured (``apriori_eval.oracle_psi_constrained``: 0.617 of the
temperature flux divergence per five-day mark, 0.551 under the monthly constraint) but nothing
has ever been FITTED inside the class. A ceiling says what the class could do with a per-cell
oracle that has seen the answer; this says what a network with the deployable feature set
actually reaches. The gap between the two numbers is the cost of features and architecture, and
it is not derivable from either one alone.

**What is predicted.** The streamfunction, not a diffusivity. ``Γ`` lives on layer interfaces
with ``Γ = 0`` at the surface and at the bottom; the layer-centred ``Ψ(k) = ½(Γ(k) + Γ(k+1))``
gives the skew flux ``F(τ) = (Ψ_y ∂_zτ, −Ψ_x ∂_zτ)``. That is exactly the parametrisation of the
``ARM B deployable`` ceiling row, and the algebra is SHARED with it
(:func:`apriori_eval.gamma_to_psi`, :func:`apriori_eval.psi_to_flux`) rather than reimplemented,
so a trained closure and the bound it is compared against cannot drift apart.

**The loss is on the flux DIVERGENCE, not on the flux.** Three reasons, and they agree.

1. It is the reported metric. Fitting one objective and reporting another leaves the reported
   number a lower bound on the class for no reason.
2. A rotational flux has ``∇·F ≡ 0``, so the divergence removes exactly the component of the
   teacher's flux that moves no tracer. The indeterminacy of the rotational/divergent split in a
   bounded domain (Fox-Kemper, Ferrari & Pedlosky, JPO 33, 478, 2003) makes any statistic
   computed on the raw flux gauge-dependent; the divergence is not. Measured here: the teacher's
   flux has ``η = |∇·F|·Δ/|F| = 0.466`` against 0.610 for a down-gradient reference field
   (``divergence_fit.py``), so the rotational excess is real and worth removing.
3. It is what the closest published analogue does. Perezhogin et al. (2025, GRL
   10.1029/2025GL117046) train on ``‖S − ∇·T̂‖²/‖S‖²`` for the subfilter momentum flux tensor,
   for the same reason.

⚠️ The ceilings in ``apriori_eval`` are fitted on the FLUX and scored on the divergence, so they
are lower bounds on their own classes under this objective. A trained network scoring near a
ceiling is therefore not evidence the class is exhausted. Say so wherever the two are compared.

⚠️ **The divergence here is HORIZONTAL ONLY, and that is a real gap, not a rounding.** The coarse
products carry no vertical eddy flux, so ``∇·F`` is missing the ``∂_z(w'τ')`` term and anything
fitted against it absorbs that omission — the same warning ``flux_divergence`` carries. It
matters more for Arm B than it did for Arm A: a bolus streamfunction fixes the vertical bolus
velocity too (``w* = ∇_h·Ψ``), so the vertical term is not free once Ψ is chosen. The teacher's
new ``wT``/``wS`` streams, first written 2026-08-09, therefore arrive as a FALSIFIABLE PREDICTION
of whatever this script fits — not as extra freedom to fit with. Complete the objective with them
before quoting a final number.

**Output scaling is local**, following the published finding that a global normalisation
destroys generalisation across resolution (Perezhogin et al.: R² −2.71 at an unseen grid spacing
with fixed coefficients, +0.87 with local scaling). The scale used is the magnitude of the
model's own GM streamfunction,

    Ψ_scale = fer_K · |S_tapered|                   [m²/s]

so the network only ever emits an O(1) number and ``a = 1, b = 0`` reproduces default GM exactly.
Two purely dimensional alternatives were tried first and both failed — one by four orders of
magnitude, one through an unbounded tail the median could not see. :func:`psi_scale` carries the
measurements; read it before replacing the scale.

**The DIRECTION is supplied, not learned**, and it has to be: all seven features are rotationally
invariant scalars, so nothing in the input distinguishes east from north and a 2-vector emitted
straight from them would be one fixed direction in the grid frame. The network's two outputs are
coefficients in the local neutral-slope frame instead,

    Γ = Ψ_scale · ( a · ŝ⊥ + b · ŝ ) ,     ŝ⊥ = (−ŝ_y, ŝ_x)

which is equivariant under rotation by construction and puts **Arm A exactly at ``b = 0, a ≥ 0``**
— GM is ``Ψ = κ(−S_y, S_x) = κ|S| ŝ⊥``. What Arm B buys is then two nameable things: the ``ŝ``
component, and the sign of ``a``. See :func:`slope_frame`.

⚠️ **Γ is UNBOUNDED here and that is a deployment question, not an oversight.** Arm A's positive
bounded multiplier makes the GM term an APE sink no matter what the weights do; a free
streamfunction has no such guarantee and can inject energy. The principled bounded form is
GEOMETRIC (Marshall, Maddison & Berloff, Ocean Modelling 52-53, 2012; implemented in Mak et al.,
JPO 48, 2018), where the eddy stress tensor's norm is bounded by the eddy energy and the
remaining unknowns are non-dimensional and bounded by one. Offline skill first, then that.

Training is per MARK, on the whole field: the divergence couples horizontal neighbours, so a
scattered sample of cells cannot be used. One mark is one gradient step.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fesom_jax  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_got = Path(fesom_jax.__file__).resolve()
if ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not from {ROOT}")

from fesom_jax import ale, gm_nn  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.state import State  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apriori_eval as _ae  # noqa: E402
import features_from_product as _fp  # noqa: E402
import flux_divergence as _fd  # noqa: E402
import kappa_hat as _kh  # noqa: E402
import nn_data as _nd  # noqa: E402

N_OUT = 2      # (a, b): the coefficients on the two directions of the local slope frame

SPINUP_EXCLUDE = ("1958_01", "1958_02", "1958_03")


# ===========================================================================
# the Arm B head — Γ from features, and the flux it implies
# ===========================================================================
def psi_scale(mesh, d):
    """``fer_K · |S_tapered|`` at layer centres — the magnitude of the model's OWN GM
    streamfunction, in m²/s, and the scale the network's O(1) output is multiplied by.

    It is the GM ansatz's own ``Ψ = κ·S`` with both factors taken from the student's operator, so
    ``a = 1, b = 0`` in the slope frame reproduces default GM exactly and Arm A is the sub-family
    ``b = 0, a ≥ 0`` with ``a`` bounded. Only the MAGNITUDE is borrowed; direction and sign stay
    free, which is the whole of what Arm B adds.

    ⚠️ **Two purely dimensional alternatives were tried first and both failed, for different
    reasons. Do not reinstate either without reading this.**

    ``Δ²M²/N`` (Visbeck) is the scale of a DIFFUSIVITY, not of a streamfunction — too large by a
    factor of the slope, ~1e4 — and the fit diverged to an explained variance of −7.7e+10 inside
    one epoch (job 26831859). Units alone cannot catch this: both combinations are m²/s.

    ``Δ²M⁴/N³``, that same expression times the slope, has the right dimensions AND the right
    median: measured 1.09 m²/s against an oracle Ψ of 0.089, a calibration of 0.08. It is still
    unusable, because **``N³`` in the denominator gives it a tail the median cannot see**:
    measured on 1959_03, p99 = 1.15e+09 and max = 8.4e+12 m²/s, with 294,445 cells — 8 % of the
    domain — above 1e+06. A least-squares objective is decided by that tail, and the fit diverged
    to −3.9e+18 (job 26831947) even though the network's output was an order of magnitude BELOW
    the oracle's in the median. Weak stratification under a finite lateral buoyancy gradient is
    exactly the regime the ODM95 taper exists to handle, and a hand-rolled dimensional scale
    simply does not have it.

    So the scale is taken from the operator that already solved this. ``fer_K`` is floored and
    depth-scaled by ``init_redi_gm``; ``slope_tapered`` is the ODM95-tapered slope the student
    actually applies. Both are bounded by construction.

    The assembly prints ``median |Ψ_oracle| / Ψ_scale`` AND the scale's own p99.9 for every mark,
    so any future change to this expression is checked against the answer and against its tail,
    not against an argument."""
    nl = mesh.nl
    slope = np.linalg.norm(np.asarray(d.slope_tapered)[:, :nl - 1, :2], axis=-1)
    return np.asarray(d.fer_K_lay) * slope                            # [m²/s]


def _to_interfaces(field_lay, xp=jnp):
    """Layer-centred field to the interface ABOVE each layer: ``½(x[k−1] + x[k])``, with the
    surface interface taking ``x[0]`` (it carries Γ = 0 and is masked out regardless).

    Same up-shift rule ``gm.fer_solve_gamma`` uses for ``sigma_xy``; kept identical so the
    staggering convention has one definition in this project."""
    above = xp.concatenate([field_lay[:, :1], field_lay[:, :-1]], axis=1)
    return 0.5 * (above + field_lay)


def slope_frame(mesh, d):
    """The unit horizontal neutral-slope direction ``ŝ`` at layer centres, ``(N, nl-1, 2)``.

    **Why the network cannot be allowed to emit Γ in the model's x–y frame.** Every one of the
    seven features is a rotationally INVARIANT scalar — ``|s|``, ``N²``, ``|∇_h b|``, ``f``,
    depth, depth fraction, strain. Nothing in the input distinguishes east from north. A network
    emitting a 2-vector from such inputs can therefore only produce one fixed direction in the
    grid's own frame times a scalar, which is not a physical field and would not survive a
    rotation of the coordinate system. Arm A never had this problem because its two outputs are
    scalars and the direction comes from the operator.

    So the direction is supplied here and the network only chooses coefficients in it:

        Γ = Ψ_scale · ( a · ŝ⊥ + b · ŝ ) ,    ŝ⊥ = (−ŝ_y, ŝ_x)

    with ``(a, b)`` the two network outputs, free in sign. That is equivariant by construction —
    rotate the frame and Γ rotates with it — and it places **Arm A exactly at ``b = 0, a ≥ 0``**,
    since GM's ``Ψ = κ(−S_y, S_x) = κ|S| ŝ⊥``. The extra freedom Arm B buys is then visible as
    two specific things: the ``ŝ`` component, and the sign of ``a``.

    Where the slope vanishes the frame is undefined; ``Ψ_scale = fer_K·|S_tapered|`` vanishes
    there too, so Γ goes to zero rather than to an arbitrary direction. The guarded norm below
    only stops a 0/0."""
    nl = mesh.nl
    s = np.asarray(d.slope_tapered)[:, :nl - 1, :2]
    n = np.linalg.norm(s, axis=-1, keepdims=True)
    return np.where(n > 0.0, s / np.where(n > 0.0, n, 1.0), 0.0)


def gamma_of(nn, feats_iface, scale_iface, shat_iface, free):
    """``Γ`` at interfaces from the network, in the local slope frame. ``free`` is the
    interior-interface mask.

    ``Γ`` is zeroed at the surface interface and at and below the bottom one, which IS the
    boundary condition — imposed by construction rather than by a penalty, so it holds exactly
    for every weight vector including the untrained one."""
    ab = gm_nn.mlp_raw(nn, feats_iface)                               # (N, nlay, 2) = (a, b)
    a, b = ab[..., 0], ab[..., 1]
    sx, sy = shat_iface[..., 0], shat_iface[..., 1]
    gx = a * (-sy) + b * sx
    gy = a * sx + b * sy
    gamma = scale_iface[..., None] * jnp.stack([gx, gy], axis=-1)
    return jnp.where(free[..., None], gamma, 0.0)


def init_head(key, hidden, init: str):
    """The network, started either at default GM or at no transport at all.

    ``init="gm"`` zeroes the last layer's WEIGHTS and sets its BIAS to ``(a, b) = (1, 0)``, so
    the untrained network emits ``Γ = Ψ_scale·ŝ⊥ = fer_K·|S|·ŝ⊥`` — the GM bolus streamfunction
    itself. Same principle as ``init_gm_nn(zero_last=True)`` on the Arm A side, where the
    untrained network reproduces default GM bit-for-bit: the fit starts at the closure being
    replaced, so every reported gain is measured against the thing actually deployed today, and
    an untrained weight file is a safe fallback rather than noise.

    ⚠️ It is the LOCAL GM streamfunction, not FESOM's. FESOM solves the Ferrari elliptic problem
    ``∂z(C ∂z Γ) − N²Γ = (g/ρ₀)∇σ·fer_K`` for Γ, which smooths the local ``κ·S`` down the column
    and enforces the boundary conditions through the operator. ``a = 1`` here is the unsmoothed
    version, so it is a good starting point and a fair baseline, but do not call it "the model's
    Γ" in a table.

    ``init="zero"`` starts at no bolus transport at all, which makes the fit's first reported
    explained variance exactly 0.000.

    ⚠️ **``zero`` is the DEFAULT, and ``gm`` is the worse starting point — measured, not assumed.**
    Job 26832107 started at ``(1, 0)`` and reported an explained variance of **−30.1** on train and
    −19.9 on holdout, i.e. the GM streamfunction at full strength is thirty times further from the
    teacher's transport than supplying nothing. That is not a contradiction with the ``formula``
    row scoring −0.169: that row is GM **plus Redi**, and with ``K_GM ≈ K_Redi`` the Redi term
    cancels the GM skew flux almost exactly, leaving horizontal diffusion. A skew-only closure has
    no Redi term to cancel it, so ``a = 1`` deploys the uncancelled half of a closure whose halves
    are meant to cancel. Keep the option: "GM alone, uncancelled" is a number worth being able to
    reproduce."""
    nn = gm_nn.init_gm_nn(key, hidden=hidden, two_scale=False, n_out=N_OUT, zero_last=True)
    if init == "zero":
        return nn
    if init != "gm":
        raise SystemExit(f"--init wants 'gm' or 'zero', got {init!r}")
    bs = list(nn.bs)
    bs[-1] = jnp.asarray([1.0, 0.0], dtype=bs[-1].dtype)
    return dataclasses.replace(nn, bs=tuple(bs))


def predict_flux(nn, b):
    """``(F_T, F_S)`` predicted by the network for one mark."""
    gamma = gamma_of(nn, b["feats_iface"], b["scale_iface"], b["shat_iface"], b["free"])
    psi_x = _ae.gamma_to_psi(gamma[..., 0], b["wet"], jnp)
    psi_y = _ae.gamma_to_psi(gamma[..., 1], b["wet"], jnp)
    return _ae.psi_to_flux(psi_x, psi_y, b["dzT"], b["dzS"], jnp)


def div_pair(mesh, pT, pS, hnode, helem, nl):
    """Horizontal divergence of both tracer fluxes, on layer cells."""
    def one(F):
        return _fd.divergence_of_node_flux(mesh, F[..., 0], F[..., 1], hnode, helem)[:, :nl - 1]
    return one(pT), one(pS)


def make_loss(mesh, hnode, helem):
    """``1 − EV`` of the joint flux divergence about zero, on the usable cells of one mark.

    About ZERO, not about the field mean: the question is what fraction of the teacher's eddy
    transport the closure supplies, so supplying none must score 0 and supplying something
    further from the teacher than nothing must go negative. Identical convention to
    ``apriori_eval.Acc.ev_div``, so the numbers are directly comparable.

    The untrained network (zeroed last layer) predicts Γ ≡ 0, hence flux ≡ 0, hence loss exactly
    1.0 — a starting point that is a measurement, not an accident."""
    nl = mesh.nl

    def loss(nn, b):
        pT, pS = predict_flux(nn, b)
        qT, qS = div_pair(mesh, pT, pS, hnode, helem, nl)
        m = b["usable"]
        w2 = b["div_w"] ** 2
        num = jnp.sum(jnp.where(m, (b["dvT"] - qT) ** 2 + w2 * (b["dvS"] - qS) ** 2, 0.0))
        den = jnp.sum(jnp.where(m, b["dvT"] ** 2 + w2 * b["dvS"] ** 2, 0.0))
        return num / jnp.maximum(den, 1e-300)

    return loss


def make_window_loss(mesh, hnode, helem):
    """``1 − EV`` of the WINDOW-MEAN flux divergence, on the usable cells of one window.

    The construction, and why it is this one. A deployed closure runs on the INSTANTANEOUS state
    every step and the ocean averages what it produces. So the network is applied to each mark's
    own features, its predicted divergence is averaged over the window, and that average is
    compared against the averaged teacher divergence. Running the network once on averaged
    features would be a different and easier problem — a closure that sees a monthly mean state,
    which no closure ever does.

    Why bother: measured on the cached marks (``averaging_scan``, job 26833431), the oracle's
    GM-direction coefficient goes from a signal-to-noise of 0.041 at one five-day mark to 0.204
    over 22 months, and the fraction of cells wanting the GM sign from 0.546 to 0.639, while the
    per-cell ceiling FALLS from 0.621 to 0.494. The predictable part and the ceiling converge, so
    this is the regime where a feature-based closure has something to find."""
    nl = mesh.nl

    def loss(nn, window):
        qT = qS = None
        for b in window:
            pT, pS = predict_flux(nn, b)
            dT, dS = div_pair(mesh, pT, pS, hnode, helem, nl)
            qT = dT if qT is None else qT + dT
            qS = dS if qS is None else qS + dS
        n = len(window)
        qT, qS = qT / n, qS / n
        tT = sum(b["dvT"] for b in window) / n
        tS = sum(b["dvS"] for b in window) / n
        m = window[0]["usable"]
        w2 = window[0]["div_w"] ** 2
        num = jnp.sum(jnp.where(m, (tT - qT) ** 2 + w2 * (tS - qS) ** 2, 0.0))
        den = jnp.sum(jnp.where(m, tT ** 2 + w2 * tS ** 2, 0.0))
        return num / jnp.maximum(den, 1e-300)

    return loss


def group_by_month(marks, batches):
    """``[(month, [batch, ...]), ...]`` in month order, from the ``(month, record)`` list the
    batches were assembled from. The window is a month because that is the unit the marks come
    in; nothing in the loss depends on it being 30 days."""
    out, order = {}, []
    for (m, _r), b in zip(marks, batches):
        if m not in out:
            out[m] = []
            order.append(m)
        out[m].append(b)
    return [(m, out[m]) for m in order]


def ev_split(mesh, nn, b, hnode, helem):
    """Per-tracer explained variance about zero, for reporting. Not the training objective —
    that one is joint, because a single Ψ serves both tracers and letting the two compete for it
    is the physics."""
    nl = mesh.nl
    pT, pS = predict_flux(nn, b)
    qT, qS = div_pair(mesh, pT, pS, hnode, helem, nl)
    m = b["usable"]
    out = {}
    for tag, tgt, pred in (("T", b["dvT"], qT), ("S", b["dvS"], qS)):
        sse = float(jnp.sum(jnp.where(m, (tgt - pred) ** 2, 0.0)))
        tot = float(jnp.sum(jnp.where(m, tgt ** 2, 0.0)))
        out[f"evdiv_{tag}"] = 1.0 - sse / max(tot, 1e-300)
    return out


def basis_divergences(mesh, b, hnode, helem, nl):
    """``(D_a, D_b)`` — the flux divergence produced by ``(a, b) = (1, 0)`` and by ``(0, 1)``.

    The head is LINEAR in its two coefficients: Γ = Ψ_scale(a ŝ⊥ + b ŝ), Ψ is a linear map of Γ,
    the skew flux is linear in Ψ and the divergence is linear in the flux. So the prediction for
    any constant pair is ``a·D_a + b·D_b``, which is what makes :func:`global_ab` an exact solve
    rather than another fit."""
    out = []
    for a, bb in ((1.0, 0.0), (0.0, 1.0)):
        sx, sy = b["shat_iface"][..., 0], b["shat_iface"][..., 1]
        gamma = b["scale_iface"][..., None] * jnp.stack(
            [a * (-sy) + bb * sx, a * sx + bb * sy], axis=-1)
        gamma = jnp.where(b["free"][..., None], gamma, 0.0)
        psi_x = _ae.gamma_to_psi(gamma[..., 0], b["wet"], jnp)
        psi_y = _ae.gamma_to_psi(gamma[..., 1], b["wet"], jnp)
        pT, pS = _ae.psi_to_flux(psi_x, psi_y, b["dzT"], b["dzS"], jnp)
        out.append(div_pair(mesh, pT, pS, hnode, helem, nl))
    return out


def global_ab(mesh, batches, hnode, helem):
    """The best SINGLE pair ``(a, b)`` for the whole ocean, and the explained variance it buys.

    The baseline every feature-based closure has to beat, and the one the Arm A gate already
    insists on in its own form ("a closure that only rescales κ globally would beat the formula
    and mean nothing"). Two numbers for the entire domain, no inputs, no network. If the trained
    network does not clear this, the features are carrying nothing and the fit is an expensive
    way to find two constants.

    Exact, not iterative: the prediction is linear in ``(a, b)``, so this is a 2×2 normal-equation
    solve accumulated over every usable cell of every mark, with the same salinity weight and the
    same about-zero reference the training loss uses."""
    nl = mesh.nl
    M = np.zeros((2, 2))
    r = np.zeros(2)
    tot = 0.0
    for b in batches:
        (aT, aS), (bT, bS) = basis_divergences(mesh, b, hnode, helem, nl)
        m = np.asarray(b["usable"])
        w2 = b["div_w"] ** 2
        cols = [(np.asarray(aT), np.asarray(aS)), (np.asarray(bT), np.asarray(bS))]
        tgt = (np.asarray(b["dvT"]), np.asarray(b["dvS"]))
        for i in range(2):
            for j in range(2):
                M[i, j] += float((cols[i][0] * cols[j][0])[m].sum()
                                 + w2 * (cols[i][1] * cols[j][1])[m].sum())
            r[i] += float((cols[i][0] * tgt[0])[m].sum() + w2 * (cols[i][1] * tgt[1])[m].sum())
        tot += float((tgt[0] ** 2)[m].sum() + w2 * (tgt[1] ** 2)[m].sum())
    ab = np.linalg.solve(M, r) if abs(np.linalg.det(M)) > 0 else np.zeros(2)
    # EV about zero of the joint divergence, = 1 − |target − pred|²/|target|², and with the
    # least-squares solution the numerator collapses to tot − aᵀr.
    ev = float(ab @ r) / max(tot, 1e-300)
    return dict(a=float(ab[0]), b=float(ab[1]), ev_joint=ev)


def oracle_ev(mesh, b, hnode, helem, n_wet):
    """The ``ARM B deployable`` ceiling on the SAME cells, scored the same way.

    Carried alongside every fit because the trained number is meaningless without it: a network
    at 0.30 is a failure against a ceiling of 0.62 and a near-success against a ceiling of 0.35,
    and the ceiling moves with the month.

    ⚠️ This row is fitted on the FLUX and scored on the DIVERGENCE, so it is a lower bound on
    its own class under the training objective used here. It can legitimately be beaten."""
    nl = mesh.nl
    pT, pS = _ae.oracle_psi_constrained(np.asarray(b["dzT"]), np.asarray(b["dzS"]),
                                        np.asarray(b["F_T"]), np.asarray(b["F_S"]),
                                        float(b["flux_w"]), n_wet)
    qT, qS = div_pair(mesh, jnp.asarray(pT), jnp.asarray(pS), hnode, helem, nl)
    m = b["usable"]
    out = {}
    for tag, tgt, pred in (("T", b["dvT"], qT), ("S", b["dvS"], qS)):
        sse = float(jnp.sum(jnp.where(m, (tgt - pred) ** 2, 0.0)))
        tot = float(jnp.sum(jnp.where(m, tgt ** 2, 0.0)))
        out[f"ceiling_evdiv_{tag}"] = 1.0 - sse / max(tot, 1e-300)
    return out


# ===========================================================================
# assembly
# ===========================================================================
def mark_batch(mesh, path: Path, record: int, n_wet, hnode, helem):
    """Everything one gradient step needs, for one teacher mark, as numpy."""
    nl = mesh.nl
    T47, S47, have47, tsec = _fp.load_product_mark(path, record)
    d = _fp.derive(mesh, T47, S47, have47, verbose=False)
    F_T, F_S = _kh.load_fluxes(path, record, nl)
    F_T, F_S = np.asarray(F_T), np.asarray(F_S)

    dzT, dzS = _ae.dz_tracers(mesh, d)
    scale = psi_scale(mesh, d)
    shat = slope_frame(mesh, d)
    feats = np.asarray(d.feats)
    usable = np.asarray(d.usable, bool)

    dvT, dvS = div_pair(mesh, jnp.asarray(F_T), jnp.asarray(F_S), hnode, helem, nl)
    dvT, dvS = np.asarray(dvT), np.asarray(dvS)
    # Two weights, and they are NOT the same number. `flux_w` rescales the salinity components
    # of the FLUX and belongs to the oracle, which is fitted on the flux; `div_w` rescales the
    # DIVERGENCE and belongs to the training loss. Using one for both would silently reweight
    # one of the two objectives.
    div_w = _rms(dvT[usable]) / max(_rms(dvS[usable]), 1e-300)
    # Interface staggering is done ONCE, here, so the cache holds exactly what a gradient step
    # consumes: the layer-centred originals are never needed again and storing both would double
    # a 300 MB-per-mark footprint for nothing.
    # Calibration of the output scale, carried with every mark. The network emits an O(1) number
    # that `scale` multiplies, so if this ratio is not near 1 the network is being asked to make
    # up the difference and the fit is badly conditioned from the first step. The free per-cell
    # oracle is the right yardstick: it is the best Ψ the class can use, so its magnitude is the
    # magnitude the scale should have.
    psi_o = _ae.oracle_psi_vec(dzT, dzS, F_T, F_S, _ae.flux_weight(F_T, F_S))
    mag = np.linalg.norm(psi_o, axis=-1)
    sel = usable & (scale > 0)
    calib = float(np.median(mag[sel] / scale[sel])) if sel.any() else float("nan")
    # The tail, not just the median. A scale with a good median and a 1e9 p99 trains to garbage,
    # and only the tail says so — measured, see the warning on psi_scale.
    scale_p999 = float(np.percentile(scale[sel], 99.9)) if sel.any() else float("nan")

    lay = np.arange(nl - 1)[None, :]
    # The slope VECTOR is averaged to the interface and then re-normalised. Averaging the two
    # unit vectors instead would give a non-unit result wherever the slope turns with depth,
    # and a zero one where it reverses.
    s_if = _to_interfaces(shat, np)
    s_n = np.linalg.norm(s_if, axis=-1, keepdims=True)
    s_if = np.where(s_n > 0.0, s_if / np.where(s_n > 0.0, s_n, 1.0), 0.0)
    return dict(feats_iface=_to_interfaces(feats, np).astype(np.float32),
                shat_iface=s_if.astype(np.float32),
                calib=np.float32(calib), psi_oracle_med=np.float32(np.median(mag[sel])),
                scale_med=np.float32(np.median(scale[sel])),
                scale_p999=np.float32(scale_p999),
                scale_iface=_to_interfaces(scale, np).astype(np.float32),
                dzT=dzT.astype(np.float32), dzS=dzS.astype(np.float32),
                F_T=F_T.astype(np.float32), F_S=F_S.astype(np.float32),
                dvT=dvT.astype(np.float32), dvS=dvS.astype(np.float32),
                usable=usable, wet=lay < n_wet[:, None],
                free=(lay >= 1) & (lay < n_wet[:, None]),
                flux_w=np.float32(_ae.flux_weight(F_T, F_S)),
                div_w=np.float32(div_w), mark=np.int32(_nd.mark_index(tsec)))


_CACHED = ("feats_iface", "scale_iface", "shat_iface", "dzT", "dzS", "F_T", "F_S",
           "dvT", "dvS",
           "usable", "wet", "free", "flux_w", "div_w", "mark",
           "calib", "psi_oracle_med", "scale_med", "scale_p999")


def _rms(v):
    v = np.asarray(v)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0


def to_device(b):
    """One cached mark to the jnp batch the loss consumes.

    The big fields stay float32 on the device; the MLP weights are float64, so the matmul
    promotes and Γ, the flux and the divergence are all computed in double. Keeping the inputs
    single halves a ~600 MB-per-mark residency for a quantity whose own accuracy is set by the
    teacher's float32 output anyway."""
    out = {k: jnp.asarray(b[k]) for k in
           ("feats_iface", "scale_iface", "shat_iface", "dzT", "dzS", "F_T", "F_S",
            "dvT", "dvS", "usable", "wet", "free")}
    out["flux_w"] = float(b["flux_w"])
    out["div_w"] = float(b["div_w"])
    return out


def month_list(coarse_dir: Path) -> list:
    return sorted(p.name[len("coarse_ng5_"):-3]
                  for p in Path(coarse_dir).glob("coarse_ng5_[0-9]*.nc"))


def select_marks(coarse_dir: Path, months, per_month: int):
    """``(month, record)`` pairs, ``per_month`` evenly spaced marks from each month.

    Evenly spaced rather than random so the set is reproducible from the arguments alone and
    covers the seasonal cycle inside a month rather than clustering."""
    import netCDF4
    out = []
    for m in months:
        p = Path(coarse_dir) / f"coarse_ng5_{m}.nc"
        if not p.exists():
            continue
        with netCDF4.Dataset(p) as ds:
            nrec = len(ds.dimensions["time"])
        if nrec == 0:
            continue
        k = min(per_month, nrec)
        idx = np.unique(np.linspace(0, nrec - 1, k).round().astype(int))
        out.extend((m, int(r)) for r in idx)
    return out


def cache_name(month: str, record: int, version: int = 4) -> str:
    """One file per mark, not one per mark LIST.

    A list-keyed cache is thrown away entirely when a single month is added, which is the normal
    way this gets used; per-mark files make adding months cost only the new months. ``version``
    is in the name because a change to what a mark carries — a new field, a different
    stratification floor — must not be served out of a file written before it."""
    return f"armb_mark_v{version}_{month}_r{record}.npz"


def assemble(mesh, marks, coarse_dir: Path, n_wet, hnode, helem, *, cache_dir: Path,
             verbose=True):
    """One numpy batch per mark, each cached on disk on its own."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    out = []
    for m, r in marks:
        f = Path(cache_dir) / cache_name(m, r)
        if f.exists():
            z = np.load(f, allow_pickle=False)
            out.append({k: z[k] for k in _CACHED})
            if verbose:
                print(f"  {m}[{r}]: cache hit", flush=True)
            continue
        t = time.time()
        b = mark_batch(mesh, Path(coarse_dir) / f"coarse_ng5_{m}.nc", r, n_wet, hnode, helem)
        # Write via a per-process temporary name: sibling jobs sharing a cache directory are
        # routine here, and a fixed temp name is a race the moment two of them overlap.
        tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, **b)
        tmp.replace(f)
        out.append(b)
        if verbose:
            print(f"  {m}[{r}] mark {int(b['mark']):>3d}: usable {int(b['usable'].sum()):>8d}, "
                  f"div_w {float(b['div_w']):.3f}, |psi_oracle|/scale {float(b['calib']):.3g} "
                  f"(psi {float(b['psi_oracle_med']):.3g}, scale {float(b['scale_med']):.3g} m2/s, "
                  f"scale p99.9 {float(b['scale_p999']):.3g})"
                  f"  [{time.time()-t:.0f}s]", flush=True)
    return out


# ===========================================================================
# fit
# ===========================================================================
def fit(mesh, batches, hold, *, hidden, seed, epochs, lr, clip, init, hnode, helem,
        windows=None, windows_hold=None, verbose=True):
    import optax
    nn = init_head(jax.random.PRNGKey(seed), hidden, init)
    loss = make_loss(mesh, hnode, helem)
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    state = opt.init(nn)

    @jax.jit
    def step(nn_, st_, b):
        l, g = jax.value_and_grad(loss)(nn_, b)
        upd, st_ = opt.update(g, st_, nn_)
        return optax.apply_updates(nn_, upd), st_, l

    # ⚠️ Build ONE of the two, never both. A cached mark is ~400 MB on the device and there are
    # 132 of them in the time-averaged configuration; materialising the flat list first and then
    # the window list holds 2 x 53 GB at once, which is an OOM rather than a slowdown.
    if windows is not None:
        dev = [[to_device(b) for b in w] for _m, w in windows]
        devh = [[to_device(b) for b in w] for _m, w in windows_hold]
        loss = make_window_loss(mesh, hnode, helem)
    else:
        dev = [to_device(b) for b in batches]
        devh = [to_device(b) for b in hold]
    hist = []
    rng = np.random.default_rng(seed + 1)
    for ep in range(epochs):
        order = rng.permutation(len(dev))
        tot = 0.0
        for i in order:
            nn, state, l = step(nn, state, dev[i])
            tot += float(l)
        tr_ev = 1.0 - tot / max(len(dev), 1)
        ho_ev = None
        if devh:
            ho_ev = 1.0 - float(np.mean([float(loss(nn, b)) for b in devh]))
        hist.append(dict(epoch=ep, train_ev=tr_ev, holdout_ev=ho_ev))
        if verbose:
            print(f"  epoch {ep:>3d}  train EV(div) {tr_ev:+.4f}"
                  + (f"   holdout {ho_ev:+.4f}" if ho_ev is not None else ""), flush=True)
    return nn, hist, dev, devh


def save_armb(path: Path, nn, hidden, seed):
    """Weights plus the tag that stops an Arm A loader accepting this file.

    The two arms emit the same shaped array from the same MLP code and mean entirely different
    things by it — a bounded log-multiplier on κ versus a streamfunction in m²/s. Nothing in the
    numbers distinguishes them, so the file says which it is."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = dict(kind="armB_gamma", n_layers=len(nn.Ws), hidden=np.asarray(hidden),
               seed=int(seed))
    for i, (W, b) in enumerate(zip(nn.Ws, nn.bs)):
        out[f"W{i}"] = np.asarray(W)
        out[f"b{i}"] = np.asarray(b)
    np.savez(path, **out)


def load_armb(path: Path):
    z = np.load(path, allow_pickle=False)
    kind = str(z["kind"]) if "kind" in z else "<absent>"
    if kind != "armB_gamma":
        raise SystemExit(f"{path} is not an Arm B streamfunction network (kind={kind!r}); an "
                         f"Arm A multiplier file would load silently and be scaled as m²/s")
    n = int(z["n_layers"])
    return gm_nn.GmNN(Ws=tuple(jnp.asarray(z[f"W{i}"]) for i in range(n)),
                      bs=tuple(jnp.asarray(z[f"b{i}"]) for i in range(n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coarse-dir", type=Path, default=_nd.COARSE_DIR)
    ap.add_argument("--train", default="", help="comma-separated months; empty = all landed "
                                                "months except the spin-up cut and the holdout")
    ap.add_argument("--holdout", default="1960_11,1960_12")
    ap.add_argument("--marks-per-month", type=int, default=2)
    ap.add_argument("--hidden", default="16,16")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="global-norm gradient clip. The loss is quadratic in an unbounded "
                         "output, so a single badly scaled mark can take one step far enough "
                         "that adam never recovers; the clip bounds that step without changing "
                         "the objective.")
    ap.add_argument("--init", default="zero", choices=("zero", "gm"),
                    help="where the fit starts. 'zero' (default) at no bolus transport, so the "
                         "first explained variance is exactly 0.000 and every gain is measured "
                         "from nothing. 'gm' at the GM streamfunction (a=1, b=0) — measured at "
                         "EV -30.1, because a skew-only closure has no Redi term to cancel it")
    ap.add_argument("--time-average", action="store_true",
                    help="fit the WINDOW-MEAN divergence: the network runs on each mark's own "
                         "features, its predicted divergence is averaged over the month, and "
                         "that is compared against the averaged teacher divergence. One month "
                         "per gradient step. This is the regime the predictable signal lives in "
                         "— see make_window_loss and averaging_scan.")
    ap.add_argument("--baseline-only", action="store_true",
                    help="compute the no-features (a, b) baseline on the cached marks and stop; "
                         "no training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/work/ba1583/a270088/nn_eddy/armb"))
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    hold_months = [m for m in args.holdout.split(",") if m.strip()]
    if args.train.strip():
        train_months = [m for m in args.train.split(",") if m.strip()]
    else:
        train_months = [m for m in month_list(args.coarse_dir)
                        if m not in SPINUP_EXCLUDE and m not in hold_months]
    overlap = set(train_months) & set(hold_months)
    if overlap:
        raise SystemExit(f"train and holdout share months: {sorted(overlap)}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "holdout_declared.json").write_text(json.dumps(
        dict(holdout=hold_months, train=train_months, marks_per_month=args.marks_per_month,
             declared_before_fitting=True), indent=2))

    t0 = time.time()
    print(f"[plan] train {len(train_months)} months, holdout {hold_months}, "
          f"{args.marks_per_month} marks/month", flush=True)
    mesh = load_mesh(_fp.MESH_DIR)
    n_wet = _ae.wet_layer_count(mesh)
    hnode = State.rest(mesh).hnode
    _, helem = ale.commit_thickness(mesh, hnode)
    cache_dir = args.cache_dir or (args.out.parent / "armb_cache")

    tr_marks = select_marks(args.coarse_dir, train_months, args.marks_per_month)
    ho_marks = select_marks(args.coarse_dir, hold_months, args.marks_per_month)
    print(f"[plan] {len(tr_marks)} train marks, {len(ho_marks)} holdout marks", flush=True)
    batches = assemble(mesh, tr_marks, args.coarse_dir, n_wet, hnode, helem,
                       cache_dir=cache_dir)
    hold = assemble(mesh, ho_marks, args.coarse_dir, n_wet, hnode, helem, cache_dir=cache_dir)

    # The no-features baseline, on the SAME cells, before anything is trained. Two constants for
    # the whole ocean; a network that does not clear this has learned nothing from its inputs.
    g_tr = global_ab(mesh, [to_device(b) for b in batches], hnode, helem)
    g_ho = global_ab(mesh, [to_device(b) for b in hold], hnode, helem)
    print(f"[baseline] best single (a, b) on train: ({g_tr['a']:+.4f}, {g_tr['b']:+.4f})  "
          f"joint EV {g_tr['ev_joint']:+.4f}", flush=True)
    print(f"[baseline] the same pair refitted on holdout: ({g_ho['a']:+.4f}, {g_ho['b']:+.4f})  "
          f"joint EV {g_ho['ev_joint']:+.4f}", flush=True)
    if args.baseline_only:
        (args.out / f"armb_baseline_seed{args.seed}.json").write_text(
            json.dumps(dict(train=g_tr, holdout=g_ho, train_months=train_months,
                            holdout_months=hold_months), indent=2))
        print("M3_ARMB_FIT_DONE", flush=True)
        return

    hidden = tuple(int(x) for x in args.hidden.split(","))
    win = win_hold = None
    if args.time_average:
        win = group_by_month(tr_marks, batches)
        win_hold = group_by_month(ho_marks, hold)
        sizes = sorted({len(w) for _m, w in win} | {len(w) for _m, w in win_hold})
        print(f"[plan] TIME-AVERAGED: {len(win)} train windows, {len(win_hold)} holdout, "
              f"{sizes} marks each", flush=True)
        if min(sizes) < 2:
            raise SystemExit("a window of one mark is not an average; pass --marks-per-month 6")
    nn, hist, dev, devh = fit(mesh, batches, hold, hidden=hidden, seed=args.seed,
                              epochs=args.epochs, lr=args.lr, clip=args.clip,
                              init=args.init, hnode=hnode, helem=helem,
                              windows=win, windows_hold=win_hold)

    if args.time_average:
        wl = make_window_loss(mesh, hnode, helem)
        rows = [dict(window=i, ev_joint=1.0 - float(wl(nn, w))) for i, w in enumerate(devh)]
        for r in rows:
            print(f"  window {r['window']}: joint EV(div) {r['ev_joint']:+.4f}", flush=True)
        rec = dict(train_months=train_months, holdout=hold_months, hidden=list(hidden),
                   epochs=args.epochs, lr=args.lr, clip=args.clip, init=args.init,
                   seed=args.seed, time_average=True, history=hist, holdout_windows=rows,
                   holdout_mean_ev=float(np.mean([r["ev_joint"] for r in rows])))
        tag = f"armb_tavg_seed{args.seed}"
        save_armb(args.out / f"gm_nn_{tag}.npz", nn, hidden, args.seed)
        (args.out / f"{tag}.json").write_text(json.dumps(rec, indent=2))
        print(f"[result] holdout window-mean joint EV: {rec['holdout_mean_ev']:+.4f}", flush=True)
        print(f"[done] {time.time()-t0:.0f}s", flush=True)
        print("M3_ARMB_FIT_DONE", flush=True)
        return

    print("[eval] holdout, per tracer, against the deployable ceiling on the same cells",
          flush=True)
    rows = []
    for b, dv in zip(hold, devh):
        r = dict(mark=int(b["mark"]))
        r |= ev_split(mesh, nn, dv, hnode, helem)
        r |= oracle_ev(mesh, dv, hnode, helem, n_wet)
        rows.append(r)
        print(f"  mark {r['mark']:>3d}: network T {r['evdiv_T']:+.4f} S {r['evdiv_S']:+.4f}  |  "
              f"ceiling T {r['ceiling_evdiv_T']:+.4f} S {r['ceiling_evdiv_S']:+.4f}", flush=True)

    tag = f"armb_seed{args.seed}"
    save_armb(args.out / f"gm_nn_{tag}.npz", nn, hidden, args.seed)
    rec = dict(train_months=train_months, holdout=hold_months, hidden=list(hidden),
               epochs=args.epochs, lr=args.lr, clip=args.clip, init=args.init,
               seed=args.seed,
               calib=[float(b["calib"]) for b in batches],
               n_train_marks=len(tr_marks), n_holdout_marks=len(ho_marks),
               baseline_train=g_tr, baseline_holdout=g_ho,
               history=hist, holdout_rows=rows,
               holdout_mean=dict(
                   network_T=float(np.mean([r["evdiv_T"] for r in rows])),
                   ceiling_T=float(np.mean([r["ceiling_evdiv_T"] for r in rows])),
                   network_S=float(np.mean([r["evdiv_S"] for r in rows])),
                   ceiling_S=float(np.mean([r["ceiling_evdiv_S"] for r in rows]))))
    (args.out / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    hm = rec["holdout_mean"]
    print(f"[result] holdout temperature flux divergence: network {hm['network_T']:+.4f}, "
          f"deployable ceiling {hm['ceiling_T']:+.4f}, "
          f"fraction of ceiling {hm['network_T']/hm['ceiling_T']:.3f}"
          if hm["ceiling_T"] > 0 else "[result] ceiling <= 0; fraction not meaningful",
          flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    print("M3_ARMB_FIT_DONE", flush=True)


if __name__ == "__main__":
    main()
