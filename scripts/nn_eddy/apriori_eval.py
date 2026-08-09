"""NN-GM child-plan M3 — the a-priori evaluation that GATE M3 is read from.

Takes one or more weight files from :mod:`offline_fit` and scores them on months that were
named as the holdout BEFORE fitting. Two measures, and they answer different questions.

**1. R² of ``log m``, on the cells the projection accepts.** This is the gate's literal
wording: can the network predict the local multiplier better than the formula (``m ≡ 1``) and
better than the best single global multiplier? Stratified by depth band, latitude band, month
and resolution, because a closure that wins globally by fixing one region is not a closure.

⚠️ Two things about this measure that decide how it may be quoted. R² is taken about each
stratum's OWN mean, so a constant that is optimal within a stratum scores exactly 0.000 there by
construction — a definition, not a measurement; the informative constant is the TRAIN-fitted one
(``--const-from``), a real deployable closure that can score below zero out of sample. And the
target is clipped to the multiplier bound, exactly as the fit clipped it, so this measure asks
"of what the architecture was asked to learn, how much did it learn" and CANNOT see the part of
the ocean the architecture cannot reach. Measure 2 is what sees that.

**2. Explained variance of the eddy flux, on EVERY usable cell.** Measure 1 lives on the small
minority of cells where a down-gradient projection is well posed, and inside that set the flux
is down-gradient by selection. That makes it silent on the question M3b raised — how much of
the teacher's actual flux any bounded-multiplier closure reproduces — and that is the question
the Arm A / Arm B decision turns on. So the closure is run FORWARD, through
:func:`kappa_hat.gm_redi_flux`, the same operator :func:`kappa_hat.project` inverts and the one
``synthetic_check`` validates against a known κ. Its output is compared against the teacher's
own ``uT_eddy``/``vT_eddy`` and salinity pair on every usable cell — no guard, no clipping, no
selection.

The reference is **zero flux**, not the field mean: the honest question is what fraction of the
eddy flux the closure explains, and a closure that explains none of it should score 0 rather
than be flattered by removing a mean it never predicted. The score is therefore unbounded below,
and a NEGATIVE value is meaningful — it says the closure's flux is further from the teacher's
than having no eddy parameterization at all.

Restricted to the seam-deployable rungs. D1 (the two-scale smoothed features already in
``gm_nn.py``) needs a mesh-Laplacian filter running INSIDE the differentiated step, which M1
does not build, so a D1 win here would not be deployable and is not screened.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fesom_jax  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_got = Path(fesom_jax.__file__).resolve()
if ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not from {ROOT}")

from fesom_jax import gm_nn  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features_from_product as _fp  # noqa: E402
import flux_divergence as _fd  # noqa: E402
import kappa_hat as _kh  # noqa: E402
import nn_data as _nd  # noqa: E402

# The five depth bands the gate names. Same edges as kappa_hat's rejection report, so the two
# tables can be read side by side.
DEPTH_EDGES = (100.0, 500.0, 1500.0, 3000.0)
DEPTH_NAMES = ("0-100m", "100-500m", "500-1500m", "1500-3000m", ">3000m")
LAT_EDGES = (-60.0, -30.0, 30.0, 60.0)
LAT_NAMES = ("S of 60S", "60S-30S", "30S-30N", "30N-60N", "N of 60N")
# Grid scale over Rossby radius, Δ·|f|/√fer_C. Below ~1 the mesh resolves the deformation
# radius and the eddies the closure stands in for are partly explicit; above ~10 it is deeply
# unresolved. The network has NO resolution input, so if skill varies across these bands the
# fixed formula is carrying resolution signal the network could be given — that is the plan's
# stated trigger for adding the feature.
RES_EDGES = (1.0, 3.0, 10.0)
RES_NAMES = ("d/Lr<1", "1-3", "3-10", ">10")

OMEGA = 7.292115e-5


class Acc:
    """Streaming sums for one stratum. Nothing is kept per cell — a holdout month carries ~21 M
    usable layer cells across its marks and the evaluation must not be bounded by that."""

    def __init__(self, n_closures: int, n_logm: int | None = None):
        # ``n_logm`` <= ``n_closures``: the two ceiling rows have no multiplier to compare, so
        # measure 1 is undefined for them and they are simply absent from its table. Sizing the
        # arrays differently rather than padding with zeros is deliberate — a zeroed row would
        # come out as a perfect R² of 1.000 for a row that was never evaluated.
        self.n = 0
        self.n_logm = n_closures if n_logm is None else n_logm
        self.sy = np.zeros(2)                   # Σ y over channels (GM, Redi), log m
        self.syy = np.zeros(2)
        self.sse = np.zeros((self.n_logm, 2))   # Σ (y − pred)² per closure, per channel
        self.fn = 0
        self.f_tot = np.zeros(2)                # Σ |F_teacher|², [T, S]
        self.f_sse = np.zeros((n_closures, 2))  # Σ |F_teacher − F_pred|²
        # Σ |F_pred|². Only used for the RMS ratio below, which exists because a closure whose
        # flux is orders of magnitude too big produces a huge NEGATIVE explained variance that
        # reads as a physical result. It happened — an untapered slope in the forward operator
        # put this ratio at ~1e+6 — and a bare score could not have shown it.
        self.f_pred = np.zeros((n_closures, 2))
        # Measure 3, the same accounting on the DIVERGENCE. Only ∇·F moves a tracer, so a flux
        # comparison can always be answered with "the difference is rotational and harmless".
        # This closes that, and it is the metric closest to what the model actually feels.
        self.dn = 0
        self.d_tot = np.zeros(2)
        self.d_sse = np.zeros((n_closures, 2))

    def add_logm(self, y, preds):
        self.n += int(y.shape[0])
        self.sy += y.sum(axis=0)
        self.syy += (y ** 2).sum(axis=0)
        for i, p in enumerate(preds):
            self.sse[i] += ((y - p) ** 2).sum(axis=0)

    def add_flux(self, n, tot, sse, pred_sq):
        self.fn += int(n)
        self.f_tot += tot
        self.f_sse += sse
        self.f_pred += pred_sq

    def add_div(self, n, tot, sse):
        self.dn += int(n)
        self.d_tot += tot
        self.d_sse += sse

    def ev_div(self):
        """Explained variance of ∇·F about ZERO, per closure, as [T, S]."""
        if self.dn == 0:
            return None
        return [[1.0 - s[k] / max(self.d_tot[k], 1e-300) for k in (0, 1)] for s in self.d_sse]

    def rms_ratio(self):
        """RMS |F_pred| / RMS |F_teacher| per closure, as [T, S]. A sane closure is O(1)."""
        if self.fn == 0:
            return None
        return [[float(np.sqrt(s[k] / max(self.f_tot[k], 1e-300))) for k in (0, 1)]
                for s in self.f_pred]

    def r2(self):
        """R² about this stratum's own mean, per closure, pooled over the two channels."""
        if self.n < 2:
            return None
        ss_tot = float((self.syy - self.sy ** 2 / self.n).sum())
        if ss_tot <= 0.0:
            return None
        return [1.0 - float(s.sum()) / ss_tot for s in self.sse]

    def ev(self):
        """Explained variance of the flux about ZERO, per closure, as [T, S]."""
        if self.fn == 0:
            return None
        return [[1.0 - s[k] / max(self.f_tot[k], 1e-300) for k in (0, 1)] for s in self.f_sse]


def _sq(v):
    """|v|² summed over the trailing vector axis."""
    return np.sum(np.asarray(v) ** 2, axis=-1)


def strata(mesh, d):
    """Per-cell stratum index arrays, all shaped (N, nl-1)."""
    nl = mesh.nl
    N = mesh.nod2D
    ones = np.ones((N, nl - 1))
    z = np.abs(np.asarray(mesh.Z))[None, :nl - 1] * ones
    # GEOGRAPHIC latitude and the mesh's own Coriolis — see nn_data.geo_latitude_deg. The
    # rotated coordinate is what comes to hand and it moves 22.8 % of nodes into a different band.
    lat = _nd.geo_latitude_deg(mesh)[:, None] * ones
    # Δ·|f|/√fer_C — grid scale over Rossby radius. Column-constant, broadcast down the column.
    delta = np.sqrt(np.maximum(np.asarray(mesh.areasvol)[:, 0], 0.0))
    f = np.abs(np.asarray(mesh.coriolis_node))
    c = np.sqrt(np.maximum(np.asarray(d.fer_C), 0.0))
    ratio = (delta * f / np.maximum(c, 1e-6))[:, None] * ones
    return dict(depth=np.digitize(z, DEPTH_EDGES),
                lat=np.digitize(lat, LAT_EDGES),
                res=np.digitize(ratio, RES_EDGES))


def _selectors(st, month):
    """(accumulator key, boolean mask) for every stratum a mark contributes to."""
    everywhere = np.ones(st["depth"].shape, bool)
    yield "all", everywhere
    yield f"month:{month}", everywhere
    for i, nm in enumerate(DEPTH_NAMES):
        yield f"depth:{nm}", st["depth"] == i
    for i, nm in enumerate(LAT_NAMES):
        yield f"lat:{nm}", st["lat"] == i
    for i, nm in enumerate(RES_NAMES):
        yield f"res:{nm}", st["res"] == i


def make_closures(weights, const_log_m):
    """``(name, fn)`` for every closure compared, where ``fn(feats, base_gm, base_redi)``
    returns ``(kappa_gm, kappa_redi)``. The NN rungs, the formula and the constant all go
    through one interface, so no closure gets a private code path into the scoring."""
    out = []
    for tag, path in weights:
        nn = gm_nn.load_gm_nn(path)

        def fn(feats, bg, br, _nn=nn):
            lm = np.asarray(_nn.log_m_max * jnp.tanh(gm_nn.mlp_raw(_nn, jnp.asarray(feats))))
            return bg * np.exp(lm[..., 0]), br * np.exp(lm[..., 1])
        out.append((tag, fn))
    out.append(("formula (m=1)", lambda f, bg, br: (bg, br)))
    if const_log_m is not None:
        cg, cr = float(const_log_m[0]), float(const_log_m[1])
        out.append((f"const m={np.exp(cg):.2f}/{np.exp(cr):.2f}",
                    lambda f, bg, br, _g=cg, _r=cr: (bg * np.exp(_g), br * np.exp(_r))))
    return out


def oracle_kappa(mesh, d, F_T, F_S):
    """The BEST κ the down-gradient family can possibly use, per cell.

    This is not a closure and it is not trainable — it is the CEILING of Arm A, and it exists
    because every other row in this report confounds two questions. A trained network scoring
    badly could mean the network is too small, the features too poor, the training too short, or
    the bound too tight. This row removes all four at once: it hands the family a per-cell
    oracle, unbounded above and below except for the sign, that has already seen the answer.
    **Whatever it scores, no network, no architecture and no amount of training inside Arm A can
    beat**, so a low value here closes the arm independently of everything else in the pipeline.

    The family is ``F(τ) = K_GM·A_τ + K_Redi·B_τ`` with ``A_τ = S ∂_zτ`` and
    ``B_τ = −(∇_hτ + S ∂_zτ)`` — exactly :func:`kappa_hat.gm_redi_flux`, on the tapered slope
    the student applies. Two unknowns against FOUR components (x and y, of the T flux and the S
    flux), so the fit is genuinely overdetermined; solving T and S separately would give two
    free parameters against a 2-vector, fit every cell exactly, and mean nothing. That
    degeneracy is the reason the two tracers are fitted jointly here, and it is why this row is
    an honest bound rather than a tautology.

    The salinity components are rescaled by the ratio of the two tracers' global flux RMS before
    the least squares, so the answer is not simply whichever tracer carries the larger numbers.
    Non-negativity is imposed exactly (2-variable active set, not a clip of the unconstrained
    solution): κ < 0 is up-gradient, which a positive bounded multiplier on a positive base
    cannot reach, so allowing it would bound a family that is not Arm A.
    """
    nl = mesh.nl
    cfg = _fp.GM_CFG
    return oracle_kappa_from_basis(*gm_redi_basis(mesh, d)[:4], F_T, F_S,
                                   flux_weight(F_T, F_S))


def flux_weight(F_T, F_S):
    """Rescaling that puts the two tracers' flux components on one footing before any joint
    least squares. Without it the answer is decided by whichever tracer carries the larger
    numbers, which is a unit choice, not physics."""
    rms_T = np.sqrt(max(float(np.mean(_sq(F_T))), 1e-300))
    rms_S = np.sqrt(max(float(np.mean(_sq(F_S))), 1e-300))
    return rms_T / rms_S if rms_S > 0 else 1.0


def dz_tracers(mesh, d):
    """``(∂_zT, ∂_zS)`` at layer centres — the ONE vertical tracer derivative in the Arm B path.

    Split out because :mod:`armb_fit` needs exactly these two fields and nothing else from
    :func:`gm_redi_basis`, and a second implementation of the same derivative in the trainer
    would let the trained closure and the ceiling it is measured against disagree about what
    ``∂_zτ`` means without either side being wrong on its own."""
    nl = mesh.nl
    return (np.asarray(_kh.to_layers(_kh.vertical_derivative(mesh, d.T), nl)),
            np.asarray(_kh.to_layers(_kh.vertical_derivative(mesh, d.S), nl)))


def gamma_to_psi(gamma, wet, xp=np):
    """Layer-centred ``Ψ`` from interface ``Γ``, ONE component, layers along the last axis.

    ``gamma[..., k]`` is Γ at the interface ABOVE layer ``k``, so ``Ψ(k) = ½(Γ(k) + Γ(k+1))``
    with Γ below the last wet layer taken as zero — the bottom boundary condition. The caller
    is responsible for Γ already being zero outside the wet range and at the surface; this only
    does the averaging and re-masks.

    ``xp`` is ``numpy`` for the ceilings and ``jax.numpy`` for the trainer. Shared rather than
    duplicated so a trained network and the ceiling it is compared against cannot drift into two
    different definitions of the same streamfunction."""
    g_below = xp.concatenate([gamma[..., 1:], xp.zeros_like(gamma[..., :1])], axis=-1)
    return xp.where(wet, 0.5 * (gamma + g_below), 0.0)


def psi_to_flux(psi_x, psi_y, dzT, dzS, xp=np):
    """The skew flux of a bolus streamfunction: ``F(τ) = (Ψ_y ∂_zτ, −Ψ_x ∂_zτ)``.

    This sign and component convention is the Arm B family's definition and it lives here only.
    Skew rather than advective ``u*·τ`` because the teacher's ``agg(uτ) − agg(u)agg(τ)`` is
    unchanged by adding a constant to τ and the skew form inherits that invariance — an advective
    comparison would change with the temperature unit."""
    pT = xp.stack([psi_y * dzT, -psi_x * dzT], axis=-1)
    pS = xp.stack([psi_y * dzS, -psi_x * dzS], axis=-1)
    return pT, pS


def gm_redi_basis(mesh, d):
    """``(A_T, B_T, A_S, B_S, dzT, dzS)`` — the fields both ceilings are fitted against.

    ``A_τ = S ∂_zτ`` and ``B_τ = −(∇_hτ + S ∂_zτ)`` are the GM and Redi directions on the
    TAPERED slope the student applies, so ``F(τ) = K_GM·A_τ + K_Redi·B_τ`` is exactly
    :func:`kappa_hat.gm_redi_flux`. ``∂_zτ`` is returned as well because the Arm B ceiling needs
    it and recomputing it there would be a second implementation of the same derivative.

    Split out of the oracle so the time-averaged path can average the BASIS over a month and fit
    one coefficient to the monthly mean. Fitting per mark and averaging the answers would be a
    different and much more permissive object — a closure that may change its coefficient every
    five days — and would not bound a parameterization at all."""
    nl = mesh.nl
    cfg = _fp.GM_CFG
    slope = np.asarray(d.slope_tapered[..., :2])
    dT, dS = dz_tracers(mesh, d)
    gT = np.asarray(_kh._tracer_gradient(mesh, d.T, cfg))[:, :nl - 1, :]
    gS = np.asarray(_kh._tracer_gradient(mesh, d.S, cfg))[:, :nl - 1, :]
    A_T = slope * dT[..., None]
    A_S = slope * dS[..., None]
    return A_T, -(gT + A_T), A_S, -(gS + A_S), dT, dS


def oracle_kappa_from_basis(A_T, B_T, A_S, B_S, F_T, F_S, w):
    """Non-negative least squares for ``(K_GM, K_Redi)``: two unknowns, four components."""
    A = np.concatenate([A_T, w * A_S], axis=-1)
    B = np.concatenate([B_T, w * B_S], axis=-1)
    Y = np.concatenate([np.asarray(F_T), w * np.asarray(F_S)], axis=-1)
    kg, kr = nnls2(np.sum(A * A, axis=-1), np.sum(A * B, axis=-1), np.sum(B * B, axis=-1),
                   np.sum(A * Y, axis=-1), np.sum(B * Y, axis=-1))
    return kg, kr


def oracle_kappa_flux(A_T, B_T, A_S, B_S, F_T, F_S, w):
    """The Arm A ceiling as a PREDICTED FLUX pair, so it can be scored like any other row."""
    kg, kr = oracle_kappa_from_basis(A_T, B_T, A_S, B_S, F_T, F_S, w)
    return (kg[..., None] * A_T + kr[..., None] * B_T,
            kg[..., None] * A_S + kr[..., None] * B_S)


def oracle_psi(dzT, dzS, F_T, F_S, w):
    """The BEST bolus streamfunction the ARM B family can possibly use, per cell.

    The counterpart of :func:`oracle_kappa`, built so the two are comparable rather than merely
    both called "ceiling". GM's eddy flux written as a skew flux is ``F(τ) = Ψ × ∇τ``, whose
    horizontal components are ``(Ψ_y ∂_zτ, −Ψ_x ∂_zτ)``. Arm A is the special case
    ``Ψ = κ·(−S_y, S_x)`` with κ >= 0 — the vector tied to the neutral slope, the coefficient
    positive. **Arm B frees exactly that**: Ψ is any 2-vector, any direction, any sign.

    So the two ceilings have **identical degrees of freedom — two free numbers per cell —
    fitted to the identical four components** (x and y of the T flux and the S flux). The only
    difference between them is whether the flux direction is chained to the density gradient.
    That is the whole Arm A / Arm B question, and this pair of rows is it, measured.

    Offset invariance is why the skew form is used and not the advective one. The teacher's
    ``agg(uτ) − agg(u)agg(τ)`` is unchanged if a constant is added to τ, but an advective flux
    ``u*·τ`` is not — with temperature in Celsius the zero point is arbitrary, so an advective
    comparison would depend on the unit. The skew form only ever sees ``∂_zτ`` and inherits the
    teacher's invariance.

    ⚠️ **This is an UPPER bound on Arm B, and slightly loose.** A deployable Ψ comes from the
    tridiagonal solve with Ψ = 0 at the surface and the bottom, which ties the levels of a
    column together and forces the depth integral of the bolus transport to vanish. Nothing here
    imposes that; each cell gets its own Ψ. If this row is LOW, Arm B is closed and the looseness
    does not matter. If it is high, the deployable fraction of it is a further question.

    ``w`` rescales the salinity components so neither tracer dominates by carrying larger
    numbers. Each component of Ψ then has one unknown against two equations, so the fit stays
    overdetermined and the closed form below is the least-squares solution."""
    psi = oracle_psi_vec(dzT, dzS, F_T, F_S, w)
    return psi_to_flux(psi[..., 0], psi[..., 1], dzT, dzS)


def oracle_psi_vec(dzT, dzS, F_T, F_S, w):
    """The free per-cell ``Ψ`` ITSELF, ``(..., 2) = (Ψ_x, Ψ_y)`` in m²/s.

    :func:`oracle_psi` turns this into a flux. It is split out so a caller can look at the
    streamfunction rather than at what it transports — ``armb_fit`` calibrates the scale of its
    network's output against ``|Ψ|`` here, and recovering Ψ by dividing a flux by ``∂_zτ`` would
    be a division by something that legitimately passes through zero."""
    den = dzT * dzT + w * w * dzS * dzS
    den = np.where(den > 0.0, den, 1.0)
    psi_y = (dzT * F_T[..., 0] + w * w * dzS * F_S[..., 0]) / den
    psi_x = -(dzT * F_T[..., 1] + w * w * dzS * F_S[..., 1]) / den
    return np.stack([psi_x, psi_y], axis=-1)


def _thomas(lo, di, up, rhs):
    """Batched tridiagonal solve, columns along the last axis. Shapes ``(..., K)``.

    Plain Thomas: the systems here are symmetric with non-negative weights on the diagonal and
    are diagonally dominant by construction (``diag_j = 0.5(c_j + c_{j-1})`` against off-diagonals
    ``0.5 c_j`` and ``0.5 c_{j-1}``), so no pivoting is needed. Rows outside a column's wet range
    are set to the identity with a zero right-hand side by the caller, which keeps every system
    the same length without letting the dry part influence the wet part."""
    K = di.shape[-1]
    cp = np.zeros_like(di)
    dp = np.zeros_like(di)
    beta = di[..., 0].copy()
    beta = np.where(np.abs(beta) > 0, beta, 1.0)
    cp[..., 0] = up[..., 0] / beta
    dp[..., 0] = rhs[..., 0] / beta
    for k in range(1, K):
        beta = di[..., k] - lo[..., k] * cp[..., k - 1]
        beta = np.where(np.abs(beta) > 1e-300, beta, 1.0)
        cp[..., k] = up[..., k] / beta
        dp[..., k] = (rhs[..., k] - lo[..., k] * dp[..., k - 1]) / beta
    x = np.zeros_like(di)
    x[..., K - 1] = dp[..., K - 1]
    for k in range(K - 2, -1, -1):
        x[..., k] = dp[..., k] - cp[..., k] * x[..., k + 1]
    return x


def oracle_psi_constrained(dzT, dzS, F_T, F_S, w, n_wet, ridge=0.0):
    """The Arm B ceiling with the DEPLOYABLE streamfunction's structure imposed.

    :func:`oracle_psi` gives every cell its own free Ψ, which bounds Arm B from above but is not
    something the model could run: a real streamfunction lives on interfaces, vanishes at the
    surface and at the bottom, and the bolus velocity is its vertical difference, so the levels of
    a column are not independent. This imposes exactly that, and the difference between the two
    rows is the price of the constraint.

    Parametrisation. ``Γ`` sits on interfaces with ``Γ(0) = Γ(n_wet) = 0``; the layer-centred
    streamfunction the skew flux uses is ``Ψ(k) = ½(Γ(k) + Γ(k+1))``. Per column and per
    horizontal direction that leaves ``n_wet − 1`` unknowns against ``2·n_wet`` residuals (the T
    and S flux components), so the fit stays overdetermined by about two — the same order as the
    unconstrained row, which is what keeps the two comparable.

    The normal equations are tridiagonal, with ``c_k = ∂_zT_k² + w²∂_zS_k²`` and
    ``r_k = ∂_zT_k F_Tx,k + w² ∂_zS_k F_Sx,k``:

        ½c_{j-1}·Γ_{j-1} + ½(c_j + c_{j-1})·Γ_j + ½c_j·Γ_{j+1} = r_j + r_{j-1}

    which is the same structure as ``gm.fer_solve_gamma`` — unsurprising, since that solves for
    the same object from the GM ansatz instead of from the teacher's flux.

    The two horizontal directions decouple: ``Γ_y`` only ever enters the x-components of the flux
    and ``Γ_x`` only the y-components. No sign constraint anywhere — being free in sign is what
    Arm B buys over Arm A."""
    nlay = dzT.shape[1]
    lay = np.arange(nlay)[None, :]
    wet = lay < n_wet[:, None]                       # (N, nlay)

    c = np.where(wet, dzT ** 2 + (w * dzS) ** 2, 0.0)
    # ⚠️ OPEN PROBLEM, and ``ridge`` is the half-built handle for it — DEFAULT OFF, because the
    # fix is not understood yet. Measured on 1960_12, this row's predicted flux is 2.41x the
    # teacher's below 1500 m and 2.83x below 3000 m, while the per-cell free fit is at 0.95 and
    # 0.93 and both agree to 0.98 in the 100-500 m band. So the column coupling damages the abyss
    # specifically. The plausible mechanism — the objective is an unweighted sum of squared flux
    # residuals, so weakly-stratified deep layers carry almost no weight and the fit is free to
    # put an arbitrary Gamma there in the service of the layers above — is NOT confirmed: a
    # synthetic column built to that description returns exactly zero in the deep WITHOUT the
    # ridge, and a nonzero 5e-4 WITH it, which is the opposite of the intended effect. A ridge at
    # 1e-3 also moves well-determined solutions by ~0.3 %, so it is not free even where it is not
    # needed. Left off until the mechanism is reproduced.
    # This does not touch the headline: the global score is dominated by the upper ocean, where
    # the constrained and free rows agree to three digits (0.5507 vs 0.5573).
    cbar = np.where(n_wet[:, None] > 0, c.sum(axis=1, keepdims=True)
                    / np.maximum(n_wet[:, None], 1), 0.0)
    lam = ridge * cbar

    def solve(Ft, Fs):
        r = np.where(wet, dzT * Ft + w * w * dzS * Fs, 0.0)     # (N, nlay)
        # Unknowns are the interior interfaces j = 1 .. n_wet-1; index them 0 .. nlay-1 with
        # j = index+1, so entry i couples layers i and i+1.
        cm = c[:, :-1]                                          # c_{j-1}, j = i+1
        cp_ = c[:, 1:]                                          # c_j
        di = 0.5 * (cm + cp_) + lam
        lo = np.zeros_like(di)
        up = np.zeros_like(di)
        lo[:, 1:] = 0.5 * cm[:, 1:]
        up[:, :-1] = 0.5 * cp_[:, :-1]
        rhs = r[:, :-1] + r[:, 1:]
        free = (lay[:, :nlay - 1] + 1) < n_wet[:, None]         # j <= n_wet-1
        di = np.where(free, di, 1.0)
        lo = np.where(free, lo, 0.0)
        up = np.where(free, up, 0.0)
        rhs = np.where(free, rhs, 0.0)
        # A zero row (a column where every c is 0) would divide by the guarded 1.0 and return 0,
        # which is the right answer there: no stratification, no recoverable streamfunction.
        di = np.where(np.abs(di) > 0.0, di, 1.0)
        g_int = _thomas(lo, di, up, rhs)                        # Γ at interfaces 1..nlay-1
        gamma = np.concatenate([np.zeros((g_int.shape[0], 1)), g_int], axis=1)   # Γ_0 = 0
        gamma = np.where(lay < n_wet[:, None], gamma, 0.0)
        psi = np.zeros_like(dzT)
        psi[:, :-1] = 0.5 * (gamma[:, :-1] + gamma[:, 1:])
        psi[:, -1] = 0.5 * gamma[:, -1]                         # Γ below the last layer is 0
        return np.where(wet, psi, 0.0)

    psi_y = solve(F_T[..., 0], F_S[..., 0])
    psi_x = -solve(F_T[..., 1], F_S[..., 1])
    pT = np.stack([psi_y * dzT, -psi_x * dzT], axis=-1)
    pS = np.stack([psi_y * dzS, -psi_x * dzS], axis=-1)
    return pT, pS


def nnls2(a11, a12, a22, b1, b2):
    """Exact non-negative least squares in two variables, elementwise over arrays.

    Minimises ``|Y − k1·A − k2·B|²`` subject to ``k1, k2 >= 0``, given the normal-equation
    entries. Exact rather than "solve then clip": clipping a negative unconstrained coefficient
    leaves the OTHER coefficient at a value optimised jointly with the one that was removed, so
    it is not the constrained optimum and can be arbitrarily far from it. With two variables the
    active set is small enough to enumerate — the interior solution if it is feasible, otherwise
    the better of the two single-variable solutions, with zero available when neither helps."""
    det = a11 * a22 - a12 * a12
    safe = np.abs(det) > 1e-300
    d_ = np.where(safe, det, 1.0)
    k1 = np.where(safe, (a22 * b1 - a12 * b2) / d_, 0.0)
    k2 = np.where(safe, (a11 * b2 - a12 * b1) / d_, 0.0)
    interior = (k1 >= 0.0) & (k2 >= 0.0) & safe

    e1 = np.maximum(np.where(a11 > 0, b1 / np.where(a11 > 0, a11, 1.0), 0.0), 0.0)
    e2 = np.maximum(np.where(a22 > 0, b2 / np.where(a22 > 0, a22, 1.0), 0.0), 0.0)
    # Each candidate's residual, dropping the |Y|² constant common to all of them.
    r1 = -2.0 * e1 * b1 + e1 * e1 * a11
    r2 = -2.0 * e2 * b2 + e2 * e2 * a22
    take1 = r1 <= r2
    kg = np.where(interior, k1, np.where(take1, e1, 0.0))
    kr = np.where(interior, k2, np.where(take1, 0.0, e2))
    return np.maximum(kg, 0.0), np.maximum(kr, 0.0)


def _div(mesh, F, hnode, helem):
    """``∇·F`` of a node-based horizontal flux ``(N, nl-1, 2)``, on layer cells."""
    nl = mesh.nl
    return np.asarray(_fd.divergence_of_node_flux(
        mesh, jnp.asarray(F[..., 0]), jnp.asarray(F[..., 1]), hnode, helem))[:, :nl - 1]


ORACLE_NAME = "ARM A ceiling"
ORACLE_B_NAME = "ARM B free"
ORACLE_B2_NAME = "ARM B deployable"
CEILINGS = (ORACLE_NAME, ORACLE_B_NAME, ORACLE_B2_NAME)


def wet_layer_count(mesh):
    """Number of wet LAYERS per column, and a check that they start at index 0.

    The constrained Arm B ceiling parametrises Γ down a contiguous column from the surface, so a
    column whose wet range began below index 0 (an ice-shelf cavity setup) would be solved with
    the wrong boundary condition. CORE2 has no cavity; this fails loudly rather than quietly
    mis-solving if that ever changes."""
    lay = np.asarray(mesh.node_layer_mask, bool)[:, :mesh.nl - 1]
    n = lay.sum(axis=1)
    first = np.argmax(lay, axis=1)
    if np.any((n > 0) & (first != 0)):
        raise SystemExit("node_layer_mask does not start at layer 0 for every wet column; the "
                         "constrained Arm B ceiling assumes a surface-anchored column")
    return n.astype(np.int64)


def evaluate(mesh, months, coarse_dir, closures, guard, log_m_max, *, verbose=True):
    import netCDF4
    from fesom_jax import ale
    from fesom_jax.state import State
    hnode = State.rest(mesh).hnode
    _, helem = ale.commit_thickness(mesh, hnode)
    nc = len(closures) + len(CEILINGS)
    nlm = len(closures)
    acc = {"all": Acc(nc, nlm)}
    for grp, names in (("depth", DEPTH_NAMES), ("lat", LAT_NAMES), ("res", RES_NAMES)):
        for nm in names:
            acc[f"{grp}:{nm}"] = Acc(nc, nlm)
    for m in months:
        acc[f"month:{m}"] = Acc(nc, nlm)

    nl = mesh.nl
    n_wet = wet_layer_count(mesh)
    totals = dict(marks=0, usable=0, accepted=0, upgradient=0)
    for m in months:
        p = Path(coarse_dir) / f"coarse_ng5_{m}.nc"
        with netCDF4.Dataset(p) as ds:
            nrec = len(ds.dimensions["time"])
        for r in range(nrec):
            t0 = time.time()
            T47, S47, have47, _ = _fp.load_product_mark(p, r)
            d = _fp.derive(mesh, T47, S47, have47, verbose=False)
            F_T, F_S = _kh.load_fluxes(p, r, nl)
            F_T, F_S = np.asarray(F_T), np.asarray(F_S)
            kh = _kh.project(mesh, d, F_T, F_S, verbose=False, **guard)

            base_gm = np.asarray(d.fer_K_lay)
            base_redi = np.asarray(d.Ki)[:, :nl - 1]
            feats = np.asarray(d.feats)
            pos_base = (base_gm > 0.0) & (base_redi > 0.0)

            # Every closure evaluated once on the whole field, then run FORWARD through the
            # library operator. Both κ and the flux it implies come from one call each.
            # ⚠️ TAPERED slope, deliberately: this scores the flux the STUDENT would produce,
            # and the student applies the ODM95 taper. The untapered slope kappa_hat.project
            # inverts with is the right choice for the inverse problem and the wrong one here —
            # see the warning on gm_redi_flux.
            tapered = jnp.asarray(d.slope_tapered[..., :2])
            kap = [fn(feats, base_gm, base_redi) for _nm, fn in closures]
            pred = [_kh.gm_redi_flux(mesh, d, jnp.asarray(kg), jnp.asarray(kr), slope=tapered)
                    for kg, kr in kap]
            pred = [(np.asarray(a), np.asarray(b)) for a, b in pred]
            # The two ceiling rows go LAST, in the order main() names them. They are the only
            # entries that have seen the teacher's flux, which is the whole point of them.
            A_T, B_T, A_S, B_S, dzT, dzS = gm_redi_basis(mesh, d)
            w = flux_weight(F_T, F_S)
            pred.append(oracle_kappa_flux(A_T, B_T, A_S, B_S, F_T, F_S, w))
            pred.append(oracle_psi(dzT, dzS, F_T, F_S, w))
            pred.append(oracle_psi_constrained(dzT, dzS, F_T, F_S, w, n_wet))

            # Measure 3 — the divergence, through the operator M3b gated to max|difference| = 0
            # against the model's own `gm_redi.diff_part_hor_redi`.
            dvT = _div(mesh, F_T, hnode, helem)
            dvS = _div(mesh, F_S, hnode, helem)
            dpred = [(_div(mesh, pt, hnode, helem), _div(mesh, ps, hnode, helem))
                     for pt, ps in pred]

            ok = np.asarray(kh.ok_gm, bool) & np.asarray(kh.ok_redi, bool) & pos_base
            r_gm = np.where(ok, np.asarray(kh.kappa_gm) / np.maximum(base_gm, 1e-300), np.nan)
            r_re = np.where(ok, np.asarray(kh.kappa_redi) / np.maximum(base_redi, 1e-300),
                            np.nan)
            keep = ok & (r_gm > 0.0) & (r_re > 0.0)
            use = np.asarray(d.usable, bool) & pos_base
            st = strata(mesh, d)

            totals["marks"] += 1
            totals["usable"] += int(use.sum())
            totals["accepted"] += int(keep.sum())
            totals["upgradient"] += int((ok & ~keep).sum())

            for key, sel in _selectors(st, m):
                k_sel = keep & sel
                if k_sel.any():
                    y = np.clip(np.stack([np.log(r_gm[k_sel]), np.log(r_re[k_sel])], axis=-1),
                                -log_m_max, log_m_max)
                    preds = [np.stack([
                        np.log(np.maximum(kg[k_sel], 1e-300) / base_gm[k_sel]),
                        np.log(np.maximum(kr[k_sel], 1e-300) / base_redi[k_sel])], axis=-1)
                        for kg, kr in kap]
                    acc[key].add_logm(y, preds)

                u_sel = use & sel
                nu = int(u_sel.sum())
                if nu:
                    tot = np.array([_sq(F_T[u_sel]).sum(), _sq(F_S[u_sel]).sum()])
                    sse = np.array([[_sq(F_T[u_sel] - pt[u_sel]).sum(),
                                     _sq(F_S[u_sel] - ps[u_sel]).sum()]
                                    for pt, ps in pred])
                    psq = np.array([[_sq(pt[u_sel]).sum(), _sq(ps[u_sel]).sum()]
                                    for pt, ps in pred])
                    acc[key].add_flux(nu, tot, sse, psq)

                    dtot = np.array([(dvT[u_sel] ** 2).sum(), (dvS[u_sel] ** 2).sum()])
                    dsse = np.array([[((dvT[u_sel] - qt[u_sel]) ** 2).sum(),
                                      ((dvS[u_sel] - qs[u_sel]) ** 2).sum()]
                                     for qt, qs in dpred])
                    acc[key].add_div(nu, dtot, dsse)

            if verbose:
                print(f"  {m}[{r}]: usable {int(use.sum()):>8d}, accepted "
                      f"{int(keep.sum()):>8d} ({100*keep.sum()/max(use.sum(),1):4.2f}%)"
                      f"  [{time.time()-t0:.0f}s]", flush=True)
    return acc, totals


def evaluate_time_averaged(mesh, months, coarse_dir, closures, guard, *, verbose=True):
    """Measures 2 and 3 on the MONTHLY MEAN of every field, one month at a time.

    Why this exists. The teacher's eddy flux in the products is a purely SPATIAL decomposition
    evaluated on one snapshot — ``agg(u·τ) − agg(u)·agg(τ)`` with no time averaging anywhere. An
    eddy parameterization represents a statistically averaged transport, so scoring it against a
    single instantaneous covariance charges it for internal waves, near-inertial motion and
    whatever the eddy field happened to be doing that day. Averaging the six five-day marks of a
    month is the best estimate of the monthly-mean subgrid flux available from what is on disk.

    ⚠️ **Expect a modest improvement, not a transformation.** Mesoscale eddies decorrelate over
    10-30 days, so six marks five days apart are far from independent and the random part falls
    by much less than sqrt(6). What five-day sampling does remove completely is everything faster
    than five days. The exact fix is accumulating the products ``uT, vT, uS, vS`` online inside
    the Fortran run; this is the estimate available without re-running the teacher.

    The two ceilings are fitted to the AVERAGED fields, not averaged over per-mark fits. That
    distinction is the whole point: one coefficient per cell for the whole month is a
    parameterization, while a coefficient free to change every five days is not, and fitting per
    mark then averaging would silently bound the second."""
    import netCDF4
    from fesom_jax import ale
    from fesom_jax.state import State
    hnode = State.rest(mesh).hnode
    _, helem = ale.commit_thickness(mesh, hnode)
    n_wet = wet_layer_count(mesh)
    nc = len(closures) + len(CEILINGS)
    acc = {"all": Acc(nc, 0)}
    for grp, nms in (("depth", DEPTH_NAMES), ("lat", LAT_NAMES), ("res", RES_NAMES)):
        for nm in nms:
            acc[f"{grp}:{nm}"] = Acc(nc, 0)
    for m in months:
        acc[f"month:{m}"] = Acc(nc, 0)

    nl = mesh.nl
    totals = dict(marks=0, usable=0, accepted=0, upgradient=0)
    for m in months:
        p = Path(coarse_dir) / f"coarse_ng5_{m}.nc"
        with netCDF4.Dataset(p) as ds:
            nrec = len(ds.dimensions["time"])
        acc_n = 0
        sums = None
        st = use = None
        for r in range(nrec):
            t0 = time.time()
            T47, S47, have47, _ = _fp.load_product_mark(p, r)
            d = _fp.derive(mesh, T47, S47, have47, verbose=False)
            F_T, F_S = _kh.load_fluxes(p, r, nl)
            F_T, F_S = np.asarray(F_T), np.asarray(F_S)

            base_gm = np.asarray(d.fer_K_lay)
            base_redi = np.asarray(d.Ki)[:, :nl - 1]
            feats = np.asarray(d.feats)
            tapered = jnp.asarray(d.slope_tapered[..., :2])
            pred = []
            for _nm, fn in closures:
                kg, kr = fn(feats, base_gm, base_redi)
                a, b = _kh.gm_redi_flux(mesh, d, jnp.asarray(kg), jnp.asarray(kr), slope=tapered)
                pred.append((np.asarray(a), np.asarray(b)))
            A_T, B_T, A_S, B_S, dzT, dzS = gm_redi_basis(mesh, d)

            cur = dict(F_T=F_T, F_S=F_S, A_T=A_T, B_T=B_T, A_S=A_S, B_S=B_S,
                       dzT=dzT, dzS=dzS)
            for i, (a, b) in enumerate(pred):
                cur[f"pT{i}"] = a
                cur[f"pS{i}"] = b
            if sums is None:
                sums = {k: v.copy() for k, v in cur.items()}
                use = np.asarray(d.usable, bool) & (base_gm > 0.0) & (base_redi > 0.0)
                st = strata(mesh, d)
            else:
                for k, v in cur.items():
                    sums[k] += v
                use &= np.asarray(d.usable, bool) & (base_gm > 0.0) & (base_redi > 0.0)
            acc_n += 1
            totals["marks"] += 1
            if verbose:
                print(f"  {m}[{r}]: accumulating  [{time.time()-t0:.0f}s]", flush=True)

        for k in sums:
            sums[k] /= acc_n
        F_T, F_S = sums["F_T"], sums["F_S"]
        w = flux_weight(F_T, F_S)
        pred = [(sums[f"pT{i}"], sums[f"pS{i}"]) for i in range(len(closures))]
        pred.append(oracle_kappa_flux(sums["A_T"], sums["B_T"], sums["A_S"], sums["B_S"],
                                      F_T, F_S, w))
        pred.append(oracle_psi(sums["dzT"], sums["dzS"], F_T, F_S, w))
        pred.append(oracle_psi_constrained(sums["dzT"], sums["dzS"], F_T, F_S, w, n_wet))

        dvT, dvS = _div(mesh, F_T, hnode, helem), _div(mesh, F_S, hnode, helem)
        dpred = [(_div(mesh, pt, hnode, helem), _div(mesh, ps, hnode, helem))
                 for pt, ps in pred]
        totals["usable"] += int(use.sum())

        for key, sel in _selectors(st, m):
            u_sel = use & sel
            nu = int(u_sel.sum())
            if not nu:
                continue
            acc[key].add_flux(
                nu,
                np.array([_sq(F_T[u_sel]).sum(), _sq(F_S[u_sel]).sum()]),
                np.array([[_sq(F_T[u_sel] - pt[u_sel]).sum(),
                           _sq(F_S[u_sel] - ps[u_sel]).sum()] for pt, ps in pred]),
                np.array([[_sq(pt[u_sel]).sum(), _sq(ps[u_sel]).sum()] for pt, ps in pred]))
            acc[key].add_div(
                nu,
                np.array([(dvT[u_sel] ** 2).sum(), (dvS[u_sel] ** 2).sum()]),
                np.array([[((dvT[u_sel] - qt[u_sel]) ** 2).sum(),
                           ((dvS[u_sel] - qs[u_sel]) ** 2).sum()] for qt, qs in dpred]))
        print(f"  {m}: averaged {acc_n} marks, {int(use.sum())} usable cells", flush=True)
    return acc, totals


def report(acc, names, totals, out_path: Path, gate_margin: float, months):
    def row(key, label):
        a = acc[key]
        r2, ev, dv = a.r2(), a.ev(), a.ev_div()
        if r2 is None and ev is None:
            return None
        pack = lambda v, k: None if v is None else {nm: v[i][k] for i, nm in enumerate(names)}  # noqa: E731
        return dict(stratum=label, n_logm=a.n, n_flux=a.fn,
                    r2=None if r2 is None else {nm: r2[i]
                                                for i, nm in enumerate(names[:len(r2)])},
                    ev_T=pack(ev, 0), ev_S=pack(ev, 1),
                    evdiv_T=pack(dv, 0), evdiv_S=pack(dv, 1))

    rows = [row("all", "ALL")]
    for grp, ns in (("depth", DEPTH_NAMES), ("lat", LAT_NAMES), ("res", RES_NAMES)):
        rows += [row(f"{grp}:{nm}", f"{grp} {nm}") for nm in ns]
    rows += [row(f"month:{m}", f"month {m}") for m in months]
    rows = [r for r in rows if r is not None]
    if not rows:
        raise SystemExit("nothing accumulated — check the months and the guard floors")

    w = max(len(r["stratum"]) for r in rows) + 1
    cw = max(14, max(len(nm) for nm in names) + 1)
    print(f"\n[totals] {totals['marks']} marks; {totals['usable']} usable layer cells; "
          f"{totals['accepted']} accepted by the projection with both κ̂ positive "
          f"({100*totals['accepted']/max(totals['usable'],1):.2f}%); "
          f"{totals['upgradient']} passed the guards but are UP-gradient")

    print(f"\n{'='*(w+12+cw*len(names))}")
    print("MEASURE 1 — R² of log m on the cells the projection accepts, target clipped to the "
          "multiplier bound")
    print("(R² is about each stratum's own mean, so a within-stratum-optimal constant is 0.000 "
          "by definition;\nthe two ceiling rows are absent because neither is a multiplier on "
          "the formula and has no log m)")
    lm_names = [nm for nm in names if nm not in CEILINGS]
    print(f"{'stratum':<{w}} {'n':>11}  " + "".join(f"{nm:>{cw}}" for nm in lm_names))
    for r in rows:
        if r["r2"] is None:
            continue
        print(f"{r['stratum']:<{w}} {r['n_logm']:>11d}  "
              + "".join(f"{r['r2'][nm]:>{cw}.4f}" for nm in lm_names))

    rr_rows = [(lab, acc[key].rms_ratio())
               for key, lab in [("all", "ALL")]
               + [(f"depth:{nm}", f"depth {nm}") for nm in DEPTH_NAMES]]
    rr = acc["all"].rms_ratio()
    print(f"\n{'='*(w+12+cw*len(names))}")
    print("SCALE CHECK — RMS |F_predicted| / RMS |F_teacher| over all usable cells")
    print("(a closure with the right physics and the wrong magnitude scores a huge NEGATIVE "
          "explained\nvariance below, which reads like a result; this row is what says it is a "
          "scale error instead)")
    if rr is not None:
        print("Broken out by depth as well, because a closure can be globally the right size and "
              "locally\nabsurd: the objective is an unweighted sum of squares, so cells with weak "
              "fluxes carry almost\nno weight and a fit is free to put anything there.")
        print(f"  {'stratum':<{w}} {'tracer':>11}  " + "".join(f"{nm:>{cw}}" for nm in names))
        for lab, rrow in rr_rows:
            if rrow is None:
                continue
            for k, tr in ((0, "T"), (1, "S")):
                print(f"  {lab:<{w}} {tr:>11}  "
                      + "".join(f"{rrow[i][k]:>{cw}.4g}" for i in range(len(names))))
        bad = [names[i] for i in range(len(names))
               if max(rr[i]) > 1e3 or (max(rr[i]) < 1e-3 and max(rr[i]) > 0)]
        if bad:
            print(f"  ⚠️ OFF BY MORE THAN 1000x: {bad} — measure 2 below is not readable as "
                  f"skill until this is fixed")

    print(f"\n{'='*(w+12+cw*len(names))}")
    print("MEASURE 2 — explained variance of the eddy flux about ZERO, on EVERY usable cell")
    print("(negative = the closure's flux is further from the teacher's than no closure at all)")
    for fld, ttl in (("ev_T", "temperature flux (uT_eddy, vT_eddy)"),
                     ("ev_S", "salinity flux (uS_eddy, vS_eddy)")):
        print(f"\n  {ttl}")
        print(f"  {'stratum':<{w}} {'n':>11}  " + "".join(f"{nm:>{cw}}" for nm in names))
        for r in rows:
            if r[fld] is None:
                continue
            print(f"  {r['stratum']:<{w}} {r['n_flux']:>11d}  "
                  + "".join(f"{r[fld][nm]:>{cw}.4f}" for nm in names))

    print(f"\n{'='*(w+12+cw*len(names))}")
    print("MEASURE 3 — explained variance of the flux DIVERGENCE about ZERO, every usable cell")
    print("(only ∇·F moves a tracer, so this is the one a rotational-difference objection "
          "cannot be raised\nagainst; same operator M3b gated against gm_redi.diff_part_hor_redi)")
    for fld, ttl in (("evdiv_T", "div of the temperature flux"),
                     ("evdiv_S", "div of the salinity flux")):
        print(f"\n  {ttl}")
        print(f"  {'stratum':<{w}} {'n':>11}  " + "".join(f"{nm:>{cw}}" for nm in names))
        for r in rows:
            if r.get(fld) is None:
                continue
            print(f"  {r['stratum']:<{w}} {r['n_flux']:>11d}  "
                  + "".join(f"{r[fld][nm]:>{cw}.4f}" for nm in names))

    # --- the gate ---------------------------------------------------------
    base_names = [nm for nm in names if nm.startswith(("formula", "const"))]
    nn_names = [nm for nm in names if nm not in base_names and nm not in CEILINGS]
    print(f"\n{'='*80}\nGATE M3 — on the holdout months named before fitting: {months}")
    verdicts = {}
    if acc["all"].r2() is None:
        print("  (not read here: measure 1 is a per-mark quantity and this run is "
              "time-averaged.\n   The gate verdict comes from the per-mark run; what this run "
              "carries is measures 2 and 3.)")
        nn_names = []
    for nm in nn_names:
        i = names.index(nm)
        rg = acc["all"].r2()
        glob = min(rg[i] - rg[names.index(b)] for b in base_names)
        bands = {}
        for b in DEPTH_NAMES:
            rb = acc[f"depth:{b}"].r2()
            if rb is not None:
                bands[b] = min(rb[i] - rb[names.index(x)] for x in base_names)
        worst = min(bands.values(), default=float("nan"))
        ok = glob >= gate_margin and all(v > 0.0 for v in bands.values())
        ev, dv = acc["all"].ev(), acc["all"].ev_div()
        j = names.index("formula (m=1)")
        verdicts[nm] = dict(global_margin=glob, depth_bands=bands, worst_band=worst,
                            flux_ev_T=ev[i][0], flux_ev_S=ev[i][1],
                            flux_ev_T_formula=ev[j][0], flux_ev_S_formula=ev[j][1],
                            div_ev_T=None if dv is None else dv[i][0],
                            div_ev_S=None if dv is None else dv[i][1],
                            div_ev_T_formula=None if dv is None else dv[j][0],
                            div_ev_S_formula=None if dv is None else dv[j][1],
                            passed=bool(ok))
        print(f"  {nm}: global margin {glob:+.4f} (needs >= {gate_margin}); "
              f"worst depth band {worst:+.4f} (needs > 0)  ->  {'PASS' if ok else 'FAIL'}")
        print(f"      flux EV over all usable cells: T {ev[i][0]:+.4f} S {ev[i][1]:+.4f}"
              f"   (formula: T {ev[j][0]:+.4f} S {ev[j][1]:+.4f})")
        if dv is not None:
            print(f"      div  EV over all usable cells: T {dv[i][0]:+.4f} S {dv[i][1]:+.4f}"
                  f"   (formula: T {dv[j][0]:+.4f} S {dv[j][1]:+.4f})")
    evo, dvo = acc["all"].ev(), acc["all"].ev_div()
    print("\n  THE TWO CEILINGS — same two free numbers per cell, same four flux components,")
    print("  both fitted against the answer. The only difference is whether the flux direction")
    print("  is chained to the density gradient with a positive coefficient (A) or free (B).")
    for nm, note in ((ORACLE_NAME, "kappa >= 0, direction fixed by the neutral slope"),
                     (ORACLE_B_NAME, "streamfunction free per cell in direction and sign"),
                     (ORACLE_B2_NAME, "streamfunction on interfaces, zero at surface and bottom, "
                                      "levels coupled — the deployable structure")):
        i = names.index(nm)
        line = f"      flux EV  T {evo[i][0]:+.4f}  S {evo[i][1]:+.4f}"
        if dvo is not None:
            line += f"      div EV  T {dvo[i][0]:+.4f}  S {dvo[i][1]:+.4f}"
        print(f"  {nm}  ({note})")
        print(line)
    ia, ib = names.index(ORACLE_NAME), names.index(ORACLE_B_NAME)
    if dvo is not None:
        print(f"\n  Freeing the direction buys {dvo[ib][0] - dvo[ia][0]:+.4f} on the temperature "
              f"divergence ({dvo[ia][0]:+.4f} -> {dvo[ib][0]:+.4f}).")
        print("  That difference IS the Arm A / Arm B question, with both sides measured rather "
              "than argued.")

    print("\nThe gate is measure 1. Measures 2 and 3 do not gate anything — they are what the "
          "Arm A / Arm B\ndecision is read from, and the ceiling rows are what make that reading "
          "independent of how well\nany particular network was trained. All are printed "
          "together so they are never separated.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(months=months, totals=totals, rows=rows,
                                        gate=verdicts, closures=names), indent=1))
    print(f"\n[out] {out_path}")
    return verdicts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coarse-dir", type=Path, default=_nd.COARSE_DIR)
    ap.add_argument("--months", default="",
                    help="comma-separated months to score on; empty = the holdout recorded in "
                         "--const-from's holdout_declared.json")
    ap.add_argument("--weights", action="append", default=[], metavar="TAG=PATH",
                    help="repeatable; TAG labels the rung in the table (e.g. D0=/path.npz)")
    ap.add_argument("--const-from", type=Path, default=None,
                    help="a stage0 output DIRECTORY: its holdout_declared.json fixes the months "
                         "and its record fixes the train-fitted constant baseline")
    ap.add_argument("--time-average", action="store_true",
                    help="score measures 2 and 3 on the MONTHLY MEAN of every field instead of "
                         "per mark. The products carry an instantaneous spatial covariance with "
                         "no time averaging, and a parameterization represents a time-averaged "
                         "transport; this is the closest estimate available without re-running "
                         "the teacher with online-accumulated products.")
    ap.add_argument("--gate-margin", type=float, default=0.05)
    ap.add_argument("--m-max", type=float, default=3.0)
    ap.add_argument("--grad-floor", type=float, default=_kh.GRAD_FLOOR)
    ap.add_argument("--slope-floor", type=float, default=_kh.SLOPE_FLOOR)
    ap.add_argument("--resid-max", type=float, default=_kh.RESID_MAX)
    ap.add_argument("--b-sigma-max", type=float, default=_kh.B_SIGMA_MAX)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.weights:
        raise SystemExit("at least one --weights TAG=PATH is required")
    weights = []
    for spec in args.weights:
        tag, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--weights wants TAG=PATH, got {spec!r}")
        weights.append((tag, Path(path)))

    months, const_log_m = None, None
    if args.const_from is not None:
        decl = json.loads((args.const_from / "holdout_declared.json").read_text())
        months = decl["holdout"]
        recs = sorted(args.const_from.glob("stage0_seed*.json"))
        if recs:
            const_log_m = json.loads(recs[0].read_text()).get("train_mean_log_m")
    if args.months.strip():
        months = [m for m in args.months.split(",") if m.strip()]
    if not months:
        raise SystemExit("no months to score: pass --months or --const-from")
    if const_log_m is None:
        print("[warn] no train-fitted constant in the stage0 record; the constant baseline is "
              "OMITTED and the gate compares against the formula only")

    t0 = time.time()
    print(f"[plan] scoring on {months}")
    print(f"[plan] rungs: {[t for t, _ in weights]}")
    mesh = load_mesh(_fp.MESH_DIR)
    guard = dict(grad_floor=args.grad_floor, slope_floor=args.slope_floor,
                 resid_max=args.resid_max, b_sigma_max=args.b_sigma_max)
    closures = make_closures(weights, const_log_m)
    names = [nm for nm, _ in closures] + list(CEILINGS)
    if args.time_average:
        print("[plan] TIME-AVERAGED mode: measures 2 and 3 on the monthly mean; measure 1 is a "
              "per-mark quantity and is not computed here")
        acc, totals = evaluate_time_averaged(mesh, months, args.coarse_dir, closures, guard)
    else:
        acc, totals = evaluate(mesh, months, args.coarse_dir, closures, guard,
                               float(np.log(args.m_max)))
    out = args.out or _nd.data_path("stage0", "apriori_eval.json")
    report(acc, names, totals, out, args.gate_margin, months)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    print("M3_APRIORI_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
