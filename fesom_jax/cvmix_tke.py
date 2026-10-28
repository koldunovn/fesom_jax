"""CVMix classical-TKE pure column core — Phase 9b (the analogue of
``fesom_cvmix_tke.c``, the 404-LOC literal port of ``cvmix_tke.F90``).

A **pure, mesh-free, vectorized-over-nodes** column solver: one prognostic TKE equation
per water column, solved implicitly. It mirrors ``fesom_cvmix_integrate_tke``
(``fesom_cvmix_tke.c:141-404``) formula-for-formula; the driver :mod:`fesom_jax.tke`
(JT.2) assembles the per-column inputs from mesh/state and wires the outputs into
``Kv``/``Av``. Same split the C chose for dump traceability.

Layout convention (the seam from the driver): every array is ``[N, nl]`` with the
**column-local interface index ``k`` == the global interface index** — i.e. the surface
interface is ``k=0`` (``ulevels_nod2D == 1``, no cavities in CORE2; the driver asserts
this). ``nlev`` ``[N]`` is the per-node layer count, so the **bottom interface is
``k=nlev``** (``= nlevels_nod2D-1``) and ``k>nlev`` is below-bottom (dry, masked-inert).
Layer quantities (``dzw``) are valid ``k=0..nlev-1``; interface quantities (everything
else) ``k=0..nlev``.

Bit-fidelity notes (ported from the C header, all verified by controlled replay):
 - the Intel reference build is ``-r8`` ⇒ every default-real literal (``6.6``, ``0.5``,
   ``√2``, ``1.5``) is a DOUBLE — ported as float64 (the one real bug the C port had).
 - ``forc_tke_surf**(3/2)`` ⇒ ``x**1.5`` (Intel ``-fp-model precise`` emits a ``pow`` call;
   expect ≤1-ulp libm residue vs C, tolerated by the ≤1e-13 replay gate).
 - the tridiagonal solve uses the C's **reciprocal-multiply** form (``fxa=1/m; cp=c*fxa``),
   NOT ``cp=c/m`` — :func:`_solve_tridiag` below.

AD-safety (the differentiability contract, §4): safe-sqrt at 0 (cold start / dry lanes),
safe ``pow`` at 0 (zero wind), clamped denominators (``max(1e-12,N²)``, ``max(S²,1e-12)``,
``mxl≥mxl_min``); the masked/dry rows are padded to **identity** in the tridiagonal so the
backward pass stays finite and no gradient leaks into dry lanes.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import lax

from . import ops
from .adjoint_stab import AdjointConfig, cut_where, dcut, dcut_where, differentiate_at

# the 6.6 Prandtl-law literal (:717) — a DOUBLE under -r8 (the C TKE_C66). Static.
TKE_C66 = 6.6
# a large finite sentinel for dry/below-bottom mxl padding before the min-scans (NOT inf:
# inf poisons the AD backward pass through the masked lanes). Masked to 0 after.
_MXL_BIG = 1.0e30
# depth (m) below which the ``tke.deep`` diagnostic cut stops the derivative of every closure
# output (a negative control: a near-surface amplification must not depend on it).
TKE_DEEP_CUT_Z = 500.0
# KappaH (m²/s) below which the ``tke.kh_floor`` diagnostic cut omits the derivative of the
# KappaH handed to Kv: the derivative a background floor max(KappaH, 1e-5) would have (a
# closure with such a floor, e.g. MITgcm's GGL90 with diffKr = 1e-5, has none there).
TKE_KH_FLOOR = 1.0e-5


def _safe_sqrt(x):
    """``sqrt(max(x,0))`` with a finite gradient at ``x=0`` (the kpp/gm double-where
    idiom). A bare ``sqrt(0)`` has grad ``∞`` a downstream mask does not stop."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(safe), 0.0)


def _safe_pow32(x):
    """``x**1.5`` for ``x>0``, ``0`` for ``x<=0`` — forward-identical to the C
    ``pow(x,1.5)`` (``pow(0,1.5)=0``) but with a finite gradient at ``x=0`` (a bare
    ``x**1.5`` autodiffs through ``log x`` ⇒ NaN backward at 0)."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.power(safe, 1.5), 0.0)


def _shift_down(x):
    """``out[...,k] = x[...,k-1]``, edge-replicated at k=0 (the value at k=0 is never used
    where this feeds a k≥1 row)."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


def _shift_up(x):
    """``out[...,k] = x[...,k+1]``, edge-replicated at the last level."""
    return jnp.concatenate([x[..., 1:], x[..., -1:]], axis=-1)


def _fwd_min_scan(mxl, dzw_dn, interior):
    """Forward (downward) min-scan ``mxl[k] = min(mxl[k], mxl[k-1]+dzw[k-1])`` over the
    interior rows, carrying the **updated** ``mxl[k-1]`` — the literal C loop
    (``:678``). ``dzw_dn[k] = dzw[k-1]`` (``_shift_down(dzw)``); ``interior[k]`` selects
    ``1≤k≤nlev-1``. Sequential (the carry is the just-updated value), so a ``lax.scan``;
    the per-node ``nlev`` lives in the ``interior`` mask. Inputs/outputs ``[N, nl]``."""
    def body(prev, xs):                        # prev = updated mxl[k-1]  [N]
        mxl_k, dzw_k, int_k = xs
        cand = prev + dzw_k                     # mxl[k-1] + dzw[k-1]
        new = jnp.where(int_k, jnp.minimum(mxl_k, cand), mxl_k)
        return new, new                         # carry the updated mxl[k]

    init = jnp.zeros(mxl.shape[:-1], mxl.dtype)  # k=0 row never updates ⇒ init unused
    if mxl.ndim == 2 and ops.column_kernel.enabled():
        return ops.column_kernel.column_scan(body, init, (mxl, dzw_dn, interior))[1]
    if ops.column_unroll():
        return ops.unrolled_scan(body, init, (mxl, dzw_dn, interior))[1]
    mxlT = jnp.moveaxis(mxl, -1, 0)            # [nl, N]
    dzwT = jnp.moveaxis(dzw_dn, -1, 0)
    intT = jnp.moveaxis(interior, -1, 0)
    _, out = lax.scan(body, init, (mxlT, dzwT, intT))
    return jnp.moveaxis(out, 0, -1)


def _bwd_min_scan(mxl, dzw, interior):
    """Backward (upward) min-scan ``mxl[k] = min(mxl[k], mxl[k+1]+dzw[k])`` over
    ``1≤k≤nlev-2``, carrying the **updated** ``mxl[k+1]`` (the C ``:682`` loop, run AFTER
    the special pre-step seeds ``mxl[nlev-1]``). The carry at ``k=nlev-1`` is that special
    value — NOT the bottom ``mxl[nlev]=0`` — exactly as the sequential C scan, which is
    why a ``lax.scan`` (not a reverse cummin) is correct here (a cummin would let the
    ``mxl_min``-floored bottom undercut the special value by ``mxl_min``)."""
    def body(nxt, xs):                          # nxt = updated mxl[k+1]  [N]
        mxl_k, dzw_k, int_k = xs
        cand = nxt + dzw_k                       # mxl[k+1] + dzw[k]
        new = jnp.where(int_k, jnp.minimum(mxl_k, cand), mxl_k)
        return new, new

    init = jnp.zeros(mxl.shape[:-1], mxl.dtype)
    if mxl.ndim == 2 and ops.column_kernel.enabled():
        return ops.column_kernel.column_scan(body, init, (mxl, dzw, interior), reverse=True)[1]
    if ops.column_unroll():
        return ops.unrolled_scan(body, init, (mxl, dzw, interior), reverse=True)[1]
    mxlT = jnp.moveaxis(mxl, -1, 0)
    dzwT = jnp.moveaxis(dzw, -1, 0)
    intT = jnp.moveaxis(interior, -1, 0)
    _, out = lax.scan(body, init, (mxlT, dzwT, intT), reverse=True)
    return jnp.moveaxis(out, 0, -1)


def _solve_tridiag(a, b, c, d):
    """Thomas solve per column (over the interface axis ``-1``), in the C's
    **reciprocal-multiply** form (``fxa=1/m; cp=c*fxa; dp=(d-dp_prev*a)*fxa``,
    ``cvmix_utils_addon.F90:116-149`` / ``fesom_cvmix_tke.c:119-135``) for forward
    bit-parity — NOT ``cp=c/m``. Forward elimination + back-substitution scans (the
    ``ops.tdma`` structure, AD-stable). Padded/dry rows must be identity (``a=c=0,b=1``)
    so they solve to ``d`` and leak no gradient."""
    zero = jnp.zeros(b.shape[:-1], b.dtype)

    def forward(carry, xs):
        cp_prev, dp_prev = carry
        ai, bi, ci, di = xs
        m = bi - ai * cp_prev                    # row 0: cp_prev=0 ⇒ m=b[0]
        fxa = 1.0 / m                            # reciprocal first (the C form)
        cp = ci * fxa
        dp = (di - dp_prev * ai) * fxa
        return (cp, dp), (cp, dp)

    def backward(x_next, xs):
        cp_i, dp_i = xs
        x_i = dp_i - cp_i * x_next                # row K-1: x_next=0 ⇒ x=dp[K-1]
        return x_i, x_i

    if b.ndim == 2 and ops.column_kernel.enabled():
        abcd = tuple(jnp.broadcast_to(v, b.shape) for v in (a, b, c, d))
        return ops.column_kernel.thomas(forward, backward, *abcd)
    if ops.column_unroll():
        _, (cp, dp) = ops.unrolled_scan(forward, (zero, zero), (a, b, c, d))
        return ops.unrolled_scan(backward, zero, (cp, dp), reverse=True)[1]
    aT, bT, cT, dT = (jnp.moveaxis(v, -1, 0) for v in (a, b, c, d))
    _, (cp, dp) = lax.scan(forward, (zero, zero), (aT, bT, cT, dT))
    _, xT = lax.scan(backward, zero, (cp, dp), reverse=True)
    return jnp.moveaxis(xT, 0, -1)


def _prandtl_fn(Nsqr, Ssqr):
    """The Prandtl number ``clip(6.6·N²/max(S², 1e-12), 1, 10)`` (``fesom_cvmix_tke.c:705-717``)
    as a module-level function of its two inputs (for :func:`differentiate_at`)."""
    return jnp.maximum(1.0, jnp.minimum(10.0, TKE_C66 * (Nsqr / jnp.maximum(Ssqr, 1.0e-12))))


def mixing_length(tke_old, Nsqr, dzw, *, is_surf, is_bot, is_interior, is_wet_iface,
                  mxl_min, cuts=frozenset(), ml_n2_star=None, ml_ratio=None, ml_raw_cut=None,
                  n2_raw=None):
    """Part 1 — the Blanke–Delecluse mixing length ``mxl`` (``fesom_cvmix_tke.c:222-237``).
    Returns ``(mxl, sqrttke)`` ``[N, nl]``. ``alpha_tke`` does NOT enter here (it is only
    the TKE-diffusivity multiplier). All level info via the boolean masks (per-node
    ``nlev``). ``cuts``: the derivative cut points ``tke.ml_e`` / ``tke.ml_n2`` /
    ``tke.ml_wall`` (see :func:`integrate_tke_column`). ``ml_n2_star``: no N² derivative of
    the free length where N² < ``ml_n2_star`` (``AdjointConfig.tke_ml_cut``); ``ml_ratio`` /
    ``ml_raw_cut``: nor where ``|n2_raw| > ml_ratio·|N²|`` or ``n2_raw < ml_raw_cut``
    (``AdjointConfig.tke_ml_ratio`` / ``tke_ml_raw_cut``; ``n2_raw`` the unsmoothed N²)."""
    sqrttke = _safe_sqrt(tke_old)                                        # :668
    n2_ml = dcut(cuts, "tke.ml_n2", Nsqr)
    if ml_n2_star is not None or ml_ratio is not None or ml_raw_cut is not None:
        # the mixing-length cut: near-neutral cells, where dmxl0/dN² = -mxl0/(2N²) is huge,
        # and inverted cells whose smoothed N² is a small residual of the raw one
        n2_fwd = lax.stop_gradient(Nsqr)
        mask = None if ml_n2_star is None else n2_fwd < ml_n2_star
        if ml_ratio is not None or ml_raw_cut is not None:
            if n2_raw is None:
                raise ValueError("the tke_ml_ratio / tke_ml_raw_cut cuts need the unsmoothed "
                                 "N² (n2_raw)")
            raw_fwd = lax.stop_gradient(n2_raw)
            if ml_ratio is not None:
                cancel = jnp.abs(raw_fwd) > ml_ratio * jnp.abs(n2_fwd)
                mask = cancel if mask is None else mask | cancel
            if ml_raw_cut is not None:
                low = raw_fwd < ml_raw_cut
                mask = low if mask is None else mask | low
        n2_ml = cut_where(n2_ml, mask)
    # unconstrained length √2·sqrttke/√max(1e-12,N²) (:671). At the surface+bottom the
    # driver zeroes N² ⇒ this blows up there, but the C overrides those endpoints to 0:
    mxl0 = (jnp.sqrt(2.0) * dcut(cuts, "tke.ml_e", sqrttke)
            / jnp.sqrt(jnp.maximum(1.0e-12, n2_ml)))
    mxl = jnp.where(is_surf | is_bot, 0.0, mxl0)                         # :676-677

    # forward + backward Blanke–Delecluse wall constraints. dzw padded to a finite value
    # on dry layers so the scan carry stays finite (those rows never update).
    dzw_safe = jnp.where(jnp.isfinite(dzw), dzw, 0.0)
    mxl = _fwd_min_scan(mxl, _shift_down(dzw_safe), is_interior)         # :678
    # special pre-step at k=nlev-1: min(mxl, mxl_min+dzw[nlev-1]) (:681). dzw at this row
    # IS dzw[nlev-1] (the bottom layer), so the per-k slice is exactly right.
    is_above_bot = is_interior & jnp.logical_not(_shift_up(is_interior))  # k==nlev-1
    mxl = jnp.where(is_above_bot, jnp.minimum(mxl, mxl_min + dzw_safe), mxl)
    # backward loop runs k=nlev-2..1 (:682) — it must NOT touch k=nlev-1 (the just-set
    # special value), else `min(mxl_min+dzw, mxl[nlev]+dzw)=dzw` silently drops it by mxl_min.
    is_interior_bwd = is_interior & jnp.logical_not(is_above_bot)        # 1≤k≤nlev-2
    mxl = _bwd_min_scan(mxl, dzw_safe, is_interior_bwd)                  # :682
    mxl = jnp.maximum(mxl, mxl_min)                                      # :685 floor
    mxl = jnp.where(is_wet_iface, mxl, 0.0)                              # dry → 0
    # tke.ml_wall: where the walls / min-scans set the length below mxl0 it is treated as a
    # fixed length (no inherited tke/N² of the level that bounds it, no dzw); free lengths keep
    # their derivative. MITgcm's GGL90 adjoint does the same with adMxlMaxFlag = 0/1.
    mxl = dcut_where(cuts, "tke.ml_wall", lambda: mxl < mxl0 * (1.0 - 1.0e-13), mxl)
    return mxl, sqrttke


def integrate_tke_column(tke_old, Ssqr, Nsqr, dzw, dzt, forc_tke_surf, nlev,
                         *, c_k, c_eps, cd, alpha_tke, mxl_min, tke_min, kappaM_max,
                         dt, with_diags=False, cuts=frozenset(), adj=None, n2_raw=None,
                         ice_conc=None, lat=None, turner=None):
    """The classical-TKE column solver, vectorized over nodes. Mirrors
    ``fesom_cvmix_integrate_tke`` (``fesom_cvmix_tke.c:141-404``) for the reference config
    (``only_tke``, Neumann BCs, ``mxl_choice=2``; the gate-only IDEMIX/Langmuir/Dirichlet
    branches are absent). Returns ``(tke_new, KappaM, KappaH)`` ``[N, nl]`` and, when
    ``with_diags``, a dict of the 10 budget/aux diagnostics (the JT.1 replay tags).

    Inputs (``[N, nl]`` unless noted), the seam from the driver:
      * ``tke_old`` — previous TKE at interfaces (the recurrent state; 0 at cold start);
      * ``Ssqr`` — vertical shear² (``vshear2``; 0 at surface+bottom interfaces);
      * ``Nsqr`` — buoyancy frequency² (``bvfreq2``; 0 at surface+bottom interfaces);
      * ``dzw`` — layer thickness (``hnode``), valid ``k=0..nlev-1``;
      * ``dzt`` — interface spacing (``dz_trr``; hnode/2 end caps), valid ``k=0..nlev``;
      * ``forc_tke_surf`` ``[N]`` — surface TKE flux ``|stress|/ρ₀`` (≥0);
      * ``nlev`` ``[N]`` int — per-node layer count (bottom interface at ``k=nlev``).
    The trainable constants ``c_k``/``c_eps``/``cd``/``alpha_tke`` are scalars (``Params``
    leaves); the structural statics ``mxl_min``/``tke_min``/``kappaM_max`` from ``TkeConfig``.

    ``cuts`` (static set of names from :data:`fesom_jax.adjoint_stab.KNOWN_CUTS`,
    ``AdjointConfig.diag_cuts``): diagnostic derivative cuts inside the closure, each a
    ``stop_gradient`` on one term that leaves the forward bitwise — ``tke.km_e`` (√tke in
    KappaM), ``tke.ml_e`` / ``tke.ml_n2`` (√tke / N² in the unconstrained mixing length),
    ``tke.ml_wall`` (the length where walls / min-scans bound it below mxl0, masked),
    ``tke.shear`` / ``tke.n2`` (S² / N² in production resp. buoyancy and in Ri), ``tke.pr``
    (the Prandtl number), ``tke.prod`` (S²·KappaM), ``tke.buoy`` (N²·KappaH), ``tke.diss``
    (the Patankar dissipation), ``tke.ke_diff`` (the TKE diffusivity), ``tke.surf`` (the
    surface flux), ``tke.deep`` (every output below :data:`TKE_DEEP_CUT_Z`) and
    ``tke.kh_floor`` (the output KappaH where it is below :data:`TKE_KH_FLOOR`). An empty set
    traces the identical program.

    ``adj`` (:class:`fesom_jax.adjoint_stab.TkeAdjoint` or ``None``, from
    ``AdjointConfig.tke_options()``): the derivative-only candidate settings — the Prandtl-
    number cut (``pr_n2_star``/``pr_mode``) and the near-floor km_e cut (``kme_tke``/
    ``kme_scope``/``kme_term``). Masks are built from forward values; the forward is bitwise
    unchanged, and ``None`` traces the identical program.

    ``n2_raw`` (``[N, nl]`` or ``None``): the UNSMOOTHED N² at the same interfaces as ``Nsqr``
    (zero where ``Nsqr`` is zeroed), read only by the smoothing-cancellation and unsmoothed-N²
    Prandtl and mixing-length cuts (``adj.pr_ratio`` / ``adj.pr_raw_cut`` / ``adj.ml_ratio`` /
    ``adj.ml_raw_cut``, ``AdjointConfig.tke_pr_ratio`` / ``tke_pr_raw_cut`` / ``tke_ml_ratio`` /
    ``tke_ml_raw_cut``) and only through its forward value.


    ``ice_conc`` (``[N]`` or ``None``): the sea-ice concentration of each column, read only by
    the sea-ice Prandtl cut (``adj.pr_ice``, ``AdjointConfig.tke_pr_ice_cut``) and only
    through its forward value. ``lat`` (``[N]`` radians or ``None``): the geographic latitude
    of each column, read only by the latitude Prandtl cut (``adj.pr_lat``,
    ``AdjointConfig.tke_pr_lat_cut``). ``turner`` (``[N, nl]`` degrees or ``None``): the Turner
    angle at the interfaces, read only by the diffusive-regime Prandtl cut (``adj.pr_tu``,
    ``AdjointConfig.tke_pr_tu_cut``).
    """
    if cuts:
        cuts = AdjointConfig(diag_cuts=cuts).validate().diag_cuts   # coerce + reject unknown
    nl = tke_old.shape[-1]
    k = jnp.arange(nl, dtype=jnp.int32)[None, :]            # [1, nl]
    nlev_c = jnp.asarray(nlev).astype(jnp.int32)[:, None]   # [N, 1]
    is_surf = (k == 0)                           # uln0 = 0 (no cavities)
    is_bot = (k == nlev_c)
    is_interior = (k >= 1) & (k <= nlev_c - 1)
    is_wet_iface = (k <= nlev_c)                 # interfaces 0..nlev
    fcol = forc_tke_surf[:, None]

    # ---- Part 1: mixing length (:666-698) ----
    mxl, sqrttke = mixing_length(
        tke_old, Nsqr, dzw, is_surf=is_surf, is_bot=is_bot, is_interior=is_interior,
        is_wet_iface=is_wet_iface, mxl_min=mxl_min, cuts=cuts,
        ml_n2_star=None if adj is None else adj.ml_n2_star,
        ml_ratio=None if adj is None else adj.ml_ratio,
        ml_raw_cut=None if adj is None else adj.ml_raw_cut, n2_raw=n2_raw)

    # the S² / N² the production, buoyancy and Ri terms see (the mixing length's N² is cut
    # separately, as tke.ml_n2). With no cut these are the inputs themselves.
    Ssqr_p = dcut(cuts, "tke.shear", Ssqr)
    Nsqr_b = dcut(cuts, "tke.n2", Nsqr)

    # ---- Part 2: diffusivities (:700-720) ----
    sqrttke_km = dcut(cuts, "tke.km_e", sqrttke)
    KappaM = jnp.minimum(kappaM_max, c_k * mxl * sqrttke_km)             # :704
    KappaM_ke = KappaM                     # the KappaM the TKE diffusivity ke reads (below)
    kme_tke = None if adj is None else adj.kme_tke
    if kme_tke is not None:
        # near-floor km_e cut (AdjointConfig.tke_kme_*): no d KappaM / d sqrt(tke) in columns
        # whose first interior interface has tke_old < tke* (any mixing-length regime). Same
        # arithmetic on the same values ⇒ KappaM_cut == KappaM bitwise.
        if kme_tke == math.inf and adj.kme_scope == "column":
            sqrt_cut = lax.stop_gradient(sqrttke_km)     # every column: the tke.km_e cut
        else:
            flag = lax.stop_gradient(tke_old[:, 1:2]) < kme_tke               # [N, 1]
            mask = flag if adj.kme_scope == "column" else flag & (k == 1)
            sqrt_cut = cut_where(sqrttke_km, mask)
        KappaM_cut = jnp.minimum(kappaM_max, c_k * mxl * sqrt_cut)
        if adj.kme_term == "km_e":
            KappaM = KappaM_ke = KappaM_cut
        else:                                                           # "ke_diff"
            KappaM_ke = KappaM_cut
    Rinum = Nsqr_b / jnp.maximum(Ssqr_p, 1.0e-12)                       # :705
    prandtl = jnp.maximum(1.0, jnp.minimum(10.0, TKE_C66 * Rinum))      # :717
    n2_star = None if adj is None else adj.pr_n2_star
    pr_ratio = None if adj is None else adj.pr_ratio
    pr_raw = None if adj is None else adj.pr_raw_cut
    if n2_star is not None or pr_ratio is not None or pr_raw is not None:
        # Prandtl-number cut (AdjointConfig.tke_pr_cut / tke_pr_mode / tke_pr_ratio /
        # tke_pr_raw_cut), forward bitwise
        n2_fwd = lax.stop_gradient(Nsqr)
        if adj.pr_mode == "cut" and n2_star == math.inf:
            prandtl = lax.stop_gradient(prandtl)        # everywhere: the tke.pr cut itself
        elif adj.pr_mode == "cut":
            mask = None if n2_star is None else n2_fwd < n2_star
            if pr_ratio is not None or pr_raw is not None:
                if n2_raw is None:
                    raise ValueError("the tke_pr_ratio / tke_pr_raw_cut cuts need the unsmoothed "
                                     "N² (n2_raw)")
                raw_fwd = lax.stop_gradient(n2_raw)
            if pr_ratio is not None:
                # the smoothing-cancellation cut: the smoothed N² is a small residual of a
                # much larger raw N² (|raw| > R*·|smoothed|)
                cancel = jnp.abs(raw_fwd) > pr_ratio * jnp.abs(n2_fwd)
                mask = cancel if mask is None else mask | cancel
            if pr_raw is not None:
                # the unsmoothed-N² cut: no Pr derivative where the local N² is below N²_raw*
                # (inverted / neutral cells, whatever the smoothed N² there)
                low = raw_fwd < pr_raw
                mask = low if mask is None else mask | low
            prandtl = cut_where(prandtl, mask)
        else:                                                           # "at"
            alt = (jnp.maximum(n2_fwd, n2_star), lax.stop_gradient(Ssqr_p))
            base = differentiate_at(_prandtl_fn, (Nsqr_b, Ssqr_p), alt)
            prandtl = lax.stop_gradient(prandtl) + (base - lax.stop_gradient(base))
    pr_ice = None if adj is None else adj.pr_ice
    if pr_ice is not None:
        # the sea-ice Prandtl cut (AdjointConfig.tke_pr_ice_cut): no Pr derivative in the
        # columns whose ice concentration exceeds a*, forward bitwise
        if ice_conc is None:
            raise ValueError("the tke_pr_ice_cut cut needs the sea-ice concentration "
                             "(ice_conc)")
        prandtl = cut_where(prandtl, lax.stop_gradient(ice_conc)[:, None] > pr_ice)
    pr_lat = None if adj is None else adj.pr_lat
    if pr_lat is not None:
        # the latitude Prandtl cut (AdjointConfig.tke_pr_lat_cut): no Pr derivative poleward
        # of phi*, forward bitwise
        if lat is None:
            raise ValueError("the tke_pr_lat_cut cut needs the column latitude (lat)")
        prandtl = cut_where(prandtl, jnp.abs(lax.stop_gradient(lat))[:, None]
                            > math.radians(pr_lat))
    pr_tu = None if adj is None else adj.pr_tu
    if pr_tu is not None:
        # the diffusive-regime Prandtl cut (AdjointConfig.tke_pr_tu_cut): no Pr derivative at
        # the interfaces with -90 < Tu < Tu* (stable, cold fresh over warm salty), forward bitwise
        if turner is None:
            raise ValueError("the tke_pr_tu_cut cut needs the Turner angle (turner)")
        tu = lax.stop_gradient(turner)
        prandtl = cut_where(prandtl, (tu > -90.0) & (tu < pr_tu))
    prandtl = dcut(cuts, "tke.pr", prandtl)
    KappaH = KappaM / prandtl                                           # :720

    # ---- Part 3: forcing (:723-744) ----
    K_diss_v = dcut(cuts, "tke.prod", Ssqr_p * KappaM)                   # :729-730
    P_diss_v = dcut(cuts, "tke.buoy", Nsqr_b * KappaH)                   # :731-732
    P_diss_v = jnp.where(is_surf, 0.0, P_diss_v)   # :733 = −forc_rho_surf·g/ρ (≡0 here)
    forc = K_diss_v - P_diss_v                                          # :734
    # surface Neumann flux cd·forc_surf^{3/2}/dzt[0] added to forc[0] (:793). The `dzt>0` guard
    # (mirroring `dzt_s` in Part 4) keeps the denom finite on a degenerate/padding column whose
    # dzt[0]=0. The driver only sets dz_trr[0]=hnode/2 where its `is_surf=(k==nzmin)` fires, but a
    # SHARDED padding node has nlevels_nod2D=0 (int _default_pad) ⇒ nzmin=-1 ⇒ dz_trr[0] stays 0,
    # while the core's `is_surf=(k==0)` still computes surf_flux here. Without `>0` that is x/0=inf
    # (or 0/0): the forward masks it away (is_wet_iface all-False on the all-dry padding column) but
    # REVERSE-mode hits 0·inf=NaN that leaks into c_k/c_eps/cd — and via the NN multiplier into the
    # NN weights (the §3 sharded-twin |g|=nan). Real wet cols have dzt[0]=hnode_surf/2>0 ⇒ untouched
    # (bit-identical; the Phase-9b replay gate is unaffected, it uses padding-free meshes).
    dzt_surf = jnp.where(is_surf & (dzt > 0.0), dzt, 1.0)
    surf_flux = dcut(cuts, "tke.surf", cd * _safe_pow32(fcol) / dzt_surf)
    forc = jnp.where(is_surf, forc + surf_flux, forc)

    # ---- Part 4: implicit vertical diffusion + dissipation (:746-827) ----
    # ke[k] = alpha·0.5·(KappaM[min(k+1,nlev-1)] + KappaM[max(k,1)]) for k=0..nlev-1 (:750-754)
    idx_kp1 = jnp.minimum(k + 1, nlev_c - 1)
    idx_kk = jnp.maximum(k, 1)
    KappaM_kp1 = jnp.take_along_axis(KappaM_ke, idx_kp1, axis=-1)
    KappaM_kk = jnp.take_along_axis(KappaM_ke, idx_kk, axis=-1)
    is_layer = (k <= nlev_c - 1)                                        # ke valid k=0..nlev-1
    ke = dcut(cuts, "tke.ke_diff",
              jnp.where(is_layer, alpha_tke * 0.5 * (KappaM_kp1 + KappaM_kk), 0.0))

    # safe layer/interface spacings (dry/padded denominators → 1, masked rows are zeroed)
    dzw_s = jnp.where(dzw > 0.0, dzw, 1.0)
    dzt_s = jnp.where(dzt > 0.0, dzt, 1.0)
    ke_dn = _shift_down(ke)                         # ke[k-1]
    dzw_dn = _shift_down(dzw_s)                      # dzw[k-1]

    # c_dif[k]=ke[k]/(dzt[k]·dzw[k]) (k=0..nlev-1), c_dif[nlev]=0 (:757-761)
    c_dif = jnp.where(is_layer, ke / (dzt_s * dzw_s), 0.0)
    # a_dif[k]=ke[k-1]/(dzt[k]·dzw[k-1]) (k=1..nlev), a_dif[0]=0 (:770-774)
    a_dif = jnp.where(is_surf, 0.0, ke_dn / (dzt_s * dzw_dn))
    a_dif = jnp.where(is_wet_iface, a_dif, 0.0)
    # b_dif interior k=1..nlev-1 (:764-767); Neumann surface (:794) + bottom (:813) overrides
    b_dif_int = ke_dn / (dzt_s * dzw_dn) + ke / (dzt_s * dzw_s)
    b_dif = jnp.where(is_interior, b_dif_int, 0.0)
    b_dif = jnp.where(is_surf, ke / (dzt_s * dzw_s), b_dif)              # :794 ke[0]/(dzt0·dzw0)
    b_dif = jnp.where(is_bot, ke_dn / (dzt_s * dzw_dn), b_dif)           # :813 ke[nlev-1]/(dztN·dzwN-1)

    # tridiagonal coefficients (:818-821): a/b/c_tri = −dt·a_dif / 1+dt·b_dif / −dt·c_dif
    a_tri = -dt * a_dif
    b_tri = 1.0 + dt * b_dif
    c_tri = -dt * c_dif
    # Patankar quasi-implicit dissipation on interior rows only (:820)
    mxl_s = jnp.where(mxl > 0.0, mxl, 1.0)          # mxl≥mxl_min on wet; safe on dry
    patankar = dcut(cuts, "tke.diss", dt * c_eps * sqrttke / mxl_s)
    b_tri = jnp.where(is_interior, b_tri + patankar, b_tri)
    d_tri = tke_old + dt * forc                                         # :824

    # padded/dry rows → identity (a=c=0, b=1, d=carry) so they solve inertly
    a_tri = jnp.where(is_wet_iface, a_tri, 0.0)
    c_tri = jnp.where(is_wet_iface, c_tri, 0.0)
    b_tri = jnp.where(is_wet_iface, b_tri, 1.0)
    d_tri = jnp.where(is_wet_iface, d_tri, 0.0)

    tke_solve = _solve_tridiag(a_tri, b_tri, c_tri, d_tri)              # :827 (pre-floor)

    # ---- Part 5: floor (:856-869) ----
    tke_unrest = tke_solve                                              # :860 pre-floor save
    tke_new = jnp.maximum(tke_solve, tke_min)                          # :867-869 only_tke
    tke_new = jnp.where(is_wet_iface, tke_new, 0.0)                     # dry stays 0

    KappaM = jnp.where(is_wet_iface, KappaM, 0.0)
    KappaH = jnp.where(is_wet_iface, KappaH, 0.0)

    # tke.deep: no derivative of any closure output below TKE_DEEP_CUT_Z (value unchanged);
    # the mask is built only when the cut is active, and once
    _deep_mask = []

    def _deep():
        if not _deep_mask:
            _deep_mask.append(_iface_depth(dzw) > TKE_DEEP_CUT_Z)
        return _deep_mask[0]
    tke_new = dcut_where(cuts, "tke.deep", _deep, tke_new)
    KappaM = dcut_where(cuts, "tke.deep", _deep, KappaM)
    KappaH = dcut_where(cuts, "tke.deep", _deep, KappaH)
    KappaH = dcut_where(cuts, "tke.kh_floor", lambda: KappaH < TKE_KH_FLOOR, KappaH)

    if not with_diags:
        return tke_new, KappaM, KappaH

    diags = _diagnostics(
        tke_old, tke_solve, tke_new, tke_unrest, K_diss_v, P_diss_v,
        a_dif, b_dif, c_dif, mxl, prandtl, surf_flux, sqrttke, c_eps, mxl_s, dt,
        is_surf=is_surf, is_bot=is_bot, is_interior=is_interior, is_wet_iface=is_wet_iface)
    return tke_new, KappaM, KappaH, diags


def _iface_depth(dzw):
    """Depth (m, positive down) of each interface from the layer thicknesses of forward
    values: ``z[k] = Σ_{j<k} dzw[j]`` over wet layers (non-finite / non-positive thicknesses
    count as 0). Used only for the ``tke.deep`` mask."""
    d = lax.stop_gradient(dzw)
    d = jnp.where(jnp.isfinite(d) & (d > 0.0), d, 0.0)
    return jnp.cumsum(d, axis=-1) - d


def _diagnostics(tke_old, tke_solve, tke_new, tke_unrest, K_diss_v, P_diss_v,
                 a_dif, b_dif, c_dif, mxl, prandtl, surf_flux, sqrttke, c_eps, mxl_s, dt,
                 *, is_surf, is_bot, is_interior, is_wet_iface):
    """The 10 budget/aux diagnostics (``fesom_cvmix_tke.c:829-903``), in Fortran order.
    ``Tdif``/``Tdis`` use the **pre-floor** ``tke_solve``; ``Tbck`` is the floor increment
    (post-floor ``tke_new`` − pre-floor ``tke_unrest``); ``Ttot`` the total tendency. Each
    masked to the wet interfaces. Returned only under ``with_diags`` (test-only — never in
    model State); the closure identity ``Ttot ≈ Σ(7 terms)`` is a free standing oracle."""
    # implicit-tendency Tdif (:831-837), on the PRE-floor solve. The interior stencil with
    # surface/bottom one-sided rows; diff_surf_forc/diff_bott_forc ≡ 0 (Neumann).
    tke_dn = _shift_down(tke_solve)               # tke[k-1]
    tke_up = _shift_up(tke_solve)                 # tke[k+1]
    Tdif_int = a_dif * tke_dn - b_dif * tke_solve + c_dif * tke_up
    Tdif_surf = -b_dif * tke_solve + c_dif * tke_up                    # :834
    Tdif_bot = a_dif * tke_dn - b_dif * tke_solve                      # :835
    Tdif = jnp.where(is_surf, Tdif_surf, jnp.where(is_bot, Tdif_bot, Tdif_int))
    Tdif = jnp.where(is_interior | is_surf | is_bot, Tdif, 0.0)

    # dissipation Tdis = −c_eps/mxl·sqrttke·tke_new — interior only (:852-853). The C uses
    # tke_new here, but at :852 tke_new is still the PRE-floor solve (the floor is :867):
    Tdis = jnp.where(is_interior, -c_eps / mxl_s * sqrttke * tke_solve, 0.0)

    Tbpr = jnp.where(is_wet_iface, -P_diss_v, 0.0)                      # :876-877
    Tspr = jnp.where(is_wet_iface, K_diss_v, 0.0)                       # :878
    Tbck = jnp.where(is_wet_iface, (tke_new - tke_unrest) / dt, 0.0)    # :880 floor increment
    Twin = jnp.where(is_surf, surf_flux, 0.0)                           # :886 surface Neumann
    Tiwf = jnp.zeros_like(tke_old)                                     # :898 iw_diss ≡ 0 (only_tke)
    Ttot = jnp.where(is_wet_iface, (tke_new - tke_old) / dt, 0.0)       # :899
    Lmix = jnp.where(is_wet_iface, mxl, 0.0)                            # :901
    Pr = jnp.where(is_wet_iface, prandtl, 0.0)                          # :902

    return {"Tbpr": Tbpr, "Tspr": Tspr, "Tdif": Tdif, "Tdis": Tdis, "Twin": Twin,
            "Tiwf": Tiwf, "Tbck": Tbck, "Ttot": Ttot, "Lmix": Lmix, "Pr": Pr}


# ---------------------------------------------------------------------------
# per-interface regime logger (forward values only)

REGIMES = ("free", "wall_top", "wall_other", "floor")
"""Mixing-length regime labels of :func:`tke_regimes` (``regime`` holds the index; ``-1`` on
dry interfaces)."""


def tke_regimes(tke_old, Ssqr, Nsqr, dzw, dzt, forc_tke_surf, nlev, *, c_k, c_eps, cd,
                alpha_tke, mxl_min, tke_min, kappaM_max, dt):
    """Per-interface regime of the TKE column core — forward values only.

    Same inputs as :func:`integrate_tke_column` (all ``stop_gradient``-ed here: this is a
    logger, not part of the model), so ``tke_regimes(*(d[k] for k in
    tke.COLUMN_INPUTS), **kw)`` describes exactly the model's solve. Returns a dict of
    ``[N, nl]`` arrays (floats 0 and ``regime`` -1 on dry interfaces):

    * ``tke`` (= ``tke_old``), ``tke_excess`` (``tke_old − tke_min``), ``floor_hit`` (the
      pre-floor solve fell below ``tke_min``, i.e. the floor ``max(tke_solve, tke_min)``
      binds and ``tke_new`` has zero derivative there);
    * ``regime`` — index into :data:`REGIMES`, decided from forward values: ``floor`` where
      ``mxl == mxl_min``; ``free`` where ``mxl == mxl0`` to 1e-13 relative (the unconstrained
      ``√2·√tke/N``); ``wall_top`` at the first interior interface where ``mxl == dzw[0]``
      (the surface wall bound, independent of the state); ``wall_other`` for any other
      wall-bound length (``mxl < mxl0``; such a length can still depend on the tke of a
      neighbour, through that neighbour's ``mxl0``). The surface and bottom interfaces have
      ``mxl = mxl_min`` and are labelled ``floor`` (their KappaM is never read downstream);
    * ``mxl``, ``mxl0``, ``at_cap`` (``c_k·mxl·√tke ≥ kappaM_max``);
    * ``dKm_dtke`` — the analytic diagonal ``∂KappaM[k]/∂tke_old[k]``: ``c_k·mxl/(2√tke)``
      where the length does not depend on the own tke (wall-bound, floor), ``√2·c_k/N`` with
      ``N = √max(1e-12, N²)`` where it does (free), 0 at the cap and where ``tke ≤ 0``;
    * ``Ssqr``, ``Nsqr``, ``Ri = N²/max(S², 1e-12)``, ``Pr = clip(6.6·Ri, 1, 10)``;
    * ``F`` — the stiffness ``dt·dKm_dtke·|S² − N²/Pr|``: the one-step change of the
      production-minus-buoyancy forcing per unit tke through KappaM.
    """
    sg = lax.stop_gradient
    tke_old, Ssqr, Nsqr, dzw, dzt, forc_tke_surf = (
        sg(jnp.asarray(x)) for x in (tke_old, Ssqr, Nsqr, dzw, dzt, forc_tke_surf))
    c_k, c_eps, cd, alpha_tke = (sg(jnp.asarray(x)) for x in (c_k, c_eps, cd, alpha_tke))

    nl = tke_old.shape[-1]
    k = jnp.arange(nl, dtype=jnp.int32)[None, :]
    nlev_c = jnp.asarray(nlev).astype(jnp.int32)[:, None]
    is_wet = (k <= nlev_c)

    _, _, _, diags = integrate_tke_column(
        tke_old, Ssqr, Nsqr, dzw, dzt, forc_tke_surf, nlev, c_k=c_k, c_eps=c_eps, cd=cd,
        alpha_tke=alpha_tke, mxl_min=mxl_min, tke_min=tke_min, kappaM_max=kappaM_max,
        dt=dt, with_diags=True)
    mxl = diags["Lmix"]                        # the core's own mixing length (0 on dry)
    floor_hit = is_wet & (diags["Tbck"] > 0.0)  # tke_new − tke_solve > 0 ⇔ the floor binds

    sqrttke = _safe_sqrt(tke_old)
    Nfreq = jnp.sqrt(jnp.maximum(1.0e-12, Nsqr))
    mxl0 = jnp.sqrt(2.0) * sqrttke / Nfreq      # the same expression as mixing_length
    raw_km = c_k * mxl * sqrttke
    at_cap = is_wet & (raw_km >= kappaM_max)

    dzw0 = dzw[..., :1]
    is_floor = is_wet & (mxl == mxl_min)
    # free: the core's mxl IS mxl0. Compared to a relative 1e-13, not bitwise: under jit XLA
    # may evaluate this copy of √2·√tke/N differently (x/√y vs x·rsqrt(y)) by an ulp, and a
    # wall bound within 1e-13 of mxl0 is a tie whose label does not matter.
    is_free = is_wet & ~is_floor & (jnp.abs(mxl - mxl0) <= 1.0e-13 * mxl0)
    is_wtop = is_wet & ~is_floor & ~is_free & (k == 1) & (mxl == dzw0)
    regime = jnp.where(is_floor, 3, jnp.where(is_free, 0, jnp.where(is_wtop, 1, 2)))
    regime = jnp.where(is_wet, regime, -1).astype(jnp.int8)

    tpos = is_wet & (tke_old > 0.0)
    st_safe = jnp.where(tpos, sqrttke, 1.0)
    dkm_wall = c_k * mxl / (2.0 * st_safe)
    dkm_free = jnp.sqrt(2.0) * c_k / Nfreq
    dKm = jnp.where(is_free, dkm_free, dkm_wall)
    dKm = jnp.where(tpos & ~at_cap, dKm, 0.0)

    Ri = Nsqr / jnp.maximum(Ssqr, 1.0e-12)
    Pr = jnp.maximum(1.0, jnp.minimum(10.0, TKE_C66 * Ri))
    F = dt * dKm * jnp.abs(Ssqr - Nsqr / Pr)

    def wet(x):
        return jnp.where(is_wet, x, 0.0)
    return {"tke": wet(tke_old), "tke_excess": wet(tke_old - tke_min), "floor_hit": floor_hit,
            "regime": regime, "mxl": wet(mxl), "mxl0": wet(mxl0), "at_cap": at_cap,
            "dKm_dtke": wet(dKm), "Ssqr": wet(Ssqr), "Nsqr": wet(Nsqr), "Ri": wet(Ri),
            "Pr": wet(Pr), "F": wet(F)}
