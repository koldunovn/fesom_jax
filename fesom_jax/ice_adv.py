"""Sea-ice FCT advection (Phase 6, Task 6.5) — an AD-safe port of ``fesom_ice_fct.c``.

A 2-D (surface-only) flux-corrected-transport advection of the three ice tracers
(``m_ice``, ``a_ice``, ``m_snow``) with the Zalesak limiter. Unlike the 3-D ocean FCT
(:mod:`fesom_jax.tracer_adv`, edge-centric), the ice FCT is **element-centric**:
Taylor–Galerkin RHS, an FE mass-matrix low/high-order solve, and the antidiffusive flux +
limiter all live on element triangles. The whole thing is expressed with element
gather/scatter — **no explicit CSR** is needed:

* the mass-matrix product ``(mm·X)[row] = Σ_{elem∋row} area/12·(X[row] + ΣX_elem)`` (the FE
  consistent-mass block ``area/12·(I + 11ᵀ)`` scattered to nodes — its row sum is the node CV
  area, so ``mm·1 = area``), and
* the Zalesak cluster bounds ``min/max`` over a node's graph neighbours == ``segment_min/max``
  over the elements touching the node (edge-neighbours == element-co-vertices on a triangle mesh).

Driver order (``fesom_ice_fct.c:524-543``): ``tg_rhs`` → high-order → low-order →
``fem_fct(m_ice)`` → ``fem_fct(a_ice)`` → ``fem_fct(m_snow)``. ``ice_gamma_fct=0.5``,
``ice_diff=10``. The limiter ratio floor is the C's ``1e-12`` (match the C, NOT the ocean
FCT's 1e-16 — bitwise to the C dump). AD: ``min``/``max``/``where`` subgradients (NaN-safe via
the floor), masked area divides; no positivity clip (match the C — let negatives show).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import ops
from .ice import IceConfig
from .mesh import Mesh

# scale_area lives in IceConfig (namelist value 5.8e9; was the 2.0e8 module default here until 2026-09-18)
_ICE_FLUX_EPS = 1.0e-12        # fesom_ice_fct.c:458,466 (the C floor — match it, not 1e-16)


def _mm_times(mesh: Mesh, X):
    """FE consistent-mass product ``(mm·X)[row]`` via elements (no CSR):
    ``Σ_{elem∋row} area/12·(X[row] + ΣX_elem)``. Cavity elements contribute 0."""
    en = mesh.elem_nodes
    Xv = X[en]                                              # [elem,3]
    contrib = (mesh.elem_area / 12.0)[:, None] * (Xv + Xv.sum(axis=1, keepdims=True))
    contrib = jnp.where((mesh.ulevels <= 1)[:, None], contrib, 0.0)
    return ops.scatter_add(contrib, en, mesh.nod2D, inv=getattr(mesh, "inv_elem_nodes", None))


def _tg_rhs(cfg: IceConfig, mesh: Mesh, u_ice, v_ice, a_ice, m_ice, m_snow):
    """Taylor–Galerkin RHS for the 3 tracers (``fesom_ice_tg_rhs``, ``:147-220``)."""
    en = mesh.elem_nodes
    gs = mesh.gradient_sca
    dx = gs[:, 0:3]                                         # [elem,3] ∂N/∂x
    dy = gs[:, 3:6]
    vol = mesh.elem_area
    dt = cfg.ice_dt
    U = u_ice[en]; V = v_ice[en]                            # [elem,3]
    um = U.sum(axis=1); vm = V.sum(axis=1)                  # [elem]  (NOT /3 — C line 189)
    diff = cfg.ice_diff * jnp.sqrt(vol / cfg.scale_area)    # [elem]  (vol>0); ice_fct.F90:175
    # a/b/c[n,q] (C lines 198-203)
    a = (dx[:, :, None] * (um[:, None, None] + U[:, None, :])
         + dy[:, :, None] * (vm[:, None, None] + V[:, None, :])) / 12.0
    b = diff[:, None, None] * (dx[:, :, None] * dx[:, None, :] + dy[:, :, None] * dy[:, None, :])
    udn = um[:, None] * dx + vm[:, None] * dy               # [elem,3] = um·∂N_n/∂x + vm·∂N_n/∂y
    c = 0.5 * dt * (udn[:, :, None] * udn[:, None, :]) / 9.0
    entries = vol[:, None, None] * dt * (a - b - c)         # [elem, n, q]
    on = (mesh.ulevels <= 1)[:, None]

    def rhs_of(X):
        Xv = X[en]                                          # [elem,3] (q)
        contrib = jnp.where(on, (entries * Xv[:, None, :]).sum(axis=2), 0.0)   # [elem, n]
        return ops.scatter_add(contrib, en, mesh.nod2D, inv=getattr(mesh, "inv_elem_nodes", None))

    return rhs_of(m_ice), rhs_of(a_ice), rhs_of(m_snow)


def _solve_low_order(cfg: IceConfig, mesh: Mesh, a_ice, m_ice, m_snow, rhs_a, rhs_m, rhs_ms):
    """Monotone low-order solution (``ice_solve_low_order``): ``X_l = (rhs + γ·mm·X)/area +
    (1-γ)·X``."""
    g = cfg.ice_gamma_fct
    area = mesh.area[:, 0]
    inv_area = 1.0 / jnp.where(area > 0, area, 1.0)
    nc = mesh.ulevels_nod2D <= 1

    def low(X, rhs):
        return jnp.where(nc, (rhs + g * _mm_times(mesh, X)) * inv_area + (1.0 - g) * X, X)

    return low(a_ice, rhs_a), low(m_ice, rhs_m), low(m_snow, rhs_ms)


def _solve_high_order(cfg: IceConfig, mesh: Mesh, rhs_a, rhs_m, rhs_ms, exch=None):
    """High-order increment ``dvalues`` (``ice_solve_high_order``): first approx ``rhs/area``,
    then 2 residual-correction passes ``d ← d + (rhs - mm·d)/area``.

    Sharding: ``_mm_times`` reads ``d`` at the element's HALO vertices (then scatters), so each
    pass's OWNED output needs ``d`` complete on the halo — and the scatter leaves the new halo
    INCOMPLETE for the next pass. ``exch`` refreshes ``d`` BEFORE each ``mm·d`` (the smoother
    idiom; the C's per-iter ``dvalues`` halo, ``SYNC_MAP`` M4.3c)."""
    area = mesh.area[:, 0]
    inv_area = 1.0 / jnp.where(area > 0, area, 1.0)
    nc = mesh.ulevels_nod2D <= 1
    _exch = (lambda f, k: f) if exch is None else exch       # noqa: E731

    def hi(rhs):
        d = jnp.where(nc, rhs * inv_area, 0.0)
        for _ in range(2):                                  # num_iter_solve-1 = 2 (C line 309)
            d = _exch(d, "nod")                             # refresh halo before mm·d reads it
            d = jnp.where(nc, d + (rhs - _mm_times(mesh, d)) * inv_area, d)
        return d

    return hi(rhs_a), hi(rhs_m), hi(rhs_ms)


def _fem_fct(cfg: IceConfig, mesh: Mesh, vals, vals_l, dvals, exch=None,
             freeze_limiter=False, freeze_limited_flux=False, pin_bounds=False,
             freeze_bounds=False, freeze_extremes=False, freeze_extremes_at=None,
             freeze_bounds_at=None):
    """The Zalesak limiter for one tracer (``ice_fem_fct``, ``:351-518``). Returns the
    flux-corrected tracer ``= vals_l + limited antidiffusive flux``.

    Sharding: the clipping ratios ``pplus``/``pminus`` (``icepplus``/``icepminus``) are read at
    the element's HALO vertices by the limiter (``cand = pplus[en]``), but are built from the
    SCATTER sums ``fp``/``fm`` (incomplete on the halo) — so ``exch`` refreshes them before the
    element gather (``fesom_ice_fct.c:474``, ``SYNC_MAP`` M4.3c). ``vals_l``/``dvals`` are
    exchanged by the caller (``fct_solve``) before this.

    ``freeze_limiter`` (derivative only, the forward is unchanged): the limiter factors ``ae``
    enter the tangent as constants, as ``AdjointConfig.freeze_fct_limiter`` does for the ocean
    FCT; the antidiffusive fluxes and the low-order solution stay differentiated.
    ``freeze_limited_flux`` (derivative only): the whole limited antidiffusive correction
    enters as a constant, so the tangent is that of the low-order solution, as
    ``AdjointConfig.freeze_advection`` does for the ocean FCT.
    ``pin_bounds`` (derivative only): a bound the node attains itself (``tmax = 0`` or
    ``tmin = 0``) has no tangent. There the forward value is a tie (open water: every value of
    the patch is 0, and the clipping ratios of all three vertices are 0), the min/max
    derivatives average over the tied entries, and a vertex whose flux sum is ~1e-11 gets into
    the average: the farc one-step m_ice tangent has a top singular value of 2e2-4e3 from
    this alone, 5-10 with it removed.
    ``freeze_bounds`` (derivative only): the bounds ``tmax``/``tmin`` enter the clipping ratios
    as constants everywhere; their flux-sum denominators stay differentiated, so a clipped
    node's limited correction still saturates in the tangent (unlike ``freeze_limiter``). The
    one-step top singular value is 1.6-2.2 and the 24-step reverse product contracts (0.8) at
    every farc state tried, including the Gulf of Anadyr pattern that ``pin_bounds`` leaves
    growing (x2.4 per step).
    ``freeze_extremes`` (derivative only): the cluster extremes ``hi``/``lo`` enter as constants,
    the low-order value inside ``tmax = hi - vals_l`` stays differentiated. Alone it leaves the
    open-water ties (σ 1e3-6e3); with ``pin_bounds`` σ is 5-7 and the 24-step reverse product
    0.8 at three of four farc states (4.4 at the fourth, day 200).
    ``freeze_extremes_at`` (a ``[nod2D]`` bool mask, derivative only): ``freeze_extremes`` on
    those nodes only (the coast, ``IceConfig.adjoint_cuts`` "fct_hilo_coast"). With
    ``pin_bounds`` it removes the Gulf of Anadyr pattern (24-step product 7e8 -> 0.2) while 96 %
    of the farc nodes keep the exact bound derivative.
    ``freeze_bounds_at`` (a ``[nod2D]`` bool mask, derivative only): ``freeze_bounds`` on those
    nodes only (the coast and its first ring, :func:`coast_ring`, ``IceConfig.adjoint_cuts``
    "fct_bound_coast"). Freezing only the extremes there is not enough: at a node that attains
    its own cluster extreme, a constant ``hi`` leaves ``-d vals_l / f`` in the clipping ratio.
    On farc that grew a snow pattern on a Baffin Island fjord (day 50 of 2013, x8.3 over 24
    steps with the extremes frozen on the coast, x94 at an Antarctic ring node with them frozen
    on the first ring as well); with the whole bound constant on the coast and its first ring
    (7.6 % of the nodes) every state tried contracts or stays at the pin-only level."""
    _exch = (lambda f, k: f) if exch is None else exch       # noqa: E731
    en = mesh.elem_nodes
    g = cfg.ice_gamma_fct
    area = mesh.area[:, 0]
    inv_area = 1.0 / jnp.where(area > 0, area, 1.0)
    nc_e = mesh.ulevels <= 1
    nc_n = mesh.ulevels_nod2D <= 1
    vol = mesh.elem_area

    # antidiffusive flux (icoef = I·(-3) + 1 ⇒ s[q] = Σv - 3·v[q]); /area(en[q]) /12, negated.
    w = g * vals + dvals
    wv = w[en]                                              # [elem,3]
    s = wv.sum(axis=1, keepdims=True) - 3.0 * wv           # [elem,3] (q)
    icefluxes = jnp.where(nc_e[:, None], -s * (vol[:, None] * inv_area[en]) / 12.0, 0.0)

    # cluster bounds: min/max over a node's element-neighbours of max(vals_l, vals).
    elem_hi = jnp.maximum(vals_l[en].max(1), vals[en].max(1))   # [elem]
    elem_lo = jnp.minimum(vals_l[en].min(1), vals[en].min(1))
    hi_flat = jnp.broadcast_to(jnp.where(nc_e, elem_hi, -jnp.inf)[:, None], (en.shape[0], 3))
    lo_flat = jnp.broadcast_to(jnp.where(nc_e, elem_lo, jnp.inf)[:, None], (en.shape[0], 3))
    hi = jax.ops.segment_max(hi_flat.reshape(-1), en.reshape(-1), num_segments=mesh.nod2D)
    lo = jax.ops.segment_min(lo_flat.reshape(-1), en.reshape(-1), num_segments=mesh.nod2D)
    # Pad / all-non-cell nodes keep the ±inf sentinel (empty/masked segment). The clipping
    # ratios below mask them (`fp>0`/`fm<0`), but ±inf poisons the AD backward (0·inf=NaN —
    # the sharded reverse pass exposes it). Clamp to finite (forward byte-identical; the
    # masked ratios are unchanged on every live node, where hi/lo are finite).
    hi = jnp.where(jnp.isfinite(hi), hi, 0.0)
    lo = jnp.where(jnp.isfinite(lo), lo, 0.0)
    if freeze_extremes:
        hi, lo = jax.lax.stop_gradient(hi), jax.lax.stop_gradient(lo)
    if freeze_extremes_at is not None:
        hi = jnp.where(freeze_extremes_at, jax.lax.stop_gradient(hi), hi)
        lo = jnp.where(freeze_extremes_at, jax.lax.stop_gradient(lo), lo)
    tmax = hi - vals_l
    tmin = lo - vals_l
    if pin_bounds:
        tmax = jnp.where(tmax > 0.0, tmax, jax.lax.stop_gradient(tmax))
        tmin = jnp.where(tmin < 0.0, tmin, jax.lax.stop_gradient(tmin))
    if freeze_bounds:
        tmax, tmin = jax.lax.stop_gradient(tmax), jax.lax.stop_gradient(tmin)
    if freeze_bounds_at is not None:
        tmax = jnp.where(freeze_bounds_at, jax.lax.stop_gradient(tmax), tmax)
        tmin = jnp.where(freeze_bounds_at, jax.lax.stop_gradient(tmin), tmin)

    # +/- flux sums per node, then the clipping ratios (C lines 440-472).
    fp = ops.scatter_add(jnp.maximum(icefluxes, 0.0), en, mesh.nod2D, inv=getattr(mesh, "inv_elem_nodes", None))
    fm = ops.scatter_add(jnp.minimum(icefluxes, 0.0), en, mesh.nod2D, inv=getattr(mesh, "inv_elem_nodes", None))
    pplus = jnp.where(fp > 0.0,
                      jnp.minimum(tmax / jnp.where(fp > _ICE_FLUX_EPS, fp, _ICE_FLUX_EPS), 1.0),
                      0.0)
    pminus = jnp.where(fm < 0.0,
                       jnp.minimum(tmin / jnp.where(fm < -_ICE_FLUX_EPS, fm, -_ICE_FLUX_EPS), 1.0),
                       0.0)
    pplus = jnp.where(nc_n, pplus, 0.0)
    pminus = jnp.where(nc_n, pminus, 0.0)
    pplus = _exch(pplus, "nod")        # icepplus: read at HALO vertices by the limiter below
    pminus = _exch(pminus, "nod")      # icepminus

    # limit element fluxes: ae = min over the 3 vertices of (pplus|pminus by flux sign).
    cand = jnp.where(icefluxes >= 0.0, pplus[en], pminus[en])   # [elem,3]
    ae = cand.min(axis=1)                                       # [elem]
    if freeze_limiter:
        ae = jax.lax.stop_gradient(ae)
    limited = jnp.where(nc_e[:, None], icefluxes * ae[:, None], 0.0)

    # apply: vals_l + scatter(limited).  vals_l already == vals on cavity nodes.
    corr = ops.scatter_add(limited, en, mesh.nod2D, inv=getattr(mesh, "inv_elem_nodes", None))
    if freeze_limited_flux:
        corr = jax.lax.stop_gradient(corr)
    return vals_l + corr


def coast_ring(mesh: Mesh, boundary_node) -> jnp.ndarray:
    """The coastal nodes and their first ring (``[nod2D]`` bool): ``boundary_node`` plus every
    vertex of a cell element (``ulevels <= 1``, the elements the limiter reads) that has a
    coastal vertex. Sharding: pass the partitioned global ``boundary_node``; the ring is right
    on the owned nodes (all their elements are local), and the halo copies of the clipping
    ratios it masks are refreshed from the owners."""
    bn = jnp.asarray(boundary_node, bool)
    en = mesh.elem_nodes
    touch = jnp.where(mesh.ulevels <= 1, bn[en].any(axis=1), False)              # [elem]
    hits = ops.scatter_add(jnp.broadcast_to(touch[:, None], en.shape).astype(jnp.int32), en,
                           mesh.nod2D)
    return bn | (hits > 0)


def fct_solve(cfg: IceConfig, mesh: Mesh, a_ice, m_ice, m_snow, u_ice, v_ice, exch=None,
              freeze_limiter=False, freeze_limited_flux=False, pin_bounds=False,
              freeze_bounds=False, freeze_extremes=False, freeze_extremes_at=None,
              freeze_bounds_at=None):
    """Advance the 3 ice tracers one step by FCT (``fesom_ice_fct_solve``). Returns the
    advected ``(a_ice, m_ice, m_snow)`` (before ``cut_off``).

    Sharding (``exch=None`` ⇒ byte-identical): ``_fem_fct`` reads both the low-order ``X_l``
    (cluster min/max + ``tmax=hi-X_l``) and the high-order ``dvals`` (``w=γ·vals+dvals``) at
    the element's HALO vertices, and both are SCATTER results (incomplete on the halo) — so
    refresh them before the limiter (``SYNC_MAP`` M4.3c: the low-order ``a_l/m_l/ms_l`` + the
    high-order ``dvalues`` halos). The high-order solve's per-iteration ``dvalues`` refresh +
    the limiter's ``icepplus/icepminus`` refresh are owned by ``_solve_high_order``/``_fem_fct``.
    ``freeze_limiter``, ``freeze_limited_flux``, ``pin_bounds``, ``freeze_bounds``,
    ``freeze_extremes``, ``freeze_extremes_at``, ``freeze_bounds_at``: see ``_fem_fct``
    (``IceConfig.adjoint_cuts`` "fct_limiter", "fct_flux", "fct_pin", "fct_bound", "fct_hilo",
    "fct_hilo_coast", "fct_bound_coast")."""
    _exch = (lambda f, k: f) if exch is None else exch       # noqa: E731
    rhs_m, rhs_a, rhs_ms = _tg_rhs(cfg, mesh, u_ice, v_ice, a_ice, m_ice, m_snow)
    da, dm, dms = _solve_high_order(cfg, mesh, rhs_a, rhs_m, rhs_ms, exch=exch)
    a_l, m_l, ms_l = _solve_low_order(cfg, mesh, a_ice, m_ice, m_snow, rhs_a, rhs_m, rhs_ms)
    # refresh the low- + high-order halos read at element vertices inside the limiter.
    a_l, m_l, ms_l = _exch(a_l, "nod"), _exch(m_l, "nod"), _exch(ms_l, "nod")
    da, dm, dms = _exch(da, "nod"), _exch(dm, "nod"), _exch(dms, "nod")
    kw = dict(exch=exch, freeze_limiter=freeze_limiter, freeze_limited_flux=freeze_limited_flux,
              pin_bounds=pin_bounds, freeze_bounds=freeze_bounds, freeze_extremes=freeze_extremes,
              freeze_extremes_at=freeze_extremes_at, freeze_bounds_at=freeze_bounds_at)
    m_new = _fem_fct(cfg, mesh, m_ice, m_l, dm, **kw)
    a_new = _fem_fct(cfg, mesh, a_ice, a_l, da, **kw)
    ms_new = _fem_fct(cfg, mesh, m_snow, ms_l, dms, **kw)
    return a_new, m_new, ms_new
