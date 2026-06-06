"""Upwind tracer advection — substep 15 (advection part), Task 2.10.

Literal vectorized port of the **upwind (non-FCT)** tracer-advection path of
``fesom_tracer_adv.c`` for the Phase-2 pi config (linfs ALE, ``use_wsplit=0`` ⇒
``w_e = w``, single MPI rank). Drives one tracer per call, mirroring
``fesom_tracer_advect_one`` (``fesom_tracer_adv.c:1269``):

1. :func:`tracer_ab` — AB2 extrapolation ``ttfAB = -(0.5+ε)·valuesold +
   (1.5+ε)·values`` and the ``valuesold := values`` save (``init_tracers_AB_one``,
   ``:174``; ε=0.1). The fluxes use ``ttfAB``; the ALE reconstruction uses
   ``values``. At **step 1** ``valuesold == values`` ⇒ ``ttfAB == values``.
2. :func:`adv_flux_hor` — horizontal upwind edge flux (``adv_tra_hor_upw1``,
   ``:212``): the volume flux ``vflux`` across each edge (summed over the two
   adjacent cells, each masked to its own layer range — the C's 5 level "zones"
   collapse to one masked sum) drives the upwind face value
   ``-½(T₁(vflux+|vflux|) + T₂(vflux−|vflux|))``.
3. :func:`adv_flux_ver` — vertical upwind flux (``adv_tra_ver_upw1``, ``:701``)
   using ``w_e``: ``-½(T[nz](w+|w|) + T[nz−1](w−|w|))·area``; the edge-replicated
   ``T[nz−1]`` reproduces the C's surface term ``-w·T·area`` at ``nzmin`` and the
   no-flux bottom ``flux_v[nzmax]=0`` falls out of the layer mask.
4. :func:`flux2dtracer` — divergence assembly (``flux2dtracer_upwind``, ``:740``):
   vertical ``(flux_v[nz]−flux_v[nz+1])·dt/areasvol`` + the antisymmetric edge→node
   horizontal scatter ``±flux_h·dt/areasvol``.
5. :func:`ale_reconstruct` — ``del_ttf += T·(hnode−hnode_new)`` (≡0 in linfs) then
   ``T += del_ttf/hnode_new`` (``ale_reconstruct``, ``:792``).

⚠️ The C dump runs **FCT**, this runs **upwind**. ``S=35`` is horizontally constant
⇒ advects trivially (upwind == FCT == the dump, a clean step-1 gate). ``T`` has the
Gaussian blob ⇒ upwind ≠ FCT in the curved region, so the step-1 ``T`` dump match is
a Phase-4 (FCT) gate; here ``T`` is verified against an independent numpy upwind
reference + a constant-tracer property. (See ``docs/REFERENCE_RUNS.md``.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import ops
from .config import DT_DEFAULT, R_EARTH
from .mesh import Mesh

# AB2 stabilization offset (ε=0.1, oce_modules.F90:92 / fesom_tracer_adv.c:185-187).
_EPS_AB = 0.1
_AB_OLD = -(0.5 + _EPS_AB)   # valuesold coefficient
_AB_NEW = (1.5 + _EPS_AB)    # values   coefficient


def _shift_down(x):
    """``out[...,k] = x[...,k-1]``, edge-replicated at k=0 (surface)."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


def _shift_up(x):
    """``out[...,k] = x[...,k+1]``, zero-padded at the last level."""
    return jnp.concatenate([x[..., 1:], jnp.zeros_like(x[..., :1])], axis=-1)


def tracer_ab(values, valuesold):
    """AB2 extrapolation + the ``valuesold := values`` save (``init_tracers_AB_one``,
    ``fesom_tracer_adv.c:174``). Returns ``(ttfAB, valuesold_new)`` where
    ``ttfAB = -(0.5+ε)·valuesold + (1.5+ε)·values`` and ``valuesold_new = values``.
    At step 1 ``valuesold == values`` (``ic.initial_state`` sets ``T_old=T``) so
    ``ttfAB == values``."""
    ttfAB = _AB_OLD * valuesold + _AB_NEW * values
    return ttfAB, values


def edge_vflux(mesh: Mesh, uv, helem):
    """Per-edge volume flux ``[edge2D, nl]`` = the sum of each adjacent cell's
    contribution, masked to that cell's layer range.

    The C's 5 explicit level-zones (el1-only above / el2-only above / both /
    el1-only below / el2-only below) collapse to ``vflux = mask₁·flux₁ +
    mask₂·flux₂`` per level. ``el1`` uses ``(u·dy₁ − v·dx₁)·h`` and ``el2`` uses
    ``(v·dx₂ − u·dy₂)·h`` (the C sign convention). Shared by the upwind face
    (:func:`adv_flux_hor`) and the high-order MFCT flux (:func:`adv_flux_hor_ho`)
    — both transport the *same* volume across the edge."""
    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s = jnp.where(has1, el1, 0)
    el2s = jnp.where(has2, el2, 0)
    cross = mesh.edge_cross_dxdy
    dx1, dy1 = cross[:, 0:1], cross[:, 1:2]
    dx2, dy2 = cross[:, 2:3], cross[:, 3:4]
    U, V = uv[:, :, 0], uv[:, :, 1]
    lm = mesh.elem_layer_mask
    t1 = jnp.where(lm[el1s] & has1[:, None],
                   (U[el1s] * dy1 - V[el1s] * dx1) * helem[el1s], 0.0)
    t2 = jnp.where(lm[el2s] & has2[:, None],
                   (V[el2s] * dx2 - U[el2s] * dy2) * helem[el2s], 0.0)
    return t1 + t2                                        # (edge, nl)


def adv_flux_hor(mesh: Mesh, uv, helem, ttf):
    """Horizontal upwind flux ``[edge2D, nl]`` (``adv_tra_hor_upw1``,
    ``fesom_tracer_adv.c:212``). The upwind face value is
    ``-½(T₁(vflux+|vflux|) + T₂(vflux−|vflux|))`` (T₁ if ``vflux>0`` flows n1→n2,
    else T₂), with ``vflux`` the per-edge volume flux (:func:`edge_vflux`)."""
    vflux = edge_vflux(mesh, uv, helem)
    T1 = ttf[mesh.edges[:, 0]]                            # (edge, nl)
    T2 = ttf[mesh.edges[:, 1]]
    av = jnp.abs(vflux)
    return -0.5 * (T1 * (vflux + av) + T2 * (vflux - av))


def adv_flux_ver(mesh: Mesh, w_e, ttf):
    """Vertical upwind flux ``[nod2D, nl]`` at interfaces (``adv_tra_ver_upw1``,
    ``fesom_tracer_adv.c:701``), using ``W = w_e``. The unified interior formula
    ``-½(T[nz](w+|w|) + T[nz−1](w−|w|))·area`` with the edge-replicated ``T[nz−1]``
    reproduces the C's surface flux ``-W·T·area`` at ``nzmin`` (½·2w·T = w·T); the
    bottom interface ``flux_v[nzmax]=0`` falls out of the layer mask (support
    ``[nzmin, nzmax)``)."""
    w = w_e
    aw = jnp.abs(w)
    T_below = ttf                                         # T[nz]
    T_above = _shift_down(ttf)                            # T[nz-1] (→ T[nzmin] at top)
    flux = -0.5 * (T_below * (w + aw) + T_above * (w - aw)) * mesh.area
    return ops.mask_below_bottom(flux, mesh.node_layer_mask)


def flux2dtracer(mesh: Mesh, flux_h, flux_v, *, dt: float = DT_DEFAULT):
    """Assemble ``del_ttf`` ``[nod2D, nl]`` from the horizontal + vertical fluxes
    (``flux2dtracer_upwind``, ``fesom_tracer_adv.c:740``).

    Vertical divergence ``(flux_v[nz] − flux_v[nz+1])·dt/areasvol[nz]``; horizontal
    antisymmetric edge→node scatter ``flux_h → +n1 / −n2`` then ``·dt/areasvol``
    (the per-receiver ``÷areasvol`` factors out of the edge scatter since it depends
    only on the node+level). The C's ``if(area>0)`` guard → a safe divide + the
    layer mask."""
    areasvol = mesh.areasvol
    safe = jnp.where(areasvol > 0.0, areasvol, 1.0)
    lm = mesh.node_layer_mask
    dttf_v = jnp.where(lm, (flux_v - _shift_up(flux_v)) * dt / safe, 0.0)
    vals = jnp.stack([flux_h, -flux_h], axis=1)          # (edge, 2, nl)
    raw = ops.scatter_add(vals, mesh.edges, mesh.nod2D)  # (nod2D, nl)
    dttf_h = jnp.where(lm, raw * dt / safe, 0.0)
    return dttf_h + dttf_v


def ale_reconstruct(mesh: Mesh, T, del_ttf, hnode, hnode_new):
    """ALE tracer reconstruction (``ale_reconstruct``, ``fesom_tracer_adv.c:792``):
    ``del_ttf += T·(hnode − hnode_new)`` then ``T += del_ttf/hnode_new`` on valid
    layers with ``hnode_new>0``. In **linfs** ``hnode == hnode_new`` ⇒ the thickness
    term vanishes and ``T_new = T + del_ttf/hnode``."""
    del_ttf = del_ttf + T * (hnode - hnode_new)
    safe_hn = jnp.where(hnode_new > 0.0, hnode_new, 1.0)
    upd = mesh.node_layer_mask & (hnode_new > 0.0)
    dT = jnp.where(upd, del_ttf / safe_hn, 0.0)
    return T + dT


def advect_one(mesh: Mesh, uv, w_e, helem, hnode, hnode_new, T, T_old,
               *, dt: float = DT_DEFAULT):
    """One tracer's upwind advection + ALE reconstruction
    (``fesom_tracer_advect_one``, ``fesom_tracer_adv.c:1269``). Returns
    ``(T_new, T_old_new)`` with ``T_old_new = T`` (the AB2 ``valuesold`` save). The
    fluxes use the AB2-extrapolated ``ttfAB``; the reconstruction updates ``T``."""
    ttfAB, T_old_new = tracer_ab(T, T_old)
    flux_h = adv_flux_hor(mesh, uv, helem, ttfAB)
    flux_v = adv_flux_ver(mesh, w_e, ttfAB)
    del_ttf = flux2dtracer(mesh, flux_h, flux_v, dt=dt)
    T_new = ale_reconstruct(mesh, T, del_ttf, hnode, hnode_new)
    return T_new, T_old_new


# ==========================================================================
# FCT (flux-corrected transport, Zalesak limiter) — Task 4.1
# --------------------------------------------------------------------------
# The dump's *live* tracer advection. The monotone upwind solution above is the
# **low-order (LO)** scheme; FCT adds a high-order (HO) flux and limits the
# antidiffusive part ``HO − LO`` so the update introduces no new extrema. Driver
# ``fesom_tracer_advect_one_fct`` (``fesom_tracer_adv.c:1199``):
#
#   1. AB2 ``ttfAB`` (:func:`tracer_ab`).
#   2. LO upwind fluxes — from **values** (T), NOT ttfAB (the C calls upw1 with
#      ``vals``).  3. :func:`compute_fct_lo` advances the LO ALE solution.
#   4. HO fluxes (``init_zero=false`` ⇒ ``adf := HO − LO``): horizontal MFCT
#      3rd-order (:func:`adv_flux_hor_ho`) using ttfAB + element/up-dn gradients
#      built from **values**; vertical QR 4th-order (:func:`adv_flux_ver_ho`).
#   5. :func:`zalesak_limit` — the limiter (the AD-hard kinks; see
#      ``docs/LIMITER_GRADIENTS.md``).  6. :func:`flux2dtracer_fct` adds the LO
#      transition + the limited antidiffusive divergence.  8. ALE reconstruct.
#
# Result: ``T_new = LO + limited_antidiff / areasvol / hnode_new``. ``S=35``
# (constant) ⇒ HO==LO==35, antidiff 0, so S is preserved exactly as in upwind.
# pi config: no cavities (``ulevels≡1`` ⇒ ``nzmin=0``), single rank
# (``myDim_edge2D == edge2D``). ``edge_up_dn_tri`` carries −1 at boundary edges.
# ==========================================================================

_FCT_FLUX_EPS = 1e-16    # divide-by-zero guard in the limiter ratio (C flux_eps)
_FCT_BIG = 1e3           # ±bignumber pad below a cell's bottom (C bignumber)


def adv_flux_hor_ho(mesh: Mesh, uv, helem, ttf, eud):
    """High-order MFCT (3rd-order) horizontal flux ``[edge2D, nl]``
    (``adv_tra_hor_mfct``, ``fesom_tracer_adv.c:547``, ``num_ord=0``).

    Same edge volume flux ``vflux`` as upwind, but the face values are the
    reconstructed ``Tmean1/Tmean2`` built from ``ttf`` (= ttfAB) plus the
    up/down-triangle gradient correction ``eud`` (``[edge2D, nl, 4]``, built from
    **values** by :func:`fill_up_dn_grad`). ``a = R·cos`` averaged over the two
    cells; ``edx,edy = edge_dxdy`` (rotated radians)."""
    vflux = edge_vflux(mesh, uv, helem)
    n1, n2 = mesh.edges[:, 0], mesh.edges[:, 1]
    T1, T2 = ttf[n1], ttf[n2]                            # (edge, nl)
    diff = T2 - T1
    g0, g1, g2, g3 = eud[:, :, 0], eud[:, :, 1], eud[:, :, 2], eud[:, :, 3]

    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    has2 = el2 >= 0
    el1s = jnp.where(el1 >= 0, el1, 0)
    el2s = jnp.where(has2, el2, 0)
    a = R_EARTH * mesh.elem_cos[el1s]
    a = jnp.where(has2, 0.5 * (a + R_EARTH * mesh.elem_cos[el2s]), a)
    a = a[:, None]                                        # (edge, 1)
    edx = mesh.edge_dxdy[:, 0:1]
    edy = mesh.edge_dxdy[:, 1:2]

    Tmean1 = T1 + (2.0 * diff + edx * a * g0 + edy * R_EARTH * g2) / 6.0
    Tmean2 = T2 - (2.0 * diff + edx * a * g1 + edy * R_EARTH * g3) / 6.0
    av = jnp.abs(vflux)
    return -0.5 * ((vflux + av) * Tmean1 + (vflux - av) * Tmean2)


def _z_stencil(Z, nl):
    """Pad ``Z`` (mid-layer depths, ``[nl-1]``) to ``[nl]`` and return the four
    shifted views ``(Z[nz], Z[nz-1], Z[nz-2], Z[nz+1])`` along the level axis,
    edge-replicated at the boundaries. The padded tail duplicates ``Z[-1]`` so the
    only zero stencil differences sit at masked-out levels (guarded downstream)."""
    Zp = jnp.concatenate([Z, Z[-1:]])                    # (nl,)
    Z_nz = Zp
    Z_m1 = jnp.concatenate([Zp[:1], Zp[:-1]])            # Z[nz-1]
    Z_m2 = jnp.concatenate([Zp[:2], Zp[:-2]])            # Z[nz-2]
    Z_p1 = jnp.concatenate([Zp[1:], Zp[-1:]])            # Z[nz+1]
    return Z_nz, Z_m1, Z_m2, Z_p1


def adv_flux_ver_ho(mesh: Mesh, w_e, ttf):
    """High-order QR 4th-order vertical flux ``[nod2D, nl]`` at interfaces
    (``adv_tra_ver_qr4c``, ``fesom_tracer_adv.c:621``, ``num_ord=1``).

    Surface ``-ttf·W·area``; the 2nd and bottom-minus-1 interfaces use centred
    differences ``-½(T[nz-1]+T[nz])·W·area``; the interior uses the quadratic
    4th-order reconstruction; the bottom interface is 0. ``ttf = ttfAB``."""
    W = w_e
    area = mesh.area
    T = ttf
    Tm1 = _shift_down(T)                                 # T[nz-1]
    Tm2 = _shift_down(Tm1)                               # T[nz-2]
    Tp1 = _shift_up(T)                                   # T[nz+1]

    surf = -T * W * area
    cent = -0.5 * (Tm1 + T) * W * area

    Z_nz, Z_m1, Z_m2, Z_p1 = _z_stencil(mesh.Z, mesh.nl)
    zb = mesh.zbar                                        # (nl,) interface depths

    def _safe(x):                                        # guard zero stencil gaps
        return jnp.where(x == 0.0, 1.0, x)
    qc = (Tm1 - T) / _safe(Z_m1 - Z_nz)
    qu = (T - Tp1) / _safe(Z_nz - Z_p1)
    qd = (Tm2 - Tm1) / _safe(Z_m2 - Z_m1)
    Tmean1 = T + (2.0 * qc + qu) * (zb - Z_nz) / 3.0
    Tmean2 = Tm1 + (2.0 * qc + qd) * (zb - Z_m1) / 3.0
    quad = -0.5 * (Tmean1 + Tmean2) * W * area           # num_ord=1

    k = jnp.arange(mesh.nl)[None, :]
    nzmin = (mesh.ulevels_nod2D - 1)[:, None]
    nzmax = (mesh.nlevels_nod2D - 1)[:, None]
    is_surf = k == nzmin
    is_bot = k == nzmax
    is_cent = (k == nzmin + 1) | (k == nzmax - 1)
    is_int = (k >= nzmin + 2) & (k <= nzmax - 2)
    hi = jnp.where(is_surf, surf,
                   jnp.where(is_bot, 0.0,
                             jnp.where(is_cent, cent,
                                       jnp.where(is_int, quad, 0.0))))
    return ops.mask_below_bottom(hi, mesh.node_iface_mask)


def compute_fct_lo(mesh: Mesh, flux_h, flux_v, T, hnode, hnode_new,
                   *, dt: float = DT_DEFAULT):
    """The low-order (upwind) ALE solution advanced one step
    (``fesom_tracer_compute_fct_LO``, ``fesom_tracer_adv.c:109``):
    ``LO = (T·hnode + (div_h + (f_top−f_bot))·dt/areasvol) / hnode_new``.
    The LO fluxes are the upwind fluxes evaluated on **values** (T)."""
    vals = jnp.stack([flux_h, -flux_h], axis=1)          # (edge, 2, nl)
    div_h = ops.scatter_add(vals, mesh.edges, mesh.nod2D)
    div_v = flux_v - _shift_up(flux_v)
    areasvol = mesh.areasvol
    safe_a = jnp.where(areasvol > 0.0, areasvol, 1.0)
    safe_hn = jnp.where(hnode_new > 0.0, hnode_new, 1.0)
    numer = T * hnode + (div_h + div_v) * dt / safe_a
    lo = numer / safe_hn
    valid = mesh.node_layer_mask & (hnode_new > 0.0) & (areasvol > 0.0)
    return jnp.where(valid, lo, 0.0)


def tracer_gradient_elements(mesh: Mesh, ttf):
    """Per-element tracer gradient ``[elem2D, nl, 2]``
    (``tracer_gradient_elements``, ``fesom_tracer_adv.c:445``): contract the
    element's 3 node values with ``gradient_sca`` (``[:3]=∂N/∂x``, ``[3:]=∂N/∂y``).
    Built from **values** (the C passes ``vals``)."""
    t = ttf[mesh.elem_nodes]                             # (elem, 3, nl)
    gs = mesh.gradient_sca
    gx = gs[:, 0:1] * t[:, 0] + gs[:, 1:2] * t[:, 1] + gs[:, 2:3] * t[:, 2]
    gy = gs[:, 3:4] * t[:, 0] + gs[:, 4:5] * t[:, 1] + gs[:, 5:6] * t[:, 2]
    m = mesh.elem_layer_mask
    return jnp.stack([jnp.where(m, gx, 0.0), jnp.where(m, gy, 0.0)], axis=-1)


def _node_avg_grad(mesh: Mesh, tr_xy):
    """Area-weighted mean of ``tr_xy`` over each node's surrounding cells active
    at that level → ``[nod2D, nl, 2]`` (``node_avg_grad``,
    ``fesom_tracer_adv.c:472``). An element→node area-weighted scatter."""
    aw = jnp.where(mesh.elem_layer_mask, mesh.elem_area[:, None], 0.0)  # (elem, nl)
    contrib = tr_xy * aw[:, :, None]                     # (elem, nl, 2)
    en = mesh.elem_nodes                                 # (elem, 3)
    e3 = (mesh.elem2D, 3)
    cb = jnp.broadcast_to(contrib[:, None], e3 + tr_xy.shape[1:])
    wb = jnp.broadcast_to(aw[:, None], e3 + (mesh.nl,))
    num = ops.scatter_add(cb, en, mesh.nod2D)            # (nod, nl, 2)
    den = ops.scatter_add(wb, en, mesh.nod2D)            # (nod, nl)
    return num / jnp.where(den > 0.0, den, 1.0)[:, :, None]


def fill_up_dn_grad(mesh: Mesh, tr_xy):
    """Per-edge MFCT gradient ``[edge2D, nl, 4]`` (``fill_up_dn_grad``,
    ``fesom_tracer_adv.c:494``): in the **shared** level range of an interior edge
    use the up/down-triangle gradient; elsewhere (above/below the shared range, and
    on boundary edges) use the node-averaged gradient. Slots: ``[0,2]`` = (x,y) for
    n1, ``[1,3]`` = (x,y) for n2."""
    gnode = _node_avg_grad(mesh, tr_xy)                  # (nod, nl, 2)
    up, dn = mesh.edge_up_dn_tri[:, 0], mesh.edge_up_dn_tri[:, 1]
    interior = (up >= 0) & (dn >= 0)
    gup = tr_xy[jnp.where(up >= 0, up, 0)]               # (edge, nl, 2)
    gdn = tr_xy[jnp.where(dn >= 0, dn, 0)]
    n1, n2 = mesh.edges[:, 0], mesh.edges[:, 1]
    gn1, gn2 = gnode[n1], gnode[n2]

    k = jnp.arange(mesh.nl)[None, :]
    nzmin = jnp.maximum(mesh.ulevels_nod2D_max[n1],
                        mesh.ulevels_nod2D_max[n2])[:, None] - 1
    nzmax = jnp.minimum(mesh.nlevels_nod2D_min[n1],
                        mesh.nlevels_nod2D_min[n2])[:, None] - 1
    use_tri = interior[:, None] & (k >= nzmin) & (k < nzmax)   # (edge, nl)
    g0 = jnp.where(use_tri, gup[:, :, 0], gn1[:, :, 0])
    g1 = jnp.where(use_tri, gdn[:, :, 0], gn2[:, :, 0])
    g2 = jnp.where(use_tri, gup[:, :, 1], gn1[:, :, 1])
    g3 = jnp.where(use_tri, gdn[:, :, 1], gn2[:, :, 1])
    return jnp.stack([g0, g1, g2, g3], axis=-1)          # (edge, nl, 4)


def zalesak_limit(mesh: Mesh, T, LO, adf_h, adf_v, hnode_new, *,
                  dt: float = DT_DEFAULT):
    """Zalesak flux limiter (``oce_tra_adv_fct``, ``fesom_tracer_adv.c:851``).
    Limits the antidiffusive fluxes ``adf_h``/``adf_v`` so ``T_new = LO + limited``
    introduces no new local extrema. Returns ``(adf_h_lim, adf_v_lim)``.

    The min/max/sign-select kinks are differentiated as subgradients (the default
    ``jnp`` VJP); the ``flux_eps`` floor keeps every ratio finite so the backward
    pass is NaN-free. See ``docs/LIMITER_GRADIENTS.md``."""
    nl = mesh.nl
    nlm = mesh.node_layer_mask

    # a1: per-node admissible window between LO and the old field
    ttf_max = jnp.maximum(LO, T)
    ttf_min = jnp.minimum(LO, T)

    # a2: per-element max/min over the 3 vertices; pad outside the cell layer range
    # with ∓bignumber so a shallow cell never wins the per-node cluster reduction
    em = mesh.elem_layer_mask
    aux_max = jnp.where(em, jnp.max(ttf_max[mesh.elem_nodes], axis=1), -_FCT_BIG)
    aux_min = jnp.where(em, jnp.min(ttf_min[mesh.elem_nodes], axis=1), _FCT_BIG)

    # a3: per-node cluster max/min over surrounding cells (element→node)
    seg = mesh.elem_nodes.reshape(-1)
    e3 = mesh.elem2D * 3
    cmax = jnp.broadcast_to(aux_max[:, None], (mesh.elem2D, 3, nl)).reshape(e3, nl)
    cmin = jnp.broadcast_to(aux_min[:, None], (mesh.elem2D, 3, nl)).reshape(e3, nl)
    tvert_max = jax.ops.segment_max(cmax, seg, num_segments=mesh.nod2D)
    tvert_min = jax.ops.segment_min(cmin, seg, num_segments=mesh.nod2D)

    # a4: admissible increment relative to LO; the interior also clusters over the
    # 3 vertical neighbours (nz-1, nz, nz+1); surface & bottom layer use own level
    k = jnp.arange(nl)[None, :]
    nzmin = (mesh.ulevels_nod2D - 1)[:, None]
    nzmax = (mesh.nlevels_nod2D - 1)[:, None]
    use_own = (k == nzmin) | (k == nzmax - 1)
    tmax3 = jnp.maximum(jnp.maximum(_shift_down(tvert_max), tvert_max),
                        _shift_up(tvert_max))
    tmin3 = jnp.minimum(jnp.minimum(_shift_down(tvert_min), tvert_min),
                        _shift_up(tvert_min))
    fct_ttf_max = jnp.where(use_own, tvert_max, tmax3) - LO
    fct_ttf_min = jnp.where(use_own, tvert_min, tmin3) - LO

    # b1: sum positive / negative antidiffusive contributions per node
    fv_bot = _shift_up(adf_v)                            # adf_v[nz+1]
    pos_v = jnp.maximum(adf_v, 0.0) + jnp.maximum(-fv_bot, 0.0)
    neg_v = jnp.minimum(adf_v, 0.0) + jnp.minimum(-fv_bot, 0.0)
    pos_v = jnp.where(nlm, pos_v, 0.0)
    neg_v = jnp.where(nlm, neg_v, 0.0)
    ph = jnp.stack([jnp.maximum(adf_h, 0.0), jnp.maximum(-adf_h, 0.0)], axis=1)
    nh = jnp.stack([jnp.minimum(adf_h, 0.0), jnp.minimum(-adf_h, 0.0)], axis=1)
    fct_plus = pos_v + ops.scatter_add(ph, mesh.edges, mesh.nod2D)
    fct_minus = neg_v + ops.scatter_add(nh, mesh.edges, mesh.nod2D)

    # b2: per-node limiter factors in [0, 1]
    a = mesh.areasvol
    safe_a = jnp.where(a > 0.0, a, 1.0)
    safe_hn = jnp.where(hnode_new > 0.0, hnode_new, 1.0)
    flux_pos = fct_plus * dt / safe_a / safe_hn + _FCT_FLUX_EPS
    flux_neg = fct_minus * dt / safe_a / safe_hn - _FCT_FLUX_EPS
    plus_fac = jnp.minimum(1.0, fct_ttf_max / flux_pos)
    minus_fac = jnp.minimum(1.0, fct_ttf_min / flux_neg)
    valid = nlm & (a > 0.0) & (hnode_new > 0.0)
    plus_fac = jnp.where(valid, plus_fac, 1.0)
    minus_fac = jnp.where(valid, minus_fac, 1.0)

    # b3: apply limits — vertical (surface vs interior), then horizontal
    ae_surf = jnp.where(adf_v >= 0.0, plus_fac, minus_fac)
    ae_int = jnp.where(adf_v >= 0.0,
                       jnp.minimum(_shift_down(minus_fac), plus_fac),
                       jnp.minimum(_shift_down(plus_fac), minus_fac))
    ae_v = jnp.where(k == nzmin, ae_surf, ae_int)
    adf_v_lim = ops.mask_below_bottom(ae_v * adf_v, mesh.node_iface_mask)

    n1, n2 = mesh.edges[:, 0], mesh.edges[:, 1]
    ae_h = jnp.where(adf_h >= 0.0,
                     jnp.minimum(plus_fac[n1], minus_fac[n2]),
                     jnp.minimum(minus_fac[n1], plus_fac[n2]))
    adf_h_lim = ae_h * adf_h
    return adf_h_lim, adf_v_lim


def flux2dtracer_fct(mesh: Mesh, T, LO, adf_h, adf_v, hnode, hnode_new, *,
                     dt: float = DT_DEFAULT):
    """Assemble ``del_ttf`` for the FCT path (``flux2dtracer_fct``,
    ``fesom_tracer_adv.c:1141``): the LO transition ``-T·hnode + LO·hnode_new``
    plus the limited antidiffusive divergence (vertical ``(adf_v[nz]−adf_v[nz+1])``
    + the horizontal edge→node scatter), each ``·dt/areasvol``."""
    nlm = mesh.node_layer_mask
    a = mesh.areasvol
    safe_a = jnp.where(a > 0.0, a, 1.0)
    lo_trans = -T * hnode + LO * hnode_new
    vdiv = jnp.where(a > 0.0, (adf_v - _shift_up(adf_v)) * dt / safe_a, 0.0)
    dttf_v = jnp.where(nlm, lo_trans + vdiv, 0.0)
    hvals = jnp.stack([adf_h, -adf_h], axis=1)
    hsc = ops.scatter_add(hvals, mesh.edges, mesh.nod2D)
    dttf_h = jnp.where(nlm, hsc * dt / safe_a, 0.0)
    return dttf_h + dttf_v


def advect_one_fct(mesh: Mesh, uv, w_e, helem, hnode, hnode_new, T, T_old,
                   *, dt: float = DT_DEFAULT):
    """One tracer's FCT advection + ALE reconstruction
    (``fesom_tracer_advect_one_fct``, ``fesom_tracer_adv.c:1199``). Returns
    ``(T_new, T_old_new)`` with ``T_old_new = T`` (the AB2 ``valuesold`` save).

    LO fluxes & the element/up-dn gradient are built from **values** (T); the HO
    fluxes use the AB2-extrapolated ``ttfAB`` (== T at step 1)."""
    ttfAB, T_old_new = tracer_ab(T, T_old)
    # 2 — LO upwind fluxes from values
    flux_h_lo = adv_flux_hor(mesh, uv, helem, T)
    flux_v_lo = adv_flux_ver(mesh, w_e, T)
    # 3 — low-order ALE solution
    LO = compute_fct_lo(mesh, flux_h_lo, flux_v_lo, T, hnode, hnode_new, dt=dt)
    # 4 — antidiffusive fluxes HO − LO
    tr_xy = tracer_gradient_elements(mesh, T)
    eud = fill_up_dn_grad(mesh, tr_xy)
    adf_h = adv_flux_hor_ho(mesh, uv, helem, ttfAB, eud) - flux_h_lo
    adf_v = adv_flux_ver_ho(mesh, w_e, ttfAB) - flux_v_lo
    # 5 — Zalesak limit
    adf_h, adf_v = zalesak_limit(mesh, T, LO, adf_h, adf_v, hnode_new, dt=dt)
    # 6-7 — assemble del_ttf  8 — reconstruct (T_new = LO + limited antidiff)
    del_ttf = flux2dtracer_fct(mesh, T, LO, adf_h, adf_v, hnode, hnode_new, dt=dt)
    T_new = ale_reconstruct(mesh, T, del_ttf, hnode, hnode_new)
    return T_new, T_old_new
