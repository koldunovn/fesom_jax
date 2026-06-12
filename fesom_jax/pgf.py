"""Pressure-gradient force at elements (substep 3 / Task 2.2).

Literal vectorized port of ``fesom_pressure_force_linfs_fullcell``
(``fesom_eos.c:285-313``, driven from ``fesom_step.c:113``) for the linfs
full-cell config. Per element ``e`` and layer ``nz``:

    pgf_x[e,nz] = (Σ_i ∂N_i/∂x · hpressure[V_i(e), nz]) / ρ0
    pgf_y[e,nz] = (Σ_i ∂N_i/∂y · hpressure[V_i(e), nz]) / ρ0

with ``∂N_i/∂x = gradient_sca[e, 0:3]``, ``∂N_i/∂y = gradient_sca[e, 3:6]``. A clean
gather (node→element via ``elem_nodes``) + linear shape-function contraction →
map/gather class (~1e-15). Output is an element **layer** field
(``elem_layer_mask``). The 3-term sum is written in the C's association order.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ops
from .config import DENSITY_0, G
from .mesh import Mesh


def pressure_force_linfs(mesh: Mesh, hpressure):
    """``(pgf_x, pgf_y)`` element layer fields ``[elem2D, nl]`` from the node
    hydrostatic pressure ``hpressure`` ``[nod2D, nl]``."""
    inv_r = 1.0 / DENSITY_0
    hp = ops.gather_nodes_to_elem(hpressure, mesh.elem_nodes)   # (elem2D, 3, nl)
    g = mesh.gradient_sca                                       # (elem2D, 6)
    hp0, hp1, hp2 = hp[:, 0], hp[:, 1], hp[:, 2]                # each (elem2D, nl)

    # keep the C's left-to-right association: ((g0·hp0 + g1·hp1) + g2·hp2)·inv_r
    pgf_x = (g[:, 0:1] * hp0 + g[:, 1:2] * hp1 + g[:, 2:3] * hp2) * inv_r
    pgf_y = (g[:, 3:4] * hp0 + g[:, 4:5] * hp1 + g[:, 5:6] * hp2) * inv_r

    pgf_x = ops.mask_below_bottom(pgf_x, mesh.elem_layer_mask)
    pgf_y = ops.mask_below_bottom(pgf_y, mesh.elem_layer_mask)
    return pgf_x, pgf_y


# ===========================================================================
# zstar (Phase 9a, JZ.5) — Shchepetkin density-Jacobian PGF on live geometry
# ===========================================================================
def _safe_div(num, den):
    """Double-``where`` safe divide: forward-identical where ``den≠0``, gradient finite
    where ``den==0`` (the masked/overridden stencil lanes). The C reads strictly-decreasing
    live ``Z_3d_n`` so real denominators are nonzero; this guards the edge-padded lanes that
    the case masks override (their NaN would otherwise poison the backward pass)."""
    den_safe = jnp.where(den != 0.0, den, 1.0)
    return jnp.where(den != 0.0, num / den_safe, 0.0)


def _dn(x):
    """Shift toward the surface: ``out[...,k] = x[...,k-1]``, top edge-replicated."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


def _up(x):
    """Shift toward the bottom: ``out[...,k] = x[...,k+1]``, bottom edge-replicated."""
    return jnp.concatenate([x[..., 1:], x[..., -1:]], axis=-1)


def _drho_dz(R0, Rm, Rp, Z0, Zm, Zp, Zn):
    """Quadratic-Newton vertical density gradient at element mid-depth ``Zn``, from the
    three stencil points ``(Zm,Rm) (Z0,R0) (Zp,Rp)`` (low→high index). Mirrors the C
    ``df10/dx10 + (dx10·df21−dx21·df10)/(dx20·dx21·dx10)·((Zn−Z0)+(Zn−Zm))`` with
    ``dx10=Z0−Zm, dx21=Zp−Z0, dx20=Zp−Zm, df10=R0−Rm, df21=Rp−R0``. Safe denominators."""
    dx10 = Z0 - Zm
    dx21 = Zp - Z0
    dx20 = Zp - Zm
    df10 = R0 - Rm
    df21 = Rp - R0
    lin = _safe_div(df10, dx10)
    curv = _safe_div(dx10 * df21 - dx21 * df10, dx20 * dx21 * dx10)
    return lin + curv * ((Zn - Z0) + (Zn - Zm))


def pressure_force_shchepetkin(mesh: Mesh, density_m_rho0, Z_3d_n, helem):
    """Shchepetkin density-Jacobian PGF (substep 3 under zstar; ``fesom_eos.c:348-503``).
    ``(pgf_x, pgf_y)`` element layer fields ``[elem2D, nl]`` from the node density anomaly
    ``density_m_rho0 = ρ − ρ0`` ``[nod2D, nl]``, the live per-node mid-depths ``Z_3d_n``
    ``[nod2D, nl]``, and the element thickness ``helem`` ``[elem2D, nl]``.

    Per element the vertical pressure-gradient integral is built TOP→BOTTOM:
    ``pgf[k] = Σ_{j<k} aux_j + ½·aux_k`` with ``aux_k = (∂ₓρ − ⟨∂_zρ⟩·∂ₓZ)·helem_k·g/ρ0``,
    ``∂ₓρ = Σ_i gs_i·ρ_i``, ``∂ₓZ = Σ_i gs_i·Z3D_i`` (the horizontal shape-function
    gradient), and ``⟨∂_zρ⟩`` the 3-vertex mean of the quadratic-Newton vertical gradient
    :func:`_drho_dz` evaluated at the ELEMENT mid-depth ``Z_n`` (built from ``helem``,
    anchored at the **static** bottom ``zbar[nlevels−1]`` — the C reads static there, mirror
    it). The running integral is the cumulative sum ``pgf = cumsum(aux) − ½·aux``.

    Three stencils, selected by **static** level masks (precomputed from the integer level
    arrays): **forward** ``(k,k+1,k+2)`` at the surface layer of each owning node, **backward**
    ``(k−2,k−1,k)`` at the bottom layer, **centered** ``(k−1,k,k+1)`` elsewhere. Edge-padded
    shifts + safe denominators keep the overridden/below-bottom lanes finite (AD-safe).
    nlevels≤2 single-mid-layer elements (surface≡bottom) are a degenerate C edge case (UB in
    the C's backward stencil) — masked/rare on CORE2; gate tests on ``nlevels≥3``."""
    inv_r = 1.0 / DENSITY_0
    g = float(G)
    nl = mesh.zbar.shape[0]
    en = mesh.elem_nodes                                   # (elem2D,3)

    # gather node fields to elements: R/Zd are [elem2D, 3, nl] (vertex × level)
    R = ops.gather_nodes_to_elem(density_m_rho0, en)       # ρ−ρ0 at the 3 vertices
    Zd = ops.gather_nodes_to_elem(Z_3d_n, en)              # live node mid-depths

    # element mid-depth Z_n[e,k] from helem, anchored at the STATIC bottom zbar[nlevels-1].
    hel = jnp.where(mesh.elem_layer_mask, helem, 0.0)      # 0 below bottom
    zbar_bot = mesh.zbar[mesh.nlevels - 1][:, None]        # (elem2D,1) static bottom interface
    revcum = jnp.cumsum(hel[:, ::-1], axis=1)[:, ::-1]     # Σ_{j≥k} hel[j] (reverse cumsum)
    zbar_n = zbar_bot + revcum                             # interface depths [e,k]
    zbar_n_below = jnp.concatenate([zbar_n[:, 1:], zbar_bot], axis=1)   # zbar_n[k+1]
    Z_n = zbar_n_below + 0.5 * hel                         # element mid-depth of layer k

    # shifted stencil neighbours (vertex axis kept; shift the level axis)
    Zm, Zp = _dn(Zd), _up(Zd)
    Zp2 = _up(Zp)
    Zm2 = _dn(Zm)
    Rm, Rp = _dn(R), _up(R)
    Rp2 = _up(Rp)
    Rm2 = _dn(Rm)
    Zn = Z_n[:, None, :]                                   # (elem2D,1,nl) broadcast over vertex

    dz_ctr = _drho_dz(R, Rm, Rp, Zd, Zm, Zp, Zn)           # centered (k-1,k,k+1)
    dz_fwd = _drho_dz(Rp, R, Rp2, Zp, Zd, Zp2, Zn)         # forward  (k,k+1,k+2)
    dz_bwd = _drho_dz(Rm, Rm2, R, Zm, Zm2, Zd, Zn)         # backward (k-2,k-1,k)

    # static stencil-case masks [elem2D, 3, nl] from the integer level arrays.
    k = jnp.arange(nl)[None, None, :]                      # (1,1,nl) 0-based layer
    surf_e = (mesh.ulevels - 1)[:, None, None]             # element surface layer (0-based)
    bot_e = (mesh.nlevels - 2)[:, None, None]              # element bottom mid-layer (0-based)
    surf_nod = (mesh.ulevels_nod2D[en] - 1)[:, :, None]    # each vertex's surface layer
    bot_nod = (mesh.nlevels_nod2D[en] - 2)[:, :, None]     # each vertex's bottom mid-layer
    fwd_m = (k == surf_e) & (k == surf_nod)                # surface block, node-surface matches
    bwd_m = (k == bot_e) & (k == bot_nod)                  # bottom block, node-bottom matches
    bwd_m = bwd_m & ~fwd_m                                 # bottom wins single-layer overlap

    drho_dz = jnp.where(bwd_m, dz_bwd, jnp.where(fwd_m, dz_fwd, dz_ctr))
    mdz = (drho_dz[:, 0] + drho_dz[:, 1] + drho_dz[:, 2]) / 3.0          # (elem2D,nl)

    gx = mesh.gradient_sca[:, 0:3]                         # ∂N/∂x (elem2D,3)
    gy = mesh.gradient_sca[:, 3:6]                         # ∂N/∂y
    drho_dx = gx[:, 0:1] * R[:, 0] + gx[:, 1:2] * R[:, 1] + gx[:, 2:3] * R[:, 2]
    drho_dy = gy[:, 0:1] * R[:, 0] + gy[:, 1:2] * R[:, 1] + gy[:, 2:3] * R[:, 2]
    dz_dx = gx[:, 0:1] * Zd[:, 0] + gx[:, 1:2] * Zd[:, 1] + gx[:, 2:3] * Zd[:, 2]
    dz_dy = gy[:, 0:1] * Zd[:, 0] + gy[:, 1:2] * Zd[:, 1] + gy[:, 2:3] * Zd[:, 2]

    fac = hel * g * inv_r                                  # helem·g/ρ0 (0 below bottom)
    aux_x = (drho_dx - mdz * dz_dx) * fac
    aux_y = (drho_dy - mdz * dz_dy) * fac
    # vertical integral: pgf[k] = Σ_{j<k} aux_j + ½ aux_k = cumsum_incl(aux) − ½ aux.
    pgf_x = jnp.cumsum(aux_x, axis=1) - 0.5 * aux_x
    pgf_y = jnp.cumsum(aux_y, axis=1) - 0.5 * aux_y
    pgf_x = ops.mask_below_bottom(pgf_x, mesh.elem_layer_mask)
    pgf_y = ops.mask_below_bottom(pgf_y, mesh.elem_layer_mask)
    return pgf_x, pgf_y
