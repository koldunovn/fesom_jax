"""Momentum RHS: Coriolis(AB2) + SSH grad + PGF + momentum advection (substep 5 / Task 2.4).

Literal vectorized port of ``fesom_compute_vel_rhs`` + ``fesom_momentum_adv_scalar``
(``fesom_momentum.c:49-271``, driven from ``fesom_step.c:142``) for the Phase-2 pi
config (linfs, AB_order=2 single-slot history, momadv_opt=2). Output is the
substep-5 element velocity RHS ``uv_rhs`` ``[elem2D, nl, 2]``.

Structure (``compute_vel_rhs``):

1. AB history shift: ``uv_rhs = ab1·uv_rhsAB``  (``ab1 = -(0.5+ε)``, ε=0.1 AB2 offset),
2. add ``(F_ssh − pgf)·area`` per layer  (``F_ssh`` = ∇N · (−g·η) at the 3 vertices),
3. overwrite ``uv_rhsAB = (v·ff, −u·ff)``  (this step's Coriolis; ``ff = coriolis·area``),
4. ``+= momentum_adv_scalar`` into ``uv_rhsAB``  (momadv_opt=2; an edge→node scatter),
5. assemble: ``uv_rhs = dt·(uv_rhs + uv_rhsAB·ff_step)/area``, ``ff_step = ab2`` (or 1
   on the first step).

At rest (uv=η=uv_rhsAB=w_e=0) only the PGF term survives → ``uv_rhs = −dt·pgf`` (the
step-1 dump gate). Coriolis/SSH/advection are exercised by the synthetic + AD tests.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ops
from .config import (
    C_D,
    DENSITY_0,
    DT_DEFAULT,
    G,
    SSH_THETA,
    ViscConfig,
)
from .mesh import Mesh

# Default γ's = the module constants ⇒ ``visc_cfg=None`` is bit-identical to the bare path.
_DEFAULT_VISC = ViscConfig()

_EPS_AB = 0.1                     # FESOM AB2 stabilization offset (oce_modules.F90:92)
_AB1 = -(0.5 + _EPS_AB)
_AB2 = (1.5 + _EPS_AB)


def _safe_sqrt(x):
    """``sqrt(x)`` whose gradient is finite at ``x=0`` (double-``where`` trick).
    Forward-identical to ``jnp.sqrt`` for ``x≥0`` (``sqrt(0)=0``), but the backward
    pass returns 0 instead of ∞ at 0 — needed because the flow-aware biharmonic
    viscosity depends on ``|∇u|`` (non-smooth at rest). AD-safe by construction."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(safe), 0.0)


def _shift_down(x):
    """``out[..., k] = x[..., k-1]``, edge-replicated at k=0."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


def _shift_up(x):
    """``out[..., k] = x[..., k+1]``, zero-padded at the last level."""
    return jnp.concatenate([x[..., 1:], jnp.zeros_like(x[..., :1])], axis=-1)


def _scatter3(mesh: Mesh, contrib):
    """Element→node sum over the 3 vertices: ``contrib`` ``[elem2D, *rest]`` →
    ``[nod2D, *rest]`` (each element adds its value to all 3 of its nodes)."""
    e = mesh.elem2D
    vals = jnp.broadcast_to(contrib[:, None], (e, 3) + contrib.shape[1:])
    return ops.scatter_add(vals, mesh.elem_nodes, mesh.nod2D)


def _horizontal_adv(mesh: Mesh, uv):
    """Horizontal momentum advection via the edge loop (``fesom_momentum.c:202-244``):
    an antisymmetric edge→node scatter. Returns ``un`` ``[nod2D, nl, 2]``."""
    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s = jnp.where(has1, el1, 0)
    el2s = jnp.where(has2, el2, 0)
    cx = mesh.edge_cross_dxdy
    dx1, dy1, dx2, dy2 = cx[:, 0:1], cx[:, 1:2], cx[:, 2:3], cx[:, 3:4]   # (edge,1)

    u1 = ops.gather(uv[:, :, 0], el1s)        # (edge, nl)
    v1 = ops.gather(uv[:, :, 1], el1s)
    u2 = ops.gather(uv[:, :, 0], el2s)
    v2 = ops.gather(uv[:, :, 1], el2s)
    lm1 = ops.gather(mesh.elem_layer_mask, el1s)
    lm2 = ops.gather(mesh.elem_layer_mask, el2s)

    # normal-flux factors (zero outside each element's layer range / for absent el2)
    un1 = jnp.where(has1[:, None] & lm1, v1 * dx1 - u1 * dy1, 0.0)        # (edge, nl)
    un2 = jnp.where(has2[:, None] & lm2, -v2 * dx2 + u2 * dy2, 0.0)
    flux_u = un1 * u1 + un2 * u2              # (edge, nl); un2=0 ⇒ el2 term drops
    flux_v = un1 * v1 + un2 * v2
    flux = jnp.stack([flux_u, flux_v], axis=-1)                          # (edge, nl, 2)

    # n1 += flux, n2 -= flux
    vals = jnp.stack([flux, -flux], axis=1)                              # (edge, 2, nl, 2)
    return ops.scatter_add(vals, mesh.edges, mesh.nod2D)                 # (nod2D, nl, 2)


def momentum_adv_scalar(mesh: Mesh, uv, w_e, hnode, *, exch=None):
    """Momentum advection on scalar control volumes (momadv_opt=2). Returns the
    contribution added to ``uv_rhsAB`` at elements, ``[elem2D, nl, 2]``.

    Vertical ``w·∂u/∂z`` (element→node scatter + flux divergence) + horizontal
    (edge→node scatter), divided by ``areasvol``, then vertex→element averaged.

    ``exch`` (Phase 8, S.7): the per-node advection ``un_u``/``un_v`` are SCATTER
    results (incomplete on the halo), and the final vertex→element average gathers them
    at the cell's 3 vertices (incl. HALO nodes) — so the C exchanges the node field
    (``uvnode_rhs``) before that gather (the Kokkos SYNC_MAP substep-4 D21 bracket).
    ``exch=None`` ⇒ identity ⇒ byte-identical to ``v1.0``."""
    _exch = exch if exch is not None else (lambda f, kind: f)
    lm_e = mesh.elem_layer_mask
    lm_n = mesh.node_layer_mask

    # --- vertical: interface velocity 0.5(uv[j]+uv[j-1]) (surface = uv[0]), ×area,
    #     scatter to nodes, × w_e, then −d/dz over 3·hnode -----------------------
    uvu, uvv = uv[:, :, 0], uv[:, :, 1]
    cu = jnp.where(lm_e, 0.5 * (uvu + _shift_down(uvu)), 0.0)
    cv = jnp.where(lm_e, 0.5 * (uvv + _shift_down(uvv)), 0.0)
    area = mesh.elem_area[:, None]
    wu = _scatter3(mesh, area * cu) * w_e        # (nod2D, nl)
    wv = _scatter3(mesh, area * cv) * w_e
    h3 = jnp.where(lm_n, 3.0 * hnode, 1.0)
    vert_u = jnp.where(lm_n, -(wu - _shift_up(wu)) / h3, 0.0)
    vert_v = jnp.where(lm_n, -(wv - _shift_up(wv)) / h3, 0.0)

    # --- horizontal edge scatter ---------------------------------------------
    horiz = _horizontal_adv(mesh, uv)            # (nod2D, nl, 2)

    # --- combine, divide by scalar control-volume area ------------------------
    inv_av = 1.0 / jnp.where(lm_n, mesh.areasvol, 1.0)
    un_u = jnp.where(lm_n, (vert_u + horiz[:, :, 0]) * inv_av, 0.0)
    un_v = jnp.where(lm_n, (vert_v + horiz[:, :, 1]) * inv_av, 0.0)
    un_u = _exch(un_u, "nod")        # S.7: scatter result, gathered to cells below
    un_v = _exch(un_v, "nod")

    # --- vertex → element: area·mean(3 vertices) ------------------------------
    cu3 = ops.gather_nodes_to_elem(un_u, mesh.elem_nodes).sum(axis=1) / 3.0   # (elem2D,nl)
    cv3 = ops.gather_nodes_to_elem(un_v, mesh.elem_nodes).sum(axis=1) / 3.0
    contrib_u = jnp.where(lm_e, area * cu3, 0.0)
    contrib_v = jnp.where(lm_e, area * cv3, 0.0)
    return jnp.stack([contrib_u, contrib_v], axis=-1)


def compute_vel_rhs(mesh: Mesh, uv, uv_rhsAB, eta_n, pgf_x, pgf_y, w_e, hnode,
                    *, is_first_step, dt=DT_DEFAULT, exch=None):
    """``(uv_rhs, uv_rhsAB_new)`` element fields ``[elem2D, nl, 2]``.

    ``uv``, ``w_e``, ``hnode`` and ``eta_n`` are the *previous-step* state (lagged,
    as in the C). ``uv_rhsAB`` is the incoming AB slot (overwritten on return)."""
    g = G
    lm = mesh.elem_layer_mask
    area = mesh.elem_area[:, None]                                  # (elem2D,1)

    # (1) AB history shift (uses the OLD uv_rhsAB)
    rhs = _AB1 * uv_rhsAB                                           # (elem2D,nl,2)

    # (2) SSH gradient F = ∇N·(−g·η) at the 3 vertices, then += (F − pgf)·area
    eta_c = ops.gather_nodes_to_elem(eta_n, mesh.elem_nodes)        # (elem2D,3)
    pre = -g * eta_c
    gs = mesh.gradient_sca
    Fx = gs[:, 0] * pre[:, 0] + gs[:, 1] * pre[:, 1] + gs[:, 2] * pre[:, 2]   # (elem2D,)
    Fy = gs[:, 3] * pre[:, 0] + gs[:, 4] * pre[:, 1] + gs[:, 5] * pre[:, 2]
    add_u = jnp.where(lm, (Fx[:, None] - pgf_x) * area, 0.0)
    add_v = jnp.where(lm, (Fy[:, None] - pgf_y) * area, 0.0)
    rhs = rhs + jnp.stack([add_u, add_v], axis=-1)

    # (3) new Coriolis into the AB slot:  (v·ff, −u·ff)
    ff = (mesh.coriolis * mesh.elem_area)[:, None]                  # (elem2D,1)
    new_AB_u = jnp.where(lm, uv[:, :, 1] * ff, 0.0)
    new_AB_v = jnp.where(lm, -uv[:, :, 0] * ff, 0.0)
    uv_rhsAB_new = jnp.stack([new_AB_u, new_AB_v], axis=-1)

    # (4) + momentum advection into the AB slot
    uv_rhsAB_new = uv_rhsAB_new + momentum_adv_scalar(mesh, uv, w_e, hnode, exch=exch)

    # (5) assemble:  uv_rhs = dt·(rhs + uv_rhsAB·ff_step)/area
    ff_step = 1.0 if is_first_step else _AB2
    inv_area = (1.0 / mesh.elem_area)[:, None, None]
    uv_rhs = dt * (rhs + uv_rhsAB_new * ff_step) * inv_area
    uv_rhs = ops.mask_below_bottom(uv_rhs, lm)
    return uv_rhs, uv_rhsAB_new


def _bidiff_edge_terms(mesh: Mesh, uv, vc: ViscConfig = _DEFAULT_VISC):
    """Shared per-edge per-level quantities for both biharmonic stages: the
    velocity difference ``(u1,v1)=uv[el1]-uv[el2]``, ``coef=sqrt(max(g0,inner)·len)``
    (``inner=max(g1·|du|, g2·|du|²)``), the per-edge overlap level mask, and the
    safe element indices/areas. Interior edges only (el1≥0 AND el2≥0).

    ``vc`` (a :class:`~fesom_jax.config.ViscConfig`) supplies the γ's; the default reproduces
    the module constants exactly ⇒ bit-identical to before."""
    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    interior = (el1 >= 0) & (el2 >= 0)
    el1s = jnp.where(interior, el1, 0)
    el2s = jnp.where(interior, el2, 0)

    a1 = ops.gather(mesh.elem_area, el1s)
    a2 = ops.gather(mesh.elem_area, el2s)
    length = _safe_sqrt(a1 + a2)[:, None]                          # (edge,1)

    nzmin = jnp.maximum(ops.gather(mesh.ulevels, el1s),
                        ops.gather(mesh.ulevels, el2s)) - 1
    nzmax = jnp.minimum(ops.gather(mesh.nlevels, el1s),
                        ops.gather(mesh.nlevels, el2s)) - 1
    k = jnp.arange(mesh.nl)[None, :]
    emask = interior[:, None] & (k >= nzmin[:, None]) & (k < nzmax[:, None])

    u1 = ops.gather(uv[:, :, 0], el1s) - ops.gather(uv[:, :, 0], el2s)   # (edge,nl)
    v1 = ops.gather(uv[:, :, 1], el1s) - ops.gather(uv[:, :, 1], el2s)
    vi2 = u1 * u1 + v1 * v1
    sq = _safe_sqrt(vi2)
    inner = jnp.maximum(vc.gamma1 * sq, vc.gamma2 * vi2)
    coef = _safe_sqrt(jnp.maximum(vc.gamma0, inner) * length)            # (edge,nl)
    return el1s, el2s, a1, a2, length, emask, u1, v1, sq, coef


def visc_filt_bidiff(mesh: Mesh, uv, uv_rhs, *, dt=DT_DEFAULT, exch=None, visc_cfg=None):
    """Biharmonic flow-aware horizontal viscosity (opt_visc=7), two edge→element
    scatter stages added into ``uv_rhs``. Returns the updated ``uv_rhs``
    ``[elem2D, nl, 2]``. Mirror of ``fesom_visc_filt_bidiff`` (``fesom_momentum.c:654``).

    ``exch`` (Phase 8, S.7): a ``(field, kind) → field`` halo-exchange callable. The
    bilaplacian is a 2-ring element operator, so stage 2 (which gathers ``Uc/Vc`` at the
    edge's two cells) needs the **halo** ``Uc/Vc`` refreshed — the C exchanges them
    between the stages (``oce_dyn.F90:367``). ``exch=None`` (single device) ⇒ the
    identity ⇒ byte-identical to ``v1.0``.

    ``visc_cfg`` (a :class:`~fesom_jax.config.ViscConfig`) supplies the γ's; ``None`` ⇒ the
    module-constant defaults ⇒ byte-identical (NG5 passes ``gamma1=0.2``).
    """
    _exch = exch if exch is not None else (lambda f, kind: f)
    vc = visc_cfg if visc_cfg is not None else _DEFAULT_VISC
    el1s, el2s, a1, a2, length, emask, u1, v1, sq, coef = _bidiff_edge_terms(mesh, uv, vc)

    # Stage 1: U_c[el1] -= u1·coef, U_c[el2] += u1·coef  (antisymmetric scatter)
    du = jnp.where(emask, u1 * coef, 0.0)                          # (edge,nl)
    dv = jnp.where(emask, v1 * coef, 0.0)
    Uc = ops.scatter_add(jnp.stack([-du, du], axis=1), mesh.edge_tri, mesh.elem2D)
    Vc = ops.scatter_add(jnp.stack([-dv, dv], axis=1), mesh.edge_tri, mesh.elem2D)

    # S.7 intra-kernel exchange: refresh halo Uc/Vc before stage 2 gathers them.
    Uc = _exch(Uc, "elem")
    Vc = _exch(Vc, "elem")

    # Stage 2: update = -dt·coef·(U_c[el1]-U_c[el2]) + viLapl·u1, scatter /area
    coef2 = -dt * coef
    viLapl = dt * jnp.maximum(vc.gamma0_h, vc.gamma1_h * sq) * length   # 0 for CORE2
    Uc_d = ops.gather(Uc, el1s) - ops.gather(Uc, el2s)
    Vc_d = ops.gather(Vc, el1s) - ops.gather(Vc, el2s)
    upd_u = jnp.where(emask, coef2 * Uc_d + viLapl * u1, 0.0)
    upd_v = jnp.where(emask, coef2 * Vc_d + viLapl * v1, 0.0)

    du_u = jnp.stack([-upd_u / a1[:, None], upd_u / a2[:, None]], axis=1)
    du_v = jnp.stack([-upd_v / a1[:, None], upd_v / a2[:, None]], axis=1)
    delta_u = ops.scatter_add(du_u, mesh.edge_tri, mesh.elem2D)    # (elem2D,nl)
    delta_v = ops.scatter_add(du_v, mesh.edge_tri, mesh.elem2D)

    out = uv_rhs + jnp.stack([delta_u, delta_v], axis=-1)
    return ops.mask_below_bottom(out, mesh.elem_layer_mask)


def impl_vert_visc(mesh: Mesh, uv, uv_rhs, Av, stress_surf, *, dt=DT_DEFAULT,
                   elem_geo=None, w_i=None):
    """Implicit vertical viscosity per element (substep 7). Returns the velocity
    **increment** ``du`` ``[elem2D, nl, 2]`` (the C stores it back into ``uv_rhs``;
    ``update_vel`` later does ``uv += du``).

    Mirror of ``fesom_impl_vert_visc`` (``fesom_momentum.c:291``) for Phase-2 pi:
    no cavity / no partial cells (so ``zbar_n=zbar``, ``Z_n=Z`` globally). Builds the
    tridiagonal (a,b,c) from ``Av`` + geometry, adds wind-stress (surface) and quadratic
    bottom-drag (bottom) forcing, converts to the ``du`` system, and solves with
    :func:`ops.tdma` (vectorized over elements).

    ``w_i`` (``[nod2D, nl]`` or ``None``) is the implicit part of the split vertical
    velocity (``compute_wvel_split``). When given, its upwind advection terms are added to
    the tridiagonal (``oce_ale.F90:2696-2742``; node→element 3-vertex mean, ``w>0``=upward);
    the increment-form rhs then carries the implicit advection of the old velocity. ``None``
    (use_wsplit off) ⇒ the terms drop ⇒ byte-identical to the no-split path.

    **zstar (Phase 9a, JZ.6).** ``elem_geo`` is the live per-element ``(zbar_n, Z_n)``
    ``[elem2D, nl]`` from :func:`fesom_jax.ale.live_geometry_elem` (built from the carried
    ``st.helem`` — the element-geometry "old" side; the C rebuilds them from ``helem`` per
    element, ``fesom_momentum.c:321-333``). ``elem_geo=None`` ⇒ the static column-uniform
    ``zbar``/``Z`` (byte-identical; live==static at cold start)."""
    nl = mesh.nl
    inv_rho0 = 1.0 / DENSITY_0
    # Layer thickness ``h`` (for ``zinv=dt/h``) + the layer-center spacings ``dZ_up``/
    # ``dZ_dn``. Static column-uniform geometry, or — under zstar — the live per-element
    # ``zbar_n``/``Z_n`` (``h[nz]=zbar_n[nz]−zbar_n[nz+1]``, exactly the C's ``zinv`` denom).
    # The masked surface/bottom-pad lanes get ``dZ==0`` ⇒ guard the divisors (the kpp.py:352
    # idiom): a bare 0 yields inf/NaN that the masks hide forward but that poisons the AD
    # backward of the masked tridiagonal (the sharded reverse pass exposes it).
    if elem_geo is None:
        zbar = mesh.zbar
        Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])
        hh = zbar[:-1] - zbar[1:]
        h = jnp.concatenate([hh, hh[-1:]])          # layer thickness (nl,), tail padded
        zinv = dt / h
        dZ_up = (_shift_down(Zp) - Zp).at[0].set(1.0)   # Z[nz-1]-Z[nz]; [0] unused (a=0)
        dZ_dn = (Zp - _shift_up(Zp)).at[-1].set(1.0)    # Z[nz]-Z[nz+1]; [-1] unused
        dZ_up = jnp.where(dZ_up != 0.0, dZ_up, 1.0)
        dZ_dn = jnp.where(dZ_dn != 0.0, dZ_dn, 1.0)
        zinv_b, dZ_up_b, dZ_dn_b = zinv[None, :], dZ_up[None, :], dZ_dn[None, :]
        zinv_top = ops.gather(zinv, mesh.ulevels - 1)[:, None]      # (elem2D,1)
        zinv_bot = ops.gather(zinv, mesh.nlevels - 2)[:, None]
    else:
        zbar_n, Z_n = elem_geo                       # (elem2D,nl) live element depths
        zbar_n_below = jnp.concatenate([zbar_n[:, 1:], zbar_n[:, -1:]], axis=1)  # zbar_n[nz+1]
        h = zbar_n - zbar_n_below                    # layer thickness = helem (0 at pad)
        zinv = dt / jnp.where(h != 0.0, h, 1.0)      # (elem2D,nl)
        dZ_up = _shift_down(Z_n) - Z_n               # Z_n[nz-1]-Z_n[nz]
        dZ_dn = Z_n - _shift_up(Z_n)                 # Z_n[nz]-Z_n[nz+1]
        dZ_up = jnp.where(dZ_up != 0.0, dZ_up, 1.0)
        dZ_dn = jnp.where(dZ_dn != 0.0, dZ_dn, 1.0)
        zinv_b, dZ_up_b, dZ_dn_b = zinv, dZ_up, dZ_dn
        zinv_top = jnp.take_along_axis(zinv, (mesh.ulevels - 1)[:, None], axis=1)
        zinv_bot = jnp.take_along_axis(zinv, (mesh.nlevels - 2)[:, None], axis=1)

    a_full = -Av / dZ_up_b * zinv_b
    c_full = -_shift_up(Av) / dZ_dn_b * zinv_b

    k = jnp.arange(nl)[None, :]
    nzmin = (mesh.ulevels - 1)[:, None]
    nzmax = (mesh.nlevels - 1)[:, None]
    valid = mesh.elem_layer_mask
    surf = k == nzmin
    bot = k == (nzmax - 1)

    a = jnp.where(surf, 0.0, a_full)
    c = jnp.where(bot, 0.0, c_full)
    b = jnp.where(surf, -c + 1.0, jnp.where(bot, -a + 1.0, -a - c + 1.0))
    a = jnp.where(valid, a, 0.0)
    c = jnp.where(valid, c, 0.0)
    b = jnp.where(valid, b, 1.0)

    # w_split: implicit vertical advection of momentum by w_i (oce_ale.F90:2696-2742). Upwind
    # (w>0 = upward): wu = element 3-vertex mean of w_i at face nz, wd at face nz+1. The terms
    # add to the SAME (a,b,c) AFTER the viscosity diagonal (b=-a-c+1) is set, exactly as the C.
    # surface uses the full wu top-face; the bottom row keeps only the top face. w_i=None ⇒ skip
    # (byte-identical); w_i≡0 ⇒ min/max(0,0)=0 ⇒ all terms vanish.
    if w_i is not None:
        wu = ops.gather_nodes_to_elem(w_i, mesh.elem_nodes).sum(axis=1) / 3.0   # (elem2D, nl)
        wd = _shift_up(wu)                                                       # face nz+1
        a_wi = jnp.where(surf, 0.0, jnp.minimum(0.0, wu)) * zinv_b
        b_wi = (jnp.where(surf, wu, jnp.maximum(0.0, wu))          # top face (full at surface)
                - jnp.where(bot, 0.0, jnp.minimum(0.0, wd))) * zinv_b   # bottom face (none at bottom)
        c_wi = jnp.where(bot, 0.0, -jnp.maximum(0.0, wd)) * zinv_b
        a = a + jnp.where(valid, a_wi, 0.0)
        b = b + jnp.where(valid, b_wi, 0.0)
        c = c + jnp.where(valid, c_wi, 0.0)

    u_old, v_old = uv[:, :, 0], uv[:, :, 1]
    fu, fv = uv_rhs[:, :, 0], uv_rhs[:, :, 1]

    # surface wind stress at the top row (zinv_top from the geometry branch above)
    fu = fu + jnp.where(surf, zinv_top * (stress_surf[:, 0:1] * inv_rho0), 0.0)
    fv = fv + jnp.where(surf, zinv_top * (stress_surf[:, 1:2] * inv_rho0), 0.0)

    # quadratic bottom drag at the bottom row (safe-sqrt: AD-finite at |u|=0)
    bot_idx = (mesh.nlevels - 2)[:, None]
    u_bot = jnp.take_along_axis(u_old, bot_idx, axis=1)
    v_bot = jnp.take_along_axis(v_old, bot_idx, axis=1)
    spd = _safe_sqrt(u_bot * u_bot + v_bot * v_bot)            # (elem2D,1)
    fric = -C_D * spd
    fu = fu + jnp.where(bot, zinv_bot * fric * u_bot, 0.0)
    fv = fv + jnp.where(bot, zinv_bot * fric * v_bot, 0.0)

    # convert "u_new" system to the increment "du":  rhs = forcing − M·u_old + u_old
    Mu = a * _shift_down(u_old) + b * u_old + c * _shift_up(u_old)
    Mv = a * _shift_down(v_old) + b * v_old + c * _shift_up(v_old)
    rhs_u = jnp.where(valid, fu - Mu + u_old, 0.0)
    rhs_v = jnp.where(valid, fv - Mv + v_old, 0.0)

    du = ops.tdma(a, b, c, rhs_u)
    dv = ops.tdma(a, b, c, rhs_v)
    return ops.mask_below_bottom(jnp.stack([du, dv], axis=-1), valid)


def update_vel(mesh: Mesh, uv, du, d_eta, *, dt=DT_DEFAULT, theta=SSH_THETA):
    """Velocity update (substep 10). Returns the new ``uv`` ``[elem2D, nl, 2]``.

    Mirror of ``fesom_update_vel`` (``fesom_momentum.c:474``):
    ``uv += du + (Fx, Fy)`` where ``du`` is the substep-7 increment from
    :func:`impl_vert_visc` (the C keeps it in ``uv_rhs``) and ``(Fx, Fy)`` is the
    SSH-gradient correction ``coef·∇N·d_eta`` evaluated at the element from the CG
    output ``d_eta`` at its 3 vertices, ``coef = −g·θ·dt``. ``Fx,Fy`` are uniform
    over the element's layers (a barotropic correction), so they broadcast over
    ``nz``. ``uv`` is the *previous-step* velocity (0 at rest), so step 1 gives the
    first wind-driven ``uv``.

    ``d_eta`` is read but not modified — it warm-starts the next step's CG solve
    (``solve_ssh``'s ``x0``); see the ssh/warmstart lesson."""
    lm = mesh.elem_layer_mask
    coef = -G * theta * dt
    ec = coef * ops.gather_nodes_to_elem(d_eta, mesh.elem_nodes)   # (elem2D,3)
    gs = mesh.gradient_sca
    Fx = gs[:, 0] * ec[:, 0] + gs[:, 1] * ec[:, 1] + gs[:, 2] * ec[:, 2]   # (elem2D,)
    Fy = gs[:, 3] * ec[:, 0] + gs[:, 4] * ec[:, 1] + gs[:, 5] * ec[:, 2]
    F = jnp.stack([Fx, Fy], axis=-1)[:, None, :]                   # (elem2D,1,2) → bcast nz
    return ops.mask_below_bottom(uv + du + F, lm)
