"""Redi neutral-diffusion tracer terms (Phase 6B, Task G.6).

The Redi tensor's off-diagonal contributions enter the tracer equation as two
EXPLICIT flux terms plus the implicit K33 diagonal augmentation:

* :func:`diff_ver_part_redi_expl` (G7a, ``fesom_gm.c:646-790``) — the vertical
  projection of the horizontal tracer gradient.
* :func:`diff_part_hor_redi` (G7b, ``fesom_gm.c:824-1022``) — the horizontal edge
  flux with the 5 partial-cell level-mismatch branches.
* the K33 augmentation lives in :mod:`fesom_jax.tracer_diff` (it modifies the
  vertical-diffusion TDMA diagonal).

Both explicit terms read ``T_old`` (the pre-step tracer — the AB2 ``valuesold``)
for their gradients and add their flux to the post-advection ``T`` with the
``/hnode_new`` reconstruction factor (the C composes the Fortran ``del_ttf``
accumulation + ``ale_reconstruct``). Full-cell linfs ⇒ the vertical geometry
(``zbar_n=zbar``, ``z_n=Z``) is static.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ops
from .config import DT_DEFAULT
from .mesh import Mesh


def _grad_elem(mesh: Mesh, field):
    """Per-element horizontal gradient ``∇field`` from ``gradient_sca`` — returns
    ``(gx, gy)`` each ``(E, nl)``. (Same contraction as ``gm.compute_sigma_xy``.)"""
    g = mesh.gradient_sca
    f_e = ops.gather_nodes_to_elem(field, mesh.elem_nodes)     # (E,3,nl)
    gx = jnp.sum(g[:, 0:3][:, :, None] * f_e, axis=1)          # (E,nl)
    gy = jnp.sum(g[:, 3:6][:, :, None] * f_e, axis=1)
    return gx, gy


def tr_xy_elem(mesh: Mesh, T_old):
    """``tr_xy`` = per-element ∇(T_old), masked to the element layer range
    (``(E, nl, 2)``). Built by G7a, reused by G7b. ``fesom_gm.c:683-697``."""
    gx, gy = _grad_elem(mesh, T_old)
    txy = jnp.stack([gx, gy], axis=-1)                         # (E,nl,2)
    return jnp.where(mesh.elem_layer_mask[..., None], txy, 0.0)


def diff_ver_part_redi_expl(mesh: Mesh, T_old, slope_tapered, Ki, hnode_new,
                            *, dt: float = DT_DEFAULT, zbar3=None, Z3d=None):
    """G7a — the Redi vertical-explicit flux's contribution to T.

    Returns ``delta`` ``(N, nl)`` to ADD to the post-advection T. Per node:
    ``tr_xynodes = (1/3areasvol)·Σ_{el∋n} ∇(T_old)·area``; the interface flux
    ``vd_flux[nz] = (up·term[nz-1] + dn·term[nz])/mid · area[nz]`` with
    ``term[nz] = (s·tr_xynodes)·Ki``; then ``ΔT[nz] = (vd_flux[nz]−vd_flux[nz+1])·
    dt/(areasvol·hnode_new)``. Static vertical geometry (``zbar_n=zbar``, ``z_n=Z``).
    """
    nl = mesh.nl
    E = mesh.elem2D

    # tr_xy (∇T_old per element) → area-weighted tr_xynodes (÷ 3·areasvol).
    gx, gy = _grad_elem(mesh, T_old)
    area_e = mesh.elem_area[:, None]                           # (E,1)
    cx = jnp.where(mesh.elem_layer_mask, area_e * gx, 0.0)
    cy = jnp.where(mesh.elem_layer_mask, area_e * gy, 0.0)
    stacked = jnp.stack([cx, cy], axis=-1)                     # (E,nl,2)
    vals = jnp.broadcast_to(stacked[:, None, :, :], (E, 3, nl, 2))
    txn_sum = ops.scatter_add(vals, mesh.elem_nodes, mesh.nod2D)   # (N,nl,2)
    asv = mesh.areasvol
    inv3 = jnp.where(asv > 0.0, 1.0 / (3.0 * jnp.where(asv > 0.0, asv, 1.0)), 0.0)
    txn = txn_sum * inv3[..., None]                            # (N,nl,2)

    # term[n,nz] = (s_x·txn_x + s_y·txn_y)·Ki  on layer nz (slope_tapered is nl-1).
    st = slope_tapered                                         # (N,nl-1,3)
    txn_l = txn[:, : nl - 1, :]
    sdot = st[:, :, 0] * txn_l[:, :, 0] + st[:, :, 1] * txn_l[:, :, 1]
    term = sdot * Ki[:, : nl - 1]                              # (N,nl-1)
    term_full = jnp.pad(term, ((0, 0), (0, 1)))               # (N,nl)
    term_up = jnp.concatenate([term_full[:, :1], term_full[:, :-1]], axis=1)

    # interface geometry (valid nz∈[1,nl-1)). Static z_n=Z, zbar_n=zbar; or — under zstar
    # (JZ.6) — the OLD-mesh (st.hnode) live per-node zbar3/Z3d (the C builds zbar_n/z_n from
    # hnode, fesom_gm.c:739-753; the ÷hnode_new divisor below is the NEW side).
    if zbar3 is None:
        zbar, Z = mesh.zbar, mesh.Z
        up_geo = jnp.zeros(nl).at[1:nl - 1].set(Z[: nl - 2] - zbar[1:nl - 1])[None, :]
        dn_geo = jnp.zeros(nl).at[1:nl - 1].set(zbar[1:nl - 1] - Z[1:nl - 1])[None, :]
        mid_geo = jnp.ones(nl).at[1:nl - 1].set(Z[: nl - 2] - Z[1:nl - 1])[None, :]
    else:
        N = mesh.nod2D
        Zm = Z3d[:, : nl - 1]                              # (N,nl-1) live mid-depths
        up_geo = jnp.zeros((N, nl)).at[:, 1:nl - 1].set(Zm[:, : nl - 2] - zbar3[:, 1:nl - 1])
        dn_geo = jnp.zeros((N, nl)).at[:, 1:nl - 1].set(zbar3[:, 1:nl - 1] - Zm[:, 1:nl - 1])
        mid_geo = jnp.ones((N, nl)).at[:, 1:nl - 1].set(Zm[:, : nl - 2] - Zm[:, 1:nl - 1])

    k = jnp.arange(nl)[None, :]
    ule = (mesh.ulevels_nod2D - 1)[:, None]
    nle = (mesh.nlevels_nod2D - 1)[:, None]
    iface = (k >= ule + 1) & (k < nle)                        # vd_flux interfaces
    vd_flux = jnp.where(
        iface, (up_geo * term_up + dn_geo * term_full) / mid_geo * mesh.area, 0.0)

    # ΔT[nz] = (vd_flux[nz] − vd_flux[nz+1])·dt / (areasvol·hnode_new)
    vd_dn = jnp.concatenate([vd_flux[:, 1:], vd_flux[:, :1]], axis=1)   # vd_flux[nz+1]
    divergence = vd_flux - vd_dn
    layer = mesh.node_layer_mask
    denom = asv * hnode_new
    safe = jnp.where(denom > 0.0, denom, 1.0)
    return jnp.where(layer, divergence * dt / safe, 0.0)      # (N,nl)


def _shift_down(x):
    """``out[...,k] = x[...,k-1]``, edge-replicated at k=0."""
    return jnp.concatenate([x[:, :1], x[:, :-1]], axis=1)


def diff_part_hor_redi(mesh: Mesh, T_old, slope_tapered, Ki, hnode, hnode_new,
                       helem, *, dt: float = DT_DEFAULT):
    """G7b — the horizontal Redi edge flux's contribution to T (``(N, nl)`` delta).

    Per edge, the C's 5 partial-cell branches A/B/C/D/E (level mismatch between the
    two adjacent elements el1/el2) collapse to **3 cases by level-membership**
    ``(in1=nz∈el1, in2=nz∈el2)``: el1-only (A/D), el2-only (B/E), both (C). The
    Redi flux ``c = (CX·Fx + CY·Fy)·dz`` (``Fx=Kh(Tx+SxTz)``, the node SxTz from the
    edge endpoints, the element Tx from el1/el2) is scattered antisymmetrically
    (``+c→e1, −c→e2``), then ``ΔT = rhs·dt/(areasvol·hnode_new)``. ``fesom_gm.c:824``.
    """
    nl, nl1 = mesh.nl, mesh.nl - 1
    txy = tr_xy_elem(mesh, T_old)                             # (E,nl,2) = ∇T_old

    # tr_z = ∂(T_old)/∂z at interfaces (OLD mesh hnode).
    dz_node = 0.5 * (_shift_down(hnode) + hnode)
    safe_dz = jnp.where(dz_node > 0.0, dz_node, 1.0)
    k = jnp.arange(nl)[None, :]
    trz_mask = (k >= mesh.ulevels_nod2D[:, None]) & (k < (mesh.nlevels_nod2D - 1)[:, None])
    trz = jnp.where(trz_mask, (_shift_down(T_old) - T_old) / safe_dz, 0.0)   # (N,nl)

    e1, e2 = mesh.edges[:, 0], mesh.edges[:, 1]
    el1, el2 = mesh.edge_tri[:, 0], mesh.edge_tri[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s, el2s = jnp.where(has1, el1, 0), jnp.where(has2, el2, 0)
    cross = mesh.edge_cross_dxdy
    dxL, dyL = cross[:, 0:1], cross[:, 1:2]                   # (Ed,1)
    dxR = jnp.where(has2[:, None], cross[:, 2:3], 0.0)
    dyR = jnp.where(has2[:, None], cross[:, 3:4], 0.0)

    # node-endpoint quantities (the COMPUTE_KH_TZ_S macro), at layer nz.
    Kh = 0.5 * (Ki[e1][:, :nl1] + Ki[e2][:, :nl1])           # (Ed,nl1)
    Tz1 = 0.5 * (trz[e1][:, :nl1] + trz[e1][:, 1:nl])
    Tz2 = 0.5 * (trz[e2][:, :nl1] + trz[e2][:, 1:nl])
    st1, st2 = slope_tapered[e1], slope_tapered[e2]          # (Ed,nl1,3)
    SxTz = 0.5 * (Tz1 * st1[:, :, 0] + Tz2 * st2[:, :, 0])
    SyTz = 0.5 * (Tz1 * st1[:, :, 1] + Tz2 * st2[:, :, 1])

    # element quantities + the 3-case level membership.
    txy1, txy2 = txy[el1s][:, :nl1, :], txy[el2s][:, :nl1, :]
    h1, h2 = helem[el1s][:, :nl1], helem[el2s][:, :nl1]
    in1 = mesh.elem_layer_mask[el1s][:, :nl1] & has1[:, None]
    in2 = mesh.elem_layer_mask[el2s][:, :nl1] & has2[:, None]
    both, only1, only2 = in1 & in2, in1 & ~in2, in2 & ~in1

    Tx = jnp.where(both, 0.5 * (txy1[:, :, 0] + txy2[:, :, 0]),
                   jnp.where(only1, txy1[:, :, 0], txy2[:, :, 0]))
    Ty = jnp.where(both, 0.5 * (txy1[:, :, 1] + txy2[:, :, 1]),
                   jnp.where(only1, txy1[:, :, 1], txy2[:, :, 1]))
    dz = jnp.where(both, 0.5 * (h1 + h2), jnp.where(only1, h1, h2))
    CX = jnp.where(both, dyL - dyR, jnp.where(only1, dyL, -dyR))   # (Ed,nl1)
    CY = jnp.where(both, dxR - dxL, jnp.where(only1, -dxL, dxR))

    Fx = Kh * (Tx + SxTz)
    Fy = Kh * (Ty + SyTz)
    c = (CX * Fx + CY * Fy) * dz
    c = jnp.where(has1[:, None] & (both | only1 | only2), c, 0.0)   # (Ed,nl1)

    # antisymmetric edge→node scatter (+c→e1, −c→e2), then ÷(areasvol·hnode_new).
    vals = jnp.stack([c, -c], axis=1)                        # (Ed,2,nl1)
    rhs = ops.scatter_add(vals, mesh.edges, mesh.nod2D)      # (N,nl1)
    rhs = jnp.pad(rhs, ((0, 0), (0, 1)))                     # (N,nl)
    denom = mesh.areasvol * hnode_new
    safe = jnp.where(denom > 0.0, denom, 1.0)
    return jnp.where(mesh.node_layer_mask, rhs * dt / safe, 0.0)   # (N,nl)


# ============================================================================
# G.6 — K33 isoneutral augmentation (fesom_tracer_diff.c:167-246)
# ============================================================================
def k33_augmentation(mesh: Mesh, slope_tapered, Ki, zbar3=None, Z3d=None):
    """The Redi K33 (3,3-tensor) augmentation of the vertical tracer diffusivity:
    a per-interface ``K33_aug`` to ADD to ``Kv`` before ``impl_vert_diff`` (the C's
    ``Ty/Ty1`` terms; note ``Ty(nz) == Ty1(nz-1)`` ⇒ one per-interface value).

    ``K33_aug[k] = (Z[k-1]−zbar[k])·zinv·s[k-1]²·Ki[k-1] + (zbar[k]−Z[k])·zinv·
    s[k]²·Ki[k]`` with ``zinv=1/(Z[k-1]−Z[k])`` and ``s = slope_tapered[...,2]`` (the
    |slope| magnitude). Static linfs vertical geometry. Returns ``(N, nl)``, on the
    interior interfaces (0 at surface/bottom; ``impl_vert_diff`` masks anyway).

    The vertical diffusion then uses ``Kv + K33_aug`` — since ``impl_vert_diff``
    builds ``a[nz]∝Kv[nz]``, ``c[nz]∝Kv[nz+1]``, the augmented Kv reproduces the C's
    ``a∝(Kv[nz]+Ty)``, ``c∝(Kv[nz+1]+Ty1)`` exactly.
    """
    nl = mesh.nl
    s = slope_tapered[:, :, 2]                                # (N,nl-1) |slope|
    stKi = s * s * Ki[:, : nl - 1]                            # (N,nl-1) s²·Ki at layer k
    stKi_full = jnp.pad(stKi, ((0, 0), (0, 1)))             # (N,nl)
    stKi_up = _shift_down(stKi_full)                         # s²Ki[k-1]

    # vertical geometry. Static z_n=Z, zbar_n=zbar; or — under zstar (JZ.6) — the NEW-mesh
    # (hnode_new) live per-node zbar3/Z3d (the C K33 builds zbar_n/Z_n from hnode_new,
    # fesom_tracer_diff.c:134-158 — the impl-diff "new" side, matching impl_vert_diff).
    if zbar3 is None:
        zbar, Z = mesh.zbar, mesh.Z
        zinv = jnp.zeros(nl).at[1:nl - 1].set(1.0 / (Z[: nl - 2] - Z[1:nl - 1]))
        gu_zinv = (jnp.zeros(nl).at[1:nl - 1].set(Z[: nl - 2] - zbar[1:nl - 1]) * zinv)[None, :]
        gd_zinv = (jnp.zeros(nl).at[1:nl - 1].set(zbar[1:nl - 1] - Z[1:nl - 1]) * zinv)[None, :]
    else:
        N = mesh.nod2D
        Zm = Z3d[:, : nl - 1]                              # (N,nl-1) live mid-depths
        zinv = jnp.zeros((N, nl)).at[:, 1:nl - 1].set(1.0 / (Zm[:, : nl - 2] - Zm[:, 1:nl - 1]))
        gu_zinv = jnp.zeros((N, nl)).at[:, 1:nl - 1].set(Zm[:, : nl - 2] - zbar3[:, 1:nl - 1]) * zinv
        gd_zinv = jnp.zeros((N, nl)).at[:, 1:nl - 1].set(zbar3[:, 1:nl - 1] - Zm[:, 1:nl - 1]) * zinv
    K33 = gu_zinv * stKi_up + gd_zinv * stKi_full

    k = jnp.arange(nl)[None, :]
    iface = (k >= mesh.ulevels_nod2D[:, None]) & (k < (mesh.nlevels_nod2D - 1)[:, None])
    return jnp.where(iface, K33, 0.0)                        # (N,nl)
