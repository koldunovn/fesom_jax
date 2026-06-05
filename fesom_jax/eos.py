"""Equation of state, hydrostatic pressure, Brunt-Väisälä N² (substep 1 / Task 2.1).

Literal, vectorized port of the C port's ``fesom_pressure_bv`` + the N² smoother
``fesom_smooth_nod3D`` (``fesom_eos.c:78-277``, driven from ``fesom_step.c:77-96``)
for the Phase-2 pi config: JM-EOS (``state_equation=1``), ``linfs`` (uses the
``hpressure`` linfs branch), no cavity (``nzmin=0``), ``use_density_ref=.false.``
(→ ``density_ref ≡ density_0``), PP mixing (so ``dbsfc``/``MLD1`` are unused and
skipped — they feed only KPP). ``sw_alpha_beta`` (substep 2) is GM/KPP-only →
deferred to Phase 6.

Outputs (all ``[nod2D, nl]``, matching the substep-1 dump fields):

* ``density`` = in-situ ρ − ρ0 (``density_m_rho0``), **layer** field,
* ``hpressure`` = hydrostatic pressure, **layer** field,
* ``bvfreq``   = N², **interface** field; the dump value is **post-smooth**, so
  the caller applies :func:`smooth_nod3D` (one sweep) before comparing.

The Golden Rule: the math + association order mirror the C exactly; the per-node /
per-layer C loops become array ops over :mod:`fesom_jax.ops`. EOS/pressure are
map/gather (~1e-15); the N² smoother is an element→node scatter (~1e-12).
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ops
from .config import DENSITY_0, G
from .mesh import Mesh

# --- JM-EOS coefficients (densityJM_components, oce_ale_pressure_bv.F90:2605-2669)
# Verbatim from fesom_eos.c:19-42 — do NOT round, fold, or "simplify". (`as`, `ass`
# are Python-safe-renamed a_s, a_ss.)
_A0, _AT, _AT2, _AT3, _AT4 = 19092.56, 209.8925, -3.041638, -1.852732e-3, -1.361629e-5
_A_S, _AST, _AST2, _AST3 = 104.4077, -6.500517, 0.1553190, 2.326469e-4
_A_SS, _ASST, _ASST2 = -5.587545, 0.7390729, -1.909078e-2
_AP, _APT, _APT2, _APT3 = -4.721788e-1, -1.028859e-2, 2.512549e-4, 5.939910e-7
_APS, _APST, _APST2, _APSS = 1.571896e-2, 2.598241e-4, -7.267926e-6, -2.042967e-3
_AP2, _AP2T, _AP2T2 = 1.045941e-5, -5.782165e-10, 1.296821e-7
_AP2S, _AP2ST, _AP2ST2 = -2.595994e-7, -1.248266e-9, -3.508914e-9

_B0, _BT, _BT2, _BT3, _BT4, _BT5 = (
    999.842594, 6.793952e-2, -9.095290e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9,
)
_BS, _BST, _BST2, _BST3, _BST4 = 0.824493, -4.08990e-3, 7.64380e-5, -8.24670e-7, 5.38750e-9
_B_SS, _BSST, _BSST2, _BSS2 = -5.72466e-3, 1.02270e-4, -1.65460e-6, 4.8314e-4

_STATE_EQ_INT = 1.0  # JM-EOS (state_equation=1) in Phase 1


def jm_components(T, S):
    """Jackett-McDougall bulk-modulus + potential-density components per point.

    Returns ``(bulk_0, bulk_pz, bulk_pz2, rhopot)``, each the shape of ``T``/``S``.
    Association order mirrors ``fesom_eos_jm_components`` (Horner form) exactly.
    """
    t = jnp.asarray(T)
    s = jnp.asarray(S)
    s_sqrt = jnp.sqrt(s)

    bulk_0 = (
        _A0 + t * (_AT + t * (_AT2 + t * (_AT3 + t * _AT4)))
        + s * (_A_S + t * (_AST + t * (_AST2 + t * _AST3))
               + s_sqrt * (_A_SS + t * (_ASST + t * _ASST2)))
    )
    bulk_pz = (
        _AP + t * (_APT + t * (_APT2 + t * _APT3))
        + s * (_APS + t * (_APST + t * _APST2) + s_sqrt * _APSS)
    )
    bulk_pz2 = (
        _AP2 + t * (_AP2T + t * _AP2T2)
        + s * (_AP2S + t * (_AP2ST + t * _AP2ST2))
    )
    rhopot = (
        _B0 + t * (_BT + t * (_BT2 + t * (_BT3 + t * (_BT4 + t * _BT5))))
        + s * (_BS + t * (_BST + t * (_BST2 + t * (_BST3 + t * _BST4))))
        + s * s_sqrt * (_B_SS + t * (_BSST + t * _BSST2))
        + s * s * _BSS2
    )
    return bulk_0, bulk_pz, bulk_pz2, rhopot


def _insitu(bulk_0, bulk_pz, bulk_pz2, rhopot, z):
    """Full in-situ density at depth ``z`` (NOT minus ρ0): the C
    ``bulk·rhopot/(bulk + 0.1·z·state_eq_int)``. ``z`` broadcasts."""
    bulk = bulk_0 + z * (bulk_pz + z * bulk_pz2)
    return bulk * rhopot / (bulk + 0.1 * z * _STATE_EQ_INT)


def _shift_down(x):
    """``x`` shifted one along the level axis with edge replication:
    ``out[..., k] = x[..., k-1]``, ``out[..., 0] = x[..., 0]`` (the replicated
    surface keeps the unused ``k=0`` N² finite — and exactly 0 — for AD safety)."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


def pressure_bv(mesh: Mesh, T, S, hnode):
    """Raw (pre-smooth) ``density``, ``hpressure``, ``bvfreq`` columns.

    ``T``, ``S``, ``hnode`` are ``[nod2D, nl]`` (``hnode`` zero below bottom). The
    returned ``bvfreq`` still needs :func:`smooth_nod3D` to match the dump. Mirrors
    ``fesom_pressure_bv`` for ``nzmin = ulevels-1``, ``nzmax = nlevels-1``.
    """
    g = G
    rho_ref = DENSITY_0
    Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])          # (nl,) pad invalid tail
    z = Zp[None, :]                                       # (1, nl)

    b0, bpz, bpz2, rhopot = jm_components(T, S)           # each (nod2D, nl)

    # --- in-situ density − ρ0 (layer field) ------------------------------------
    density = _insitu(b0, bpz, bpz2, rhopot, z) - rho_ref
    density = ops.mask_below_bottom(density, mesh.node_layer_mask)

    # --- hydrostatic pressure (layer field; downward cumulative) ---------------
    # surface BC at nzmin:  hp = -Z[nzmin]·ρ[nzmin]·g
    # interior nz>nzmin:    hp += 0.5·g·(ρ[nz-1]·h[nz-1] + ρ[nz]·h[nz])
    h = jnp.asarray(hnode)
    rho_up = _shift_down(density)
    h_up = _shift_down(h)
    interior = 0.5 * g * (rho_up * h_up + density * h)
    surf_bc = -z * density * g
    nzmin = (mesh.ulevels_nod2D - 1).reshape(-1, 1)      # (nod2D,1); 0 for pi
    k = jnp.arange(mesh.nl).reshape(1, -1)
    incr = jnp.where(k == nzmin, surf_bc, interior)
    incr = jnp.where(mesh.node_layer_mask, incr, 0.0)    # zero outside valid layers
    hpressure = jnp.cumsum(incr, axis=1)
    hpressure = ops.mask_below_bottom(hpressure, mesh.node_layer_mask)

    # --- Brunt-Väisälä N² (interface field) ------------------------------------
    # ρ_up, ρ_dn evaluated at the SAME depth zmean (compressibility cancels):
    #   zmean[nz] = ½(Z[nz-1]+Z[nz]);  bv[nz] = -g/(Z[nz-1]-Z[nz])·(ρ_up-ρ_dn)/ρ0
    Zd = _shift_down(Zp)                                  # Z[nz-1], edge-replicated
    zmean = (0.5 * (Zd + Zp))[None, :]
    zdiff = Zd - Zp
    zdiff = zdiff.at[0].set(1.0)                          # nz=0 unused; avoid 0/0
    bulk_up = _shift_down(b0) + zmean * (_shift_down(bpz) + zmean * _shift_down(bpz2))
    bulk_dn = b0 + zmean * (bpz + zmean * bpz2)
    rho_up_n = bulk_up * _shift_down(rhopot) / (bulk_up + 0.1 * zmean * _STATE_EQ_INT)
    rho_dn_n = bulk_dn * rhopot / (bulk_dn + 0.1 * zmean * _STATE_EQ_INT)
    bv = -g * (1.0 / zdiff[None, :]) * (rho_up_n - rho_dn_n) / rho_ref   # (nod2D,nl)

    # pad surface/bottom interfaces: bvfreq[nzmin]=bv[nzmin+1], bvfreq[nzmax]=bv[nzmax-1].
    # Equivalent to gathering bv at clip(nz, nzmin+1, nzmax-1) over the interface range.
    lo = mesh.ulevels_nod2D.reshape(-1, 1)               # = nzmin+1
    hi = (mesh.nlevels_nod2D - 2).reshape(-1, 1)         # = nzmax-1
    idx = jnp.clip(k, lo, hi)
    bvfreq = jnp.take_along_axis(bv, idx, axis=1)
    bvfreq = ops.mask_below_bottom(bvfreq, mesh.node_iface_mask)

    return density, hpressure, bvfreq


def smooth_nod3D(mesh: Mesh, arr, n_smooth: int = 1):
    """Area-weighted node-patch horizontal smoother (``fesom_smooth_nod3D``).

    One sweep = for every element ``el`` and interface level ``nz`` in its valid
    range, scatter ``area_el·(arr[v0]+arr[v1]+arr[v2])`` to each of ``el``'s three
    vertices, then divide each node's accumulation by ``3·Σarea``. The element's
    level range ⊆ its vertices' ranges (node ``nlevels``=MAX, ``ulevels``=MIN over
    cells), so the per-element clamp is exactly ``elem_iface_mask`` — no extra
    node-side level clamp. ``arr`` is ``[nod2D, nl]``. Scatter class → ~1e-12.
    """
    e, three, nl = mesh.elem2D, 3, mesh.nl
    area = mesh.elem_area[:, None]                        # (elem2D, 1)
    arr_s = jnp.asarray(arr)
    for _ in range(n_smooth):
        corners = ops.gather_nodes_to_elem(arr_s, mesh.elem_nodes)   # (elem2D,3,nl)
        bsum = corners.sum(axis=1)                                   # (elem2D, nl)
        contrib = jnp.where(mesh.elem_iface_mask, area * bsum, 0.0)  # (elem2D, nl)
        area_lev = jnp.where(mesh.elem_iface_mask, jnp.broadcast_to(area, (e, nl)), 0.0)

        vals = jnp.broadcast_to(contrib[:, None, :], (e, three, nl))
        work = ops.scatter_add(vals, mesh.elem_nodes, mesh.nod2D)    # (nod2D, nl)
        avals = jnp.broadcast_to(area_lev[:, None, :], (e, three, nl))
        vol = ops.scatter_add(avals, mesh.elem_nodes, mesh.nod2D)    # (nod2D, nl)

        denom = 3.0 * vol
        safe = jnp.where(denom > 0.0, denom, 1.0)
        arr_s = jnp.where(mesh.node_iface_mask, work / safe, 0.0)
    return arr_s


def compute_pressure_bv(mesh: Mesh, T, S, hnode, n_smooth: int = 1):
    """Driver mirror of ``fesom_step.c:77-92``: raw EOS/pressure/N² then the N²
    smoother. Returns ``(density, hpressure, bvfreq_smoothed)`` — the substep-1
    dump fields, ready to compare."""
    density, hpressure, bvfreq = pressure_bv(mesh, T, S, hnode)
    bvfreq = smooth_nod3D(mesh, bvfreq, n_smooth)
    return density, hpressure, bvfreq
