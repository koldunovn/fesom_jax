"""Equation of state, hydrostatic pressure, Brunt-Väisälä N² (substep 1 / Task 2.1).

Literal, vectorized port of the C port's ``fesom_pressure_bv`` + the N² smoother
``fesom_smooth_nod3D`` (``fesom_eos.c:78-277``, driven from ``fesom_step.c:77-96``)
for the Phase-2 pi config: JM-EOS (``state_equation=1``), ``linfs`` (uses the
``hpressure`` linfs branch), no cavity (``nzmin=0``), ``use_density_ref=.false.``
(→ ``density_ref ≡ density_0``), PP mixing (so ``dbsfc``/``MLD1`` are unused and
skipped — they feed only KPP).

:func:`compute_sw_alpha_beta` (substep 2; McDougall 1987 thermal-expansion /
saline-contraction coefficients) is added in **Phase 6B** — GM/Redi (and KPP)
read ``sw_alpha``/``sw_beta``. It is a pure per-node polynomial map (sibling to
:func:`jm_components`), ``fesom_eos.c:323-375``.

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
    # zdiff == 0 at BOTH unused interfaces — the surface (nz=0, edge-replicated) and
    # the bottom padding (Zp duplicates Z[-1] in its tail) — where 1/zdiff would be
    # inf. Both are clipped out of the forward bvfreq, but a forward inf still poisons
    # the BACKWARD pass (0·inf = NaN flows to d/dT at the masked lanes — the classic
    # masked-NaN trap, cf. tracer_diff's where(dZ==0,1,dZ)). Replace with 1.0 so the
    # divide is finite both ways; the forward output is unchanged (these lanes are
    # never read after the clip).
    zdiff = jnp.where(zdiff == 0.0, 1.0, zdiff)
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


# --- sw_alpha_beta (McDougall 1987) — substep 2, Phase 6B ----------------------
# Verbatim from fesom_eos.c:336-369 (= oce_ale_pressure_bv.F90:2751-2846). The two
# polynomials (saline contraction `beta`, the ratio `a_over_b`) are written term by
# term in the C — NOT Horner — so we mirror that exact grouping (left-to-right `+`)
# for bit-for-bit agreement. Pure per-node MAP, no sqrt/divide ⇒ trivially AD-finite.


def compute_sw_alpha_beta(mesh: Mesh, T, S):
    """Thermal-expansion (``sw_alpha``, 1/K) and saline-contraction (``sw_beta``,
    1/(g/kg)) coefficients per node/level — McDougall (1987), ``fesom_eos.c:323``.

    ``T``, ``S`` are ``[nod2D, nl]``. Pressure proxy ``p1 = |Z[nz]|`` (the static
    layer-midpoint depth, broadcast over nodes; linfs full-cell ⇒ ``Z_3d_n = Z``).
    Returns ``(sw_alpha, sw_beta)``, both ``[nod2D, nl]``, masked to the layer
    range. Consumed by GM/Redi (:mod:`fesom_jax.gm`) and KPP (Phase 6C).
    """
    t1 = jnp.asarray(T) * 1.00024
    s1 = jnp.asarray(S)
    # Z is the (nl-1,) layer-midpoint depth; pad the invalid tail to (nl,) like
    # pressure_bv (the padded lane is masked out below). p1 = |Z[nz]| broadcast.
    Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])
    p1 = jnp.abs(Zp)[None, :]                             # (1, nl) pressure proxy |Z|

    t1_2 = t1 * t1
    t1_3 = t1_2 * t1
    t1_4 = t1_3 * t1
    p1_2 = p1 * p1
    p1_3 = p1_2 * p1
    s35 = s1 - 35.0
    s35_2 = s35 * s35

    beta = (
        0.785567e-3
        - 0.301985e-5 * t1
        + 0.555579e-7 * t1_2
        - 0.415613e-9 * t1_3
        + s35 * (-0.356603e-6 + 0.788212e-8 * t1
                 + 0.408195e-10 * p1 - 0.602281e-15 * p1_2)
        + s35_2 * (0.515032e-8)
        + p1 * (-0.121555e-7 + 0.192867e-9 * t1 - 0.213127e-11 * t1_2)
        + p1_2 * (0.176621e-12 - 0.175379e-14 * t1)
        + p1_3 * (0.121551e-17)
    )

    a_over_b = (
        0.665157e-1
        + 0.170907e-1 * t1
        - 0.203814e-3 * t1_2
        + 0.298357e-5 * t1_3
        - 0.255019e-7 * t1_4
        + s35 * (0.378110e-2 - 0.846960e-4 * t1
                 - 0.164759e-6 * p1 - 0.251520e-11 * p1_2)
        + s35_2 * (-0.678662e-5)
        + p1 * (0.380374e-4 - 0.933746e-6 * t1 + 0.791325e-8 * t1_2)
        + p1_2 * t1_2 * (0.512857e-12)
        - p1_3 * (0.302285e-13)
    )

    sw_beta = ops.mask_below_bottom(beta, mesh.node_layer_mask)
    sw_alpha = ops.mask_below_bottom(a_over_b * beta, mesh.node_layer_mask)
    return sw_alpha, sw_beta
