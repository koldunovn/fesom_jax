"""GM/Redi — the mesoscale eddy parameterization (Phase 6B).

Ports the FESOM2 C GM/Redi subsystem (``fesom_gm.c``): an eddy-induced **bolus
advection** (Gent-McWilliams) + **neutral (isopycnal) diffusion** (Redi). The
parameterization is a pure **diagnostic of the current density field** — every
field is recomputed each step from T/S/N² (no prognostic carry, unlike the ice EVP
σ memory), so there are **no new** :class:`~fesom_jax.state.State` fields.

Pipeline (data-flow order; the C ``fesom_step.c`` integration seam):

1. ``eos.compute_sw_alpha_beta`` — α/β (substep 2; in :mod:`fesom_jax.eos`).
2. :func:`compute_sigma_xy` → :func:`compute_neutral_slope` — neutral slopes + the
   ODM95 taper.  *(Task G.2)*
3. :func:`init_redi_gm` — the per-step GM/Redi coefficient builder (``fer_K``/``Ki``/
   ``fer_C``/``fer_scal``); reads the differentiable ``k_gm``/``redi_kmax`` from
   :class:`~fesom_jax.params.Params` (the 2nd ML-hook seam).  *(Task G.3)*
4. :func:`fer_solve_gamma` → :func:`fer_gamma2vel` — the streamfunction TDMA + the
   bolus velocity ``fer_uv``.  *(Task G.4)*

The bolus vertical velocity ``fer_w`` + the bolus-advection wrap live in
:mod:`fesom_jax.ale` / :mod:`fesom_jax.step` (Task G.5); the Redi explicit terms +
the K33 augmentation in :mod:`fesom_jax.gm_redi` / :mod:`fesom_jax.tracer_diff`
(Task G.6). Everything is wired behind a static ``gm_cfg`` arg (``None`` ⇒ the
pi/Phase-5/ice path stays bit-identical — the ``ice_cfg`` precedent).

The active namelist is a small subset (default ``namelist.oce``): ``Fer_GM=Redi=
Redi_Ktaper=scaling_ODM95=scaling_resolution=scaling_GMzexp=T``,
``K_GM_resscalorder=2``. See ``docs/plans/20260607-fesom-jax-gmredi.md`` §0/§3.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from . import ops
from .config import DENSITY_0, G, K_GM_MAX, REDI_KMAX
from .mesh import Mesh


class GMConfig(NamedTuple):
    """Static GM/Redi constants (closed over the step; the differentiable ceilings
    ``k_gm``/``redi_kmax`` live in :class:`~fesom_jax.params.Params` instead).

    Defaults = the active ``namelist.oce`` values, cross-checked against
    ``fesom_gm.c`` (the per-constant citations are in the §3 config table of the
    sub-plan). All SI.
    """

    # --- master switches (active branches only) ---
    redi: bool = True            # Redi neutral diffusion on
    fer_gm: bool = True          # GM bolus advection on
    redi_ktaper: bool = True     # taper Ki by sqrt(fer_tapfac)

    # --- diffusivity floors/scales (the ceilings are in Params) ---
    k_gm_min: float = 2.0        # fer_K floor [m²/s]            (fesom_gm.c:358)
    redi_kmin: float = 100.0     # Redi taper floor [m²/s]       (fesom_gm.c:361)
    k_gm_cmin: float = 0.1       # baroclinic-speed floor        (fesom_gm.c:359)
    k_gm_cm: float = 3.0         # cm divisor                    (fesom_gm.c:360)

    # --- resolution scaling (K_GM_resscalorder=2 → sqrt) ---
    refscalresol: float = 100000.0   # 100 km                    (fesom_gm.c:365)

    # --- depth scaling (scaling_GMzexp) ---
    gmzexp_zref: float = 500.0       # e-folding depth [m]       (fesom_gm.c:363)
    gmzexp_smin: float = 0.6         # floor of the depth scale  (fesom_gm.c:364)

    # --- ODM95 slope taper (scaling_ODM95; LDD97 off so c2≡1) ---
    odm95_scr: float = 0.2e-2        # critical slope            (fesom_gm.c:237)
    odm95_sd: float = 1.0e-3         # taper width               (fesom_gm.c:238)
    slope_eps: float = 5.0e-6        # neutral-slope floor       (fesom_gm.c:233)

    # --- numerical floors ---
    gamma_bv_floor: float = 1.0e-8   # N² floor in the Γ solve   (fesom_gm.c:558)

    # --- physical constants (mirror config; here for a self-contained bundle) ---
    g: float = G
    rho_ref: float = DENSITY_0

    # --- defaults for the differentiable ceilings (used when params is None) ---
    k_gm_max: float = K_GM_MAX       # = Params.k_gm default     (fesom_gm.c:357)
    redi_kmax_default: float = REDI_KMAX  # = Params.redi_kmax    (fesom_gm.c:362)


def _safe_sqrt(x):
    """Double-``where`` safe sqrt: forward-identical to ``sqrt(max(x,0))`` but with a
    finite gradient at ``x=0`` (a bare ``sqrt(0)`` has grad ``1/(2·0)=∞`` that a
    downstream mask does not stop in the backward pass)."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(safe), 0.0)


# ============================================================================
# G.2 — compute_sigma_xy (fesom_gm.c:124-202)
# ============================================================================
def compute_sigma_xy(mesh: Mesh, T, S, sw_alpha, sw_beta, cfg: GMConfig):
    """Density gradient on neutral surfaces, per node/level, both components.

    ``sigma_xy(c,nz,n) = (-α·∂_c⟨T⟩ + β·∂_c⟨S⟩)·ρ0`` where ``∂_c⟨·⟩`` is the
    **area-weighted mean over surrounding elements** of the per-element ∇T/∇S
    (``gradient_sca`` contraction). An element→node area-weighted scatter (the
    ``eos.smooth_nod3D`` pattern, but ÷Σarea, not 3·Σarea). Returns ``(N, nl, 2)``
    (comp 0=x, 1=y), masked to the layer range. ``fesom_gm.c:124``.
    """
    E, nl = mesh.elem2D, mesh.nl
    g = mesh.gradient_sca                                   # (E,6): 0:3=∂N/∂x, 3:6=∂N/∂y
    gx, gy = g[:, 0:3], g[:, 3:6]                           # (E,3)
    T_e = ops.gather_nodes_to_elem(T, mesh.elem_nodes)      # (E,3,nl)
    S_e = ops.gather_nodes_to_elem(S, mesh.elem_nodes)
    gradTx = jnp.sum(gx[:, :, None] * T_e, axis=1)          # (E,nl)
    gradTy = jnp.sum(gy[:, :, None] * T_e, axis=1)
    gradSx = jnp.sum(gx[:, :, None] * S_e, axis=1)
    gradSy = jnp.sum(gy[:, :, None] * S_e, axis=1)

    area = mesh.elem_area[:, None]                          # (E,1)
    one = jnp.broadcast_to(area, (E, nl))
    # 5 element contributions: the 4 area-weighted grads + the area itself (→ vol).
    contribs = jnp.stack(
        [area * gradTx, area * gradTy, area * gradSx, area * gradSy, one], axis=-1)
    contribs = jnp.where(mesh.elem_layer_mask[..., None], contribs, 0.0)  # (E,nl,5)
    vals = jnp.broadcast_to(contribs[:, None, :, :], (E, 3, nl, 5))
    acc = ops.scatter_add(vals, mesh.elem_nodes, mesh.nod2D)             # (N,nl,5)
    tx, ty, sx, sy, vol = (acc[..., k] for k in range(5))

    inv_vol = jnp.where(vol > 0.0, 1.0 / jnp.where(vol > 0.0, vol, 1.0), 0.0)
    rho0 = cfg.rho_ref
    sig_x = (-sw_alpha * tx + sw_beta * sx) * inv_vol * rho0
    sig_y = (-sw_alpha * ty + sw_beta * sy) * inv_vol * rho0
    sigma_xy = jnp.stack([sig_x, sig_y], axis=-1)          # (N,nl,2)
    return jnp.where(mesh.node_layer_mask[..., None], sigma_xy, 0.0)


# ============================================================================
# G.2 — compute_neutral_slope (fesom_gm.c:223-310)
# ============================================================================
def compute_neutral_slope(mesh: Mesh, sigma_xy, bvfreq, cfg: GMConfig):
    """Neutral slopes from ``sigma_xy`` / N², the ODM95 tanh taper ``c1``, and the
    tapered slope. Active config: ``scaling_ODM95=T``, ``LDD97=F`` (c2≡1).

    Returns ``(neutral_slope, slope_tapered, fer_tapfac)``:
    ``neutral_slope``/``slope_tapered`` are ``(N, nl-1, 3)`` (comp 0=x, 1=y, 2=|s|);
    ``fer_tapfac`` is ``(N, nl)`` (= c1 at the layer levels). ``fesom_gm.c:223``.
    """
    nl = mesh.nl
    eps_sq = cfg.slope_eps ** 2
    bv0 = bvfreq[:, : nl - 1]                               # bvfreq[nz]   (N,nl-1)
    bv1 = bvfreq[:, 1:nl]                                   # bvfreq[nz+1]
    denom = jnp.maximum(bv0 + bv1, eps_sq)
    ro_z_inv = 2.0 * cfg.g / cfg.rho_ref / denom
    sx = sigma_xy[:, : nl - 1, 0] * ro_z_inv
    sy = sigma_xy[:, : nl - 1, 1] * ro_z_inv
    sm = _safe_sqrt(sx * sx + sy * sy)

    # ODM95 c1 = ½(1+tanh((Scr-|s|)/Sd)); forced 0 where either N² ≤ 0.
    c1 = 0.5 * (1.0 + jnp.tanh((cfg.odm95_scr - sm) / cfg.odm95_sd))
    c1 = jnp.where((bv0 <= 0.0) | (bv1 <= 0.0), 0.0, c1)
    smask = mesh.node_layer_mask[:, : nl - 1]              # (N,nl-1) layer range
    c1 = jnp.where(smask, c1, 0.0)

    neutral_slope = jnp.where(
        smask[..., None], jnp.stack([sx, sy, sm], axis=-1), 0.0)   # (N,nl-1,3)
    slope_tapered = neutral_slope * _safe_sqrt(c1)[..., None]
    fer_tapfac = jnp.pad(c1, ((0, 0), (0, 1)))             # (N,nl); padded col below-bottom
    return neutral_slope, slope_tapered, fer_tapfac


# ============================================================================
# G.3 — init_redi_gm (fesom_gm.c:345-468)
# ============================================================================
def init_redi_gm(mesh: Mesh, bvfreq, hnode_new, fer_tapfac, params, cfg: GMConfig):
    """Per-step GM/Redi coefficient builder: ``fer_K`` (GM thickness diffusivity),
    ``Ki`` (Redi diffusivity), ``fer_C`` (baroclinic-wave-speed²), ``fer_scal``
    (resolution scaling). Reads the differentiable ``params.k_gm``/``params.redi_kmax``
    (the 2nd ML-hook). Active config (default namelist): ``K_GM_resscalorder=2``,
    ``scaling_resolution=T``, ``scaling_GMzexp=T``, ``Redi=GM sync``, ``Redi_Ktaper=T``.

    Two passes:
      * F1 (CONSERVATIVE bounds ``ulevels_nod2D_max``/``nlevels_nod2D_min``): the
        resolution scaling + the depth-integral wave speed ``cm`` → the top-level
        ``fer_K``/``Ki`` and ``fer_C``.
      * F2 (REGULAR bounds): the depth-exp ``zscaling`` applied to all levels, then
        the ``Redi_Ktaper`` √(fer_tapfac) tapering of ``Ki``.

    ``fer_K`` is on the iface range (``node_iface_mask``); ``Ki``/``fer_C``/``fer_scal``
    are layer/scalar. Returns ``(fer_K, Ki, fer_C, fer_scal)``. ``fesom_gm.c:345``.
    """
    nl = mesh.nl
    pi = 3.14159265358979323846
    k_gm = params.k_gm
    redi_kmax = params.redi_kmax

    # --- F1: resolution scaling + the wave-speed cm (conservative bounds) -----
    area_n = mesh.area[:, 0]                                # (N,) surface CV area
    inv_ref_sq = 1.0 / (cfg.refscalresol * cfg.refscalresol)
    scaling = jnp.minimum(_safe_sqrt(area_n * inv_ref_sq * 2.0), 1.0)   # (N,)
    fer_scal = scaling
    k_top = jnp.maximum(scaling * k_gm, cfg.k_gm_min)       # (N,) fer_K[nzmin]
    ki_top = jnp.maximum(scaling * redi_kmax, cfg.k_gm_min)  # (N,) Ki[nzmin]

    # cm = depth integral of |N| over the CONSERVATIVE level range.
    sqrt_bv = _safe_sqrt(bvfreq)                            # (N,nl); 0 where bv≤0
    bv_lo = sqrt_bv[:, : nl - 1]                            # √bv[nz]   (N,nl-1)
    bv_hi = sqrt_bv[:, 1:nl]                                # √bv[nz+1]
    term = hnode_new[:, : nl - 1] * 0.5 * (bv_lo + bv_hi)   # (N,nl-1) layer terms
    k_layer = jnp.arange(nl - 1)[None, :]
    cons_lo = (mesh.ulevels_nod2D_max - 1)[:, None]
    cons_hi = (mesh.nlevels_nod2D_min - 1)[:, None]
    cons_mask = (k_layer >= cons_lo) & (k_layer < cons_hi)  # conservative layer range
    cm_sum = jnp.sum(jnp.where(cons_mask, term, 0.0), axis=1)            # (N,)
    cm = jnp.maximum(cm_sum / pi / cfg.k_gm_cm, cfg.k_gm_cmin)
    fer_C = cm * cm                                         # (N,)

    # --- F2: depth-exp zscaling (regular bounds) -----------------------------
    z = jnp.abs(mesh.zbar_3d_n)                             # (N,nl) |interface depth|
    zscaling = cfg.gmzexp_smin + (1.0 - cfg.gmzexp_smin) * jnp.exp(-z / cfg.gmzexp_zref)
    zscaling = jnp.clip(zscaling, cfg.gmzexp_smin, 1.0)     # (N,nl); below-bottom→1 (z=0)

    fer_K = jnp.where(mesh.node_iface_mask, k_top[:, None] * zscaling, 0.0)

    # Ki uses the interface-average 0.5(zscaling[nz]+zscaling[nz+1]) at layer nz.
    zsc_avg = 0.5 * (zscaling[:, : nl - 1] + zscaling[:, 1:nl])         # (N,nl-1)
    zsc_avg = jnp.pad(zsc_avg, ((0, 0), (0, 1)))           # (N,nl); pad below-bottom
    Ki = ki_top[:, None] * zsc_avg
    # Redi_Ktaper: Ki = Ki·√tapfac + Redi_Kmin·|√tapfac − 1|.
    s = _safe_sqrt(fer_tapfac)                             # (N,nl) √c1
    Ki = Ki * s + cfg.redi_kmin * jnp.abs(s - 1.0)
    Ki = jnp.where(mesh.node_layer_mask, Ki, 0.0)

    return fer_K, Ki, fer_C, fer_scal


# fer_solve_gamma + fer_gamma2vel (G.4) land next.
