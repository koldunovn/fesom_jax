"""Sea-ice mEVP dynamics (Phase 9c) — an AD-safe port of ``fesom_ice_maevp.c``.

mEVP (``whichEVP=1``, Bouillon et al. 2013) replaces the standard-EVP elastic subcycling
(:mod:`fesom_jax.ice_evp`) by a **pseudo-time fixed-point iteration for the backward-Euler VP
problem over the FULL ice step** ``rdt = ice_dt``, stabilized by two relaxation constants
``alpha_evp`` (stress) / ``beta_evp`` (momentum). 120 fixed iterations, no CFL/adaptivity (that
is aEVP), no ``theta_io`` rotation, no ``Tevp_inv``/``dte``/``zeta``.

This module mirrors the C monolith ``fesom_ice_maevp.c`` (363 LOC) for 1:1 traceability but
**imports** the genuinely shared pieces from :mod:`fesom_jax.ice_evp` (``strain_rates``,
``stress_div_scatter``, ``_safe_sqrt``, ``boundary_node_mask``, ``bc_index_nod2D``). The setup /
relaxation / solve are rewritten against the C (the formulas genuinely differ). ``whichEVP=0``
keeps the EVP path; this kernel is reached only at ``whichEVP=1`` (the ``ice_step`` dispatch).

The 14 fidelity traps (plan ``20260611-fesom-jax-mevp.md``) are cited inline by number ``[Tn]``:
  T1  ``rdt = FULL ice_dt`` and **drag carries rdt**; ``drag·u_w`` is OUTSIDE the rdt group.
  T2  ``pressure_fac`` has NO 0.5 (det2 folded); the 0.5 sits in the σ11/σ22 updates only.
  T3  NO ``theta_io`` rotation anywhere.
  T4  masks: element ``mean₃(m_ice) > 0.01`` (mean, m-only); node ``a_ice ≥ 0.01``.
  T5  ``mass = M/((1+M²)·area)`` verbatim (smooth small-mass regularizer — never simplify).
  T6  ssh-tilt scatter is **UNMASKED** (all non-cavity elements); the σ scatter is owned-guarded.
  T7  rhs zeroing/scaling are C-MPI artifacts — pure values here (documented, not replicated).
  T8  aux init AND final copy over the full node extent; no exchange after the final copy.
  T9  ``bc_index`` from the GLOBAL-edge-id mask (never a local recompute).
  T10 ``delta_min`` is **ADDITIVE** (``/(Δ+δmin)``), not EVP's ``max(Δ, δmin)``.
  T11 σ is NOT zeroed on entry (persists across steps; decays by ``det1¹²⁰``).
  T12 all literals are doubles; ``meancos = metric_factor/3`` (we keep EVP's ``mfac·ΣV/3``, ~1e-16).
  T13 non-ice nodes are SKIPPED (identity velocity carry), not zeroed.
  T14 no ``uice_old``/``vice_old`` saves.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from . import ops
from .config import DENSITY_0
from .forcing import _safe_speed
from .halo import exchange_pair
from .ice import IceConfig
from .ice_evp import _safe_sqrt, bc_index_nod2D, boundary_node_mask, strain_rates, stress_div_scatter
from .mesh import Mesh

FESOM_G = 9.81           # gravity (the C FESOM_G; std-EVP uses the same literal)


# --------------------------------------------------------------------------
# Setup: per-call precompute (constant across the 120 iterations) — fesom_ice_maevp.c:138-220
# --------------------------------------------------------------------------
class MevpStatic(NamedTuple):
    ice_nod: jnp.ndarray        # [nod2D] bool — a_ice ≥ 0.01 & non-cavity (T4)
    inv_thickness: jnp.ndarray  # [nod2D] 1/max((ρᵢm+ρₛms)/a, 9), masked
    mass: jnp.ndarray           # [nod2D] M/((1+M²)·area), masked (T5)
    tilt_u: jnp.ndarray         # [nod2D] ssh-tilt rhs_a (scaled /area on ice nodes; T6)
    tilt_v: jnp.ndarray         # [nod2D] ssh-tilt rhs_m
    ice_el: jnp.ndarray         # [elem2D] bool — mean₃(m_ice) > 0.01 & non-cavity (T4)
    pressure_fac: jnp.ndarray   # [elem2D] det2·pstar·msum·exp(-C(1-asum)), NO 0.5 (T2)
    bc_index: jnp.ndarray       # [nod2D] 1.0 interior / 0.0 coastal (T9)


def mevp_setup(cfg: IceConfig, mesh: Mesh, a_ice, m_ice, m_snow, elevation,
               boundary_node) -> MevpStatic:
    """The per-call mEVP precompute (masks / mass regularizer / pressure_fac / ssh-tilt) that
    does not change across the 120 iterations (``fesom_ice_maevp.c:138-220``)."""
    nod2D = mesh.nod2D
    en = mesh.elem_nodes
    gs = mesh.gradient_sca
    area_n = mesh.area[:, 0]
    area_safe = jnp.where(area_n > 0.0, area_n, 1.0)
    non_cavity_n = jnp.asarray(mesh.ulevels_nod2D) <= 1
    non_cavity_el = jnp.asarray(mesh.ulevels) <= 1
    val3 = 1.0 / 3.0

    # --- ssh-tilt scatter (inlined ssh2rhs, levitating branch :148-162) — UNMASKED over all
    #     non-cavity elements (T6: NO has_ice mask, unlike std-EVP's tilt). Each element pushes
    #     -aa/-bb to ALL 3 vertices (same value). ---
    e_v = elevation[en]
    edx = gs[:, 0] * e_v[:, 0] + gs[:, 1] * e_v[:, 1] + gs[:, 2] * e_v[:, 2]
    edy = gs[:, 3] * e_v[:, 0] + gs[:, 4] * e_v[:, 1] + gs[:, 5] * e_v[:, 2]
    bb0 = FESOM_G * val3 * mesh.elem_area                       # G·area/3  (:156)
    aa = jnp.where(non_cavity_el, bb0 * edx, 0.0)              # :157 (cavity elements contribute 0)
    bb = jnp.where(non_cavity_el, bb0 * edy, 0.0)              # :158
    cu = (-aa)[:, None] * jnp.ones((1, 3))                     # rhs_a[nk] -= aa  (:159-161)
    cv = (-bb)[:, None] * jnp.ones((1, 3))
    rhs_a = ops.scatter_add(cu, en, nod2D)
    rhs_m = ops.scatter_add(cv, en, nod2D)

    # --- node precompute (:168-187): masks + mass. T4 node mask a_ice≥0.01; T5 verbatim mass. ---
    ice_nod = (a_ice >= 0.01) & non_cavity_n
    a_safe = jnp.where(a_ice >= 0.01, a_ice, 1.0)             # masked divide (AD)
    it = (cfg.rhoice * m_ice + cfg.rhosno * m_snow) / a_safe   # mean ice+snow thickness (:175)
    inv_thickness = jnp.where(ice_nod, 1.0 / jnp.maximum(it, 9.0), 0.0)   # limit small mass (:176)
    M = m_ice * cfg.rhoice + m_snow * cfg.rhosno              # per-area mass M (NOT /a_ice) (:178)
    mass = jnp.where(ice_nod, M / ((1.0 + M * M) * area_safe), 0.0)        # T5 (:179)
    # scale the ssh-tilt by 1/area ON ICE NODES (:182-183); non-ice keep the raw scatter (unused)
    tilt_u = jnp.where(ice_nod, rhs_a / area_safe, rhs_a)
    tilt_v = jnp.where(ice_nod, rhs_m / area_safe, rhs_m)

    # --- element precompute (:192-207): T4 element mask mean₃(m_ice)>0.01; pressure_fac with
    #     det2 folded and NO 0.5 (T2). ---
    m_v = m_ice[en]
    a_v = a_ice[en]
    msum = (m_v[:, 0] + m_v[:, 1] + m_v[:, 2]) * val3          # mean (:200)
    asum = (a_v[:, 0] + a_v[:, 1] + a_v[:, 2]) * val3          # mean (:203)
    ice_el = (msum > 0.01) & non_cavity_el
    pressure_fac = jnp.where(
        ice_el, cfg.mevp_det2 * cfg.pstar * msum * jnp.exp(-cfg.c_pressure * (1.0 - asum)), 0.0)

    return MevpStatic(ice_nod=ice_nod, inv_thickness=inv_thickness, mass=mass,
                      tilt_u=tilt_u, tilt_v=tilt_v, ice_el=ice_el,
                      pressure_fac=pressure_fac, bc_index=bc_index_nod2D(boundary_node))


# --------------------------------------------------------------------------
# One pseudo-time iteration — fesom_ice_maevp.c:224-322 (element σ + scatter, node solve, edge-BC)
# --------------------------------------------------------------------------
def mevp_iterate(cfg: IceConfig, mesh: Mesh, u_aux, v_aux, s11, s12, s22, st: MevpStatic,
                 u_anchor, v_anchor, srfoce_u, srfoce_v, stress_ax, stress_ay, rdt):
    """One mEVP pseudo-time iteration. Returns the new ``(u_aux, v_aux, σ11, σ12, σ22)`` AFTER
    the edge-BC, BEFORE the halo exchange (the C dump placement, :324-337).

    ``u_anchor``/``v_anchor`` are the backward-Euler rhs anchor — in production the **FROZEN
    entry** ``(u_ice, v_ice)`` (closed over, constant across iterations; Decisions #4 / T1).
    std-EVP anchors on the current iterate; passing ``u_aux`` here reproduces that (wrong) fixed
    point — exercised by the it2 entry-anchor test."""
    vale = cfg.vale

    # --- element: strain (shared) → Δ (additive δmin, T10) → α-relaxed σ (T2: 0.5 in σ11/σ22) ---
    eps11, eps22, eps12 = strain_rates(mesh, u_aux, v_aux)     # shared block (:241-246)
    eps1 = eps11 + eps22                                       # :249
    eps2 = eps11 - eps22                                       # :250
    radicand = eps1 * eps1 + vale * (eps2 * eps2 + 4.0 * eps12 * eps12)   # :253
    delta = _safe_sqrt(radicand)                              # AD-safe sqrt (Δ=0 at cold start)
    pressure = st.pressure_fac / (delta + cfg.delta_min)      # T10 ADDITIVE δmin (:254)
    det1 = cfg.mevp_det1
    s12n = det1 * s12 + pressure * eps12 * vale               # :258  (NO 0.5 — T2)
    s11n = det1 * s11 + 0.5 * pressure * (eps1 - delta + eps2 * vale)    # :259
    s22n = det1 * s22 + 0.5 * pressure * (eps1 - delta - eps2 * vale)    # :260
    # element skip (`!ice_el` ⇒ σ frozen, carry old) — :693 continue
    s11 = jnp.where(st.ice_el, s11n, s11)
    s12 = jnp.where(st.ice_el, s12n, s12)
    s22 = jnp.where(st.ice_el, s22n, s22)

    # --- stress-divergence scatter (shared raw scatter; act=ice_el guards ice-free elems, T6
    #     owned-guard is implicit on a single device — scatter-to-all, read-at-owned) (:265-272) ---
    u_rhs, v_rhs = stress_div_scatter(mesh, s11, s12, s22, st.ice_el)

    # --- node solve (:281-309): β-relaxed, bc_index-masked, drag carries rdt (T1), frozen-entry
    #     rhs anchor (T1/Decisions #4), non-ice identity carry (T13), NO theta_io (T3). ---
    u_rhs = u_rhs * st.mass + st.tilt_u                        # ·mass + ssh-tilt tail (:284)
    v_rhs = v_rhs * st.mass + st.tilt_v                        # (:285)
    du = u_aux - srfoce_u                                      # :287
    dv = v_aux - srfoce_v                                      # :288
    umod = _safe_speed(du, dv)                                 # AD-safe |Δu| (:289)
    drag = rdt * cfg.cd_oce_ice * umod * DENSITY_0 * st.inv_thickness    # T1: rdt-carrying (:290)
    # rhs: entry anchor + drag·u_w OUTSIDE the rdt group + rdt·(air stress + internal) + β·u_aux
    rhsu = u_anchor + drag * srfoce_u + rdt * (st.inv_thickness * stress_ax + u_rhs) \
        + cfg.beta_evp * u_aux                                 # :294-296
    rhsv = v_anchor + drag * srfoce_v + rdt * (st.inv_thickness * stress_ay + v_rhs) \
        + cfg.beta_evp * v_aux                                 # :297-299
    obd = 1.0 + cfg.beta_evp + drag                           # :302
    rf = rdt * mesh.coriolis_node                             # :303
    det = st.bc_index / (obd * obd + rf * rf)                 # T9 bc_index in the determinant (:304)
    un = det * (obd * rhsu + rf * rhsv)                       # :306
    vn = det * (obd * rhsv - rf * rhsu)                       # :307
    u_aux = jnp.where(st.ice_nod, un, u_aux)                  # T13 identity carry on non-ice (:283)
    v_aux = jnp.where(st.ice_nod, vn, v_aux)

    # --- edge BC (:315-322): zero aux velocity at coastal boundary-edge endpoints. ---
    u_aux = jnp.where(st.bc_index > 0.0, u_aux, 0.0)
    v_aux = jnp.where(st.bc_index > 0.0, v_aux, 0.0)
    return u_aux, v_aux, s11, s12, s22


# --------------------------------------------------------------------------
# Full mEVP driver — the 120-iteration scan — fesom_ice_maevp.c:66-363
# --------------------------------------------------------------------------
def mevp_dynamics(cfg: IceConfig, mesh: Mesh, *, a_ice, m_ice, m_snow,
                  u_ice, v_ice, sigma11, sigma12, sigma22,
                  srfoce_u, srfoce_v, elevation, stress_ax, stress_ay,
                  boundary_node=None, exch=None):
    """Run the mEVP momentum solver (setup + 120 pseudo-time iterations). Same signature /
    outputs as :func:`fesom_jax.ice_evp.evp_dynamics`: returns the updated
    ``(u_ice, v_ice, sigma11, sigma12, sigma22)``. ``stress_ax/ay`` = the atm-ice wind stress;
    ``elevation`` = ``srfoce_ssh`` (``hbar``); ``srfoce_u/v`` = the ocean surface velocity.

    The **scan carry is ``(u_aux, v_aux, σ11, σ12, σ22)``** initialised to the entry
    ``(u_ice, v_ice, σ…)`` (T8/T11 — σ NOT zeroed on entry), while the **frozen entry
    ``(u_ice, v_ice)``** is closed over as the backward-Euler rhs anchor (Decisions #4 / T1).
    ``rdt = cfg.ice_dt`` is the FULL ice step (T1; ``ice_dt`` is force-derived in ``ice_step`` to
    track the ocean dt — the historic config-desync lesson).

    **Sharding.** ``exch`` (``None`` ⇒ byte-identical) is the per-iteration ``u_aux``/``v_aux``
    node-halo refresh wired INSIDE the scan (the EVP precedent): the next iteration's
    ``strain_rates`` reads ``u_aux`` at the element's HALO vertices, so the halo must be refreshed
    each iteration. ``boundary_node`` MUST be the GLOBAL coastal mask (partitioned in), never a
    local recompute (T9). No exchange after the final copy — the aux halo is current from the last
    iteration's exchange (T8)."""
    if boundary_node is None:
        boundary_node = boundary_node_mask(mesh)
    st = mevp_setup(cfg, mesh, a_ice, m_ice, m_snow, elevation, boundary_node)
    rdt = cfg.ice_dt                                          # T1: the FULL ice step

    def body(carry, _):
        u_i, v_i, s11, s12, s22 = carry
        u_i, v_i, s11, s12, s22 = mevp_iterate(
            cfg, mesh, u_i, v_i, s11, s12, s22, st,
            u_ice, v_ice, srfoce_u, srfoce_v, stress_ax, stress_ay, rdt)   # anchor = frozen entry
        u_i, v_i = exchange_pair(exch, u_i, v_i, "nod")      # fused halo refresh for next strain read
        return (u_i, v_i, s11, s12, s22), None

    init = (u_ice, v_ice, sigma11, sigma12, sigma22)          # T11: σ carried, NOT zeroed
    body = jax.checkpoint(body)                               # cap the 120-iteration backward memory
    (u_ice, v_ice, sigma11, sigma12, sigma22), _ = lax.scan(
        body, init, None, length=cfg.evp_rheol_steps)
    return u_ice, v_ice, sigma11, sigma12, sigma22            # T8 final copy; no extra exchange
