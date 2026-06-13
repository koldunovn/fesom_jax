"""Sea-ice EVP dynamics (Phase 6, Task 6.4) — an AD-safe port of ``fesom_ice_evp.c``.

The elastic-viscous-plastic momentum solver for the ice velocity ``u_ice``/``v_ice``. The
hardest AD piece in Phase 6: a **fixed 120-subcycle** loop (``evp_rheol_steps``) → a
checkpointed :func:`jax.lax.scan`. Each subcycle: element strain rates → the Hunke EVP stress
update (σ carried as elastic memory) → scatter the stress divergence to nodes → a 2×2 implicit
Coriolis+ocean-drag velocity update → the coastal boundary condition.

Structure (mirrors ``fesom_ice_evp_dynamics``, ``fesom_ice_evp.c:212-457``):
  :func:`evp_setup`     — per-node mass / ``inv_areamass`` / ``inv_mass``; per-element
                          ``ice_strength`` (Hunke) + the SSH-tilt velocity forcing (``rhs_a/m``).
  :func:`stress_tensor` — strain rates ε (gradient_sca + metric), ``Δ`` (safe-sqrt), ζ, the
                          Hunke σ update.  (``fesom_ice_stress_tensor``, ``:68-132``)
  :func:`stress2rhs`    — element→node scatter of the σ divergence + ``·inv_areamass + tilt``.
  :func:`velocity_update` — the 2×2 implicit solve + the ``a<0.01`` gate + coastal BC.
  :func:`evp_dynamics`  — setup + the 120-subcycle ``scan``.

⚠️ AD guards (the masked-NaN rule): the strain-rate invariant ``Δ = √(radicand)`` uses a
double-`where` safe-sqrt (``radicand→0`` in ice-free/rigid lanes); ``Δ`` is then clamped to
``delta_min`` (matching the C — do NOT raise it); ``inv_mass``/``inv_areamass`` are masked
divides; ``ice_strength`` is 0 unless all 3 vertices have ice (so ice-free elements contribute
nothing and σ is frozen there, exactly as the C skips them); the ocean-drag ``|u_ice-u_w|`` is a
safe-sqrt. Single device ⇒ no halo exchange (the C's per-subcycle ``exchange_nod2D`` is a no-op).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from . import ops
from .config import DENSITY_0
from .forcing import _safe_speed
from .ice import IceConfig
from .mesh import Mesh


def _safe_sqrt(x):
    """``√x`` with finite (0) gradient at ``x==0`` (matches C ``sqrt(0)``)."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(safe), 0.0)


def boundary_node_mask(mesh: Mesh) -> jnp.ndarray:
    """Boundary-node mask (``[nod2D]`` bool) — endpoints of boundary edges (index ≥
    ``edge2D_in``, the interior-first ordering; == the C ``myList_edge2D>edge2D_in`` on a
    single rank). The EVP zeros ``u_ice``/``v_ice`` here (``fesom_ice_evp.c:430-437``)."""
    edge2D = mesh.edges.shape[0]
    be = (jnp.arange(edge2D) >= mesh.edge2D_in).astype(jnp.int32)
    cnt = (ops.scatter_add(be, mesh.edges[:, 0], mesh.nod2D)
           + ops.scatter_add(be, mesh.edges[:, 1], mesh.nod2D))
    return cnt > 0


def bc_index_nod2D(boundary_node: jnp.ndarray) -> jnp.ndarray:
    """Interior-node indicator (``1.0`` interior / ``0.0`` coastal) — the C ``bc_index_nod2D``
    (``fesom_ice.c:249-258``: 1 everywhere, 0 at boundary-edge endpoints) == ``1.0 −
    boundary_node_mask``. mEVP multiplies it into the node-solve determinant
    (``fesom_ice_maevp.c:304``). ⚠️ Sharding: pass the GLOBAL boundary mask (partitioned in),
    never a local-mesh recompute — a partition-seam edge mis-flags an interior node as coastal
    (the C reads ``partit->myList_edge2D``; the same trap the std-EVP ``boundary_node`` guards)."""
    return 1.0 - boundary_node.astype(jnp.float64)


# --------------------------------------------------------------------------
# Shared rheology helpers (extracted JM.1) — EVP and mEVP use these IDENTICALLY.
# ⚠️ The EVP association ``mfac·(ΣV)/3`` is kept here; the C mEVP writes ``meancos=mfac/3``
# (``fesom_ice_maevp.c:236``), a ~1e-16 FP-association difference that is NOT chased in the
# mEVP it-dumps (plan JM.1). The per-module tails (``·inv_areamass+tilt`` for EVP,
# ``·mass+rhs_a`` for mEVP) stay in their own modules — only the raw blocks are shared.
# --------------------------------------------------------------------------
def strain_rates(mesh: Mesh, u_ice, v_ice):
    """Element strain-rate tensor ``(eps11, eps22, eps12)`` from the node velocities at the
    element vertices (``fesom_ice_evp.c:88-105`` / ``fesom_ice_maevp.c:241-246``). Shared,
    bit-identical formula; the metric term uses the EVP association ``mfac·(ΣV)/3``."""
    gs = mesh.gradient_sca
    mfac = mesh.metric_factor
    en = mesh.elem_nodes
    U = u_ice[en]
    V = v_ice[en]
    eps11 = (gs[:, 0] * U[:, 0] + gs[:, 1] * U[:, 1] + gs[:, 2] * U[:, 2]
             - mfac * (V[:, 0] + V[:, 1] + V[:, 2]) / 3.0)
    eps22 = gs[:, 3] * V[:, 0] + gs[:, 4] * V[:, 1] + gs[:, 5] * V[:, 2]
    eps12 = 0.5 * (gs[:, 3] * U[:, 0] + gs[:, 4] * U[:, 1] + gs[:, 5] * U[:, 2]
                   + gs[:, 0] * V[:, 0] + gs[:, 1] * V[:, 1] + gs[:, 2] * V[:, 2]
                   + mfac * (U[:, 0] + U[:, 1] + U[:, 2]) / 3.0)
    return eps11, eps22, eps12


def stress_div_scatter(mesh: Mesh, s11, s12, s22, act):
    """Element→node scatter of the σ divergence — the RAW scatter (``fesom_ice_evp.c:159-166``
    / ``fesom_ice_maevp.c:265-272``), BEFORE the per-module ``·mass/inv_areamass + tilt`` tail.
    ``act`` (per-element bool) zeroes ice-free element contributions (the C ``continue``):
    EVP passes ``ice_strength>0``, mEVP passes ``ice_el``. Returns ``(u_rhs, v_rhs)``."""
    gs = mesh.gradient_sca
    mfac = mesh.metric_factor
    en = mesh.elem_nodes
    a = mesh.elem_area
    val3 = 1.0 / 3.0
    sx11 = jnp.where(act, s11, 0.0)
    sx12 = jnp.where(act, s12, 0.0)
    sx22 = jnp.where(act, s22, 0.0)
    cu = jnp.stack(
        [-a * (sx11 * gs[:, k] + sx12 * gs[:, k + 3] + sx12 * val3 * mfac) for k in range(3)],
        axis=1)
    cv = jnp.stack(
        [-a * (sx12 * gs[:, k] + sx22 * gs[:, k + 3] - sx11 * val3 * mfac) for k in range(3)],
        axis=1)
    u_rhs = ops.scatter_add(cu, en, mesh.nod2D)
    v_rhs = ops.scatter_add(cv, en, mesh.nod2D)
    return u_rhs, v_rhs


# --------------------------------------------------------------------------
# Setup: mass / inv_mass / ice_strength / SSH-tilt rhs — fesom_ice_evp.c:263-350
# --------------------------------------------------------------------------
class EVPStatic(NamedTuple):
    inv_areamass: jnp.ndarray   # [nod2D]  1/(area·mass) masked
    inv_mass: jnp.ndarray       # [nod2D]  1/max(mass/a, 9) masked
    ice_strength: jnp.ndarray   # [elem2D] Hunke P, 0 unless all 3 vertices iced
    tilt_u: jnp.ndarray         # [nod2D]  SSH-tilt velocity forcing (rhs_a), ÷area
    tilt_v: jnp.ndarray         # [nod2D]  (rhs_m)


def evp_setup(cfg: IceConfig, mesh: Mesh, a_ice, m_ice, m_snow, elevation) -> EVPStatic:
    """Per-step EVP constants that do not change across the 120 subcycles."""
    nod2D = mesh.nod2D
    area_n = mesh.area[:, 0]
    en = mesh.elem_nodes
    gs = mesh.gradient_sca
    mfac = mesh.metric_factor

    # per-node mass (fesom_ice_evp.c:273-289)
    mass_per_area = cfg.rhoice * m_ice + cfg.rhosno * m_snow
    safe_mass = jnp.where(mass_per_area > 1.0e-3, mass_per_area, 1.0)
    inv_areamass = jnp.where(mass_per_area > 1.0e-3, 1.0 / (area_n * safe_mass), 0.0)
    m = mass_per_area / jnp.where(a_ice >= 0.01, a_ice, 1.0)
    m = jnp.maximum(m, 9.0)                                    # limit small mass (:285)
    inv_mass = jnp.where(a_ice >= 0.01, 1.0 / m, 0.0)

    # per-element ice_strength + SSH-tilt rhs (:293-343)
    m_v = m_ice[en]                                            # [elem,3]
    a_v = a_ice[en]
    has_ice = ((m_v > 0.0).all(axis=1) & (a_v > 0.0).all(axis=1)
               & (mesh.ulevels <= 1))                         # all vertices iced, non-cavity
    msum = (m_v[:, 0] + m_v[:, 1] + m_v[:, 2]) / 3.0
    asum = (a_v[:, 0] + a_v[:, 1] + a_v[:, 2]) / 3.0
    strength = jnp.where(
        has_ice, 0.5 * cfg.pstar * msum * jnp.exp(-cfg.c_pressure * (1.0 - asum)), 0.0)

    aa = 9.81 * mesh.elem_area / 3.0
    e_v = elevation[en]
    edx = gs[:, 0] * e_v[:, 0] + gs[:, 1] * e_v[:, 1] + gs[:, 2] * e_v[:, 2]
    edy = gs[:, 3] * e_v[:, 0] + gs[:, 4] * e_v[:, 1] + gs[:, 5] * e_v[:, 2]
    cu = jnp.where(has_ice, -aa * edx, 0.0)[:, None] * jnp.ones((1, 3))   # [elem,3] (same/vertex)
    cv = jnp.where(has_ice, -aa * edy, 0.0)[:, None] * jnp.ones((1, 3))
    tilt_u = ops.scatter_add(cu, en, nod2D) / jnp.where(area_n > 0, area_n, 1.0)  # ÷area (:348)
    tilt_v = ops.scatter_add(cv, en, nod2D) / jnp.where(area_n > 0, area_n, 1.0)
    return EVPStatic(inv_areamass=inv_areamass, inv_mass=inv_mass, ice_strength=strength,
                     tilt_u=tilt_u, tilt_v=tilt_v)


# --------------------------------------------------------------------------
# Stress tensor (Hunke EVP) — fesom_ice_evp.c:68-132
# --------------------------------------------------------------------------
def stress_tensor(cfg: IceConfig, mesh: Mesh, u_ice, v_ice, s11, s12, s22, ice_strength):
    """Strain rates → Δ → ζ → the Hunke det1/det2 σ update. σ frozen where ice_strength≤0."""
    eps11, eps22, eps12 = strain_rates(mesh, u_ice, v_ice)   # shared block (JM.1)
    vale = cfg.vale
    radicand = ((eps11 * eps11 + eps22 * eps22) * (1.0 + vale)
                + 4.0 * vale * eps12 * eps12
                + 2.0 * eps11 * eps22 * (1.0 - vale))
    delta = jnp.maximum(_safe_sqrt(radicand), cfg.delta_min)   # = C max(sqrt, delta_min)
    zeta = ice_strength / delta * cfg.Tevp_inv
    r1 = zeta * (eps11 + eps22) - ice_strength * cfg.Tevp_inv
    r2 = zeta * (eps11 - eps22) * vale
    r3 = zeta * eps12 * vale
    dte = cfg.dte
    det1 = 1.0 / (1.0 + 0.5 * cfg.Tevp_inv * dte)
    si1 = det1 * (s11 + s22 + dte * r1)
    si2 = det1 * (s11 - s22 + dte * r2)
    s12n = det1 * (s12 + dte * r3)
    s11n = 0.5 * (si1 + si2)
    s22n = 0.5 * (si1 - si2)
    act = ice_strength > 0.0                                   # freeze σ on ice-free elements
    return (jnp.where(act, s11n, s11), jnp.where(act, s12n, s12), jnp.where(act, s22n, s22))


# --------------------------------------------------------------------------
# Stress → node rhs — fesom_ice_evp.c:143-193
# --------------------------------------------------------------------------
def stress2rhs(cfg: IceConfig, mesh: Mesh, s11, s12, s22, ice_strength,
               inv_areamass, tilt_u, tilt_v):
    """Element→node scatter of the σ divergence (shared raw scatter), then ``·inv_areamass + tilt``."""
    act = ice_strength > 0.0                                   # skip ice-free (C continue)
    u_rhs, v_rhs = stress_div_scatter(mesh, s11, s12, s22, act)
    use = inv_areamass > 0.0
    return (jnp.where(use, u_rhs * inv_areamass + tilt_u, 0.0),
            jnp.where(use, v_rhs * inv_areamass + tilt_v, 0.0))


# --------------------------------------------------------------------------
# Velocity update — fesom_ice_evp.c:388-437
# --------------------------------------------------------------------------
def velocity_update(cfg: IceConfig, mesh: Mesh, u_ice, v_ice, u_rhs, v_rhs,
                    u_w, v_w, stress_ax, stress_ay, inv_mass, a_ice, boundary_node):
    """The 2×2 implicit Coriolis + ocean-drag solve, the ``a<0.01`` gate, the coastal BC."""
    rdt = cfg.dte
    ax = jnp.cos(cfg.theta_io)
    ay = jnp.sin(cfg.theta_io)
    du = u_ice - u_w
    dv = v_ice - v_w
    umod = _safe_speed(du, dv)
    drag = cfg.cd_oce_ice * umod * DENSITY_0 * inv_mass
    rhsu = u_ice + rdt * (drag * (ax * u_w - ay * v_w) + inv_mass * stress_ax + u_rhs)
    rhsv = v_ice + rdt * (drag * (ax * v_w + ay * u_w) + inv_mass * stress_ay + v_rhs)
    r_a = 1.0 + ax * drag * rdt
    r_b = rdt * (mesh.coriolis_node + ay * drag)
    det = 1.0 / (r_a * r_a + r_b * r_b)
    un = det * (r_a * rhsu + r_b * rhsv)
    vn = det * (r_a * rhsv - r_b * rhsu)
    iced = a_ice >= 0.01
    un = jnp.where(iced, un, 0.0)
    vn = jnp.where(iced, vn, 0.0)
    un = jnp.where(boundary_node, 0.0, un)
    vn = jnp.where(boundary_node, 0.0, vn)
    return un, vn


# --------------------------------------------------------------------------
# Full EVP driver — the 120-subcycle scan
# --------------------------------------------------------------------------
def evp_dynamics(cfg: IceConfig, mesh: Mesh, *, a_ice, m_ice, m_snow,
                 u_ice, v_ice, sigma11, sigma12, sigma22,
                 srfoce_u, srfoce_v, elevation, stress_ax, stress_ay,
                 boundary_node=None, exch=None):
    """Run the EVP momentum solver (setup + 120 subcycles). Returns the updated
    ``(u_ice, v_ice, sigma11, sigma12, sigma22)``. ``stress_ax/ay`` = the atm-ice wind stress
    (``forcing.atm_ice_stress``); ``elevation`` = ``srfoce_ssh`` (``hbar``).

    **Sharding (Phase 8, S.7 part 3).** ``exch`` (``None`` ⇒ byte-identical) is the
    ``u_ice``/``v_ice`` node halo refresh wired INSIDE the 120-subcycle ``lax.scan`` (a
    collective in a ``scan`` under ``shard_map``, ``check_vma=False``): each subcycle's
    ``velocity_update`` is a per-node update of the element→node SCATTER ``u_rhs``/``v_rhs``
    (incomplete on the halo), and the NEXT subcycle's ``stress_tensor`` reads ``u_ice``/
    ``v_ice`` at the element's HALO vertices — so without the per-subcycle refresh the halo
    error propagates into owned nodes over 120 subcycles (``fesom_ice_evp.c:446``,
    ``SYNC_MAP`` M4.3b). ``boundary_node`` MUST be the GLOBAL coastal mask (partitioned in),
    NOT the local-mesh recompute: a partition-boundary edge has ``edge_tri[:,1]==-1`` locally
    and would mis-flag an interior node as coastal (the C uses ``partit->myList_edge2D``)."""
    if boundary_node is None:
        boundary_node = boundary_node_mask(mesh)
    st = evp_setup(cfg, mesh, a_ice, m_ice, m_snow, elevation)
    _exch = (lambda f, k: f) if exch is None else exch        # noqa: E731

    def body(carry, _):
        u_i, v_i, s11, s12, s22 = carry
        s11, s12, s22 = stress_tensor(cfg, mesh, u_i, v_i, s11, s12, s22, st.ice_strength)
        u_rhs, v_rhs = stress2rhs(cfg, mesh, s11, s12, s22, st.ice_strength,
                                  st.inv_areamass, st.tilt_u, st.tilt_v)
        u_i, v_i = velocity_update(cfg, mesh, u_i, v_i, u_rhs, v_rhs, srfoce_u, srfoce_v,
                                   stress_ax, stress_ay, st.inv_mass, a_ice, boundary_node)
        u_i = _exch(u_i, "nod")                              # refresh halo for next subcycle's
        v_i = _exch(v_i, "nod")                              # stress_tensor (reads at vertices)
        return (u_i, v_i, s11, s12, s22), None

    init = (u_ice, v_ice, sigma11, sigma12, sigma22)
    body = jax.checkpoint(body)                               # cap the 120-step backward memory
    (u_ice, v_ice, sigma11, sigma12, sigma22), _ = lax.scan(
        body, init, None, length=cfg.evp_rheol_steps)
    return u_ice, v_ice, sigma11, sigma12, sigma22
