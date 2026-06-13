"""Sea-ice mEVP dynamics (Phase 9c) — an AD-safe port of ``fesom_ice_maevp.c``.

mEVP (``whichEVP=1``, Bouillon et al. 2013) replaces the standard-EVP elastic subcycling
(:mod:`fesom_jax.ice_evp`) by a **pseudo-time fixed-point iteration for the backward-Euler VP
problem over the FULL ice step** ``rdt = ice_dt``, stabilized by two relaxation constants
``alpha_evp`` (stress) / ``beta_evp`` (momentum). 120 fixed iterations, no CFL/adaptivity
(that is aEVP), no ``theta_io`` rotation, no ``Tevp_inv``/``dte``/``zeta``.

This module mirrors the C monolith ``fesom_ice_maevp.c`` (363 LOC) for 1:1 traceability but
**imports** the genuinely shared pieces from :mod:`fesom_jax.ice_evp` (strain rates, the raw
σ-divergence scatter, the safe helpers, the boundary mask). The setup / relaxation / solve are
rewritten against the C (the formulas genuinely differ — see the 14-trap checklist in the
plan). ``whichEVP=0`` keeps the EVP path; this kernel is reached only at ``whichEVP=1``.

⚠️ Decisions (plan ``20260611-fesom-jax-mevp.md``): scan carry = ``(u_aux, v_aux, σ11, σ12,
σ22)`` with the **frozen entry ``(u_ice, v_ice)`` closed over as the backward-Euler rhs anchor**
(``rhsu = u_ice + … + β·u_aux``) — UNLIKE std-EVP, whose rhs bases on the current iterate.
"""

from __future__ import annotations

# JM.2 lands the kernel; this stub is reached only via the whichEVP=1 dispatch in ice_step.


def mevp_dynamics(cfg, mesh, *, a_ice, m_ice, m_snow, u_ice, v_ice,
                  sigma11, sigma12, sigma22, srfoce_u, srfoce_v, elevation,
                  stress_ax, stress_ay, boundary_node=None, exch=None):
    """Run the mEVP momentum solver (setup + 120 pseudo-time iterations). Same signature /
    outputs as :func:`fesom_jax.ice_evp.evp_dynamics`. **STUB** — the kernel lands in JM.2."""
    raise NotImplementedError("mEVP (whichEVP=1) kernel lands in JM.2 of the Phase-9c plan")
