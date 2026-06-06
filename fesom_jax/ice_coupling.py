"""Ice ↔ ocean coupling (Phase 6, Task 6.3) — ports of ``fesom_ice_coupling.c``.

Three faithful ports of the ocean↔ice exchange:

* :func:`ocean2ice` (``fesom_ice_coupling.c:47-115``) — copy the ocean surface state into
  the ice's ``srfoce_*``. Nearly free in JAX: ``srfoce_temp/salt = T/S[:,0]``,
  ``srfoce_ssh = hbar``, ``srfoce_u/v = uvnode[:,0]`` (the C uses the same area-weighted
  surface-velocity recipe as ``uvnode``, per the ``:44-45`` comment).
* :func:`ice_oce_fluxes` (``fesom_ice_coupling.c:125-179``) — **the runoff handoff.**
  ``water_flux = -flx_fw`` (the thermo freshwater flux, which already includes runoff via
  ``prec``), ``heat_flux = -flx_h``; then ``virtual_salt = rsss·water_flux`` and ``relax_salt
  = surf_relax_S·(Ssurf - S_top)``, each balanced to net zero. Reuses the Phase-5
  :func:`sss_runoff.sss_runoff_fluxes` math with ``balance_water_flux=False`` (the ice-on path
  drops the standalone ``water_flux += ⟨water_flux+runoff⟩`` term — runoff is already in
  ``flx_fw``). This is what makes river freshwater reach the salinity BC in Phase 6.
* :func:`ice_oce_fluxes_mom` (``fesom_ice_coupling.c:213-264``) — the surface momentum-stress
  blend with the **prognostic** ``u_ice`` (Phase 5 used a static ``u_ice=0``): per node
  ``stress_iceoce = ρ·cd_oce_ice·|u_ice-u_w|·(u_ice-u_w)``; ``stress_node = stress_iceoce·a +
  atm·(1-a)``; then ``stress_surf`` = mean-of-3 vertices.

All differentiable; the ``water_flux``/``virtual_salt`` carry the SST→flux gradient (via the
thermo ``flx_fw``) and the surface-current→stress gradient (via the drag ``|u_ice-u_w|``,
safe-sqrt).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .config import DENSITY_0
from .forcing import _safe_speed
from .ice import IceConfig
from .mesh import Mesh
from .state import State
from .sss_runoff import sss_runoff_fluxes


# --------------------------------------------------------------------------
# ocean2ice — fesom_ice_coupling.c:47-115
# --------------------------------------------------------------------------
class SrfOce(NamedTuple):
    """Ocean surface state seen by the ice (all ``[nod2D]``)."""
    temp: jnp.ndarray       # srfoce_temp = T[:,0]
    salt: jnp.ndarray       # srfoce_salt = S[:,0]
    ssh: jnp.ndarray        # srfoce_ssh  = hbar
    u: jnp.ndarray          # srfoce_u    = uvnode[:,0,0]
    v: jnp.ndarray          # srfoce_v    = uvnode[:,0,1]


def ocean2ice(state: State) -> SrfOce:
    """Copy the ocean surface state into the ice (``fesom_ocean2ice``). ``u_w/v_w`` use the
    surface node velocity ``uvnode[:,0]`` (== the C area-weighted recipe; ``:44-45``)."""
    return SrfOce(temp=state.T[:, 0], salt=state.S[:, 0], ssh=state.hbar,
                  u=state.uvnode[:, 0, 0], v=state.uvnode[:, 0, 1])


# --------------------------------------------------------------------------
# ice→ocean heat/freshwater fluxes (the runoff handoff) — :125-179
# --------------------------------------------------------------------------
class IceOceFluxes(NamedTuple):
    heat_flux: jnp.ndarray      # = -flx_h        [W/m²]
    water_flux: jnp.ndarray     # = -flx_fw       [m/s] (incl. runoff; +ocean loses FW)
    virtual_salt: jnp.ndarray   # rsss·water_flux − ⟨·⟩   [PSU·m/s]
    relax_salt: jnp.ndarray     # surf_relax_S·(Ssurf − S_top) − ⟨·⟩   [PSU·m/s]


def ice_oce_fluxes(S_top, flx_fw, flx_h, Ssurf_month, runoff_node,
                   areasvol_surf, ocean_area, open_water=None) -> IceOceFluxes:
    """Ice-mediated surface heat/freshwater fluxes (``fesom_ice_oce_fluxes``).

    ``water_flux = -flx_fw`` (the thermo flux, runoff already folded in via ``prec``);
    ``heat_flux = -flx_h``; ``virtual_salt``/``relax_salt`` via the Phase-5 balance with
    ``balance_water_flux=False`` (the ice-on path drops the ``⟨water_flux+runoff⟩`` term —
    ``runoff_node`` is passed only because the shared signature takes it, and is unused here)."""
    water_flux = -flx_fw
    heat_flux = -flx_h
    sss = sss_runoff_fluxes(S_top, water_flux, Ssurf_month, runoff_node,
                            areasvol_surf, ocean_area, open_water,
                            balance_water_flux=False)
    return IceOceFluxes(heat_flux=heat_flux, water_flux=sss.water_flux,
                        virtual_salt=sss.virtual_salt, relax_salt=sss.relax_salt)


# --------------------------------------------------------------------------
# ice-mediated surface momentum stress blend — :213-264
# --------------------------------------------------------------------------
def ice_oce_fluxes_mom(mesh: Mesh, a_ice, u_ice, v_ice, u_w, v_w,
                       stress_node_atm, open_water, cfg: IceConfig = IceConfig()):
    """Blend the atm-ocean and ice-ocean surface stress and average to elements
    (``fesom_ice_oce_fluxes_mom``). ``stress_node_atm`` is the bulk ``stress_node_surf``
    ``[nod2D,2]``; returns ``stress_surf`` ``[elem2D,2]``. Prognostic ``u_ice``."""
    rho_cd = DENSITY_0 * cfg.cd_oce_ice
    du = u_ice - u_w
    dv = v_ice - v_w
    aux = _safe_speed(du, dv) * rho_cd                     # ρ·cd·|u_ice-u_w| (safe-sqrt at 0)
    ice_on = a_ice > 0.001                                 # C threshold
    sic_x = jnp.where(ice_on, aux * du, 0.0)
    sic_y = jnp.where(ice_on, aux * dv, 0.0)
    atm_x, atm_y = stress_node_atm[:, 0], stress_node_atm[:, 1]
    blend_x = jnp.where(open_water, sic_x * a_ice + atm_x * (1.0 - a_ice), atm_x)
    blend_y = jnp.where(open_water, sic_y * a_ice + atm_y * (1.0 - a_ice), atm_y)
    sns = jnp.stack([blend_x, blend_y], axis=-1)
    ev = mesh.elem_nodes
    return (sns[ev[:, 0]] + sns[ev[:, 1]] + sns[ev[:, 2]]) / 3.0
