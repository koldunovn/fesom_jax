"""Assembled sea-ice step (Phase 6, Task 6.6) — composes the five ice kernels in the C
runtime order and produces the surface BCs the ocean step consumes.

Mirrors the per-iteration flow of ``fesom_main.c`` (the ice block before the ocean step):

    bulk_compute (atm-ocean fluxes + Ch/Ce + stress_atmice)
      → fesom_ice_step:  ocean2ice → EVP → FCT → cut_off → thermo → oce_fluxes
      → oce_fluxes_mom   (ice-ocean stress blend, prognostic u_ice/a_ice)
      → cal_shortwave_rad (penetration; open-water-only gate on the prognostic a_ice)
      → bc_T = −dt·heat_flux/vcpw,  bc_S = dt·(virtual_salt + relax_salt)

Returns :class:`IceStepOut` — the surface fluxes (``stress_surf``/``bc_T``/``bc_S``/``sw_3d``,
+ diagnostics) AND the updated prognostic ice state (a/m/snow, u/v_ice, t_skin, σ) to carry.
Every kernel is individually dump-gated (Tasks 6.2–6.5); this is the wiring + the runtime
order, gated end-to-end by the config-C per-substep T/S dump (Task 6.6).

The two Phase-5 ``a_ice`` couplings (the shortwave gate + the momentum stress blend) now read
the **prognostic** post-thermo ``a_ice`` and the EVP ``u_ice`` (Phase 5 used the static mask +
``u_ice=0``). Differentiable end-to-end (the SST→flux / current→stress / runoff seams).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from . import forcing as _forcing
from . import ice_adv, ice_coupling, ice_evp, ice_thermo
from .config import VCPW
from .ice import IceConfig
from .mesh import Mesh
from .state import State


class IceStepOut(NamedTuple):
    # surface BCs the ocean step consumes
    stress_surf: jnp.ndarray        # (elem2D,2)
    bc_T: jnp.ndarray               # (nod2D,)
    bc_S: jnp.ndarray               # (nod2D,)
    sw_3d: jnp.ndarray              # (nod2D,nl)
    stress_node_surf: jnp.ndarray   # (nod2D,2) ice-blended NODE wind stress (KPP ustar)
    # diagnostics (for the dump gate)
    heat_flux: jnp.ndarray
    water_flux: jnp.ndarray
    virtual_salt: jnp.ndarray
    relax_salt: jnp.ndarray
    # updated prognostic ice state (to thread into State)
    a_ice: jnp.ndarray
    m_ice: jnp.ndarray
    m_snow: jnp.ndarray
    u_ice: jnp.ndarray
    v_ice: jnp.ndarray
    t_skin: jnp.ndarray
    sigma11: jnp.ndarray
    sigma12: jnp.ndarray
    sigma22: jnp.ndarray


def ice_surface_step(cfg: IceConfig, mesh: Mesh, state: State, sf, fs, *,
                     dt: float, boundary_node=None) -> IceStepOut:
    """One assembled ice step → the surface fluxes + the new ice state. ``sf`` is this step's
    :class:`core2_forcing.StepForcing` (atmosphere + month SSS/chl); ``fs`` the
    :class:`core2_forcing.ForcingStatic` (runoff/areas/open_water). ``boundary_node`` (the
    coastal mask) is precomputed once if given, else derived from ``mesh``."""
    # ⚠️ The ice timestep MUST track the ocean dt (C ``fesom_ice_setup``: ``ice_dt =
    # ice_ave_steps*dt``, fesom_ice.c:231). ``IceConfig.ice_dt`` is only a build-time default
    # (500 s) — if the run's ocean dt differs (e.g. the dt=1800 climate run) and ice_dt is left
    # at 500, EVERY ice rate (thermo growth/melt ×ice_dt, FCT transport ×ice_dt, EVP dte/Tevp_inv)
    # runs at the wrong timestep ⇒ the ice evolves 3.6× too slowly ⇒ a high-lat climate bias.
    # Deriving it from dt here makes the desync impossible regardless of the passed cfg.
    cfg = cfg._replace(ice_dt=cfg.ice_ave_steps * dt)
    open_water = fs.open_water
    geo_lat = jnp.asarray(mesh.geo_coord_nod2D)[:, 1]

    # --- bulk: atm-ocean fluxes + exchange coeffs + atm-ice wind stress ---
    u_w0 = state.uvnode[:, 0, 0]
    v_w0 = state.uvnode[:, 0, 1]
    bulk = _forcing.bulk_surface_fluxes(
        mesh, sf.u_air, sf.v_air, sf.shum, sf.shortwave, sf.longwave, sf.Tair,
        sf.prec_rain, sf.prec_snow, state.T[:, 0], u_w0, v_w0)
    stress_ax, stress_ay = _forcing.atm_ice_stress(
        sf.u_air, sf.v_air, state.u_ice, state.v_ice)         # prev-step u_ice (bulk runs first)

    # --- fesom_ice_step: ocean2ice → EVP → FCT → cut_off → thermo → oce_fluxes ---
    srf = ice_coupling.ocean2ice(state)
    u_ice, v_ice, s11, s12, s22 = ice_evp.evp_dynamics(
        cfg, mesh, a_ice=state.a_ice, m_ice=state.m_ice, m_snow=state.m_snow,
        u_ice=state.u_ice, v_ice=state.v_ice,
        sigma11=state.sigma11, sigma12=state.sigma12, sigma22=state.sigma22,
        srfoce_u=srf.u, srfoce_v=srf.v, elevation=srf.ssh,
        stress_ax=stress_ax, stress_ay=stress_ay, boundary_node=boundary_node)

    a_adv, m_adv, ms_adv = ice_adv.fct_solve(
        cfg, mesh, state.a_ice, state.m_ice, state.m_snow, u_ice, v_ice)
    a_co, m_co, ms_co = ice_thermo.cut_off(a_adv, m_adv, ms_adv)

    th = ice_thermo.thermodynamics(
        cfg, m_ice=m_co, m_snow=ms_co, a_ice=a_co, u_ice=u_ice, v_ice=v_ice,
        t_skin=state.t_skin, srfoce_temp=srf.temp, srfoce_salt=srf.salt,
        srfoce_u=srf.u, srfoce_v=srf.v,
        fsh=sf.shortwave, flo=sf.longwave, Tair=sf.Tair, qa=sf.shum,
        u_wind=sf.u_air, v_wind=sf.v_air, rain=sf.prec_rain, snow=sf.prec_snow,
        runo=fs.runoff_node, ch=bulk.ch, ce=bulk.ce, geo_lat=geo_lat, non_cavity=open_water)

    icef = ice_coupling.ice_oce_fluxes(
        srf.salt, th.flx_fw, th.flx_h, sf.Ssurf_month, fs.runoff_node,
        fs.areasvol_surf, fs.ocean_area, open_water)

    # --- oce_fluxes_mom: ice-ocean stress blend (prognostic a_ice/u_ice) ---
    stress_surf, stress_node_surf = ice_coupling.ice_oce_fluxes_mom(
        mesh, th.a_ice, u_ice, v_ice, srf.u, srf.v, bulk.stress_node_surf, open_water, cfg)

    # --- shortwave penetration (open-water-only gate on the prognostic a_ice) ---
    pene_open = open_water & (th.a_ice <= 0.0)
    heat_flux, sw_3d = _forcing.cal_shortwave_rad(
        mesh, icef.heat_flux, sf.shortwave, sf.chl, pene_open)

    # --- surface BCs (linfs) ---
    bc_T = -dt * heat_flux / VCPW
    bc_S = dt * (icef.virtual_salt + icef.relax_salt)

    return IceStepOut(
        stress_surf=stress_surf, bc_T=bc_T, bc_S=bc_S, sw_3d=sw_3d,
        stress_node_surf=stress_node_surf,
        heat_flux=heat_flux, water_flux=icef.water_flux,
        virtual_salt=icef.virtual_salt, relax_salt=icef.relax_salt,
        a_ice=th.a_ice, m_ice=th.m_ice, m_snow=th.m_snow,
        u_ice=u_ice, v_ice=v_ice, t_skin=th.t_skin,
        sigma11=s11, sigma12=s12, sigma22=s22)
