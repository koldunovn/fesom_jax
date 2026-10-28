"""REcoM's per-step forcing (parent plan Task 10): the solar incidence angle, the surface gas exchange
of ``REcoM_Forcing``, the atmospheric deposition and the alkalinity restoring of ``bio_fluxes``.

Ports of ``recom_extra.F90`` (``Cobeta``, ``calculate_solar_declination``, ``solar_incidence_cosine``),
``recom_forcing.F90:146-301`` (the surface CO2 and O2 fluxes, computed from the tracers BEFORE the
sources and sinks of the step) and ``recom_main.F90`` (``bio_fluxes`` :16-86, the per-node shortwave,
wind, sea-level pressure and dust, :326-475). Element-wise over nodes, f64, jit/grad safe.

Cavities (``nzmin > 1``, Fortran's zero fluxes and NaN fills) are out of scope; :func:`surface_fluxes`
takes the surface layer.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from . import carbonate as cb
from .params import PA2ATM, SECONDS_PER_DAY, TINY, RecomParams
from .tracers import ialk, idic, idin, ioxy, isi

PI_RECOM = 3.14159265358979      # REcoM_declarations pi (recom_modules.F90:25), the truncated literal
RAD = math.pi / 180.0            # FESOM2 rad = pi/180 (oce_modules.F90:34-35)
N_WATER = 1.33                   # recom_extra.F90 solar_incidence_cosine


def solar_declination(daynew, ndpyr):
    """Paltridge & Platt (1976), ``calculate_solar_declination`` (``recom_extra.F90``):
    ``yearfrac = mod(daynew, ndpyr) / ndpyr`` (so the last day of the year is day 0)."""
    d = jnp.asarray(daynew, jnp.float64)
    n = jnp.asarray(ndpyr, jnp.float64)
    yearfrac = jnp.fmod(d, n) / n
    yday = 2.0 * PI_RECOM * yearfrac
    return (0.006918 - 0.399912 * jnp.cos(yday) + 0.070257 * jnp.sin(yday)
            - 0.006758 * jnp.cos(2.0 * yday) + 0.000907 * jnp.sin(2.0 * yday)
            - 0.002697 * jnp.cos(3.0 * yday) + 0.001480 * jnp.sin(3.0 * yday))


def cobeta(daynew, ndpyr, lat_rad):
    """``cosAI``: the cosine of the noon solar incidence angle after refraction into water
    (``Cobeta`` + ``solar_incidence_cosine``, ``recom_extra.F90``), per node."""
    decl = solar_declination(daynew, ndpyr)
    cos_noon = jnp.sin(lat_rad) * jnp.sin(decl) + jnp.cos(lat_rad) * jnp.cos(decl)
    return jnp.sqrt(1.0 - (1.0 - cos_noon ** 2) / (N_WATER * N_WATER))


def bio_fluxes(alk_top, alk_surf, area_top, ocean_area, surf_relax_alk,
               reduce_sum: Callable = jnp.sum):
    """Alkalinity restoring with the global mean removed, ``bio_fluxes`` (``recom_main.F90:63-84``):
    ``relax_alk = surf_relax_Alk (Alk_surf - Alk(1)) - net / ocean_area``,
    ``net = sum(relax_alk * areasvol(1, :))`` over the owned nodes [mmol m-2 s-1].

    ``alk_top`` the surface Alk of the step's INPUT state (before the sources and sinks); ``area_top`` =
    ``areasvol(ulevels, :)`` of the owned nodes (zero on halo copies, so a sharded caller passes the owned
    mask folded in); ``reduce_sum`` sums over the local nodes and, sharded, over the devices (e.g.
    ``lambda x: jax.lax.psum(jnp.sum(x), axis)``). Fortran sums sequentially; ``jnp.sum`` reassociates
    (round-off in ``net``)."""
    relax = surf_relax_alk * (alk_surf - alk_top)
    net = reduce_sum(relax * area_top)
    return relax - net / ocean_area


class SurfaceFluxes(NamedTuple):
    """Stage 1's surface quantities per node (Fortran's module globals of the same name)."""
    GloCO2flux_seaicemask: jax.Array   # [mmol m-2 s-1] air to sea, DIC bc_surface
    GloO2flux_seaicemask: jax.Array    # [mmol m-2 s-1], O2 bc_surface
    AtmFeInput: jax.Array              # [umol m-2 s-1 as REcoM's GloFeDust units], Fe bc_surface
    AtmNInput: jax.Array               # [mmol m-2 s-1], DIN bc_surface
    relax_alk: jax.Array               # [mmol m-2 s-1], Alk bc_surface
    GloCO2flux: jax.Array              # [mmol m-2 d-1] (output)
    GloO2flux: jax.Array               # [mmol m-2 d-1] (output)
    GloPCO2surf: jax.Array             # [uatm]
    GlodPCO2surf: jax.Array            # [uatm] ocean - atmosphere
    GloHplus: jax.Array                # the surface pH (total scale; restart HPlus, output only)
    PistonVelocity: jax.Array          # kw660 [m s-1]
    alphaCO2: jax.Array                # K0 [(mol/kg)/atm]
    x_co2atm: jax.Array                # [ppm]
    GlodecayBenthos: jax.Array         # (nod2D, 4) [mmol m-2 s-1], the benthic return flux (dtr_bf)


def surface_gas_exchange(top, temp_top, sali_top, u_wind, v_wind, a_ice, press_air, co2_atm, lat_deg,
                         *, n_iter: int = cb.PH_MAXITER):
    """CO2 and O2 air-sea fluxes of ``REcoM_Forcing`` (``recom_forcing.F90:154-293``) from the surface
    layer ``top`` (nod2D, ntr) of the tracers before the sources and sinks. Returns a dict of the flux
    outputs (Fortran names)."""
    dic = jnp.maximum(TINY * 1e-3, top[:, idic] * 1e-3)
    alk = jnp.maximum(TINY * 1e-3, top[:, ialk] * 1e-3)
    si = jnp.maximum(TINY * 1e-3, top[:, isi] * 1e-3)
    phos = jnp.maximum(TINY * 1e-3, top[:, idin] * 1e-3) / 16.0
    o2 = jnp.maximum(TINY * 1e-3, top[:, ioxy] * 1e-3)
    t = jnp.maximum(2.0, temp_top)
    t = jnp.minimum(40.0, t)
    s = jnp.maximum(21.0, sali_top)                      # no upper clamp at the surface (:174-177)
    patm = press_air / PA2ATM
    uloc = jnp.sqrt(u_wind ** 2 + v_wind ** 2)           # recom_main.F90:365
    kw660 = cb.pistonvel(uloc, a_ice)                    # :195
    f = cb.flxco2(t, s, alk, dic, si, phos, kw660, co2_atm, patm, lat_deg, n_iter=n_iter)
    o2ex = cb.o2flux(t, s, kw660, patm, o2)              # ppo = Loc_slp / Pa2atm (:274-278)
    return dict(
        GloCO2flux=f.co2flux * 1.e3 * SECONDS_PER_DAY,
        GloCO2flux_seaicemask=f.co2flux * 1.e3,
        GloO2flux=o2ex * 1.e3 * SECONDS_PER_DAY,
        GloO2flux_seaicemask=o2ex * 1.e3,
        GloPCO2surf=f.carb.pco2, GlodPCO2surf=f.dpco2, GloHplus=f.carb.ph,
        PistonVelocity=kw660, alphaCO2=f.k0)


def deposition(fe_dust, n_dust, a_ice, p: RecomParams):
    """``(AtmFeInput, AtmNInput)`` = ``GloFeDust (1 - a_ice) dust_sol``, ``GloNDust (1 - a_ice)``
    (``recom_main.F90:466-473``)."""
    return fe_dust * (1.0 - a_ice) * p.dust_sol, n_dust * (1.0 - a_ice)


def surface_shortwave(shortwave, a_ice, p: RecomParams):
    """``SW = parFrac shortwave (1 - a_ice)`` (``recom_main.F90:408-411``), the PAR at the surface."""
    sw = p.parFrac * shortwave
    return sw * (1.0 - a_ice)
