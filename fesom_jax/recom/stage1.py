"""Stage 1 of a REcoM step: everything FESOM2's ``call recom(...)`` does (``fesom_module.F90:1203``,
``recom_main.F90:recom``) as ONE pure function, :func:`recom_sources_sinks` (parent plan Task 10).

Order, as Fortran's ``recom()``:

1. ``bio_fluxes``: ``relax_alk`` from the INPUT surface alkalinity, global mean removed
   (``recom_main.F90:279-283``; the only global reduction, through ``reduce_sum``);
2. the DICremin surface reset: tracer 1037 = 0 in the top layer (``recom_main.F90:300-310``);
3. per column (``REcoM_Forcing``): the surface CO2 and O2 fluxes from the surface tracers after the reset
   and BEFORE the sources and sinks (``recom_forcing.F90:154-293``), then ``REcoM_sms`` with the
   per-layer carbonate refresh and the clip ``max(tiny, state + sms)`` + floors (:mod:`fesom_jax.recom.sms`);
4. the 2-D outputs: dust deposition, ``GloHplus`` = surface pH, ``GlodecayBenthos = decayBenthos / 86400``.

``dt_bgc`` is an argument [s]: REcoM's ``dt_b = dt_bgc / 86400 / biostep`` (``recom_sms.F90:352-355``).
With the BGC stepped every k-th physics step the caller passes ``dt_bgc = k dt``; the carbonate refresh
periods are then ``int(86400 / dt_bgc * 7)`` and ``... * 30`` BGC steps and ``inputs.mstep`` must count
BGC steps (Fortran has no such mode; with k = 1 it is FESOM's ``mstep``).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from . import carbonate as cb
from .forcing import (SurfaceFluxes, bio_fluxes, cobeta, deposition, surface_gas_exchange,
                      surface_shortwave)
from .params import SECONDS_PER_DAY, RecomParams
from .sms import CarbCache, SmsResult, recom_sms
from .tracers import ialk, idicremin


class RecomGeometry(NamedTuple):
    """The per-step geometry stage 1 reads (``hnode``, ``zbar_3d_n`` change with zstar every step)."""
    hnode: jax.Array          # (nod2D, nl-1) layer thickness [m]
    zbar: jax.Array           # (nod2D, nl) interface depths zbar_3d_n [m, negative down]
    nlev: jax.Array           # (nod2D,) int: nlevels_nod2D - 1, the wet layers are k < nlev
    lat_rad: jax.Array        # (nod2D,) geo latitude [rad]
    area_top: jax.Array       # (nod2D,) areasvol(1, :) [m2] of the OWNED nodes, 0 on halo copies
    ocean_area: jax.Array     # scalar: FESOM's ocean_area [m2]


class BgcStepInputs(NamedTuple):
    """The per-step input record (calendar and atmosphere at this step), all per node unless stated."""
    daynew: jax.Array         # scalar int: day of year (FESOM daynew)
    ndpyr: jax.Array          # scalar int: days in the year
    mstep: jax.Array          # scalar int: step within the run segment, 1-based (FESOM mstep)
    fe_dust: jax.Array        # GloFeDust: the monthly Albani dust [as the forcing reader holds it]
    n_dust: jax.Array         # GloNDust (zero: useAeolianN off)
    co2_atm: jax.Array        # scalar: atmospheric xCO2 [ppm] (constant_CO2: CO2_for_spinup)
    press_air: jax.Array      # sea-level pressure [Pa] (l_mslp)
    u_wind: jax.Array         # 10-m wind [m s-1]
    v_wind: jax.Array
    shortwave: jax.Array      # [W m-2]
    a_ice: jax.Array          # sea-ice concentration


class Stage1Result(NamedTuple):
    bgc: jax.Array            # (nod2D, nl-1, ntr) tracers after stage 1
    carb: CarbCache           # the carbonate cache
    sfc: SurfaceFluxes        # surface fluxes and 2-D outputs for stage 3 / output
    benthos: jax.Array        # (nod2D, 4) benthic pools after the decay [mmol m-2]
    sms: SmsResult            # the full sources-and-sinks result (sms, clip, PAR, diagnostics)


def recom_sources_sinks(bgc, carb: CarbCache, benthos, alk_surf, temp, salt, geom: RecomGeometry,
                        inputs: BgcStepInputs, dt_bgc, params: RecomParams, *,
                        reduce_sum: Callable = jnp.sum, n_iter: int = cb.PH_MAXITER) -> Stage1Result:
    """FESOM2's ``recom()`` for every column: see the module docstring for the order.

    ``bgc`` (nod2D, nl-1, ntr) the BGC tracers at the start of the step; ``carb`` the carbonate cache;
    ``benthos`` (nod2D, 4); ``alk_surf`` (nod2D,) the alkalinity restoring target; ``temp``, ``salt``
    (nod2D, nl-1) the ocean T and S at this point of the step; ``dt_bgc`` the BGC step [s]; ``params`` a
    :class:`~fesom_jax.recom.params.RecomParams` (its leaves may be traced: differentiable).
    """
    p = params
    f64 = jnp.float64
    bgc, benthos, alk_surf, temp, salt = (jnp.asarray(a, f64) for a in (bgc, benthos, alk_surf, temp, salt))
    lat_deg = geom.lat_rad / (jnp.pi / 180.0)
    relax_alk = bio_fluxes(bgc[:, 0, ialk], alk_surf, geom.area_top, geom.ocean_area, p.surf_relax_Alk,
                           reduce_sum=reduce_sum)
    bgc = bgc.at[:, 0, idicremin].set(0.0)
    gas = surface_gas_exchange(bgc[:, 0, :], temp[:, 0], salt[:, 0], inputs.u_wind, inputs.v_wind,
                               inputs.a_ice, inputs.press_air, inputs.co2_atm, lat_deg, n_iter=n_iter)
    surf_sr = surface_shortwave(inputs.shortwave, inputs.a_ice, p)
    cos_ai = cobeta(inputs.daynew, inputs.ndpyr, geom.lat_rad)
    dt_b = dt_bgc / SECONDS_PER_DAY / 1.0                     # dt_d / real(biostep), biostep = 1
    r = recom_sms(bgc, temp, salt, carb, geom.hnode, geom.zbar, geom.nlev, surf_sr, cos_ai, lat_deg,
                  geom.lat_rad, inputs.press_air, benthos, inputs.mstep, inputs.daynew, dt_b, p, n_iter=n_iter)
    fe_in, n_in = deposition(inputs.fe_dust, inputs.n_dust, inputs.a_ice, p)
    sfc = SurfaceFluxes(
        GloCO2flux_seaicemask=gas["GloCO2flux_seaicemask"], GloO2flux_seaicemask=gas["GloO2flux_seaicemask"],
        AtmFeInput=fe_in, AtmNInput=n_in, relax_alk=relax_alk,
        GloCO2flux=gas["GloCO2flux"], GloO2flux=gas["GloO2flux"], GloPCO2surf=gas["GloPCO2surf"],
        GlodPCO2surf=gas["GlodPCO2surf"], GloHplus=gas["GloHplus"], PistonVelocity=gas["PistonVelocity"],
        alphaCO2=gas["alphaCO2"], x_co2atm=jnp.broadcast_to(jnp.asarray(inputs.co2_atm, jnp.float64),
                                                            relax_alk.shape),
        GlodecayBenthos=r.decay_benthos / SECONDS_PER_DAY)
    return Stage1Result(bgc=r.state, carb=r.carb, sfc=sfc, benthos=r.benthos, sms=r)
