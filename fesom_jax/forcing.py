"""Surface forcing for the FESOM2 → JAX port.

Two paths live here:

* **Phase-2 analytical wind** (:func:`surface_stress`) — the pi reference forcing.
* **Phase-5 L&Y09 bulk formulae** (:func:`bulk_surface_fluxes` + helpers) — the
  CORE2 forcing, an **AD-safe** JAX port of ``fesom_bulk.c``
  (``ncar_ocean_fluxes_mode`` + ``obudget`` + the wind-stress / node→elem assembly).
  This is the **differentiable** SST→flux / surface-current→stress seam: it consumes
  the JRA55 atmosphere (host-numpy reader :mod:`fesom_jax.jra55`, a per-step device
  constant) plus the model's surface T and surface current, and produces
  ``heat_flux``/``water_flux``/``stress_node_surf``/``stress_surf``. The whole bulk
  is traced, so ``d(heat_flux)/d(SST)`` and ``d(stress)/d(current)`` flow for the
  hybrid-ML training objective.

Analytical path. Port of ``fesom_forcing_set_analytical``
(``fesom_forcing_analytical.c``) **plus** the per-step element re-averaging in
``fesom_ice_oce_fluxes_mom`` (``fesom_ice_coupling.c:256-264``, no-ice blend is a
no-op but the re-average runs every step before the ocean step —
``fesom_main.c:983``). Phase-2 pi config: ``tau0=0.05`` N/m², ``Ly_factor=2.0``.
The raw element stress is a steady zonal cosine pattern
``raw[e] = (−tau0·cos((2/Ly_factor)·lat_e), 0)`` (``lat_e`` = mean **geographic**
latitude of the element's 3 vertices); the stress ``impl_vert_visc`` reads is
**double-averaged** (raw element → area-weighted node → simple mean of 3 vertices).
Use :func:`surface_stress`.

Bulk path — fidelity notes (``fesom_bulk.c``).

* **Fixed 5-iteration Monin-Obukhov loop, unrolled, NO early break.** The C breaks
  on ``|Δcd|/(cd+1e-8) < 1e-4`` (data-dependent ⇒ not AD-safe). Running all 5 iters
  is a physical no-op (post-convergence iters re-evaluate the fixed point) but
  differs from the early-break value by the residual drift; we verify against a
  **fixed-5 C dump** (``FESOM_BULK_FIXED_ITERS`` path) so the comparison is pure FP
  reassociation, and separately confirm fixed-5 vs early-break is physically tiny.
* **The deliberate Fortran mismatch is preserved:** the exchange coefficients and
  the wind stress use the **relative** wind ``|u_atm − u_ocn|`` (floored at 0.3),
  but ``obudget``'s ``ug`` uses the **absolute** wind ``|u_atm|`` (``fesom_bulk.c``
  L283, ``ice_thermo_oce.F90``). Do not "fix" it.
* **AD guards:** ``u`` and ``mag`` use a double-``where`` safe-sqrt (the
  ``current→stress`` gradient at ``Δu=0`` would otherwise be ``0·inf`` NaN);
  ``x2 = sqrt(|1−16ζ|)`` (singular at ζ=1/16) is computed as
  ``sqrt(max(|1−16ζ|, 1))`` — bit-identical to the C ``x2=sqrt(..); if(x2<1)x2=1``
  yet smooth (the floored region maps to the constant 1, gradient 0); the
  ``copysign`` step-switches are ported literally via :func:`jnp.copysign`
  (gradient 0, exact at ±0). ``albw=0.1`` (CORE2 ``namelist.ice``, not LY2004 0.066).
  ``heat_flux = qns − qsr`` is BEFORE shortwave penetration (a later step).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from . import ops
from .mesh import Mesh


def raw_element_stress(mesh: Mesh, tau0: float = 0.05, Ly_factor: float = 2.0):
    """Raw analytical element wind stress ``[elem2D, 2]`` (the ``set_analytical``
    element write, before re-averaging). v-component is 0."""
    inv_period = 2.0 / Ly_factor
    lat = mesh.geo_coord_nod2D[:, 1]                       # (nod2D,) geographic radians
    lat_e = lat[mesh.elem_nodes].mean(axis=1)             # (elem2D,)
    sx = -tau0 * jnp.cos(inv_period * lat_e)
    return jnp.stack([sx, jnp.zeros_like(sx)], axis=-1)


def node_stress(mesh: Mesh, raw):
    """Area-weighted element→node stress average ``[nod2D, 2]`` (``set_analytical``
    node interpolation): ``sns[n] = Σ_{el∋n} area_el·raw[el] / Σ area_el``."""
    e = mesh.elem2D
    area = mesh.elem_area[:, None]                         # (elem2D,1)
    num = ops.scatter_add(jnp.broadcast_to((area * raw)[:, None], (e, 3, 2)),
                          mesh.elem_nodes, mesh.nod2D)      # (nod2D,2)
    den = ops.scatter_add(jnp.broadcast_to(mesh.elem_area[:, None], (e, 3)),
                          mesh.elem_nodes, mesh.nod2D)      # (nod2D,)
    safe = jnp.where(den > 0.0, den, 1.0)[:, None]
    return num / safe


def surface_stress(mesh: Mesh, tau0: float = 0.05, Ly_factor: float = 2.0):
    """Element surface stress ``[elem2D, 2]`` as ``impl_vert_visc`` reads it:
    raw → area-weighted node average → simple mean of the element's 3 vertices."""
    raw = raw_element_stress(mesh, tau0, Ly_factor)
    sns = node_stress(mesh, raw)
    return sns[mesh.elem_nodes].mean(axis=1)               # (elem2D,2)


# ==========================================================================
# Phase-5 L&Y09 open-water bulk formulae (AD-safe port of fesom_bulk.c).
# ==========================================================================
# Constants — fesom_bulk.c:21-39 (MOD_ICE.F90 ice%thermo defaults). Note the
# bulk's own gravity 9.80 (NOT config.G=9.81) and albw=0.1 (CORE2 namelist.ice).
BULK_RHOAIR = 1.3
BULK_INV_RHOAIR = 1.0 / 1.3
BULK_CPAIR = 1005.0
BULK_CLHW = 2.501e6                 # J/kg, water → vapor
BULK_TMELT = 273.15
BULK_BOLTZMANN = 5.67e-8
BULK_EMISS_WAT = 0.97
BULK_ALBW = 0.1                     # open-water albedo (CORE2, not LY2004 0.066)
BULK_INV_RHOWAT = 1.0 / 1000.0
BULK_GRAV = 9.80
BULK_VONKARM = 0.40
BULK_Q1 = 640380.0
BULK_Q2 = -5107.4
BULK_U10MIN = 0.3                   # m/s relative-wind floor
BULK_N_ITTS = 5                     # fixed iteration count (no early break)

Z_MEAS = 10.0                       # measurement height for wind/tair/shum (jra->z_*)


def _safe_speed(sx, sy):
    """``sqrt(sx²+sy²)`` with a double-``where`` so the ``arg=0`` lane is finite in
    the backward pass (returns 0 there, matching the C ``sqrt(0)``). A bare
    ``sqrt(0)`` has an ``inf`` derivative ⇒ ``0·inf`` NaN even behind a forward
    ``where`` — see the AD masked-NaN rule."""
    s = sx * sx + sy * sy
    spd = jnp.sqrt(jnp.where(s > 0.0, s, 1.0))
    return jnp.where(s > 0.0, spd, 0.0)


def _cd_n10(u10):
    """Neutral 10 m drag coefficient, LY2009 eqn 11a/b (``fesom_bulk.c:78-80``).
    The ``copysign`` selector (``hl1`` for u10<33, 2.34e-3 for u10≥33) is ported
    literally; ``u10`` is floored at 0.3 upstream so ``2.7/u10`` and ``u10**6`` are
    finite (hence ``hl1`` is finite even on the non-selected branch)."""
    hl1 = (2.7 / u10 + 0.142 + 0.0764 * u10 - 3.14807e-10 * u10**6) / 1.0e3
    sw = jnp.copysign(0.5, u10 - 33.0)
    return (0.5 - sw) * hl1 + (0.5 + sw) * 2.34e-3


def ncar_ocean_fluxes_mode(tair_C, shum, u_wind, v_wind, T_oc_C, u_w, v_w,
                           z_wind=Z_MEAS, z_tair=Z_MEAS, z_shum=Z_MEAS):
    """NCAR L&Y04/09 exchange coefficients ``(cd, ce, ch)`` — AD-safe port of
    ``ncar_ocean_fluxes_mode`` (``fesom_bulk.c:49-177``). Elementwise over any
    broadcast shape (the node axis). All inputs in the C's units (°C, kg/kg, m/s).

    Differentiable w.r.t. ``T_oc_C`` (via ``ts→qs→qstar→bstar→ζ→ψ``) and the surface
    current ``u_w,v_w`` (via the relative wind ``u``). Runs a **fixed** 5-iteration
    Monin-Obukhov loop (no data-dependent break)."""
    t = tair_C + BULK_TMELT
    ts = T_oc_C + BULK_TMELT
    q = shum
    qs = 0.98 * BULK_Q1 * BULK_INV_RHOAIR * jnp.exp(BULK_Q2 / ts)   # L-Y eqn 5
    tv = t * (1.0 + 0.608 * q)

    # relative wind, floored at 0.3 (safe-sqrt: current→u gradient finite at Δu=0)
    u = jnp.maximum(_safe_speed(u_wind - u_w, v_wind - v_w), BULK_U10MIN)

    u10, t10, q10 = u, t, q

    cd_n10 = _cd_n10(u10)
    cd_n10_rt = jnp.sqrt(cd_n10)
    ce_n10 = 34.6 * cd_n10_rt * 1.0e-3
    stab = 0.5 + jnp.copysign(0.5, t - ts)                 # init: copysign(t-ts)
    ch_n10 = (18.0 * stab + 32.7 * (1.0 - stab)) * cd_n10_rt * 1.0e-3

    cd, ch, ce = cd_n10, ch_n10, ce_n10

    def _psi(zeta, z):
        # ζ for height z, clamped to ±10 (== C copysign(10,ζ) for |ζ|>10).
        zeta = jnp.clip(zeta, -10.0, 10.0)
        # x2 = sqrt(|1-16ζ|), floored to 1 — written as sqrt(max(|1-16ζ|,1)):
        # bit-identical to the C floor yet smooth through the ζ=1/16 singularity
        # (the floored region maps to the constant 1, gradient 0).
        x2 = jnp.sqrt(jnp.maximum(jnp.abs(1.0 - 16.0 * zeta), 1.0))
        x = jnp.sqrt(x2)
        psi_m_un = (jnp.log((1.0 + 2.0 * x + x2) * (1.0 + x2) / 8.0)
                    - 2.0 * (jnp.arctan(x) - jnp.arctan(1.0)))
        psi_h_un = 2.0 * jnp.log((1.0 + x2) / 2.0)
        psi_m = jnp.where(zeta > 0.0, -5.0 * zeta, psi_m_un)
        psi_h = jnp.where(zeta > 0.0, -5.0 * zeta, psi_h_un)
        return psi_m, psi_h

    for _ in range(BULK_N_ITTS):
        cd_rt = jnp.sqrt(cd)
        ustar = cd_rt * u                                  # L-Y 7a
        tstar = (ch / cd_rt) * (t10 - ts)                  # L-Y 7b
        qstar = (ce / cd_rt) * (q10 - qs)                  # L-Y 7c
        bstar = BULK_GRAV * (tstar / tv + qstar / (q10 + 1.0 / 0.608))

        zeta_u = BULK_VONKARM * bstar * z_wind / (ustar * ustar)
        zeta_t = BULK_VONKARM * bstar * z_tair / (ustar * ustar)
        zeta_q = BULK_VONKARM * bstar * z_shum / (ustar * ustar)
        psi_m_u, psi_h_u = _psi(zeta_u, z_wind)
        _, psi_h_t = _psi(zeta_t, z_tair)
        _, psi_h_q = _psi(zeta_q, z_shum)

        # shift wind/temp/humidity to 10 m + reference levels. NB cd_n10_rt here is
        # still the PREVIOUS iteration's value (the C updates it only below).
        u10 = u / (1.0 + cd_n10_rt * (jnp.log(z_wind / 10.0) - psi_m_u) / BULK_VONKARM)
        u10 = jnp.maximum(u10, BULK_U10MIN)
        t10 = t - tstar / BULK_VONKARM * (jnp.log(z_tair / z_wind) + psi_h_u - psi_h_t)
        q10 = q - qstar / BULK_VONKARM * (jnp.log(z_shum / z_wind) + psi_h_u - psi_h_q)
        tv = t10 * (1.0 + 0.608 * q10)

        cd_n10 = _cd_n10(u10)
        cd_n10_rt = jnp.sqrt(cd_n10)
        ce_n10 = 34.6 * cd_n10_rt * 1.0e-3
        stab = 0.5 + jnp.copysign(0.5, zeta_u)             # loop: copysign(zeta_u)
        ch_n10 = (18.0 * stab + 32.7 * (1.0 - stab)) * cd_n10_rt * 1.0e-3

        xx = (jnp.log(z_wind / 10.0) - psi_m_u) / BULK_VONKARM
        cd = cd_n10 / (1.0 + cd_n10_rt * xx)**2
        xx = (jnp.log(z_wind / 10.0) - psi_h_u) / BULK_VONKARM
        ch = ch_n10 / (1.0 + ch_n10 * xx / cd_n10_rt) * jnp.sqrt(cd / cd_n10)
        ce = ce_n10 / (1.0 + ce_n10 * xx / cd_n10_rt) * jnp.sqrt(cd / cd_n10)

    return cd, ce, ch


def obudget(qa, fsh, flo, t, ug, ta, ch, ce):
    """Open-water heat & freshwater fluxes — port of ``obudget``
    (``fesom_bulk.c:187-221``, standard-saturation branch). Returns
    ``(qsr, qns, evap)``: ``qsr`` downward shortwave to ocean (W/m², +down);
    ``qns`` non-solar surface heat (W/m², +up = ocean loses);
    ``evap`` (m/s, +up). ``ug`` is the **absolute** wind speed (the deliberate
    mismatch vs the relative wind in the coefficients). Differentiable w.r.t. ``t``
    (SST) and ``ch``/``ce`` (which carry the SST/current dependence)."""
    b = 3.8e-3 * jnp.exp(17.27 * t / (t + 237.3))
    hfswrow = (1.0 - BULK_ALBW) * fsh
    hflwrow = flo
    hflwrdout = -BULK_EMISS_WAT * BULK_BOLTZMANN * (t + BULK_TMELT)**4
    hfsenow = BULK_RHOAIR * BULK_CPAIR * ch * ug * (ta - t)
    evap = BULK_RHOAIR * ce * ug * (qa - b)                # kg/m²/s
    hflatow = BULK_CLHW * evap
    qns = -(hflwrow + hflwrdout + hfsenow + hflatow)
    qsr = hfswrow
    return qsr, qns, evap * BULK_INV_RHOWAT               # evap → m/s, +up


class BulkFluxes(NamedTuple):
    """Bulk output (a pytree). ``cd``/``ce``/``ch`` and the per-node fields are
    ``[nod2D]``; ``stress_node_surf`` is ``[nod2D, 2]``, ``stress_surf`` ``[elem2D, 2]``."""
    cd: jnp.ndarray
    ce: jnp.ndarray
    ch: jnp.ndarray
    heat_flux: jnp.ndarray          # qns − qsr (W/m², +up = ocean loses); pre-sw-pene
    water_flux: jnp.ndarray         # evap − prec_rain − prec_snow (m/s, +up)
    stress_node_surf: jnp.ndarray   # cd·ρ_air·|Δu|·Δu at nodes (N/m²)
    stress_surf: jnp.ndarray        # node→elem mean-of-3 (N/m²)


def bulk_surface_fluxes(mesh: Mesh, u_air, v_air, shum, shortwave, longwave, Tair,
                        prec_rain, prec_snow, T_surf, u_w, v_w,
                        z_wind=Z_MEAS, z_tair=Z_MEAS, z_shum=Z_MEAS) -> BulkFluxes:
    """Drive the L&Y09 bulk over all nodes and assemble the forcing — JAX twin of
    ``fesom_bulk_compute`` (``fesom_bulk.c:226-343``). All atmosphere args are
    ``[nod2D]`` jnp arrays from the JRA reader (``u_air``/``v_air`` already g2r-rotated
    to the model frame, ``Tair`` °C, ``prec_*`` m/s); ``T_surf = T[:,0]`` and
    ``u_w,v_w = uvnode[:,0,:]`` tap the ocean state. Cavity nodes (``ulevels>1``) are
    zeroed (none on CORE2). The element stress is the **simple mean of 3 vertices**
    (NOT the analytical path's area-weighted double average)."""
    cd, ce, ch = ncar_ocean_fluxes_mode(Tair, shum, u_air, v_air, T_surf, u_w, v_w,
                                        z_wind, z_tair, z_shum)
    ug = _safe_speed(u_air, v_air)                         # ABSOLUTE wind (obudget)
    qsr, qns, evap = obudget(shum, shortwave, longwave, T_surf, ug, Tair, ch, ce)

    heat_flux = qns - qsr
    water_flux = evap - prec_rain - prec_snow

    dux = u_air - u_w
    dvy = v_air - v_w
    mag = _safe_speed(dux, dvy) * BULK_RHOAIR             # relative-wind stress
    sns_x = cd * mag * dux
    sns_y = cd * mag * dvy

    # cavity nodes carry zero flux (fesom_bulk.c:248-257). All-False on CORE2.
    open_water = mesh.ulevels_nod2D <= 1
    heat_flux = jnp.where(open_water, heat_flux, 0.0)
    water_flux = jnp.where(open_water, water_flux, 0.0)
    sns_x = jnp.where(open_water, sns_x, 0.0)
    sns_y = jnp.where(open_water, sns_y, 0.0)
    sns = jnp.stack([sns_x, sns_y], axis=-1)              # (nod2D, 2)

    # node → element: simple mean of the 3 vertices (left-assoc to match the C sum).
    v = mesh.elem_nodes                                   # (elem2D, 3)
    stress_surf = (sns[v[:, 0]] + sns[v[:, 1]] + sns[v[:, 2]]) / 3.0
    return BulkFluxes(cd=cd, ce=ce, ch=ch, heat_flux=heat_flux, water_flux=water_flux,
                      stress_node_surf=sns, stress_surf=stress_surf)
