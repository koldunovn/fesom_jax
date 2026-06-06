"""Sea-ice thermodynamics (Phase 6, Task 6.2) — an AD-safe per-node port of
``fesom_ice_thermo.c``.

The ice growth/melt energy budget: the open-water budget (:func:`_obudget`), the ice-
covered skin-temperature solver (:func:`_budget`, a 5-iter Newton + 4-way albedo select),
the full single-cell step (:func:`therm_ice_cell`, ``fesom_therm_ice`` /
``fesom_ice_thermo.c:234-411``), snow→ice flooding, the cutoff, and the per-node driver
(:func:`thermodynamics`). Thermo is **per-node with no scatter** — `therm_ice_cell` is a
scalar function `vmap`-ped over nodes. This kernel **caps the supercooling** (the ``o2ihf``
ocean→ice heat flux + the freezing point) and **activates runoff** (``prec = rain + runo +
snow*(1-A)``, the one line where runoff enters).

⚠️ Scope = the C (single ice class, ``use_virt_salt=1`` virtual-salt path, ``snowdist=1``,
``open_water_albedo=0``); see :mod:`fesom_jax.ice` (`IceConfig`). The skin-temperature
Newton is a **fixed 5 iterations** (no data-dependent break); the 7-class growth-rate loop
sequentially refines ``t`` — both unrolled (AD-friendly). ``obudget`` uses the OPEN-OCEAN
coeffs ``ch``/``ce`` (the bulk ``Ch``/``Ce_atm_oce``); ``budget`` uses the over-ice
constants ``ch_i``/``ce_i`` = ``cfg.ch_atm_ice``/``ce_atm_ice``.

AD guards (the masked-NaN rule — a forward ``where`` does not stop a backward ``0·inf``):
``tfrez``'s ``√(S³)`` (double-`where` safe-sqrt), ``con/hice`` (`where(hice>0,hice,1)`),
``/rsss``, the Newton ``/A3``, ``ustar``/``ug`` ``√``. The freezing/albedo/melt
``min``/``max`` kinks use subgradients (`jnp.minimum`/`maximum`/`where`).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .ice import IceConfig


# --------------------------------------------------------------------------
# AD-safe helpers
# --------------------------------------------------------------------------
def _safe_sqrt(x):
    """``√x`` with a finite (0) gradient at ``x==0`` (double-`where`; matches C ``sqrt(0)``)."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(safe), 0.0)


def tfrez(cfg: IceConfig, salinity):
    """Freezing point of seawater, Millero (1978) / UNESCO (``fesom_ice_thermo.c:20-24``):
    ``-0.0575·S + 1.7105e-3·√(S³) - 2.155e-4·S²``. Safe-sqrt for the masked S=0 lane."""
    s = salinity
    return -0.0575 * s + 1.7105e-3 * _safe_sqrt(s * s * s) - 2.155e-4 * s * s


# --------------------------------------------------------------------------
# Open-water surface energy budget — fesom_ice_obudget (fesom_ice_thermo.c:98-142)
# --------------------------------------------------------------------------
def _obudget(cfg: IceConfig, qa, fsh, flo, t, ug, ta, ch, ce):
    """Open-water growth rate ``fh`` [m ice/s] + evaporation ``evap`` [m water/s].
    ``t`` is the SST (°C); ``ch``/``ce`` the open-ocean transfer coeffs. Smooth."""
    b = 3.8e-3 * jnp.exp(17.27 * t / (t + 237.3))
    hfswr = (1.0 - cfg.albw) * fsh
    hflwrd = -cfg.emiss_wat * cfg.boltzmann * (t + cfg.tmelt) ** 4
    hfrad = hfswr + flo + hflwrd
    hfsen = cfg.rhoair * cfg.cpair * ch * ug * (ta - t)
    evap_kgm2s = cfg.rhoair * ce * ug * (qa - b)
    hflat = cfg.clhw * evap_kgm2s
    hftot = hfrad + hfsen + hflat
    fh = -hftot / cfg.cl
    evap = evap_kgm2s / cfg.rhowat
    return fh, evap


# --------------------------------------------------------------------------
# Ice skin-temperature solver + thick-ice budget — fesom_ice_budget (:160-217)
# --------------------------------------------------------------------------
def _budget(cfg: IceConfig, hice, hsn, t, ta, qa, fsh, flo, ug, S_oc, ch_i, ce_i):
    """5-iter Newton on the ice skin temperature ``t`` → ``(t_new, fh, subli)``.
    ``hice`` is the per-class ice thickness (>0 on active classes; guarded ice-free)."""
    # 4-way albedo select (t<0 freezing | t>=0 melting) × (snow | bare) — branchless.
    alb_frz = jnp.where(hsn > 0.0, cfg.albsn, cfg.albi)
    alb_melt = jnp.where(hsn > 0.0, cfg.albsnm, cfg.albim)
    alb = jnp.where(t < 0.0, alb_frz, alb_melt)

    q1 = 11637800.0
    q2 = -5897.8
    inv_rhoair = 1.0 / cfg.rhoair
    d1 = cfg.rhoair * cfg.cpair * ch_i
    d2 = cfg.rhoair * ce_i
    d3 = d2 * cfg.clhi
    A1 = (1.0 - alb) * fsh + flo + d1 * ug * ta + d3 * ug * qa
    hice_safe = jnp.where(hice > 0.0, hice, 1.0)      # con/hice finite on the ice-free lane

    B = jnp.zeros_like(t)
    for _ in range(5):                                # fixed 5-iter Newton (no break)
        tk = t + cfg.tmelt
        B = q1 * inv_rhoair * jnp.exp(q2 / tk)
        A2 = (-d1 * ug * t - d3 * ug * B
              - cfg.emiss_ice * cfg.boltzmann * (tk * tk * tk * tk))
        A3 = -d3 * ug * B * q2 / (tk * tk)
        C = cfg.con / hice_safe
        A3 = A3 + C + d1 * ug + 4.0 * cfg.emiss_ice * cfg.boltzmann * (tk * tk * tk)
        C = C * (tfrez(cfg, S_oc) - t)
        t = t + (A1 + A2 + C) / jnp.where(A3 != 0.0, A3, 1.0)
    t = jnp.minimum(t, 0.0)

    tk = t + cfg.tmelt
    hfrad = ((1.0 - alb) * fsh + flo
             - cfg.emiss_ice * cfg.boltzmann * (tk * tk * tk * tk))
    hfsen = d1 * ug * (ta - t)
    _subli = d2 * ug * (qa - B)
    hflat = cfg.clhi * _subli
    fh = -(hfrad + hfsen + hflat) / cfg.cl
    subli = _subli / cfg.rhowat
    return t, fh, subli


# --------------------------------------------------------------------------
# Snow→ice flooding — fesom_ice_flooding (fesom_ice_thermo.c:30-42)
# --------------------------------------------------------------------------
def _flooding(cfg: IceConfig, h, hsn):
    hdraft = (cfg.rhosno * hsn + h * cfg.rhoice) / cfg.rhowat
    hflood = hdraft - jnp.minimum(hdraft, h)
    return h + hflood, hsn - hflood * cfg.rhoice / cfg.rhosno


# --------------------------------------------------------------------------
# Single-cell thermodynamic step — fesom_therm_ice (fesom_ice_thermo.c:234-411)
# --------------------------------------------------------------------------
class ThermoOut(NamedTuple):
    h: jax.Array        # new ice volume per area [m]
    hsn: jax.Array      # new snow volume per area [m]
    A: jax.Array        # new concentration
    t: jax.Array        # new skin temperature [°C]
    fw: jax.Array       # total freshwater flux to ocean [m water/ice_dt] (+down; incl. runoff)
    ehf: jax.Array      # net surface heat flux to ocean [W/m²] (+down)
    thdgr: jax.Array    # dhgrowth = (h_new - h_old)/ice_dt [m/s]
    evap: jax.Array     # evap + sublimation [m water/s]
    dAgrowth: jax.Array
    iflice: jax.Array


def therm_ice_cell(cfg: IceConfig, h, hsn, A, fsh, flo, Ta, qa, rain, snow, runo,
                   rsss, ug, ustar, T_oc, S_oc, ch, ce, t, lid_clo) -> ThermoOut:
    """One node's growth/melt step (``fesom_therm_ice``, virtual-salt path). Scalar — the
    driver `vmap`s it over nodes. ``H_ML = cfg.h_ml``, ``ice_dt = cfg.ice_dt``. ``ch``/``ce``
    = open-ocean transfer coeffs (bulk ``Ch``/``Ce_atm_oce``)."""
    ice_dt = cfg.ice_dt
    inv_rhosno = 1.0 / cfg.rhosno
    h_old = h                                              # for dhgrowth (initial h)
    one_minus_A = 1.0 - A

    # effective (snow+ice) thickness, Semtner 0-layer (:261-263)
    Aclamp = jnp.maximum(A, cfg.armin)
    thick = hsn * (cfg.con / cfg.consn) / Aclamp + h / Aclamp

    # open-water growth rate (:269)
    rhow, evap = _obudget(cfg, qa, fsh, flo, T_oc, ug, Ta, ch, ce)

    # ice-covered growth rate — iclasses loop, sequentially refining t (:288-303)
    rhice = jnp.zeros_like(h)
    subli_acc = jnp.zeros_like(h)
    t_loop = t
    for k in range(1, cfg.iclasses + 1):
        thact = (2 * k - 1) * thick / cfg.iclasses
        t_loop, shice, subli_i = _budget(cfg, thact, hsn, t_loop, Ta, qa, fsh, flo,
                                         ug, S_oc, cfg.ch_atm_ice, cfg.ce_atm_ice)
        rhice = rhice + shice
        subli_acc = subli_acc + subli_i
    active = thick > cfg.hmin                              # the C `if (thick > hmin)` gate
    rhice = jnp.where(active, rhice / cfg.iclasses, 0.0)
    subli = jnp.where(active, subli_acc / cfg.iclasses, 0.0)
    t = jnp.where(active, t_loop, t)                       # skin temp updates only with ice

    # convert rates [m/s]→[m/DT] + areal weighting (:305-315)
    rhow = rhow * ice_dt
    rhice = rhice * ice_dt
    sh = rhow * one_minus_A + rhice * A
    ahf = -cfg.cl * sh / ice_dt

    prec = rain + runo + snow * one_minus_A               # ← RUNOFF enters here (:318)
    hsn = hsn + snow * ice_dt * A * 1000.0 * inv_rhosno   # snow fall (1000 = rhofwt)
    hsn_post_fall = hsn                                    # _dhsngrowth save (:322)

    evap = evap * one_minus_A
    subli = subli * A

    # snow melt by negative atmospheric heat (:328-330)
    hsntmp = jnp.minimum(-jnp.minimum(sh, 0.0) * cfg.rhoice * inv_rhosno, hsn)
    hsn = hsn - hsntmp

    rh = sh + hsntmp * cfg.rhosno / cfg.rhoice
    h = jnp.maximum(h, 0.0)
    Tfrez_S = tfrez(cfg, S_oc)
    o2ihf = ((T_oc - Tfrez_S) * 0.006 * ustar * cfg.cc * A
             + (T_oc - Tfrez_S) * cfg.h_ml / ice_dt * cfg.cc * one_minus_A)
    rh = rh - o2ihf * ice_dt / cfg.cl
    qhst = h + rh

    # melt remaining snow if ML heat content left, then final ice (:343-348)
    hsn = jnp.maximum(hsn + jnp.minimum(qhst, 0.0) * cfg.rhoice * inv_rhosno, 0.0)
    h = jnp.maximum(qhst, 0.0)
    h = jnp.where(h < 1.0e-6, 0.0, h)

    dhgrowth = (h - h_old) / ice_dt
    dhsngrowth = (hsn - hsn_post_fall) / ice_dt
    ehf = ahf + cfg.cl * (dhgrowth + (cfg.rhosno / cfg.rhoice) * dhsngrowth)

    # freshwater (virtual-salt path) (:367-372)
    rsss_safe = jnp.where(rsss != 0.0, rsss, 1.0)
    fwice = -dhgrowth * cfg.rhoice / cfg.rhowat * (rsss - cfg.Sice) / rsss_safe
    fwsnw = -dhsngrowth * cfg.rhosno / cfg.rhowat
    fw = prec + evap + fwice + fwsnw

    # compactness — Hibler 1979 eq. 16 (:377-385). A on the RHS is the original A.
    rh = -jnp.minimum(h, -rh)
    rA = rhow - o2ihf * ice_dt / cfg.cl
    Aold = A
    A = (A + cfg.c_melt * jnp.minimum(rh, 0.0) * A / jnp.maximum(h, cfg.hmin)
         + jnp.maximum(rA, 0.0) * (1.0 - A) / lid_clo)
    A = jnp.minimum(A, h * 1.0e6)
    A = jnp.minimum(jnp.maximum(A, 0.0), 1.0)
    dAgrowth = (A - Aold) / ice_dt

    # flooding (snow→ice) + the virtual-salt correction (:387-403)
    h_pre_flood = h
    h, hsn = _flooding(cfg, h, hsn)
    iflice = (h - h_pre_flood) / ice_dt
    fw = fw + iflice * cfg.rhoice / cfg.rhowat * cfg.Sice / rsss_safe

    return ThermoOut(h=h, hsn=hsn, A=A, t=t, fw=fw, ehf=ehf, thdgr=dhgrowth,
                     evap=evap + subli, dAgrowth=dAgrowth, iflice=iflice)


# --------------------------------------------------------------------------
# Per-node cutoff — fesom_ice_cut_off (fesom_ice_thermo.c:48-75)
# --------------------------------------------------------------------------
def cut_off(a_ice, m_ice, m_snow):
    """Clamp a_ice≤1; zero all three tracers where a_ice<1e-9 OR m_ice<1e-9. Branchless."""
    a = jnp.minimum(a_ice, 1.0)
    zero = (a < 1.0e-9) | (m_ice < 1.0e-9)
    a = jnp.where(zero, 0.0, a)
    m = jnp.where(zero, 0.0, m_ice)
    ms = jnp.where(zero, 0.0, m_snow)
    return a, m, ms


# --------------------------------------------------------------------------
# Per-node driver — fesom_ice_thermodynamics (fesom_ice_thermo.c:421-526)
# --------------------------------------------------------------------------
def compute_ustar(cfg: IceConfig, u_ice, v_ice, u_w, v_w):
    """Friction velocity ``√((u_ice-u_w)²+(v_ice-v_w)²)·cd_oce_ice`` (safe-sqrt at 0)."""
    du = u_ice - u_w
    dv = v_ice - v_w
    return _safe_sqrt((du * du + dv * dv) * cfg.cd_oce_ice)


class ThermoState(NamedTuple):
    """Driver output — the updated ice tracers + skin temp + ice→ocean fluxes (all [nod2D])."""
    a_ice: jax.Array
    m_ice: jax.Array
    m_snow: jax.Array
    t_skin: jax.Array
    flx_fw: jax.Array      # = therm_ice fw (+down; oce_fluxes negates to water_flux)
    flx_h: jax.Array       # = therm_ice ehf (+down; oce_fluxes negates to heat_flux)
    thdgr: jax.Array


def thermodynamics(cfg: IceConfig, *, m_ice, m_snow, a_ice, u_ice, v_ice, t_skin,
                   srfoce_temp, srfoce_salt, srfoce_u, srfoce_v,
                   fsh, flo, Tair, qa, u_wind, v_wind, rain, snow, runo,
                   ch, ce, geo_lat, non_cavity) -> ThermoState:
    """Run the per-node sea-ice thermodynamics over the whole mesh (the ``vmap`` of
    `therm_ice_cell` + the ``ustar`` pass + the cavity mask). All inputs are ``[nod2D]``.
    ``ch``/``ce`` are the bulk open-ocean coeffs; ``geo_lat`` selects ``lid_clo``."""
    ustar = compute_ustar(cfg, u_ice, v_ice, srfoce_u, srfoce_v)
    ug = _safe_sqrt(u_wind * u_wind + v_wind * v_wind)
    rsss = srfoce_salt if cfg.ref_sss_local else jnp.full_like(srfoce_salt, cfg.ref_sss)
    lid_clo = jnp.where(geo_lat > 0.0, cfg.h0, cfg.h0_s)

    out = jax.vmap(lambda *a: therm_ice_cell(cfg, *a))(
        m_ice, m_snow, a_ice, fsh, flo, Tair, qa, rain, snow, runo,
        rsss, ug, ustar, srfoce_temp, srfoce_salt, ch, ce, t_skin, lid_clo)

    # Cavity skip (ulevels_nod2D>1): leave those nodes unchanged (no-op on CORE2).
    keep = non_cavity
    sel = lambda new, old: jnp.where(keep, new, old)
    return ThermoState(
        a_ice=sel(out.A, a_ice), m_ice=sel(out.h, m_ice), m_snow=sel(out.hsn, m_snow),
        t_skin=sel(out.t, t_skin),
        flx_fw=jnp.where(keep, out.fw, 0.0), flx_h=jnp.where(keep, out.ehf, 0.0),
        thdgr=jnp.where(keep, out.thdgr, 0.0))
