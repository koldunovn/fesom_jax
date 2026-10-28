"""REcoM's sources and sinks: ``REcoM_sms`` with ``sms_initialize_variables``, the per-layer carbonate
refresh and ``sms_update_state`` (parent plan Task 9, gate G2), for the ``bin_2p3z2d`` switch set.

A literal f64 port of ``recom_sms.F90`` (REcoM_src 692f804), ``recom_sms_init.F90`` and
``recom_sms_update.F90``, vectorised over ``(nod2D, nl-1)``: Fortran loops over the nodes and, per node,
over the layers ``k = nzmin..Nn``; nothing in a layer reads another layer's sources and sinks
(``sms`` is zeroed at the start of the step and ``sms_update_state`` writes only row ``k``), so every
layer is computed at once. The two vertical couplings are explicit:

* the light, ``PARave`` = ``max(tiny, SurfSR exp(-kdz))`` with ``kdz`` the cumulative sum of
  ``kappastar * thick`` from the top (``recom_sms_init.F90:963-1040``);
* the benthic calcite dissolution ``calc_diss_ben``, which Fortran leaves at the value of the LAST layer
  of the loop (``recom_sms.F90:3637`` reads the module variable set at ``k = Nn``).

Only the switch values of ``bin_2p3z2d`` are ported (:data:`fesom_jax.recom.params.SUPPORTED`): 3 zoo +
2 detritus, no coccolithophores, detritus grazing, variable preference and efficiency, photodamage, CO2
limitation, O2-dependent remineralisation, diatom mucus, the depth-dependent calcite dissolution
(``OmegaC_diss = .false.``), one-ligand iron chemistry, ``biostep = 1``. Terms Fortran multiplies by
``is_coccos = 0`` are dropped (they add an exact zero), ``is_3zoo2det = 1`` factors are dropped (exact).
The sums keep Fortran's left-to-right order and parenthesisation, so the G2 floor is ifort's FMA
contraction and transcendental ulps (``-fimf-use-svml``) against XLA's.

``biostep == 1`` (asserted by :func:`fesom_jax.recom.params.check_supported`): the reset
``if (any(abs(sms) <= tiny)) sms = 0`` at the sub-step start (``recom_sms.F90:465``) acts on an ``sms``
that was zeroed just before (``:259``), a no-op, so it is not ported; for the same reason
``state + sms`` in ``sms_initialize_variables`` is ``state``.

Fortran's run stops (``KOchl`` NaN checks, ``recom_sms.F90:867-922``) and print-only diagnostics
(``aggregationrate > 5``, benthos runaway) are not ported. Cavities (``nzmin > 1``) are out of scope: the
column starts at layer 0 (:func:`check_no_cavities`).

Units: tracers in mmol m-3 (Fe umol m-3, Chl mg m-3), rates per day, ``dt_b`` in days.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import carbonate as cb
from .params import PA2ATM, REDO2C, SECONDS_PER_DAY, TINY, TINY_CHL, RecomParams, derived
from .transport import holds
from .tracers import (NTR, ialk, idchl, idetc, idetcal, idetn, idetsi,
                      idetz2c, idetz2calc, idetz2n, idetz2si, idiac, idian, idiasi, idic, idicremin, idin,
                      idoc, idon, ife, ihetc, ihetn, imiczooc, imiczoon, ioxy, ipchl, iphycal, iphyc,
                      iphyn, isi, izoo2c, izoo2n)

# The interior carbonate cache (recom_main.F90:429-437): State.bgc_carb's trailing axis.
CARB_FIELDS = ("CO2", "pH", "pCO2", "HCO3", "CO3", "OmegaC", "kspc", "rhoSW")

DIAG3D = ("aggn", "aggd", "docexn", "docexd", "respn", "respd", "NPPn3D", "NPPd3D", "calcdiss", "calcif",
          "respmeso", "respmacro", "respmicro", "TPhyCO2", "TDiaCO2", "TCphotLigLim_phyto", "TCphot_phyto",
          "TCphotLigLim_diatoms", "TCphot_diatoms")
DIAG2D = ("NPPn", "NPPd", "GPPn", "GPPd", "NNAn", "NNAd", "Chldegn", "Chldegd",
          "grazmeso_tot", "grazmeso_n", "grazmeso_d", "grazmeso_det", "grazmeso_mic", "grazmeso_det2",
          "grazmacro_tot", "grazmacro_n", "grazmacro_d", "grazmacro_mes", "grazmacro_det", "grazmacro_mic",
          "grazmacro_det2", "grazmicro_tot", "grazmicro_n", "grazmicro_d")


class CarbCache(NamedTuple):
    """The interior carbonate system REcoM caches between steps (``CO23D`` ... ``rhoSW3D``), each
    ``(nod2D, nl-1)``: CO2*, HCO3-, CO3-- [mol m-3], pH (total scale), pCO2 [uatm], Omega_calc,
    Kspc [mol2 kg-2], in-situ density [kg m-3]. Not in the Fortran restart (zero after a restart until the
    ``mstep == 1`` refresh fills it)."""
    CO2: jax.Array
    pH: jax.Array
    pCO2: jax.Array
    HCO3: jax.Array
    CO3: jax.Array
    OmegaC: jax.Array
    kspc: jax.Array
    rhoSW: jax.Array


class SmsResult(NamedTuple):
    """Outputs of :func:`recom_sms` (all ``(nod2D, nl-1[, ntr])`` unless stated)."""
    state: jax.Array          # tracers after state = max(tiny, state + sms) and the floors (dry layers unchanged)
    carb: CarbCache           # the cache after this step's refreshes
    sms: jax.Array            # the step's sources minus sinks [mmol m-3 per step] (0 in dry layers)
    clip: jax.Array           # state_new - (state + sms) >= 0: the non-conservative clip, for the budget
    par: jax.Array            # PAR3D [W m-2]
    benthos: jax.Array        # (nod2D, 4) benthic N, C, Si, Calc after the decay [mmol m-2]
    decay_benthos: jax.Array  # (nod2D, 4) decayBenthos [mmol m-2 d-1]
    refreshed: jax.Array      # bool, the layers whose carbonate cache was refreshed
    diag3d: dict              # DIAG3D -> (nod2D, nl-1)
    diag2d: dict              # DIAG2D -> (nod2D,): column integrals sum(vert * thick)


def check_no_cavities(ulevels_nod2D) -> None:
    """Host-side guard: the port assumes ``nzmin = ulevels_nod2D = 1`` everywhere (no ice-shelf cavities)."""
    u = np.asarray(ulevels_nod2D)
    if (u != 1).any():
        raise NotImplementedError(f"REcoM with cavities (ulevels_nod2D > 1 at {int((u != 1).sum())} nodes) "
                                  "is out of scope")


def recom_limiter(slope, qa, qb):
    """``recom_limiter`` without the Geider option, ``recom_sms.F90:3971-3989``."""
    dq = qa - qb
    return 1.0 - jnp.exp(-slope * (jnp.abs(dq) - dq) ** 2)


def iron_chemistry(fe, total_ligand, ligand_stab_const):
    """Free iron from total iron with one ligand, ``recom_sms.F90:4041-4070``."""
    a = ligand_stab_const
    b = ligand_stab_const * (fe - total_ligand) + 1.0
    c = -total_ligand
    discrim = b * b - 4.0 * a * c
    ok = (jnp.abs(a) > TINY) & (discrim >= TINY)
    ligand = (-b + jnp.sqrt(jnp.where(ok, discrim, 1.0))) / (2.0 * a)
    fel = total_ligand - ligand
    return jnp.where(ok, fe - fel, 0.0)


def photosynthesis_rate(pmax, par, chl2c, co2fac, slope):
    """``calculate_photosynthesis_rate`` with ``CO2lim`` (``recom_sms.F90:3923-3965``):
    ``(Cphot, light limitation)``."""
    dead = (pmax < TINY) | jnp.isnan(par) | jnp.isnan(chl2c)
    pm = jnp.where(dead, 1.0, pmax)
    cphot = pm * (1.0 - jnp.exp(-slope * chl2c * par / pm))
    lightlim = cphot / pm
    cphot = cphot * co2fac
    cphot = jnp.where(dead, 0.0, cphot)
    lightlim = jnp.where(dead, 0.0, lightlim)
    cphot = jnp.where(cphot < TINY, 0.0, cphot)
    return cphot, lightlim


EXP_ARG_MAX = 700.0     # below log(DBL_MAX) ~ 709.78


def chlorophyll_degradation(pmax, par, chl2c_plast, deg, slope):
    """``calculate_chlorophyll_degradation`` (photodamage, ``recom_sms.F90:3888-3921``)."""
    dead = (pmax < TINY) | jnp.isnan(par) | jnp.isnan(chl2c_plast)
    pm = jnp.where(dead, 1.0, pmax)
    # With N:C below NCmin, chl2c_plast is negative and the exponent positive: exp overflows to inf, loss is -inf and
    # the max below takes deg/10. The exponent is capped at EXP_ARG_MAX (exp ~1e304, the same max result), so the
    # reverse pass never multiplies the zero cotangent of the unselected branch by an infinite exp (0*inf = nan).
    loss = deg * (1.0 - jnp.exp(jnp.minimum(-slope * chl2c_plast * par / pm, EXP_ARG_MAX)))
    loss = jnp.maximum(deg * 0.1, loss)
    return jnp.where(dead, deg * 0.1, loss)


def co2_effect(hco3, co2, ph, a, b, c, d, cunits):
    """The CO2 response of photosynthesis, ``calculate_phytoplankton_co2_effects``
    (``recom_sms.F90:3816-3823``), clipped to [0, 3]."""
    f = a * hco3 * cunits / (b + hco3 * cunits) - jnp.exp(-c * co2 * cunits) - d * 10.0 ** (-ph)
    f = jnp.minimum(f, 3.0)
    return jnp.maximum(0.0, f)


def krill_resp(daynew, lat_rad):
    """``res_zoo2_a``, the seasonal macrozooplankton respiration factor, ``recom_extra.F90:krill_resp``;
    ``daynew`` an integer (scalar), ``lat_rad`` per node. The Fortran ``if/else if`` chains, in order."""
    d = jnp.asarray(daynew, jnp.float64)
    sh = jnp.select([d <= 105, (105 <= d) & (d <= 150), (150 < d) & (d < 250), (250 <= d) & (d <= 295), d > 295],
                    [0.0, (-1.0 / 90.0) * d + 7.0 / 6.0, -0.5, (1.0 / 90.0) * d - 59.0 / 18.0, 0.0], 0.0)
    nh = jnp.select([d <= 65, (285 <= d) & (d <= 330), 330 < d, (65 <= d) & (d <= 110), (110 < d) & (d < 285)],
                    [-0.5, (-1.0 / 90.0) * d + 57.0 / 18.0, -0.5, (1.0 / 90.0) * d - 22.0 / 18.0, 0.0], 0.0)
    return jnp.where(lat_rad < 0.0, sh, nh)


def refresh_periods(dt_b):
    """``(logfile_outfreq_7, logfile_outfreq_30)``: the 7- and 30-day refresh periods in steps,
    ``int(1/dt_b * 7)``, ``int(1/dt_b * 30)`` (``recom_carbonate_chemistry.F90:231-233``)."""
    if isinstance(dt_b, (int, float)):
        spd = 1.0 / dt_b
        return int(spd * 7.0), int(spd * 30.0)
    spd = 1.0 / jnp.asarray(dt_b, jnp.float64)
    return (jnp.trunc(spd * 7.0).astype(jnp.int32), jnp.trunc(spd * 30.0).astype(jnp.int32))


def recom_sms(state, temp, sali, carb: CarbCache, thick, zbar, nlev, surf_sr, cos_ai, lat_deg, lat_rad,
              loc_slp, benthos, mstep, daynew, dt_b, p: RecomParams, *, n_iter: int = cb.PH_MAXITER) -> SmsResult:
    """One REcoM biology step for every column (``REcoM_sms`` + the clip of ``REcoM_Forcing``,
    ``recom_forcing.F90:316-325``).

    ``state`` (nod2D, nl-1, 31) the BGC tracers (after the DICremin surface reset); ``temp``, ``sali``
    (nod2D, nl-1) potential temperature and salinity; ``carb`` the cache; ``thick`` = ``hnode`` and
    ``zbar`` = ``zbar_3d_n`` (nod2D, nl) [m, negative down]; ``nlev`` (nod2D,) int = ``nlevels_nod2D - 1``
    (the wet layers are ``k < nlev``); ``surf_sr`` (nod2D,) = ``parFrac * shortwave * (1 - a_ice)``;
    ``cos_ai`` (nod2D,) the refracted solar incidence cosine (:func:`fesom_jax.recom.forcing.cobeta`);
    ``lat_deg``/``lat_rad`` latitude; ``loc_slp`` (nod2D,) sea-level pressure [Pa]; ``benthos`` (nod2D, 4);
    ``mstep`` the step within the run segment (int, may be traced); ``daynew`` day of year (int);
    ``dt_b`` the BGC step in days; ``p`` the parameters. Returns :class:`SmsResult`.
    """
    f64 = jnp.float64
    nl1 = state.shape[1]
    kk = jnp.arange(nl1)
    nlev = jnp.asarray(nlev)
    wet = kk[None, :] < nlev[:, None]                                 # k = nzmin..Nn (nzmin = 1)
    dz = derived(p)
    sv = lambda i: jnp.where(wet, state[..., i], 1.0)                 # safe values in dry layers
    temp = jnp.where(wet, temp, 10.0)
    sali = jnp.where(wet, sali, 35.0)
    thick = thick[:, :nl1]
    zf = zbar[:, :nl1]                                                 # zF(k), the top interface of layer k
    surf = surf_sr[:, None]

    # ---- sms_initialize_variables (recom_sms_init.F90:82-252), state + sms = state (biostep 1) ----
    DIN = jnp.maximum(TINY, sv(idin))
    Si = jnp.maximum(TINY, sv(isi))
    Fe = jnp.maximum(TINY, sv(ife))
    O2 = jnp.maximum(TINY, sv(ioxy))
    DON = jnp.maximum(TINY, sv(idon))
    EOC = jnp.maximum(TINY, sv(idoc))
    PhyN = jnp.maximum(dz.tiny_N, sv(iphyn))
    PhyC = jnp.maximum(dz.tiny_C, sv(iphyc))
    PhyChl = jnp.maximum(TINY_CHL, sv(ipchl))
    PhyCalc = jnp.maximum(TINY, sv(iphycal))
    DiaN = jnp.maximum(dz.tiny_N_d, sv(idian))
    DiaC = jnp.maximum(dz.tiny_C_d, sv(idiac))
    DiaChl = jnp.maximum(TINY_CHL, sv(idchl))
    DiaSi = jnp.maximum(dz.tiny_Si, sv(idiasi))
    HetN = jnp.maximum(TINY, sv(ihetn))
    HetC = jnp.maximum(TINY, sv(ihetc))
    Zoo2N = jnp.maximum(TINY, sv(izoo2n))
    Zoo2C = jnp.maximum(TINY, sv(izoo2c))
    MicZooN = jnp.maximum(TINY, sv(imiczoon))
    MicZooC = jnp.maximum(TINY, sv(imiczooc))
    DetN = jnp.maximum(TINY, sv(idetn))
    DetC = jnp.maximum(TINY, sv(idetc))
    DetSi = jnp.maximum(TINY, sv(idetsi))
    DetCalc = jnp.maximum(TINY, sv(idetcal))
    DetZ2N = jnp.maximum(TINY, sv(idetz2n))
    DetZ2C = jnp.maximum(TINY, sv(idetz2c))
    DetZ2Si = jnp.maximum(TINY, sv(idetz2si))
    DetZ2Calc = jnp.maximum(TINY, sv(idetz2calc))

    # mocsy inputs (recom_sms_init.F90:285-342)
    T_d = jnp.minimum(jnp.maximum(2.0, temp), 40.0)
    S_d = jnp.minimum(jnp.maximum(21.0, sali), 43.0)
    DIC_d = jnp.maximum(TINY * 1e-3, sv(idic) * 1e-3)
    Alk_d = jnp.maximum(TINY * 1e-3, sv(ialk) * 1e-3)
    Si_d = jnp.maximum(TINY * 1e-3, sv(isi) * 1e-3)
    Phos_d = jnp.maximum(TINY * 1e-3, sv(idin) * 1e-3) / 16.0

    # quotas (:380-633)
    quota = PhyN / PhyC
    recipQuota = 1.0 / quota
    Chl2C = PhyChl / PhyC
    Chl2N = PhyChl / PhyN
    CHL2C_plast = Chl2C * (quota / (quota - p.NCmin))
    quota_dia = DiaN / DiaC
    recipQuota_dia = 1.0 / quota_dia
    Chl2C_dia = DiaChl / DiaC
    Chl2N_dia = DiaChl / DiaN
    CHL2C_plast_dia = Chl2C_dia * (quota_dia / (quota_dia - p.NCmin_d))
    qSiC = DiaSi / DiaC
    qSiN = DiaSi / DiaN
    recipQZoo = HetC / HetN
    recipDet = DetC / DetN
    recipQZoo2 = Zoo2C / Zoo2N
    recipQZoo3 = MicZooC / MicZooN
    recipDet2 = DetZ2C / DetZ2N

    # temperature dependences (:729-878)
    rTref = 1.0 / p.recom_Tref                                        # recom_sms.F90:358
    rTloc = 1.0 / (temp + p.C2K)
    arrFunc = jnp.exp(-p.Ae * (rTloc - rTref))
    arrFuncZoo2 = (jnp.exp(p.t1_zoo2 / p.t2_zoo2 - p.t1_zoo2 * rTloc)
                   / (1.0 + jnp.exp(p.t3_zoo2 / p.t4_zoo2 - p.t3_zoo2 * rTloc)))
    q10_mes = 1.0242 ** temp
    q10_mic = 1.04 ** temp
    q10_mes_res = 1.0887 ** temp
    q10_mic_res = 1.0897 ** temp
    reminSiT = jnp.maximum(0.023 * 2.6 ** ((temp - 10.0) / 10.0), p.reminSi)
    O2Func = O2 / (p.k_o2_remin + O2)                                  # O2dep_remin

    # light (:963-1040): PAR(nzmin) = max(tiny, SurfSR); below, the attenuation by the mean chlorophyll of
    # the layer and the one above, accumulated over the thickness of the layer above
    chl = PhyChl + DiaChl
    chlave = (chl[:, :-1] + chl[:, 1:]) * 0.5                          # Chlave of layer k = 1..
    kappa = p.k_w + p.a_chl * chlave
    kappastar = kappa / cos_ai[:, None]
    inc = kappastar * thick[:, :-1]
    kdz = jnp.concatenate([jnp.zeros_like(inc[:, :1]), jnp.cumsum(inc, axis=1)], axis=1)
    lower = jnp.maximum(TINY, surf * jnp.exp(-kdz[:, 1:]))
    PARave = jnp.concatenate([jnp.broadcast_to(jnp.maximum(TINY, surf), lower[:, :1].shape), lower], axis=1)

    # ---- the carbonate cache refresh (recom_carbonate_chemistry.F90:228-257) ----
    f7, f30 = refresh_periods(dt_b)
    mstep = jnp.asarray(mstep)
    lit = PARave > 0.01 * surf
    dark = PARave < 0.01 * surf
    refresh = wet & ((mstep == 1) | (lit & (mstep % f7 == 0)) | (dark & (mstep % f30 == 0)))
    any_step = (mstep == 1) | (mstep % f7 == 0) | (mstep % f30 == 0)
    patm = (loc_slp / PA2ATM)[:, None]

    def _do_refresh(c):
        v = cb.vars_sprac(T_d, S_d, Alk_d, DIC_d, Si_d, Phos_d, patm, -zf, lat_deg[:, None], n_iter=n_iter)
        new = CarbCache(v.co2, v.ph, v.pco2, v.hco3, v.co3, v.omega_c, v.kspc, v.rho_sw)
        return CarbCache(*(jnp.where(refresh, a, b) for a, b in zip(new, c)))

    carb = CarbCache(*(jnp.asarray(x, f64) for x in carb))
    carb = jax.lax.cond(any_step, _do_refresh, lambda c: c, carb)
    co2c = jnp.where(wet, carb.CO2, 0.0)
    hco3c = jnp.where(wet, carb.HCO3, 0.0)
    phc = jnp.where(wet, carb.pH, 8.0)

    # CO2 effects (recom_sms.F90:3816-3841) and the depth-dependent calcite dissolution (:550-575)
    PhyCO2 = co2_effect(hco3c, co2c, phc, p.a_co2_phy, p.b_co2_phy, p.c_co2_phy, p.d_co2_phy, p.Cunits)
    DiaCO2 = co2_effect(hco3c, co2c, phc, p.a_co2_dia, p.b_co2_dia, p.c_co2_dia, p.d_co2_dia, p.Cunits)
    Sink_Vel = p.Vcalc * jnp.abs(zf) + p.VDet
    calc_diss = p.calc_diss_rate * Sink_Vel / 20.0
    calc_diss2 = p.calc_diss_rate2 * Sink_Vel / 20.0
    calc_diss_ben = p.calc_diss_rate * Sink_Vel / 20.0

    # photosynthesis limitation (:625-665)
    qlim = recom_limiter(p.NMinSlope, p.NCmin, quota)
    feLimitFac = Fe / (p.k_Fe + Fe)
    qlim = jnp.minimum(qlim, feLimitFac)
    pMax = p.P_cm * qlim * arrFunc
    qlim_d = recom_limiter(p.NMinSlope, p.NCmin_d, quota_dia)
    qlimTmp = recom_limiter(p.SiMinSlope, p.SiCmin, qSiC)
    qlim_d = jnp.minimum(qlim_d, qlimTmp)
    feLimitFac_d = Fe / (p.k_Fe_d + Fe)
    qlim_d = jnp.minimum(qlim_d, feLimitFac_d)
    pMax_dia = p.P_cm_d * qlim_d * arrFunc

    Cphot, lightlim = photosynthesis_rate(pMax, PARave, Chl2C, PhyCO2, p.alfa)               # :740
    Cphot_dia, lightlim_dia = photosynthesis_rate(pMax_dia, PARave, Chl2C_dia, DiaCO2, p.alfa_d)

    # chlorophyll degradation with photodamage (:808-859)
    KOchl = chlorophyll_degradation(pMax, PARave, CHL2C_plast, p.deg_Chl, p.alfa)
    KOchl_dia = chlorophyll_degradation(pMax_dia, PARave, CHL2C_plast_dia, p.deg_Chl_d, p.alfa_d)

    # assimilation (:950-1002)
    V_cm = p.V_cm_fact
    limitFacN = recom_limiter(p.NMaxSlope, quota, p.NCmax)
    N_assim = V_cm * pMax * p.NCuptakeRatio * limitFacN * (DIN / (DIN + p.k_din))
    V_cm = p.V_cm_fact_d
    limitFacN_dia = recom_limiter(p.NMaxSlope, quota_dia, p.NCmax_d)
    N_assim_dia = V_cm * pMax_dia * p.NCUptakeRatio_d * limitFacN_dia * DIN / (DIN + p.k_din_d)
    limitFacSi = recom_limiter(p.SiMaxSlope, qSiC, p.SiCmax) * limitFacN_dia
    Si_assim = p.V_cm_fact_d * p.P_cm_d * arrFunc * p.SiCUptakeRatio * limitFacSi * Si / (Si + p.k_si)

    # iron chemistry (:1036)
    FreeFe = iron_chemistry(Fe, p.totalligand, p.ligandStabConst)

    # chlorophyll synthesis (:1059-1070); PARave >= tiny always, NaN never
    okpar = (PARave >= TINY) & ~jnp.isnan(PARave)
    den = jnp.where(okpar, p.alfa * Chl2C * PARave, 1.0)
    chlSynth = jnp.where(okpar, N_assim * p.Chl2N_max * jnp.minimum(1.0, Cphot / den), 0.0)
    den_d = jnp.where(okpar, p.alfa_d * Chl2C_dia * PARave, 1.0)
    ChlSynth_dia = jnp.where(okpar, N_assim_dia * p.Chl2N_max_d * jnp.minimum(1.0, Cphot_dia / den_d), 0.0)

    # phytoplankton respiration (:1107-1111)
    phyRespRate = p.res_phy * limitFacN + p.biosynth * N_assim
    phyRespRate_dia = p.res_phy_d * limitFacN_dia + p.biosynth * N_assim_dia + p.biosynthSi * Si_assim

    # mesozooplankton grazing (:1170-1392), variable preference, detritus grazing, 3zoo2det
    aux = p.pzPhy * PhyN + p.pzDia * DiaN
    aux = aux + p.pzDet * DetN
    aux = aux + p.pzDetZ2 * DetZ2N
    aux = aux + p.pzMicZoo * MicZooN
    fPhyN = (p.pzPhy * PhyN) / aux * PhyN
    fDiaN = (p.pzDia * DiaN) / aux * DiaN
    fDetN = (p.pzDet * DetN) / aux * DetN
    fDetZ2N = (p.pzDetZ2 * DetZ2N) / aux * DetZ2N
    fMicZooN = (p.pzMicZoo * MicZooN) / aux * MicZooN
    food = fPhyN + fDiaN
    food = food + fDetN
    food = food + fDetZ2N
    food = food + fMicZooN
    foodsq = food * food
    grazingFlux = (p.graz_max * foodsq) / (p.epsilonr + foodsq) * HetN * q10_mes
    grazingFlux_phy = grazingFlux * fPhyN / food
    grazingFlux_Dia = grazingFlux * fDiaN / food
    grazingFlux_Det = grazingFlux * fDetN / food
    grazingFlux_DetZ2 = grazingFlux * fDetZ2N / food
    grazingFlux_miczoo = grazingFlux * fMicZooN / food
    grazEff = p.gfin + 1.0 / (0.2 * food + 2.0)
    gfc_mes = (grazingFlux_phy * recipQuota * grazEff) + (grazingFlux_Dia * recipQuota_dia * grazEff)
    gfc_mes = gfc_mes + (grazingFlux_Det * recipDet * grazEff)
    gfc_mes = gfc_mes + (grazingFlux_DetZ2 * recipDet2 * grazEff)
    gfc_mes = gfc_mes + (grazingFlux_miczoo * recipQZoo3 * grazEff)

    # macrozooplankton grazing (:1455-1649)
    aux = p.pzPhy2 * PhyN + p.pzDia2 * DiaN + p.pzHet * HetN + p.pzMicZoo2 * MicZooN
    aux = aux + p.pzDet2 * DetN + p.pzDetZ22 * DetZ2N
    fPhyN2 = (p.pzPhy2 * PhyN) / aux * PhyN
    fDiaN2 = (p.pzDia2 * DiaN) / aux * DiaN
    fMicZooN2 = (p.pzMicZoo2 * MicZooN) / aux * MicZooN
    fHetN = (p.pzHet * HetN) / aux * HetN
    fDetN2 = (p.pzDet2 * DetN) / aux * DetN
    fDetZ2N2 = (p.pzDetZ22 * DetZ2N) / aux * DetZ2N
    food2 = fPhyN2 + fDiaN2 + fHetN + fMicZooN2
    food2 = food2 + fDetN2 + fDetZ2N2
    foodsq2 = food2 * food2
    grazingFlux2 = (p.graz_max2 * foodsq2) / (p.epsilon2 + foodsq2) * Zoo2N * arrFuncZoo2
    grazingFlux_phy2 = (grazingFlux2 * fPhyN2) / food2
    grazingFlux_Dia2 = (grazingFlux2 * fDiaN2) / food2
    grazingFlux_miczoo2 = (grazingFlux2 * fMicZooN2) / food2
    grazingFlux_het2 = (grazingFlux2 * fHetN) / food2
    grazingFlux_Det2 = (grazingFlux2 * fDetN2) / food2
    grazingFlux_DetZ22 = (grazingFlux2 * fDetZ2N2) / food2
    gfc_zoo2 = ((grazingFlux_phy2 * recipQuota * p.grazEff2) + (grazingFlux_Dia2 * recipQuota_dia * p.grazEff2)
                + (grazingFlux_het2 * recipQZoo * p.grazEff2) + (grazingFlux_miczoo2 * recipQZoo3 * p.grazEff2))
    gfc_zoo2 = (gfc_zoo2 + (grazingFlux_Det2 * recipDet * p.grazEff2)
                + (grazingFlux_DetZ22 * recipDet2 * p.grazEff2))

    # microzooplankton grazing (:1721-1834)
    aux = p.pzPhy3 * PhyN + p.pzDia3 * DiaN
    fPhyN3 = (p.pzPhy3 * PhyN) / aux * PhyN
    fDiaN3 = (p.pzDia3 * DiaN) / aux * DiaN
    food3 = fPhyN3 + fDiaN3
    foodsq3 = food3 * food3
    grazingFlux3 = (p.graz_max3 * foodsq3) / (p.epsilon3 + foodsq3) * MicZooN * q10_mic
    grazingFlux_phy3 = (grazingFlux3 * fPhyN3) / food3
    grazingFlux_Dia3 = (grazingFlux3 * fDiaN3) / food3

    # zooplankton respiration and losses (:1887-2125)
    HetRespFlux = p.res_het * q10_mes_res * HetC                       # het_resp_noredfield, 3zoo2det
    hetLossFlux = p.loss_het * HetN * HetN
    res_zoo2_a = krill_resp(daynew, lat_rad)[:, None]
    ratio = gfc_zoo2 / Zoo2C
    res_zoo2_f = jnp.where(ratio <= 0.1, 0.1 * (gfc_zoo2 / Zoo2C * 100.0), 1.0)
    recip_res_zoo22 = p.res_zoo2 * (1.0 + res_zoo2_f + res_zoo2_a)
    Zoo2RespFlux = recip_res_zoo22 * Zoo2C
    Zoo2LossFlux = p.loss_zoo2 * Zoo2N * Zoo2N
    MicZooRespFlux = p.res_miczoo * q10_mic_res * MicZooC
    MicZooLossFlux = p.loss_miczoo * MicZooN * MicZooN
    Zoo2fecalloss_n = p.fecal_rate_n * grazingFlux2
    Zoo2fecalloss_c = p.fecal_rate_c * gfc_zoo2
    Mesfecalloss_n = p.fecal_rate_n_mes * grazingFlux
    Mesfecalloss_c = p.fecal_rate_c_mes * gfc_mes

    # aggregation with diatom mucus (:2159-2206): the diatom limiter recomputed as at :650-659
    ql = recom_limiter(p.NMinSlope, p.NCmin_d, quota_dia)
    qlt = recom_limiter(p.SiMinSlope, p.SiCmin, qSiC)
    ql = jnp.minimum(ql, qlt)
    ql = jnp.minimum(ql, Fe / (p.k_Fe_d + Fe))
    aggregationRate = p.agg_PP * (1.0 - ql) * DiaN
    aggregationRate = aggregationRate + p.agg_PD * DetN + p.agg_PP * PhyN
    aggregationRate = aggregationRate + p.agg_PD * DetZ2N

    # calcification by small phytoplankton (:2381-2456)
    calcification = p.calc_prod_ratio * Cphot * PhyC
    calc_loss_agg = aggregationRate * PhyCalc
    aux = recipQuota / (PhyC + TINY) * PhyCalc
    calc_loss_gra = grazingFlux_phy * aux
    calc_loss_gra2 = grazingFlux_phy2 * aux
    calc_loss_gra3 = grazingFlux_phy3 * aux

    # ---- sms_update_state (recom_sms_update.F90:1183-3063), per day times dt_b ----
    s = {}
    s[idin] = (-N_assim * PhyC - N_assim_dia * DiaC
               + p.rho_N * arrFunc * O2Func * DON) * dt_b
    s[idic] = (-Cphot * PhyC - Cphot_dia * DiaC
               - calcification
               + phyRespRate * PhyC + phyRespRate_dia * DiaC
               + p.rho_C1 * arrFunc * O2Func * EOC
               + HetRespFlux + Zoo2RespFlux + MicZooRespFlux
               + calc_diss * DetCalc
               + calc_loss_gra * p.calc_diss_guts
               + calc_loss_gra2 * p.calc_diss_guts
               + calc_loss_gra3 * p.calc_diss_guts
               + calc_diss2 * DetZ2Calc) * dt_b
    s[ialk] = (+1.0625 * N_assim * PhyC + 1.0625 * N_assim_dia * DiaC
               - 1.0625 * p.rho_N * arrFunc * O2Func * DON
               + 2.0 * calc_diss * DetCalc
               + 2.0 * calc_loss_gra * p.calc_diss_guts
               + 2.0 * calc_loss_gra2 * p.calc_diss_guts
               + 2.0 * calc_loss_gra3 * p.calc_diss_guts
               + 2.0 * calc_diss2 * DetZ2Calc
               - 2.0 * calcification) * dt_b
    s[iphyn] = (+N_assim * PhyC
                - p.lossN * limitFacN * PhyN
                - aggregationRate * PhyN
                - grazingFlux_phy - grazingFlux_phy2 - grazingFlux_phy3) * dt_b
    s[iphyc] = (+Cphot * PhyC
                - phyRespRate * PhyC
                - p.lossC * limitFacN * PhyC
                - aggregationRate * PhyC
                - grazingFlux_phy * recipQuota
                - grazingFlux_phy2 * recipQuota
                - grazingFlux_phy3 * recipQuota) * dt_b
    s[ipchl] = (+chlSynth * PhyC
                - KOchl * PhyChl
                - aggregationRate * PhyChl
                - grazingFlux_phy * Chl2N
                - grazingFlux_phy2 * Chl2N
                - grazingFlux_phy3 * Chl2N) * dt_b
    s[idetn] = (+grazingFlux_phy3 - grazingFlux_phy3 * p.grazEff3
                + grazingFlux_Dia3 - grazingFlux_Dia3 * p.grazEff3
                + aggregationRate * PhyN
                + aggregationRate * DiaN
                + MicZooLossFlux
                - grazingFlux_Det * grazEff
                - grazingFlux_Det2 * p.grazEff2
                - p.reminN * arrFunc * O2Func * DetN) * dt_b
    s[idetc] = (+grazingFlux_phy3 * recipQuota * (1.0 - p.grazEff3)
                + grazingFlux_Dia3 * recipQuota_dia * (1.0 - p.grazEff3)
                + aggregationRate * PhyC
                + aggregationRate * DiaC
                + MicZooLossFlux * recipQZoo3
                - grazingFlux_Det * recipDet * grazEff
                - grazingFlux_Det2 * recipDet * p.grazEff2
                - p.reminC * arrFunc * O2Func * DetC) * dt_b
    s[ihetn] = (+grazingFlux * grazEff
                - grazingFlux_het2
                - Mesfecalloss_n
                - hetLossFlux
                - p.lossN_z * HetN) * dt_b
    s[ihetc] = (+grazingFlux_phy * recipQuota * grazEff
                + grazingFlux_Dia * recipQuota_dia * grazEff
                + grazingFlux_miczoo * recipQZoo3 * grazEff
                + grazingFlux_DetZ2 * recipDet2 * grazEff
                + grazingFlux_Det * recipDet * grazEff
                - grazingFlux_het2 * recipQZoo
                - Mesfecalloss_c
                - hetLossFlux * recipQZoo
                - p.lossC_z * HetC
                - HetRespFlux) * dt_b
    s[izoo2n] = (+grazingFlux2 * p.grazEff2
                 - Zoo2LossFlux
                 - p.lossN_z2 * Zoo2N
                 - Zoo2fecalloss_n) * dt_b
    s[izoo2c] = (+grazingFlux_phy2 * recipQuota * p.grazEff2
                 + grazingFlux_Dia2 * recipQuota_dia * p.grazEff2
                 + grazingFlux_het2 * recipQZoo * p.grazEff2
                 + grazingFlux_miczoo2 * recipQZoo3 * p.grazEff2
                 + grazingFlux_Det2 * recipDet * p.grazEff2
                 + grazingFlux_DetZ22 * recipDet2 * p.grazEff2
                 - Zoo2LossFlux * recipQZoo2
                 - p.lossC_z2 * Zoo2C
                 - Zoo2RespFlux
                 - Zoo2fecalloss_c) * dt_b
    s[imiczoon] = (+grazingFlux3 * p.grazEff3
                   - grazingFlux_miczoo
                   - grazingFlux_miczoo2
                   - MicZooLossFlux
                   - p.lossN_z3 * MicZooN) * dt_b
    s[imiczooc] = (+grazingFlux_phy3 * recipQuota * p.grazEff3
                   + grazingFlux_Dia3 * recipQuota_dia * p.grazEff3
                   - MicZooLossFlux * recipQZoo3
                   - grazingFlux_miczoo * recipQZoo3
                   - grazingFlux_miczoo2 * recipQZoo3
                   - p.lossC_z3 * MicZooC
                   - MicZooRespFlux) * dt_b
    s[idetz2n] = (+grazingFlux_phy2 * (1.0 - p.grazEff2)
                  + grazingFlux_Dia2 * (1.0 - p.grazEff2)
                  + grazingFlux_het2 * (1.0 - p.grazEff2)
                  + grazingFlux_miczoo2 * (1.0 - p.grazEff2)
                  + grazingFlux_phy * (1.0 - grazEff)
                  + grazingFlux_Dia * (1.0 - grazEff)
                  + grazingFlux_miczoo * (1.0 - grazEff)
                  + Zoo2LossFlux
                  + hetLossFlux
                  + Zoo2fecalloss_n
                  + Mesfecalloss_n
                  - grazingFlux_DetZ2 * grazEff
                  - grazingFlux_DetZ22 * p.grazEff2
                  - p.reminN * arrFunc * O2Func * DetZ2N) * dt_b
    s[idetz2c] = (+grazingFlux_phy2 * recipQuota * (1.0 - p.grazEff2)
                  + grazingFlux_Dia2 * recipQuota_dia * (1.0 - p.grazEff2)
                  + grazingFlux_het2 * recipQZoo * (1.0 - p.grazEff2)
                  + grazingFlux_miczoo2 * recipQZoo3 * (1.0 - p.grazEff2)
                  + grazingFlux_phy * recipQuota * (1.0 - grazEff)
                  + grazingFlux_Dia * recipQuota_dia * (1.0 - grazEff)
                  + grazingFlux_miczoo * recipQZoo3 * (1.0 - grazEff)
                  + Zoo2LossFlux * recipQZoo2
                  + hetLossFlux * recipQZoo
                  + Zoo2fecalloss_c
                  + Mesfecalloss_c
                  - grazingFlux_DetZ2 * recipDet2 * grazEff
                  - grazingFlux_DetZ22 * recipDet2 * p.grazEff2
                  - p.reminC * arrFunc * O2Func * DetZ2C) * dt_b
    s[idetz2si] = (+grazingFlux_Dia2 * qSiN
                   + grazingFlux_Dia * qSiN
                   - reminSiT * DetZ2Si) * dt_b
    s[idetz2calc] = (+calc_loss_gra2 * (1.0 - p.calc_diss_guts)
                     + calc_loss_gra * (1.0 - p.calc_diss_guts)
                     - calc_diss2 * DetZ2Calc) * dt_b
    s[idon] = (+p.lossN * limitFacN * PhyN
               + p.lossN_d * limitFacN_dia * DiaN
               + p.reminN * arrFunc * O2Func * DetN
               + p.reminN * arrFunc * O2Func * DetZ2N
               + p.lossN_z * HetN
               + p.lossN_z2 * Zoo2N
               + p.lossN_z3 * MicZooN
               - p.rho_N * arrFunc * O2Func * DON) * dt_b
    s[idoc] = (+p.lossC * limitFacN * PhyC
               + p.lossC_d * limitFacN_dia * DiaC
               + p.reminC * arrFunc * O2Func * DetC
               + p.reminC * arrFunc * O2Func * DetZ2C
               + p.lossC_z * HetC
               + p.lossC_z2 * Zoo2C
               + p.lossC_z3 * MicZooC
               - p.rho_C1 * arrFunc * O2Func * EOC) * dt_b
    s[idian] = (+N_assim_dia * DiaC
                - p.lossN_d * limitFacN_dia * DiaN
                - aggregationRate * DiaN
                - grazingFlux_Dia - grazingFlux_Dia2 - grazingFlux_Dia3) * dt_b
    s[idiac] = (+Cphot_dia * DiaC
                - p.lossC_d * limitFacN_dia * DiaC
                - phyRespRate_dia * DiaC
                - aggregationRate * DiaC
                - grazingFlux_Dia * recipQuota_dia
                - grazingFlux_Dia2 * recipQuota_dia
                - grazingFlux_Dia3 * recipQuota_dia) * dt_b
    s[idchl] = (+ChlSynth_dia * DiaC
                - KOchl_dia * DiaChl
                - aggregationRate * DiaChl
                - grazingFlux_Dia * Chl2N_dia
                - grazingFlux_Dia2 * Chl2N_dia
                - grazingFlux_Dia3 * Chl2N_dia) * dt_b
    s[idiasi] = (+Si_assim * DiaC
                 - p.lossN_d * limitFacN_dia * DiaSi
                 - aggregationRate * DiaSi
                 - grazingFlux_Dia * qSiN
                 - grazingFlux_Dia2 * qSiN
                 - grazingFlux_Dia3 * qSiN) * dt_b
    s[idetsi] = (+aggregationRate * DiaSi
                 + p.lossN_d * limitFacN_dia * DiaSi
                 + grazingFlux_Dia3 * qSiN
                 - reminSiT * DetSi) * dt_b
    s[isi] = (-Si_assim * DiaC
              + reminSiT * DetSi
              + reminSiT * DetZ2Si) * dt_b
    s[ife] = (p.Fe2N * (-N_assim * PhyC - N_assim_dia * DiaC
                        + p.lossN * limitFacN * PhyN
                        + p.lossN_d * limitFacN_dia * DiaN
                        + p.reminN * arrFunc * O2Func * DetN
                        + p.reminN * arrFunc * O2Func * DetZ2N
                        + p.lossN_z * HetN
                        + p.lossN_z2 * Zoo2N
                        + p.lossN_z3 * MicZooN)
              - p.kScavFe * DetC * FreeFe
              - p.kScavFe * DetZ2C * FreeFe
              - p.kScavFe2 * FreeFe) * dt_b
    s[iphycal] = (+calcification
                  - p.lossC * limitFacN * PhyCalc
                  - phyRespRate * PhyCalc
                  - calc_loss_agg
                  - calc_loss_gra
                  - calc_loss_gra2
                  - calc_loss_gra3) * dt_b
    s[idetcal] = (+p.lossC * limitFacN * PhyCalc
                  + phyRespRate * PhyCalc
                  + calc_loss_agg
                  + calc_loss_gra3
                  - calc_loss_gra3 * p.calc_diss_guts
                  - calc_diss * DetCalc) * dt_b
    s[ioxy] = (+Cphot * PhyC + Cphot_dia * DiaC
               - phyRespRate * PhyC - phyRespRate_dia * DiaC
               - p.rho_C1 * arrFunc * O2Func * EOC
               - HetRespFlux
               - Zoo2RespFlux
               - MicZooRespFlux) * REDO2C * dt_b
    s[idicremin] = (+p.rho_C1 * arrFunc * O2Func * EOC) * dt_b
    assert sorted(s) == list(range(NTR))
    sms = jnp.stack([jnp.where(wet, s[i], 0.0) for i in range(NTR)], axis=-1)

    # ---- REcoM_Forcing's update and clip (recom_forcing.F90:316-325) ----
    raw = state + sms
    new = jnp.maximum(TINY, raw)
    floors = {ipchl: TINY_CHL, iphyn: dz.tiny_N, iphyc: dz.tiny_C, idchl: TINY_CHL,
              idian: dz.tiny_N_d, idiac: dz.tiny_C_d, idiasi: dz.tiny_Si}
    cols = [jnp.maximum(floors[i], new[..., i]) if i in floors else new[..., i] for i in range(NTR)]
    if holds():
        # each column its own kernel before the stack: fused into it, the 31 tracers' expressions alternated within
        # every warp of the tracer-minor stack XLA chose (14 ms of a step at 0.42 TB/s on one GH200, job 28470554;
        # holding the stack tracer-major only added a copy after it, job 28473168)
        cols = list(jax.lax.optimization_barrier(tuple(cols)))
    new = jnp.where(wet[..., None], jnp.stack(cols, axis=-1), state)
    clip = jnp.where(wet[..., None], new - raw, 0.0)

    # ---- benthos (recom_sms.F90:3503-3640): calc_diss_ben of the last wet layer ----
    last = jnp.clip(nlev - 1, 0, nl1 - 1)
    cdb = jnp.take_along_axis(calc_diss_ben, last[:, None], axis=1)[:, 0]
    rates = jnp.stack([jnp.full_like(cdb, p.decayRateBenN), jnp.full_like(cdb, p.decayRateBenC),
                       jnp.full_like(cdb, p.decayRateBenSi), cdb], axis=-1)
    decay = rates * benthos
    benthos_new = benthos - decay * dt_b

    # ---- diagnostics (recom_sms.F90:2563-3373, recom_forcing.F90:450-505); recipbiostep = 1 ----
    z = lambda x: jnp.where(wet, x, 0.0)
    d3 = dict(
        aggn=aggregationRate * PhyC, aggd=aggregationRate * DiaC,
        docexn=p.lossC * limitFacN * PhyC, docexd=p.lossC_d * limitFacN_dia * DiaC,
        respn=phyRespRate * PhyC, respd=phyRespRate_dia * DiaC,
        NPPn3D=+Cphot * PhyC - phyRespRate * PhyC, NPPd3D=+Cphot_dia * DiaC - phyRespRate_dia * DiaC,
        calcdiss=calc_diss * DetCalc, calcif=calcification,
        respmeso=HetRespFlux, respmacro=Zoo2RespFlux, respmicro=MicZooRespFlux,
        TPhyCO2=PhyCO2, TDiaCO2=DiaCO2, TCphotLigLim_phyto=lightlim, TCphot_phyto=Cphot,
        TCphotLigLim_diatoms=lightlim_dia, TCphot_diatoms=Cphot_dia)
    vert = dict(
        NPPn=d3["NPPn3D"], NPPd=d3["NPPd3D"], GPPn=+Cphot * PhyC, GPPd=+Cphot_dia * DiaC,
        NNAn=+N_assim * PhyC - p.lossN * limitFacN * PhyN,
        NNAd=+N_assim_dia * DiaC - p.lossN * limitFacN_dia * DiaN,          # lossN, not lossN_d (:2711)
        Chldegn=KOchl, Chldegd=KOchl_dia,
        grazmeso_tot=((+grazingFlux_phy * recipQuota * grazEff + grazingFlux_Dia * recipQuota_dia * grazEff
                       + grazingFlux_Det * recipDet * grazEff)
                      + (+grazingFlux_DetZ2 * recipDet2 * grazEff + grazingFlux_miczoo * recipQZoo3 * grazEff)),
        grazmeso_n=grazingFlux_phy * recipQuota, grazmeso_d=grazingFlux_Dia * recipQuota_dia,
        grazmeso_det=grazingFlux_Det * recipDet, grazmeso_mic=grazingFlux_miczoo * recipQZoo3,
        grazmeso_det2=grazingFlux_DetZ2 * recipDet2,
        grazmacro_tot=(+grazingFlux_phy2 * recipQuota * p.grazEff2 + grazingFlux_Dia2 * recipQuota_dia * p.grazEff2
                       + grazingFlux_het2 * recipQZoo * p.grazEff2 + grazingFlux_miczoo2 * recipQZoo3 * p.grazEff2
                       + grazingFlux_Det2 * recipDet * p.grazEff2 + grazingFlux_DetZ22 * recipDet2 * p.grazEff2),
        grazmacro_n=grazingFlux_phy2 * recipQuota, grazmacro_d=grazingFlux_Dia2 * recipQuota_dia,
        grazmacro_mes=grazingFlux_het2 * recipQZoo, grazmacro_det=grazingFlux_Det2 * recipDet,
        grazmacro_mic=grazingFlux_miczoo2 * recipQZoo3, grazmacro_det2=grazingFlux_DetZ22 * recipDet2,
        grazmicro_tot=(+grazingFlux_phy3 * recipQuota * p.grazEff3
                       + grazingFlux_Dia3 * recipQuota_dia * p.grazEff3),
        grazmicro_n=grazingFlux_phy3 * recipQuota, grazmicro_d=grazingFlux_Dia3 * recipQuota_dia)
    d3 = {k: z(v) for k, v in d3.items()}
    d2 = {k: jnp.sum(z(v) * z(thick), axis=1) for k, v in vert.items()}

    return SmsResult(state=new, carb=carb, sms=sms, clip=clip, par=z(PARave), benthos=benthos_new,
                     decay_benthos=decay, refreshed=refresh, diag3d=d3, diag2d=d2)
