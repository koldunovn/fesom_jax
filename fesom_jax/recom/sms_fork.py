"""REcoM's sources and sinks as the spectral fork computes them (lane E of
``docs/plans/20261028-recom-spectral-port.md``; spec ``docs/plans/20261028-recom-spectral-lane-E.md``).

A literal f64 port of ``REcoM_sms`` of the fork (``sveta-loza/fesom2`` ``fesom2.6_recom_tra_diags_spectral`` +
the 11 upstream commits, worktree ``/albedo/work/user/sloza/SPECTRAL/recom_spectral_r692``, S =
``src/int_recom/recom_sms.F90``, F = ``recom_forcing.F90``) and of the update + clip of ``REcoM_Forcing``
(F:1004-1048), vectorised over ``(nod2D, nl-1)`` as :mod:`fesom_jax.recom.sms` (the REcoM 692f804 port, which
this module leaves untouched: a 692f804 namelist never reaches here).

What differs from 692f804, all of it in this module:

* the tracer layout is a :class:`~fesom_jax.recom.layout.Layout` (the fork has no DICremin; CDOM optional);
* ``enable_coccos``: coccolithophores and Phaeocystis (S:613-641 unpack, :953-1036 quotas, :1234-1311 temperature
  functions, :2181-2218 CO2 effects, :2434-2464 limiters, :2654-2734 photosynthesis, :2855-2890 photodamage,
  :3003-3015 N uptake, :3116-3133 Chl synthesis, :3154-3175 respiration, the grazing of all three zooplankton
  :3227-3876, aggregation :4241-4279, Phaeocystis mortality :4284-4288/6212-6218, calcification by the
  coccolithophores :4390-4481, their own sms :5988-6205 and the ``is_coccos`` terms of every other tracer), and the
  two things coccos change for the other groups: ``pMax = qlim * Temp_X`` instead of ``P_cm * qlim * arrFunc``
  (S:2393-2425) and the diatom Si uptake with ``Temp_diatoms`` (S:3035-3044);
* the broadband light in both forms: ``par_layer_mean`` (the exact layer mean, S:1466-1484) and FESOM's
  interface form (S:1490-1567), the attenuating chlorophyll including the coccos;
* Platt photoinhibition ``beta_inh*`` (S:2548, 2608, 2673, 2731), pseudo N fixation ``nfix_*`` (S:2982-2995),
  constant respiration ``res_phy_const`` (S:3154-3160), carbon overflow ``f_ovf`` / ``res_ovf*`` (S:4294-4318,
  6545-6579), ``k_din_calc`` (S:4414-4418), ``agg_fac_phaeo*`` (S:4257-4262), ``mort_phaeo`` (S:4284-4288);
* ``tiny_Si = tiny_C_d / SiCmax`` (S:362; 692f804 multiplies), no ``kScavFe2`` term in the iron sink (S:6320-6321), the DOC sms assembled in two steps (S:5783-5810);
* CDOM (``Layout.cdom``, S:6493-6538, F:1031): fixed constants :data:`FCDOM`, :data:`RHO_CDOM`,
  :data:`PHOT_CDOM`, :data:`KPHOT_CDOM` (``module REcoM_spectral``, M:2678-2681), the DOC ``(1 - fcdom)``
  factor and the overflow split ``ovf_to_cdom``.

Switch values this port assumes (the fork's ``tmp_run_kdind15``, checked by :func:`check_fork_switches`):
``enable_3zoo2det``, ``Grazing_detritus``, ``REcoM_Grazing_Variable_Preference``, ``het_resp_noredfield``,
``diatom_mucus``, ``O2dep_remin``, ``use_photodamage``, ``CO2lim`` on; ``OmegaC_diss``, ``REcoM_Geider_limiter``,
``RECOM_MARSHALL`` off; ``biostep = 1``. The N:C quota clamp of the spectral build (S:847-850, 906-909, 976-979,
1018-1021) is ``nc_clamp`` (lane W sets it with the spectral light; the broadband twin has none).

The Fortran switches on parameter VALUES (``f_ovf > 0``, ``res_ovf > 0``, ``mort_phaeo > 0``, ``beta_inh > 0``,
``k_din_calc >= 0``, ``agg_fac_phaeo_det /= 1``) are ``jnp.where`` here, so a traced parameter stays
differentiable and the zero case is Fortran's exact zero. The column cap of the N fixation (``nfix_max``) is a
``lax.scan`` down the column (a no-op when ``nfix_floor = 0``).

Units: tracers mmol m-3 (Fe umol m-3, Chl mg m-3), rates per day, ``dt_b`` in days.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import carbonate as cb
from .layout import Layout
from .params import PA2ATM, REDO2C, TINY, TINY_CHL, RecomParams, RecomSwitches, derived
from .sms import (CarbCache, EXP_ARG_MAX, check_no_cavities, chlorophyll_degradation, co2_effect,  # noqa: F401
                  iron_chemistry, krill_resp, photosynthesis_rate, recom_limiter, refresh_periods)
from .transport import holds

# module REcoM_spectral (M:2678-2681): not namelist entries in the fork
FCDOM = 0.02          # fraction of the DOC sources that is CDOM
RHO_CDOM = 0.003      # [d-1] bacterial degradation of CDOM, times arrFunc
PHOT_CDOM = 0.167     # [d-1] photochemical degradation at saturating light
KPHOT_CDOM = 13.51    # [W m-2] PAR at which the photolysis saturates

# what the fork's recom() produces per layer (MAIN:470-537, S:7163-7676): the 692f804 grazing diagnostics do not exist
DIAG3D_BASE = ("aggn", "aggd", "docexn", "docexd", "respn", "respd", "NPPn3D", "NPPd3D", "calcdiss", "calcif",
               "respmeso", "respmacro", "respmicro", "TPhyCO2", "TDiaCO2", "TCphotLigLim_phyto", "TCphot_phyto",
               "TCphotLigLim_diatoms", "TCphot_diatoms", "TTemp_phyto", "TTemp_diatoms", "TqLF_phyto",
               "TqLF_diatoms", "TSi_assimDia")
DIAG3D_COCCO = ("aggc", "aggp", "docexc", "docexp", "respc", "respp", "NPPc3D", "NPPp3D", "TCoccoCO2", "TPhaeoCO2",
                "TCphotLigLim_cocco", "TCphot_cocco", "TCphotLigLim_phaeo", "TCphot_phaeo", "TTemp_cocco",
                "TTemp_phaeo", "TqLF_cocco", "TqLF_phaeo")
DIAG2D_BASE = ("NPPn", "NPPd", "GPPn", "GPPd", "NNAn", "NNAd", "Chldegn", "Chldegd")
DIAG2D_COCCO = ("NPPc", "NPPp", "GPPc", "GPPp", "NNAc", "NNAp", "Chldegc", "Chldegp")


class SmsForkResult(NamedTuple):
    """Outputs of :func:`recom_sms_fork` (all ``(nod2D, nl-1[, ntr])`` unless stated)."""
    state: jax.Array
    carb: CarbCache
    sms: jax.Array
    clip: jax.Array
    par: jax.Array
    benthos: jax.Array
    decay_benthos: jax.Array
    refreshed: jax.Array
    diag3d: dict
    diag2d: dict


def check_fork_switches(sw: RecomSwitches, lay: Layout) -> None:
    """The switch values this module ports (the fork's reference configuration); anything else raises."""
    need = dict(enable_3zoo2det=True, Grazing_detritus=True, REcoM_Grazing_Variable_Preference=True,
                het_resp_noredfield=True, diatom_mucus=True, O2dep_remin=True, use_photodamage=True, CO2lim=True,
                OmegaC_diss=False, REcoM_Geider_limiter=False, RECOM_MARSHALL=False, biostep=1)
    bad = [f"{k} = {getattr(sw, k)!r} (need {v!r})" for k, v in need.items() if getattr(sw, k) != v]
    if bad:
        raise NotImplementedError("fesom_jax.recom.sms_fork ports the fork's reference switch set only: " + "; ".join(bad))
    if lay.enable_coccos != sw.enable_coccos or lay.enable_3zoo2det != sw.enable_3zoo2det:
        raise ValueError("layout and switches disagree on enable_coccos / enable_3zoo2det")


def _temp_phaeo(temp, p: RecomParams):
    """Blanchard et al. with the fork's guard (S:1293-1309)."""
    tmax, topt = p.Tmax_phaeo, p.Topt_phaeo
    base = jnp.maximum((tmax - temp) / (tmax - topt), 0.0)
    f = p.uopt_phaeo * base ** p.beta_phaeo * jnp.exp(-p.beta_phaeo * (topt - temp) / (tmax - topt))
    return jnp.where(temp >= tmax, TINY, jnp.maximum(f, TINY))


def _nfix(temp, thick, pmax, limfac, nassim_base, cphot, phyc, v_cm, p: RecomParams):
    """Pseudo nitrogen fixation of the small phytoplankton (S:2982-2995): ``N_fix`` per layer, 0 unless
    ``nfix_floor > 0`` and the layer is warmer than ``nfix_Tmin``; the column cap ``nfix_max`` (when > 0) is
    Fortran's running sum from the surface down, a scan over the layers. All of it ``jnp.where``, so a traced
    parameter stays differentiable and the off case is Fortran's exact zero."""
    on = (p.nfix_floor > 0.0) & (temp > p.nfix_Tmin)
    unc = (v_cm * pmax * p.NCuptakeRatio * limfac * jnp.maximum(p.nfix_floor - nassim_base, 0.0)
           * jnp.minimum(cphot / jnp.maximum(pmax, TINY), 1.0))
    unc = jnp.where(on, unc, 0.0)
    content = phyc * thick

    def layer(col, x):
        u, c, w = x
        capped = jnp.minimum(u, jnp.maximum(p.nfix_max - col, 0.0) / jnp.maximum(c, TINY))
        f = jnp.where(p.nfix_max > 0.0, capped, u)
        f = jnp.where(w, f, 0.0)
        return col + f * c, f

    _, fix = jax.lax.scan(layer, jnp.zeros(temp.shape[0]), (unc.T, content.T, on.T))
    return fix.T


def recom_sms_fork(state, temp, sali, carb: CarbCache, thick, zbar, nlev, surf_sr, cos_ai, lat_deg, lat_rad,
                   loc_slp, benthos, mstep, daynew, dt_b, p: RecomParams, sw: RecomSwitches, lay: Layout, *,
                   nc_clamp: bool = False, n_iter: int = cb.PH_MAXITER, spectral=None,
                   spectral_co2lim: bool = False) -> SmsForkResult:
    """One biology step of the fork's REcoM for every column (``REcoM_sms`` + the update and clip of
    ``REcoM_Forcing``). Arguments as :func:`fesom_jax.recom.sms.recom_sms`, plus the switches ``sw`` and the
    layout ``lay`` (both static) and ``nc_clamp`` (the spectral build's N:C quota clamp).

    ``spectral`` (a :class:`fesom_jax.recom.spectral.SpectralLight`, lane L): the ``__RECOM_WAVEBANDS`` build. The
    broadband attenuation is replaced by its ``PARave`` (S:1457-1465), and the growth, chlorophyll degradation and
    synthesis take the spectral forms ``Cphot_X = pMax_X (1 - exp(-alpha_I_X Chl2C_X / pMax_X))`` (S:2502-2550,
    2564-2734; no ``Cphot < tiny`` reset), ``KOchl_X = deg_Chl_X (1 - exp(-alpha_I_X Chl2C_X / pMax_X))`` on the
    plain ``Chl2C`` (S:2785-2840) and ``chlSynth_X = N_assim_X Chl2N_max_X min(1, Cphot_X / (alpha_I_X Chl2C_X))``
    gated by ``PARlocal`` (S:3072-3133). The fork applies no CO2 limitation in those branches even with ``CO2lim``;
    ``spectral_co2lim=True`` multiplies ``Cphot_X`` by the CO2 factor as the broadband build does (the user's stated
    deviation, 2026-10-28; off for the twin gate). ``TCphotLigLim_X`` is ``Cphot_X / pMax_X`` before that factor (the
    fork leaves the diagnostic unset in a spectral build)."""
    check_fork_switches(sw, lay)
    coccos = bool(sw.enable_coccos)
    cdom = bool(lay.cdom)
    f64 = jnp.float64
    nl1 = state.shape[1]
    kk = jnp.arange(nl1)
    nlev = jnp.asarray(nlev)
    wet = kk[None, :] < nlev[:, None]
    dz = derived(p)
    tiny_Si = dz.tiny_C_d / p.SiCmax                 # the fork divides (S:362, F:263); 692f804 multiplies
    sv = lambda i: jnp.where(wet, state[..., i], 1.0)                 # noqa: E731  safe values in dry layers
    temp = jnp.where(wet, temp, 10.0)
    sali = jnp.where(wet, sali, 35.0)
    thick = thick[:, :nl1]
    zf = zbar[:, :nl1]
    surf = surf_sr[:, None]
    L = lay

    # ---- the state at the start of the layer loop, floored (S:499-700) ----
    DIN = jnp.maximum(TINY, sv(L.idin)); Si = jnp.maximum(TINY, sv(L.isi)); Fe = jnp.maximum(TINY, sv(L.ife))
    O2 = jnp.maximum(TINY, sv(L.ioxy)); DON = jnp.maximum(TINY, sv(L.idon)); EOC = jnp.maximum(TINY, sv(L.idoc))
    cdomC = jnp.maximum(TINY, sv(L.icdom)) if cdom else None
    PhyN = jnp.maximum(dz.tiny_N, sv(L.iphyn)); PhyC = jnp.maximum(dz.tiny_C, sv(L.iphyc))
    PhyChl = jnp.maximum(TINY_CHL, sv(L.ipchl)); PhyCalc = jnp.maximum(TINY, sv(L.iphycal))
    DiaN = jnp.maximum(dz.tiny_N_d, sv(L.idian)); DiaC = jnp.maximum(dz.tiny_C_d, sv(L.idiac))
    DiaChl = jnp.maximum(TINY_CHL, sv(L.idchl)); DiaSi = jnp.maximum(tiny_Si, sv(L.idiasi))
    if coccos:
        CoccoN = jnp.maximum(dz.tiny_N_c, sv(L.icocn)); CoccoC = jnp.maximum(dz.tiny_C_c, sv(L.icocc))
        CoccoChl = jnp.maximum(TINY_CHL, sv(L.icchl))
        PhaeoN = jnp.maximum(dz.tiny_N_p, sv(L.iphan)); PhaeoC = jnp.maximum(dz.tiny_C_p, sv(L.iphac))
        PhaeoChl = jnp.maximum(TINY_CHL, sv(L.iphachl))
    HetN = jnp.maximum(TINY, sv(L.ihetn)); HetC = jnp.maximum(TINY, sv(L.ihetc))
    Zoo2N = jnp.maximum(TINY, sv(L.izoo2n)); Zoo2C = jnp.maximum(TINY, sv(L.izoo2c))
    MicZooN = jnp.maximum(TINY, sv(L.imiczoon)); MicZooC = jnp.maximum(TINY, sv(L.imiczooc))
    DetN = jnp.maximum(TINY, sv(L.idetn)); DetC = jnp.maximum(TINY, sv(L.idetc))
    DetSi = jnp.maximum(TINY, sv(L.idetsi)); DetCalc = jnp.maximum(TINY, sv(L.idetcal))
    DetZ2N = jnp.maximum(TINY, sv(L.idetz2n)); DetZ2C = jnp.maximum(TINY, sv(L.idetz2c))
    DetZ2Si = jnp.maximum(TINY, sv(L.idetz2si)); DetZ2Calc = jnp.maximum(TINY, sv(L.idetz2calc))

    # mocsy inputs (S:720-730)
    T_d = jnp.minimum(jnp.maximum(2.0, temp), 40.0)
    S_d = jnp.minimum(jnp.maximum(21.0, sali), 43.0)
    DIC_d = jnp.maximum(TINY * 1e-3, sv(L.idic) * 1e-3)
    Alk_d = jnp.maximum(TINY * 1e-3, sv(L.ialk) * 1e-3)
    Si_d = jnp.maximum(TINY * 1e-3, sv(L.isi) * 1e-3)
    Phos_d = jnp.maximum(TINY * 1e-3, sv(L.idin) * 1e-3) / 16.0

    # quotas (S:731-1036); nc_clamp = the spectral build's clamp
    def quotas(N, C, Chl, ncmin, ncmax):
        q = N / C
        if nc_clamp:
            q = jnp.maximum(ncmin, q)
            q = jnp.minimum(q, ncmax)
        return q, 1.0 / q, Chl / C, Chl / N, (Chl / C) * (q / (q - ncmin))
    quota, recipQuota, Chl2C, Chl2N, CHL2C_plast = quotas(PhyN, PhyC, PhyChl, p.NCmin, p.NCmax)
    quota_dia, recipQuota_dia, Chl2C_dia, Chl2N_dia, CHL2C_plast_dia = quotas(DiaN, DiaC, DiaChl, p.NCmin_d, p.NCmax_d)
    qSiC = DiaSi / DiaC
    qSiN = DiaSi / DiaN
    if coccos:
        quota_cocco, recipQuota_cocco, Chl2C_cocco, Chl2N_cocco, CHL2C_plast_cocco = quotas(
            CoccoN, CoccoC, CoccoChl, p.NCmin_c, p.NCmax_c)
        quota_phaeo, recipQuota_phaeo, Chl2C_phaeo, Chl2N_phaeo, CHL2C_plast_phaeo = quotas(
            PhaeoN, PhaeoC, PhaeoChl, p.NCmin_p, p.NCmax_p)
    recipQZoo = HetC / HetN
    recipDet = DetC / DetN
    recipQZoo2 = Zoo2C / Zoo2N
    recipQZoo3 = MicZooC / MicZooN
    recipDet2 = DetZ2C / DetZ2N

    # temperature dependences (S:1234-1340)
    rTref = 1.0 / p.recom_Tref
    rTloc = 1.0 / (temp + p.C2K)
    arrFunc = jnp.exp(-p.Ae * (rTloc - rTref))
    if coccos:
        Temp_phyto = jnp.exp(p.ord_phy + p.expon_phy * temp)
        Temp_diatoms = jnp.exp(p.ord_d + p.expon_d * temp)
        Temp_cocco = jnp.where(temp < 5.0, TINY, jnp.maximum(jnp.exp(p.ord_cocco + p.expon_cocco * temp), TINY))
        Temp_phaeo = _temp_phaeo(temp, p)
    arrFuncZoo2 = (jnp.exp(p.t1_zoo2 / p.t2_zoo2 - p.t1_zoo2 * rTloc)
                   / (1.0 + jnp.exp(p.t3_zoo2 / p.t4_zoo2 - p.t3_zoo2 * rTloc)))
    q10_mes = 1.0242 ** temp
    q10_mic = 1.04 ** temp
    q10_mes_res = 1.0887 ** temp
    q10_mic_res = 1.0897 ** temp
    reminSiT = jnp.maximum(0.023 * 2.6 ** ((temp - 10.0) / 10.0), p.reminSi)
    O2Func = O2 / (p.k_o2_remin + O2)

    # light (S:1466-1567): the attenuating chlorophyll is every group's; a spectral build brings its own PARave
    chl = PhyChl + DiaChl
    if coccos:
        chl = chl + CoccoChl + PhaeoChl
    if spectral is not None:
        PARave = jnp.asarray(spectral.PARave, f64)
    elif sw.par_layer_mean:
        kappa = p.k_w + p.a_chl * chl                                     # Chlave of the layer itself
        kappastar = kappa / cos_ai[:, None]
        inc = kappastar * thick
        kdzLower = jnp.cumsum(inc, axis=1)
        kdzUpper = kdzLower - inc
        PARave = jnp.maximum(TINY, (surf * jnp.exp(-kdzUpper) - surf * jnp.exp(-kdzLower)) / inc)
    else:
        chlave = (chl[:, :-1] + chl[:, 1:]) * 0.5
        kappa = p.k_w + p.a_chl * chlave
        kappastar = kappa / cos_ai[:, None]
        inc = kappastar * thick[:, :-1]
        kdz = jnp.concatenate([jnp.zeros_like(inc[:, :1]), jnp.cumsum(inc, axis=1)], axis=1)
        lower = jnp.maximum(TINY, surf * jnp.exp(-kdz[:, 1:]))
        PARave = jnp.concatenate([jnp.broadcast_to(jnp.maximum(TINY, surf), lower[:, :1].shape), lower], axis=1)

    # ---- the carbonate cache refresh (as sms.py; recom_carbonate_chemistry.F90) ----
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

    # CO2 effects (S:2170-2218) and the depth-dependent calcite dissolution (S:2220-2340, OmegaC_diss off)
    PhyCO2 = co2_effect(hco3c, co2c, phc, p.a_co2_phy, p.b_co2_phy, p.c_co2_phy, p.d_co2_phy, p.Cunits)
    DiaCO2 = co2_effect(hco3c, co2c, phc, p.a_co2_dia, p.b_co2_dia, p.c_co2_dia, p.d_co2_dia, p.Cunits)
    if coccos:
        CoccoCO2 = co2_effect(hco3c, co2c, phc, p.a_co2_cocco, p.b_co2_cocco, p.c_co2_cocco, p.d_co2_cocco, p.Cunits)
        PhaeoCO2 = co2_effect(hco3c, co2c, phc, p.a_co2_phaeo, p.b_co2_phaeo, p.c_co2_phaeo, p.d_co2_phaeo, p.Cunits)
    Sink_Vel = p.Vdet_a * jnp.abs(zf) + p.VDet
    calc_diss = p.calc_diss_rate * Sink_Vel / 20.0
    calc_diss2 = p.calc_diss_rate2 * Sink_Vel / 20.0
    calc_diss_ben = p.calc_diss_rate * Sink_Vel / 20.0

    # photosynthesis limitation (S:2380-2464)
    qlim = jnp.minimum(recom_limiter(p.NMinSlope, p.NCmin, quota), Fe / (p.k_Fe + Fe))
    pMax = qlim * Temp_phyto if coccos else p.P_cm * qlim * arrFunc
    TqLF_phyto = qlim
    qlim_d = recom_limiter(p.NMinSlope, p.NCmin_d, quota_dia)
    qlim_d = jnp.minimum(qlim_d, recom_limiter(p.SiMinSlope, p.SiCmin, qSiC))
    qlim_d = jnp.minimum(qlim_d, Fe / (p.k_Fe_d + Fe))
    pMax_dia = qlim_d * Temp_diatoms if coccos else p.P_cm_d * qlim_d * arrFunc
    TqLF_diatoms = qlim_d
    if coccos:
        qlim_c = jnp.minimum(recom_limiter(p.NMinSlope, p.NCmin_c, quota_cocco), Fe / (p.k_Fe_c + Fe))
        pMax_cocco = qlim_c * Temp_cocco
        qlim_p = jnp.minimum(recom_limiter(p.NMinSlope, p.NCmin_p, quota_phaeo), Fe / (p.k_Fe_p + Fe))
        pMax_phaeo = qlim_p * Temp_phaeo

    # photosynthesis with CO2lim, then the Platt photoinhibition (S:2500-2734)
    def inhibit(cphot, beta, chl2c, pmax):
        on = (beta > 0.0) & (cphot >= TINY) & (pmax >= TINY)
        safe = jnp.where(pmax >= TINY, pmax, 1.0)
        return jnp.where(on, cphot * jnp.exp(-beta * chl2c * PARave / safe), cphot)
    if spectral is None:
        Cphot, lightlim = photosynthesis_rate(pMax, PARave, Chl2C, PhyCO2, p.alfa)
        Cphot_dia, lightlim_dia = photosynthesis_rate(pMax_dia, PARave, Chl2C_dia, DiaCO2, p.alfa_d)
        if coccos:
            Cphot_cocco, lightlim_cocco = photosynthesis_rate(pMax_cocco, PARave, Chl2C_cocco, CoccoCO2, p.alfa_c)
            Cphot_phaeo, lightlim_phaeo = photosynthesis_rate(pMax_phaeo, PARave, Chl2C_phaeo, PhaeoCO2, p.alfa_p)
    else:
        def spec_phot(pmax, alpha_i, chl2c, co2fac):                     # S:2502-2510 (and :2564, :2628, :2690)
            ok = (pmax >= TINY) & (alpha_i >= TINY) & (chl2c >= TINY)
            pm = jnp.where(ok, pmax, 1.0)
            cphot = jnp.where(ok, pm * (1.0 - jnp.exp(-alpha_i * chl2c / pm)), 0.0)
            lightlim = jnp.where(ok, cphot / pm, 0.0)
            if spectral_co2lim:
                cphot = cphot * co2fac
            return cphot, lightlim
        Cphot, lightlim = spec_phot(pMax, spectral.alpha_I, Chl2C, PhyCO2)
        Cphot_dia, lightlim_dia = spec_phot(pMax_dia, spectral.alpha_I_dia, Chl2C_dia, DiaCO2)
        if coccos:
            Cphot_cocco, lightlim_cocco = spec_phot(pMax_cocco, spectral.alpha_I_cocco, Chl2C_cocco, CoccoCO2)
            Cphot_phaeo, lightlim_phaeo = spec_phot(pMax_phaeo, spectral.alpha_I_phaeo, Chl2C_phaeo, PhaeoCO2)
    Cphot = inhibit(Cphot, p.beta_inh, Chl2C, pMax)
    Cphot_dia = inhibit(Cphot_dia, p.beta_inh_d, Chl2C_dia, pMax_dia)
    if coccos:
        Cphot_cocco = inhibit(Cphot_cocco, p.beta_inh_c, Chl2C_cocco, pMax_cocco)
        Cphot_phaeo = inhibit(Cphot_phaeo, p.beta_inh_p, Chl2C_phaeo, pMax_phaeo)

    # chlorophyll degradation with photodamage (S:2777-2890)
    if spectral is None:
        KOchl = chlorophyll_degradation(pMax, PARave, CHL2C_plast, p.deg_Chl, p.alfa)
        KOchl_dia = chlorophyll_degradation(pMax_dia, PARave, CHL2C_plast_dia, p.deg_Chl_d, p.alfa_d)
        if coccos:
            KOchl_cocco = chlorophyll_degradation(pMax_cocco, PARave, CHL2C_plast_cocco, p.deg_Chl_c, p.alfa_c)
            KOchl_phaeo = chlorophyll_degradation(pMax_phaeo, PARave, CHL2C_plast_phaeo, p.deg_Chl_p, p.alfa_p)
    else:
        def spec_deg(pmax, alpha_i, chl2c, deg):                         # S:2785-2790 (and :2800, :2815, :2829)
            ok = (pmax >= TINY) & (alpha_i >= TINY) & (PARave >= TINY) & (chl2c >= TINY)
            pm = jnp.where(ok, pmax, 1.0)
            return jnp.where(ok, deg * (1.0 - jnp.exp(-alpha_i * chl2c / pm)), 0.0)
        KOchl = spec_deg(pMax, spectral.alpha_I, Chl2C, p.deg_Chl)
        KOchl_dia = spec_deg(pMax_dia, spectral.alpha_I_dia, Chl2C_dia, p.deg_Chl_d)
        if coccos:
            KOchl_cocco = spec_deg(pMax_cocco, spectral.alpha_I_cocco, Chl2C_cocco, p.deg_Chl_c)
            KOchl_phaeo = spec_deg(pMax_phaeo, spectral.alpha_I_phaeo, Chl2C_phaeo, p.deg_Chl_p)

    # nitrogen assimilation, N fixation, silicate (S:2976-3044)
    V_cm = p.V_cm_fact
    limitFacN = recom_limiter(p.NMaxSlope, quota, p.NCmax)
    N_assim = V_cm * pMax * p.NCuptakeRatio * limitFacN * (DIN / (DIN + p.k_din))
    N_fix = _nfix(temp, thick, pMax, limitFacN, DIN / (DIN + p.k_din), Cphot, PhyC, V_cm, p)
    N_assim = N_assim + N_fix
    limitFacN_dia = recom_limiter(p.NMaxSlope, quota_dia, p.NCmax_d)
    N_assim_dia = p.V_cm_fact_d * pMax_dia * p.NCUptakeRatio_d * limitFacN_dia * DIN / (DIN + p.k_din_d)
    if coccos:
        limitFacN_cocco = recom_limiter(p.NMaxSlope, quota_cocco, p.NCmax_c)
        N_assim_cocco = p.V_cm_fact_c * pMax_cocco * p.NCUptakeRatio_c * limitFacN_cocco * DIN / (DIN + p.k_din_c)
        limitFacN_phaeo = recom_limiter(p.NMaxSlope, quota_phaeo, p.NCmax_p)
        N_assim_phaeo = p.V_cm_fact_p * pMax_phaeo * p.NCUptakeRatio_p * limitFacN_phaeo * DIN / (DIN + p.k_din_p)
    limitFacSi = recom_limiter(p.SiMaxSlope, qSiC, p.SiCmax) * limitFacN_dia
    if coccos:
        Si_assim = p.V_cm_fact_d * Temp_diatoms * p.SiCUptakeRatio * limitFacSi * Si / (Si + p.k_si)
    else:
        Si_assim = p.V_cm_fact_d * p.P_cm_d * arrFunc * p.SiCUptakeRatio * limitFacSi * Si / (Si + p.k_si)
    FreeFe = iron_chemistry(Fe, p.totalligand, p.ligandStabConst)

    # chlorophyll synthesis (S:3078-3140)
    if spectral is None:
        okpar = (PARave >= TINY) & ~jnp.isnan(PARave)
        def synth(nassim, chl2n_max, cphot, alfa, chl2c):
            den = jnp.where(okpar, alfa * chl2c * PARave, 1.0)
            return jnp.where(okpar, nassim * chl2n_max * jnp.minimum(1.0, cphot / den), 0.0)
        chlSynth = synth(N_assim, p.Chl2N_max, Cphot, p.alfa, Chl2C)
        ChlSynth_dia = synth(N_assim_dia, p.Chl2N_max_d, Cphot_dia, p.alfa_d, Chl2C_dia)
        if coccos:
            ChlSynth_cocco = synth(N_assim_cocco, p.Chl2N_max_c, Cphot_cocco, p.alfa_c, Chl2C_cocco)
            ChlSynth_phaeo = synth(N_assim_phaeo, p.Chl2N_max_p, Cphot_phaeo, p.alfa_p, Chl2C_phaeo)
    else:
        PARl = jnp.asarray(spectral.PARl, f64)
        def spec_synth(nassim, chl2n_max, cphot, alpha_i, chl2c):        # S:3072-3077 (and :3086, :3099, :3112)
            ok = (PARl >= TINY) & (alpha_i >= TINY) & (chl2c >= TINY)
            den = jnp.where(ok, alpha_i * chl2c, 1.0)
            return jnp.where(ok, nassim * chl2n_max * jnp.minimum(1.0, cphot / den), 0.0)
        chlSynth = spec_synth(N_assim, p.Chl2N_max, Cphot, spectral.alpha_I, Chl2C)
        ChlSynth_dia = spec_synth(N_assim_dia, p.Chl2N_max_d, Cphot_dia, spectral.alpha_I_dia, Chl2C_dia)
        if coccos:
            ChlSynth_cocco = spec_synth(N_assim_cocco, p.Chl2N_max_c, Cphot_cocco, spectral.alpha_I_cocco, Chl2C_cocco)
            ChlSynth_phaeo = spec_synth(N_assim_phaeo, p.Chl2N_max_p, Cphot_phaeo, spectral.alpha_I_phaeo, Chl2C_phaeo)

    # phytoplankton respiration (S:3154-3175)
    if sw.res_phy_const:
        phyRespRate = p.res_phy + p.biosynth * N_assim
        phyRespRate_dia = p.res_phy_d + p.biosynth * N_assim_dia + p.biosynthSi * Si_assim
        if coccos:
            phyRespRate_cocco = p.res_phy_c + p.biosynth * N_assim_cocco
            phyRespRate_phaeo = p.res_phy_p + p.biosynth * N_assim_phaeo
    else:
        phyRespRate = p.res_phy * limitFacN + p.biosynth * N_assim
        phyRespRate_dia = p.res_phy_d * limitFacN_dia + p.biosynth * N_assim_dia + p.biosynthSi * Si_assim
        if coccos:
            phyRespRate_cocco = p.res_phy_c * limitFacN_cocco + p.biosynth * N_assim_cocco
            phyRespRate_phaeo = p.res_phy_p * limitFacN_phaeo + p.biosynth * N_assim_phaeo

    # mesozooplankton grazing (S:3227-3443), variable preference, detritus grazing, 3zoo2det (+ coccos)
    aux = p.pzPhy * PhyN + p.pzDia * DiaN
    aux = aux + p.pzDet * DetN
    aux = aux + p.pzDetZ2 * DetZ2N
    aux = aux + p.pzMicZoo * MicZooN
    if coccos:
        aux = aux + p.pzCocco * CoccoN + p.pzPhaeo * PhaeoN
    fPhyN = (p.pzPhy * PhyN) / aux * PhyN
    fDiaN = (p.pzDia * DiaN) / aux * DiaN
    fDetN = (p.pzDet * DetN) / aux * DetN
    fDetZ2N = (p.pzDetZ2 * DetZ2N) / aux * DetZ2N
    fMicZooN = (p.pzMicZoo * MicZooN) / aux * MicZooN
    if coccos:
        fCoccoN = (p.pzCocco * CoccoN) / aux * CoccoN
        fPhaeoN = (p.pzPhaeo * PhaeoN) / aux * PhaeoN
    food = fPhyN + fDiaN
    food = food + fDetN
    food = food + fDetZ2N
    food = food + fMicZooN
    if coccos:
        food = food + fCoccoN + fPhaeoN
    foodsq = food * food
    grazingFlux = (p.graz_max * foodsq) / (p.epsilonr + foodsq) * HetN * q10_mes
    grazingFlux_phy = grazingFlux * fPhyN / food
    grazingFlux_Dia = grazingFlux * fDiaN / food
    grazingFlux_Det = grazingFlux * fDetN / food
    grazingFlux_DetZ2 = grazingFlux * fDetZ2N / food
    grazingFlux_miczoo = grazingFlux * fMicZooN / food
    if coccos:
        grazingFlux_Cocco = grazingFlux * fCoccoN / food
        grazingFlux_Phaeo = grazingFlux * fPhaeoN / food
    grazEff = p.gfin + 1.0 / (0.2 * food + 2.0)
    gfc_mes = (grazingFlux_phy * recipQuota * grazEff) + (grazingFlux_Dia * recipQuota_dia * grazEff)
    gfc_mes = gfc_mes + (grazingFlux_Det * recipDet * grazEff)
    gfc_mes = gfc_mes + (grazingFlux_DetZ2 * recipDet2 * grazEff)
    gfc_mes = gfc_mes + (grazingFlux_miczoo * recipQZoo3 * grazEff)
    if coccos:
        gfc_mes = (gfc_mes + (grazingFlux_Cocco * recipQuota_cocco * grazEff)
                   + (grazingFlux_Phaeo * recipQuota_phaeo * grazEff))

    # macrozooplankton grazing (S:3520-3695)
    aux = p.pzPhy2 * PhyN + p.pzDia2 * DiaN + p.pzHet * HetN + p.pzMicZoo2 * MicZooN
    aux = aux + p.pzDet2 * DetN + p.pzDetZ22 * DetZ2N
    if coccos:
        aux = aux + p.pzCocco2 * CoccoN + p.pzPhaeo2 * PhaeoN
    fPhyN2 = (p.pzPhy2 * PhyN) / aux * PhyN
    fDiaN2 = (p.pzDia2 * DiaN) / aux * DiaN
    fMicZooN2 = (p.pzMicZoo2 * MicZooN) / aux * MicZooN
    fHetN = (p.pzHet * HetN) / aux * HetN
    if coccos:
        fCoccoN2 = (p.pzCocco2 * CoccoN) / aux * CoccoN
        fPhaeoN2 = (p.pzPhaeo2 * PhaeoN) / aux * PhaeoN
    fDetN2 = (p.pzDet2 * DetN) / aux * DetN
    fDetZ2N2 = (p.pzDetZ22 * DetZ2N) / aux * DetZ2N
    food2 = fPhyN2 + fDiaN2 + fHetN + fMicZooN2
    food2 = food2 + fDetN2 + fDetZ2N2
    if coccos:
        food2 = food2 + fCoccoN2 + fPhaeoN2
    foodsq2 = food2 * food2
    grazingFlux2 = (p.graz_max2 * foodsq2) / (p.epsilon2 + foodsq2) * Zoo2N * arrFuncZoo2
    grazingFlux_phy2 = (grazingFlux2 * fPhyN2) / food2
    grazingFlux_Dia2 = (grazingFlux2 * fDiaN2) / food2
    grazingFlux_miczoo2 = (grazingFlux2 * fMicZooN2) / food2
    grazingFlux_het2 = (grazingFlux2 * fHetN) / food2
    if coccos:
        grazingFlux_Cocco2 = (grazingFlux2 * fCoccoN2) / food2
        grazingFlux_Phaeo2 = (grazingFlux2 * fPhaeoN2) / food2
    grazingFlux_Det2 = (grazingFlux2 * fDetN2) / food2
    grazingFlux_DetZ22 = (grazingFlux2 * fDetZ2N2) / food2
    gfc_zoo2 = ((grazingFlux_phy2 * recipQuota * p.grazEff2) + (grazingFlux_Dia2 * recipQuota_dia * p.grazEff2)
                + (grazingFlux_het2 * recipQZoo * p.grazEff2) + (grazingFlux_miczoo2 * recipQZoo3 * p.grazEff2))
    gfc_zoo2 = (gfc_zoo2 + (grazingFlux_Det2 * recipDet * p.grazEff2)
                + (grazingFlux_DetZ22 * recipDet2 * p.grazEff2))
    if coccos:
        gfc_zoo2 = (gfc_zoo2 + (grazingFlux_Cocco2 * recipQuota_cocco * p.grazEff2)
                    + (grazingFlux_Phaeo2 * recipQuota_phaeo * p.grazEff2))

    # microzooplankton grazing (S:3775-3876)
    aux = p.pzPhy3 * PhyN + p.pzDia3 * DiaN
    if coccos:
        aux = aux + p.pzCocco3 * CoccoN + p.pzPhaeo3 * PhaeoN
    fPhyN3 = (p.pzPhy3 * PhyN) / aux * PhyN
    fDiaN3 = (p.pzDia3 * DiaN) / aux * DiaN
    if coccos:
        fCoccoN3 = (p.pzCocco3 * CoccoN) / aux * CoccoN
        fPhaeoN3 = (p.pzPhaeo3 * PhaeoN) / aux * PhaeoN
    food3 = fPhyN3 + fDiaN3
    if coccos:
        food3 = food3 + fCoccoN3 + fPhaeoN3
    foodsq3 = food3 * food3
    grazingFlux3 = (p.graz_max3 * foodsq3) / (p.epsilon3 + foodsq3) * MicZooN * q10_mic
    grazingFlux_phy3 = (grazingFlux3 * fPhyN3) / food3
    grazingFlux_Dia3 = (grazingFlux3 * fDiaN3) / food3
    if coccos:
        grazingFlux_Cocco3 = (grazingFlux3 * fCoccoN3) / food3
        grazingFlux_Phaeo3 = (grazingFlux3 * fPhaeoN3) / food3

    # zooplankton respiration and losses (S:3900-3960)
    HetRespFlux = p.res_het * q10_mes_res * HetC
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

    # aggregation with diatom mucus (S:4241-4262), the Phaeocystis rate, the mortality (S:4284-4288)
    ql = recom_limiter(p.NMinSlope, p.NCmin_d, quota_dia)
    ql = jnp.minimum(ql, recom_limiter(p.SiMinSlope, p.SiCmin, qSiC))
    ql = jnp.minimum(ql, Fe / (p.k_Fe_d + Fe))
    aggregationRate = p.agg_PP * (1.0 - ql) * DiaN
    aggregationRate = aggregationRate + p.agg_PD * DetN + p.agg_PP * PhyN
    aggregationRate = aggregationRate + p.agg_PD * DetZ2N
    if coccos:
        aggregationRate = aggregationRate + p.agg_PP * CoccoN + p.agg_PP * PhaeoN
        agg_det = p.agg_PD * DetN
        agg_det = agg_det + p.agg_PD * DetZ2N
        aggregationrate_phaeo = jnp.where(p.agg_fac_phaeo_det != 1.0,
                                          aggregationRate - (1.0 - p.agg_fac_phaeo_det) * agg_det, aggregationRate)
        aggregationrate_phaeo = p.agg_fac_phaeo * aggregationrate_phaeo
        mortRate_phaeo = jnp.where(p.mort_phaeo > 0.0,
                                   jnp.where(Fe < p.k_Fe_p, p.mort_fac_fe_phaeo * p.mort_phaeo, p.mort_phaeo), 0.0)

    # carbon overflow rates (S:4294-4318)
    ovf_on = p.f_ovf > 0.0
    lim1 = lambda ncmin, q: 1.0 - recom_limiter(p.NMinSlope, ncmin, q)        # noqa: E731
    ovfRate = jnp.where(ovf_on, p.f_ovf * lim1(p.NCmin, quota) * Cphot, 0.0)
    ovfRate_dia = jnp.where(ovf_on, p.f_ovf * lim1(p.NCmin_d, quota_dia) * Cphot_dia, 0.0)
    if coccos:
        ovfRate_cocco = jnp.where(ovf_on, p.f_ovf * lim1(p.NCmin_c, quota_cocco) * Cphot_cocco, 0.0)
        ovfRate_phaeo = jnp.where(ovf_on, p.f_ovf * lim1(p.NCmin_p, quota_phaeo) * Cphot_phaeo, 0.0)
    res_ovf_d_eff = jnp.where(p.res_ovf_d < 0.0, p.res_ovf, p.res_ovf_d)
    res_ovf_c_eff = jnp.where(p.res_ovf_c < 0.0, p.res_ovf, p.res_ovf_c)
    res_ovf_p_eff = jnp.where(p.res_ovf_p < 0.0, p.res_ovf, p.res_ovf_p)
    ovr_on = (p.res_ovf > 0.0) | (res_ovf_d_eff > 0.0) | (res_ovf_c_eff > 0.0) | (res_ovf_p_eff > 0.0)
    ovrRate = jnp.where(ovr_on, p.res_ovf * lim1(p.NCmin, quota) * Cphot, 0.0)
    ovrRate_dia = jnp.where(ovr_on, res_ovf_d_eff * lim1(p.NCmin_d, quota_dia) * Cphot_dia, 0.0)
    if coccos:
        ovrRate_cocco = jnp.where(ovr_on, res_ovf_c_eff * lim1(p.NCmin_c, quota_cocco) * Cphot_cocco, 0.0)
        ovrRate_phaeo = jnp.where(ovr_on, res_ovf_p_eff * lim1(p.NCmin_p, quota_phaeo) * Cphot_phaeo, 0.0)

    # calcification (S:4390-4481) and the calcite losses (S:4511-4556)
    if coccos:
        PICPOCtemp = jnp.maximum(TINY, jnp.where(temp < 10.6, 0.104 * temp - 0.108, 1.0))
        PICPOCCO2 = co2_effect(hco3c, co2c, phc, p.a_co2_calc, p.b_co2_calc, p.c_co2_calc, p.d_co2_calc, p.Cunits)
        kdc = jnp.where(p.k_din_calc >= 0.0, p.k_din_calc, p.k_din_c)
        PICPOCN = jnp.maximum(TINY, -0.31 * (DIN / (DIN + kdc)) + 1.31)
        calcification = 1.0 * Cphot_cocco * CoccoC * PICPOCtemp * PICPOCN
        calcification = calcification * PICPOCCO2                               # CO2lim
        calc_loss_agg = aggregationRate * PhyCalc
        aux = recipQuota_cocco / (CoccoC + TINY) * PhyCalc
        calc_loss_gra = grazingFlux_Cocco * aux
        calc_loss_gra2 = grazingFlux_Cocco2 * aux
        calc_loss_gra3 = grazingFlux_Cocco3 * aux
    else:
        calcification = p.calc_prod_ratio * Cphot * PhyC
        calc_loss_agg = aggregationRate * PhyCalc
        aux = recipQuota / (PhyC + TINY) * PhyCalc
        calc_loss_gra = grazingFlux_phy * aux
        calc_loss_gra2 = grazingFlux_phy2 * aux
        calc_loss_gra3 = grazingFlux_phy3 * aux

    # ---- the sms (S:4650-6620); the coccos-off path drops the exact-zero is_coccos terms ----
    z0 = 0.0
    cN = (N_assim_cocco * CoccoC) if coccos else z0
    pN = (N_assim_phaeo * PhaeoC) if coccos else z0
    s = {}
    s[L.idin] = (-(N_assim - N_fix) * PhyC - N_assim_dia * DiaC
                 - cN - pN
                 + p.rho_N * arrFunc * O2Func * DON) * dt_b
    s[L.idic] = (-Cphot * PhyC - Cphot_dia * DiaC
                 - ((Cphot_cocco * CoccoC) if coccos else z0) - ((Cphot_phaeo * PhaeoC) if coccos else z0)
                 - calcification
                 + phyRespRate * PhyC + phyRespRate_dia * DiaC
                 + ((phyRespRate_cocco * CoccoC) if coccos else z0) + ((phyRespRate_phaeo * PhaeoC) if coccos else z0)
                 + p.rho_C1 * arrFunc * O2Func * EOC
                 + HetRespFlux + Zoo2RespFlux + MicZooRespFlux
                 + calc_diss * DetCalc
                 + calc_loss_gra * p.calc_diss_guts
                 + calc_loss_gra2 * p.calc_diss_guts
                 + calc_loss_gra3 * p.calc_diss_guts
                 + calc_diss2 * DetZ2Calc) * dt_b
    s[L.ialk] = (+1.0625 * (N_assim - N_fix) * PhyC + 1.0625 * N_assim_dia * DiaC
                 + 1.0625 * cN + 1.0625 * pN
                 - 1.0625 * p.rho_N * arrFunc * O2Func * DON
                 + 2.0 * calc_diss * DetCalc
                 + 2.0 * calc_loss_gra * p.calc_diss_guts
                 + 2.0 * calc_loss_gra2 * p.calc_diss_guts
                 + 2.0 * calc_loss_gra3 * p.calc_diss_guts
                 + 2.0 * calc_diss2 * DetZ2Calc
                 - 2.0 * calcification) * dt_b
    s[L.iphyn] = (+N_assim * PhyC
                  - p.lossN * limitFacN * PhyN
                  - aggregationRate * PhyN
                  - grazingFlux_phy - grazingFlux_phy2 - grazingFlux_phy3) * dt_b
    s[L.iphyc] = (+Cphot * PhyC
                  - phyRespRate * PhyC
                  - p.lossC * limitFacN * PhyC
                  - aggregationRate * PhyC
                  - grazingFlux_phy * recipQuota
                  - grazingFlux_phy2 * recipQuota
                  - grazingFlux_phy3 * recipQuota) * dt_b
    s[L.ipchl] = (+chlSynth * PhyC
                  - KOchl * PhyChl
                  - aggregationRate * PhyChl
                  - grazingFlux_phy * Chl2N
                  - grazingFlux_phy2 * Chl2N
                  - grazingFlux_phy3 * Chl2N) * dt_b
    # detritus, Grazing_detritus + 3zoo2det (S:4932-4940, 5107-5115)
    s[L.idetn] = (+grazingFlux_phy3 - grazingFlux_phy3 * p.grazEff3
                  + grazingFlux_Dia3 - grazingFlux_Dia3 * p.grazEff3
                  + ((grazingFlux_Cocco3 - grazingFlux_Cocco3 * p.grazEff3) if coccos else z0)
                  + ((grazingFlux_Phaeo3 - grazingFlux_Phaeo3 * p.grazEff3) if coccos else z0)
                  + aggregationRate * PhyN
                  + aggregationRate * DiaN
                  + ((aggregationRate * CoccoN) if coccos else z0)
                  + ((aggregationrate_phaeo * PhaeoN) if coccos else z0)
                  + MicZooLossFlux
                  - grazingFlux_Det * grazEff
                  - grazingFlux_Det2 * p.grazEff2
                  - p.reminN * arrFunc * O2Func * DetN) * dt_b
    s[L.idetc] = (+grazingFlux_phy3 * recipQuota * (1.0 - p.grazEff3)
                  + grazingFlux_Dia3 * recipQuota_dia * (1.0 - p.grazEff3)
                  + ((grazingFlux_Cocco3 * recipQuota_cocco * (1.0 - p.grazEff3)) if coccos else z0)
                  + ((grazingFlux_Phaeo3 * recipQuota_phaeo * (1.0 - p.grazEff3)) if coccos else z0)
                  + aggregationRate * PhyC
                  + aggregationRate * DiaC
                  + ((aggregationRate * CoccoC) if coccos else z0)
                  + ((aggregationrate_phaeo * PhaeoC) if coccos else z0)
                  + MicZooLossFlux * recipQZoo3
                  - grazingFlux_Det * recipDet * grazEff
                  - grazingFlux_Det2 * recipDet * p.grazEff2
                  - p.reminC * arrFunc * O2Func * DetC) * dt_b
    s[L.ihetn] = (+grazingFlux * grazEff
                  - grazingFlux_het2
                  - Mesfecalloss_n
                  - hetLossFlux
                  - p.lossN_z * HetN) * dt_b
    s[L.ihetc] = (+grazingFlux_phy * recipQuota * grazEff
                  + grazingFlux_Dia * recipQuota_dia * grazEff
                  + ((grazingFlux_Cocco * recipQuota_cocco * grazEff) if coccos else z0)
                  + ((grazingFlux_Phaeo * recipQuota_phaeo * grazEff) if coccos else z0)
                  + grazingFlux_miczoo * recipQZoo3 * grazEff
                  + grazingFlux_DetZ2 * recipDet2 * grazEff
                  + grazingFlux_Det * recipDet * grazEff
                  - grazingFlux_het2 * recipQZoo
                  - Mesfecalloss_c
                  - hetLossFlux * recipQZoo
                  - p.lossC_z * HetC
                  - HetRespFlux) * dt_b
    s[L.izoo2n] = (+grazingFlux2 * p.grazEff2
                   - Zoo2LossFlux
                   - p.lossN_z2 * Zoo2N
                   - Zoo2fecalloss_n) * dt_b
    s[L.izoo2c] = (+grazingFlux_phy2 * recipQuota * p.grazEff2
                   + grazingFlux_Dia2 * recipQuota_dia * p.grazEff2
                   + ((grazingFlux_Cocco2 * recipQuota_cocco * p.grazEff2) if coccos else z0)
                   + ((grazingFlux_Phaeo2 * recipQuota_phaeo * p.grazEff2) if coccos else z0)
                   + grazingFlux_het2 * recipQZoo * p.grazEff2
                   + grazingFlux_miczoo2 * recipQZoo3 * p.grazEff2
                   + grazingFlux_Det2 * recipDet * p.grazEff2
                   + grazingFlux_DetZ22 * recipDet2 * p.grazEff2
                   - Zoo2LossFlux * recipQZoo2
                   - p.lossC_z2 * Zoo2C
                   - Zoo2RespFlux
                   - Zoo2fecalloss_c) * dt_b
    s[L.imiczoon] = (+grazingFlux3 * p.grazEff3
                     - grazingFlux_miczoo
                     - grazingFlux_miczoo2
                     - MicZooLossFlux
                     - p.lossN_z3 * MicZooN) * dt_b
    s[L.imiczooc] = (+grazingFlux_phy3 * recipQuota * p.grazEff3
                     + grazingFlux_Dia3 * recipQuota_dia * p.grazEff3
                     + ((grazingFlux_Cocco3 * recipQuota_cocco * p.grazEff3) if coccos else z0)
                     + ((grazingFlux_Phaeo3 * recipQuota_phaeo * p.grazEff3) if coccos else z0)
                     - MicZooLossFlux * recipQZoo3
                     - grazingFlux_miczoo * recipQZoo3
                     - grazingFlux_miczoo2 * recipQZoo3
                     - p.lossC_z3 * MicZooC
                     - MicZooRespFlux) * dt_b
    s[L.idetz2n] = (+grazingFlux_phy2 * (1.0 - p.grazEff2)
                    + grazingFlux_Dia2 * (1.0 - p.grazEff2)
                    + ((grazingFlux_Cocco * (1.0 - grazEff)) if coccos else z0)
                    + ((grazingFlux_Cocco2 * (1.0 - p.grazEff2)) if coccos else z0)
                    + ((grazingFlux_Phaeo * (1.0 - grazEff)) if coccos else z0)
                    + ((grazingFlux_Phaeo2 * (1.0 - p.grazEff2)) if coccos else z0)
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
    s[L.idetz2c] = (+grazingFlux_phy2 * recipQuota * (1.0 - p.grazEff2)
                    + grazingFlux_Dia2 * recipQuota_dia * (1.0 - p.grazEff2)
                    + ((grazingFlux_Cocco * recipQuota_cocco * (1.0 - grazEff)) if coccos else z0)
                    + ((grazingFlux_Cocco2 * recipQuota_cocco * (1.0 - p.grazEff2)) if coccos else z0)
                    + ((grazingFlux_Phaeo * recipQuota_phaeo * (1.0 - grazEff)) if coccos else z0)
                    + ((grazingFlux_Phaeo2 * recipQuota_phaeo * (1.0 - p.grazEff2)) if coccos else z0)
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
    s[L.idetz2si] = (+grazingFlux_Dia2 * qSiN
                     + grazingFlux_Dia * qSiN
                     - reminSiT * DetZ2Si) * dt_b
    s[L.idetz2calc] = (+calc_loss_gra2 * (1.0 - p.calc_diss_guts)
                       + calc_loss_gra * (1.0 - p.calc_diss_guts)
                       - calc_diss2 * DetZ2Calc) * dt_b
    s[L.idon] = (+p.lossN * limitFacN * PhyN
                 + p.lossN_d * limitFacN_dia * DiaN
                 + ((p.lossN_c * limitFacN_cocco * CoccoN) if coccos else z0)
                 + ((p.lossN_p * limitFacN_phaeo * PhaeoN) if coccos else z0)
                 + p.reminN * arrFunc * O2Func * DetN
                 + p.reminN * arrFunc * O2Func * DetZ2N
                 + p.lossN_z * HetN
                 + p.lossN_z2 * Zoo2N
                 + p.lossN_z3 * MicZooN
                 - p.rho_N * arrFunc * O2Func * DON) * dt_b
    doc_src = (+p.lossC * limitFacN * PhyC
               + p.lossC_d * limitFacN_dia * DiaC
               + ((p.lossC_c * limitFacN_cocco * CoccoC) if coccos else z0)
               + ((p.lossC_p * limitFacN_phaeo * PhaeoC) if coccos else z0)
               + p.reminC * arrFunc * O2Func * DetC
               + p.reminC * arrFunc * O2Func * DetZ2C
               + p.lossC_z * HetC
               + p.lossC_z2 * Zoo2C
               + p.lossC_z3 * MicZooC) * dt_b
    if cdom:
        doc_src = (1.0 - FCDOM) * doc_src
    s[L.idoc] = (-p.rho_C1 * arrFunc * O2Func * EOC) * dt_b + doc_src
    s[L.idian] = (+N_assim_dia * DiaC
                  - p.lossN_d * limitFacN_dia * DiaN
                  - aggregationRate * DiaN
                  - grazingFlux_Dia - grazingFlux_Dia2 - grazingFlux_Dia3) * dt_b
    s[L.idiac] = (+Cphot_dia * DiaC
                  - p.lossC_d * limitFacN_dia * DiaC
                  - phyRespRate_dia * DiaC
                  - aggregationRate * DiaC
                  - grazingFlux_Dia * recipQuota_dia
                  - grazingFlux_Dia2 * recipQuota_dia
                  - grazingFlux_Dia3 * recipQuota_dia) * dt_b
    s[L.idchl] = (+ChlSynth_dia * DiaC
                  - KOchl_dia * DiaChl
                  - aggregationRate * DiaChl
                  - grazingFlux_Dia * Chl2N_dia
                  - grazingFlux_Dia2 * Chl2N_dia
                  - grazingFlux_Dia3 * Chl2N_dia) * dt_b
    s[L.idiasi] = (+Si_assim * DiaC
                   - p.lossN_d * limitFacN_dia * DiaSi
                   - aggregationRate * DiaSi
                   - grazingFlux_Dia * qSiN
                   - grazingFlux_Dia2 * qSiN
                   - grazingFlux_Dia3 * qSiN) * dt_b
    if coccos:
        s[L.icocn] = (+N_assim_cocco * CoccoC
                      - p.lossN_c * limitFacN_cocco * CoccoN
                      - aggregationRate * CoccoN
                      - grazingFlux_Cocco
                      - grazingFlux_Cocco2
                      - grazingFlux_Cocco3) * dt_b
        s[L.icocc] = (+Cphot_cocco * CoccoC
                      - p.lossC_c * limitFacN_cocco * CoccoC
                      - phyRespRate_cocco * CoccoC
                      - aggregationRate * CoccoC
                      - grazingFlux_Cocco * recipQuota_cocco
                      - grazingFlux_Cocco2 * recipQuota_cocco
                      - grazingFlux_Cocco3 * recipQuota_cocco) * dt_b
        s[L.icchl] = (+ChlSynth_cocco * CoccoC
                      - KOchl_cocco * CoccoChl
                      - aggregationRate * CoccoChl
                      - grazingFlux_Cocco * Chl2N_cocco
                      - grazingFlux_Cocco2 * Chl2N_cocco
                      - grazingFlux_Cocco3 * Chl2N_cocco) * dt_b
        s[L.iphan] = (+N_assim_phaeo * PhaeoC
                      - p.lossN_p * limitFacN_phaeo * PhaeoN
                      - aggregationrate_phaeo * PhaeoN
                      - grazingFlux_Phaeo
                      - grazingFlux_Phaeo2
                      - grazingFlux_Phaeo3) * dt_b
        s[L.iphac] = (+Cphot_phaeo * PhaeoC
                      - p.lossC_p * limitFacN_phaeo * PhaeoC
                      - phyRespRate_phaeo * PhaeoC
                      - aggregationrate_phaeo * PhaeoC
                      - grazingFlux_Phaeo * recipQuota_phaeo
                      - grazingFlux_Phaeo2 * recipQuota_phaeo
                      - grazingFlux_Phaeo3 * recipQuota_phaeo) * dt_b
        s[L.iphachl] = (+ChlSynth_phaeo * PhaeoC
                        - KOchl_phaeo * PhaeoChl
                        - aggregationrate_phaeo * PhaeoChl
                        - grazingFlux_Phaeo * Chl2N_phaeo
                        - grazingFlux_Phaeo2 * Chl2N_phaeo
                        - grazingFlux_Phaeo3 * Chl2N_phaeo) * dt_b
        # Phaeocystis mortality (S:6212-6218): mortRate_phaeo is 0 unless mort_phaeo > 0
        s[L.iphan] = s[L.iphan] - mortRate_phaeo * PhaeoN * dt_b
        s[L.iphac] = s[L.iphac] - mortRate_phaeo * PhaeoC * dt_b
        s[L.iphachl] = s[L.iphachl] - mortRate_phaeo * PhaeoChl * dt_b
        s[L.idetn] = s[L.idetn] + mortRate_phaeo * PhaeoN * dt_b
        s[L.idetc] = s[L.idetc] + mortRate_phaeo * PhaeoC * dt_b
    s[L.idetsi] = (+aggregationRate * DiaSi
                   + p.lossN_d * limitFacN_dia * DiaSi
                   + grazingFlux_Dia3 * qSiN
                   + grazingFlux_Dia * qSiN * (1.0 - 1.0)                 # is_3zoo2det = 1: an exact zero
                   - reminSiT * DetSi) * dt_b
    s[L.isi] = (-Si_assim * DiaC
                + reminSiT * DetSi
                + reminSiT * DetZ2Si) * dt_b
    s[L.ife] = (p.Fe2N * (-N_assim * PhyC - N_assim_dia * DiaC
                          - cN - pN
                          + p.lossN * limitFacN * PhyN
                          + p.lossN_d * limitFacN_dia * DiaN
                          + ((p.lossN_c * limitFacN_cocco * CoccoN) if coccos else z0)
                          + ((p.lossN_p * limitFacN_phaeo * PhaeoN) if coccos else z0)
                          + p.reminN * arrFunc * O2Func * DetN
                          + p.reminN * arrFunc * O2Func * DetZ2N
                          + p.lossN_z * HetN
                          + p.lossN_z2 * Zoo2N
                          + p.lossN_z3 * MicZooN)
                - p.kScavFe * DetC * FreeFe
                - p.kScavFe * DetZ2C * FreeFe) * dt_b
    if coccos:
        s[L.iphycal] = (+calcification
                        - p.lossC_c * limitFacN_cocco * PhyCalc
                        - phyRespRate_cocco * PhyCalc
                        - calc_loss_agg
                        - calc_loss_gra
                        - calc_loss_gra2
                        - calc_loss_gra3) * dt_b
        s[L.idetcal] = (+p.lossC_c * limitFacN_cocco * PhyCalc
                        + phyRespRate_cocco * PhyCalc
                        + calc_loss_agg
                        + calc_loss_gra3
                        - calc_loss_gra3 * p.calc_diss_guts
                        - calc_diss * DetCalc) * dt_b
    else:
        s[L.iphycal] = (+calcification
                        - p.lossC * limitFacN * PhyCalc
                        - phyRespRate * PhyCalc
                        - calc_loss_agg
                        - calc_loss_gra
                        - calc_loss_gra2
                        - calc_loss_gra3) * dt_b
        s[L.idetcal] = (+p.lossC * limitFacN * PhyCalc
                        + phyRespRate * PhyCalc
                        + calc_loss_agg
                        + calc_loss_gra3
                        - calc_loss_gra3 * p.calc_diss_guts
                        - calc_diss * DetCalc) * dt_b
    if cdom:
        cdom_photo_rate = PHOT_CDOM * jnp.minimum(PARave / KPHOT_CDOM, 1.0)
        src = (+p.lossC * limitFacN * PhyC
               + p.lossC_d * limitFacN_dia * DiaC
               + ((p.lossC_c * limitFacN_cocco * CoccoC) if coccos else z0)
               + ((p.lossC_p * limitFacN_phaeo * PhaeoC) if coccos else z0)
               + p.reminC * arrFunc * O2Func * DetC
               + p.reminC * arrFunc * O2Func * DetZ2C
               + p.lossC_z * HetC
               + p.lossC_z2 * Zoo2C
               + p.lossC_z3 * MicZooC) * dt_b
        s[L.icdom] = (-RHO_CDOM * arrFunc * cdomC - cdom_photo_rate * cdomC) * dt_b + FCDOM * src
    # carbon overflow (S:6545-6579): the rates are 0 unless the switch values are set
    ovfFlux = ovfRate * PhyC + ovfRate_dia * DiaC
    s[L.iphyc] = s[L.iphyc] - ovfRate * PhyC * dt_b
    s[L.idiac] = s[L.idiac] - ovfRate_dia * DiaC * dt_b
    if coccos:
        ovfFlux = ovfFlux + ovfRate_cocco * CoccoC + ovfRate_phaeo * PhaeoC
        s[L.icocc] = s[L.icocc] - ovfRate_cocco * CoccoC * dt_b
        s[L.iphac] = s[L.iphac] - ovfRate_phaeo * PhaeoC * dt_b
    ovfFrac = 1.0
    if cdom and sw.ovf_to_cdom:
        ovfFrac = 1.0 - FCDOM
        s[L.icdom] = s[L.icdom] + FCDOM * ovfFlux * dt_b
    s[L.idoc] = s[L.idoc] + ovfFrac * ovfFlux * dt_b
    ovrFlux = ovrRate * PhyC + ovrRate_dia * DiaC
    s[L.iphyc] = s[L.iphyc] - ovrRate * PhyC * dt_b
    s[L.idiac] = s[L.idiac] - ovrRate_dia * DiaC * dt_b
    if coccos:
        ovrFlux = ovrFlux + ovrRate_cocco * CoccoC + ovrRate_phaeo * PhaeoC
        s[L.icocc] = s[L.icocc] - ovrRate_cocco * CoccoC * dt_b
        s[L.iphac] = s[L.iphac] - ovrRate_phaeo * PhaeoC * dt_b
    s[L.idic] = s[L.idic] + ovrFlux * dt_b
    oxy_ovr = -ovrFlux * REDO2C * dt_b
    s[L.ioxy] = (+Cphot * PhyC + Cphot_dia * DiaC
                 + ((Cphot_cocco * CoccoC) if coccos else z0) + ((Cphot_phaeo * PhaeoC) if coccos else z0)
                 - phyRespRate * PhyC - phyRespRate_dia * DiaC
                 - ((phyRespRate_cocco * CoccoC) if coccos else z0) - ((phyRespRate_phaeo * PhaeoC) if coccos else z0)
                 - p.rho_C1 * arrFunc * O2Func * EOC
                 - HetRespFlux
                 - Zoo2RespFlux
                 - MicZooRespFlux) * REDO2C * dt_b + oxy_ovr
    if L.dicremin:
        s[L.idicremin] = (+p.rho_C1 * arrFunc * O2Func * EOC) * dt_b
    ntr = L.ntr
    assert sorted(s) == list(range(ntr)), (sorted(s), ntr)
    sms = jnp.stack([jnp.where(wet, s[i], 0.0) for i in range(ntr)], axis=-1)

    # ---- REcoM_Forcing's update and clip (F:1004-1033) ----
    raw = state + sms
    new = jnp.maximum(TINY, raw)
    floors = {L.ipchl: TINY_CHL, L.iphyn: dz.tiny_N, L.iphyc: dz.tiny_C, L.idchl: TINY_CHL,
              L.idian: dz.tiny_N_d, L.idiac: dz.tiny_C_d, L.idiasi: tiny_Si}
    if coccos:
        floors.update({L.icchl: TINY_CHL, L.icocn: dz.tiny_N_c, L.icocc: dz.tiny_C_c,
                       L.iphachl: TINY_CHL, L.iphan: dz.tiny_N_p, L.iphac: dz.tiny_C_p})
    cols = [jnp.maximum(floors[i], new[..., i]) if i in floors else new[..., i] for i in range(ntr)]
    if holds():
        cols = list(jax.lax.optimization_barrier(tuple(cols)))
    new = jnp.where(wet[..., None], jnp.stack(cols, axis=-1), state)
    clip = jnp.where(wet[..., None], new - raw, 0.0)

    # ---- benthos (S:7780-7900): calc_diss_ben of the last wet layer ----
    last = jnp.clip(nlev - 1, 0, nl1 - 1)
    cdb = jnp.take_along_axis(calc_diss_ben, last[:, None], axis=1)[:, 0]
    rates = jnp.stack([jnp.full_like(cdb, p.decayRateBenN), jnp.full_like(cdb, p.decayRateBenC),
                       jnp.full_like(cdb, p.decayRateBenSi), cdb], axis=-1)
    decay = rates * benthos
    benthos_new = benthos - decay * dt_b

    # ---- diagnostics (S:7163-7676, MAIN:470-537, F:1122-1132); recipbiostep = 1 ----
    z = lambda x: jnp.where(wet, x, 0.0)                                       # noqa: E731
    d3 = dict(
        aggn=aggregationRate * PhyC, aggd=aggregationRate * DiaC,
        docexn=(p.lossC * limitFacN + ovfRate) * PhyC, docexd=(p.lossC_d * limitFacN_dia + ovfRate_dia) * DiaC,
        respn=phyRespRate * PhyC + ovrRate * PhyC, respd=phyRespRate_dia * DiaC + ovrRate_dia * DiaC,
        NPPn3D=+Cphot * PhyC - phyRespRate * PhyC, NPPd3D=+Cphot_dia * DiaC - phyRespRate_dia * DiaC,
        calcdiss=calc_diss * DetCalc, calcif=calcification,
        respmeso=HetRespFlux, respmacro=Zoo2RespFlux, respmicro=MicZooRespFlux,
        TPhyCO2=PhyCO2, TDiaCO2=DiaCO2, TCphotLigLim_phyto=lightlim, TCphot_phyto=Cphot,
        TCphotLigLim_diatoms=lightlim_dia, TCphot_diatoms=Cphot_dia,
        TTemp_phyto=Temp_phyto if coccos else jnp.zeros_like(temp),
        TTemp_diatoms=Temp_diatoms if coccos else jnp.zeros_like(temp),
        TqLF_phyto=TqLF_phyto, TqLF_diatoms=TqLF_diatoms,
        TSi_assimDia=Si_assim if coccos else jnp.zeros_like(temp))
    vert = dict(
        NPPn=d3["NPPn3D"], NPPd=d3["NPPd3D"], GPPn=+Cphot * PhyC, GPPd=+Cphot_dia * DiaC,
        NNAn=+N_assim * PhyC - p.lossN * limitFacN * PhyN,
        NNAd=+N_assim_dia * DiaC - p.lossN * limitFacN_dia * DiaN,          # lossN, not lossN_d (S:7314)
        Chldegn=KOchl, Chldegd=KOchl_dia)
    if coccos:
        d3.update(
            aggc=aggregationRate * CoccoC, aggp=aggregationrate_phaeo * PhaeoC,
            docexc=(p.lossC_c * limitFacN_cocco + ovfRate_cocco) * CoccoC,
            docexp=(p.lossC_p * limitFacN_phaeo + ovfRate_phaeo) * PhaeoC,
            respc=phyRespRate_cocco * CoccoC + ovrRate_cocco * CoccoC,
            respp=phyRespRate_phaeo * PhaeoC + ovrRate_phaeo * PhaeoC,
            NPPc3D=+Cphot_cocco * CoccoC - phyRespRate_cocco * CoccoC,
            NPPp3D=+Cphot_phaeo * PhaeoC - phyRespRate_phaeo * PhaeoC,
            TCoccoCO2=CoccoCO2, TPhaeoCO2=PhaeoCO2,
            TCphotLigLim_cocco=lightlim_cocco, TCphot_cocco=Cphot_cocco,
            TCphotLigLim_phaeo=lightlim_phaeo, TCphot_phaeo=Cphot_phaeo,
            TTemp_cocco=Temp_cocco, TTemp_phaeo=Temp_phaeo, TqLF_cocco=qlim_c, TqLF_phaeo=qlim_p)
        vert.update(
            NPPc=d3["NPPc3D"], NPPp=d3["NPPp3D"], GPPc=+Cphot_cocco * CoccoC, GPPp=+Cphot_phaeo * PhaeoC,
            NNAc=+N_assim_cocco * CoccoC - p.lossN * limitFacN_cocco * CoccoN,     # lossN (S:7320)
            NNAp=+N_assim_phaeo * PhaeoC - p.lossN * limitFacN_phaeo * PhaeoN,
            Chldegc=KOchl_cocco, Chldegp=KOchl_phaeo)
    d3 = {k: z(v) for k, v in d3.items()}
    d2 = {k: jnp.sum(z(v) * z(thick), axis=1) for k, v in vert.items()}

    return SmsForkResult(state=new, carb=carb, sms=sms, clip=clip, par=z(PARave), benthos=benthos_new,
                         decay_benthos=decay, refreshed=refresh, diag3d=d3, diag2d=d2)
