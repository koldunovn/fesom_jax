# Lane E: the fork's 4-PFT ecosystem (coccolithophores, Phaeocystis) and CDOM in fesom-jax

Parent plan: `docs/plans/20261028-recom-spectral-port.md`. Fortran: the fork worktree
`/albedo/work/user/sloza/SPECTRAL/recom_spectral_r692` (S = `src/int_recom/recom_sms.F90`, F = `recom_forcing.F90`,
K = `recom_sinking.F90`, M = `recom_modules.F90`, MAIN = `recom_main.F90`, OAT = `src/oce_ale_tracer.F90`).
This file is the read-only inventory made 2026-10-28 (an agent's pass over the sources; **verify every line
against the source while porting**, line numbers move). Gate: the broadband 4-PFT twin dumps (lane 0,
`install_nowb/bin/fesom.x`), G2 on `recom_in -> recom_out` per dumped step, rel 1e-11.

Three facts first:

- The fork has no `recom_sms_init.F90` / `recom_sms_update.F90`: the unpack and the per-layer computation are
  inline in `REcoM_sms` (S:499-1170); the state update and clip are in F:1004-1048 after `call REcoM_sms` (F:971).
- CDOM exists only in a spectral build (`icdom` assigned under `__RECOM_WAVEBANDS`, M:787-882; `RECOM_CDOM` M:613).
  The broadband path has no CDOM tracer and no `(1-fcdom)` factor on DOC. Section 7 is spectral code (lane W).
- The N:C clamp `quota = min(max(quota, NCmin_X), NCmax_X)` is spectral-only (S:847-850, 906-909, 976-979,
  1018-1021), for all four PFTs. Decision (user): mirror it in the spectral program.

Switches: `is_coccos = 1.0` if `enable_coccos` else 0 (recom_init.F90:745-748). 4p3z2d slots (1-based):
icocn 29, icocc 30, icchl 31, iphan 32, iphac 33, iphachl 34, imiczoon 35, imiczooc 36; IDs 1029-1034
(M:771-785). Without 3zoo2det: 23..28, IDs 1023-1028 (M:812-827), which collide with the DetZ2 IDs hard-coded
in K (do not support that configuration).

## 0. Before the layer loop (S:369-391; repeated F:265-271)
`tiny_N_c = tiny_chl/chl2N_max_c; tiny_C_c = tiny_N_c/NCmax_c; tiny_N_p = tiny_chl/chl2N_max_p; tiny_C_p = tiny_N_p/NCmax_p`
(`&paassimilation` M:318-346: Chl2N_max_c 3.5, Chl2N_max_p 3.5, NCmax_c 0.15, NCmax_p 0.2). biostep loop S:474,
layer loop S:499.

## 1. Unpack, quotas, temperature, limiters
- Unpack (S:613-641, `enable_coccos`): `CoccoN = max(tiny_N_c, state+sms)`, `CoccoC = max(tiny_C_c, .)`,
  `CoccoChl = max(tiny_chl, .)`; Phaeo likewise with `_p`. CDOM `cdomC = max(tiny, .)` S:549-556 (spectral).
  PhyCalc stays `max(tiny, .)` (S:572) and is the coccolith calcite pool with coccos.
- Quotas (S:953-1036): `quota_cocco = CoccoN/CoccoC` (spectral clamp S:976-979), `recipQuota_cocco`,
  `Chl2C_cocco = CoccoChl/CoccoC`, `Chl2N_cocco = CoccoChl/CoccoN`,
  `CHL2C_plast_cocco = Chl2C_cocco*(quota_cocco/(quota_cocco-NCmin_c))`; Phaeo S:1017-1034.
- Temperature (S:1234-1311, the whole block only with `enable_coccos`): `Temp_phyto = exp(ord_phy + expon_phy*T)`,
  `Temp_diatoms = exp(ord_d + expon_d*T)`; cocco (S:1265-1273) `T < 5: tiny else max(exp(ord_cocco + expon_cocco*T), tiny)`;
  Phaeo Blanchard (S:1293-1309): `T >= Tmax_phaeo: tiny else max(uopt_phaeo*((Tmax-T)/(Tmax-Topt))**beta_phaeo *
  exp(-beta_phaeo*(Topt-T)/(Tmax-Topt)), tiny)`. Diagnostics VTTemp_*. `&paArrhenius` M:248 (Tmax_phaeo 16,
  uopt_phaeo 0.7328, ord_cocco -0.2310, expon_cocco 0.0327). ord_phaeo/expon_phaeo, P_cm_c/P_cm_p, sCoccoNsq*/
  sPhaeoNsq* are unused. No Arrhenius factor in cocco/phaeo growth; `arrFunc` still in remineralisation and CDOM.
- CO2 effects (S:2181-2218, `enable_coccos`, always computed): `CoccoCO2 = a_co2_cocco*HCO3*Cunits/(b_co2_cocco +
  HCO3*Cunits) - exp(-c_co2_cocco*CO2*Cunits) - d_co2_cocco*10**(-pH)`, clamped to [0, 3]; VTCoccoCO2. Phaeo alike.
- Limiters (S:2434-2464): `qlimitFac = min(recom_limiter(NMinSlope, NCmin_c, quota_cocco), Fe/(k_Fe_c+Fe))`;
  `pMax_cocco = qlimitFac*Temp_cocco`; VTqlimitFac_cocco. Phaeo with NCmin_p, k_Fe_p, Temp_phaeo. No Si limitation.

## 2. Broadband light and photosynthesis
- PARave (S:1466-1569, the `#else` of WAVEBANDS). `par_layer_mean` (M:283, default .true.): `Chlave = PhyChl+DiaChl
  (+CoccoChl+PhaeoChl)`, `kappa = k_w + a_chl*Chlave`, `kappastar = kappa/cosAI`, `kdzLower = kdzUpper +
  kappastar*thick(k)`, `PARave = max(tiny, (SurfSR*exp(-kdzUpper) - SurfSR*exp(-kdzLower))/(kappastar*thick(k)))`,
  `kdzUpper = kdzLower`; `kdzUpper = 0` at each biostep start (S:478). Otherwise the interface form (S:1490-1567,
  what sms.py implements) with the cocco/phaeo Chl added at S:1507-1510, 1526-1528.
- Cphot_cocco (S:2654-2676): 0 if `pMax_cocco < tiny` or NaN; else `pMax_cocco*(1 - exp(-alfa_c*Chl2C_cocco*PARave/
  pMax_cocco))`, VTCphotLigLim_cocco = Cphot/pMax, `if (CO2lim) Cphot_cocco *= CoccoCO2`; `< tiny -> 0`; then
  unguarded `if (beta_inh_c > 0 .and. Cphot >= tiny .and. pMax >= tiny) Cphot *= exp(-beta_inh_c*CHL2C_cocco*PARave/
  pMax_cocco)`; VTCphot_cocco. Phaeo S:2712-2734 (alfa_p, PhaeoCO2, beta_inh_p). alfa_c = alfa_p = 0.10, beta_inh_* 0.
  VTCphotLigLim keeps its previous value when dark.

## 3. Uptake, chlorophyll, respiration, N fixation, overflow
- N assimilation (S:3003-3015): `N_assim_cocco = V_cm_fact_c*pMax_cocco*NCUptakeRatio_c*recom_limiter(NMaxSlope,
  quota_cocco, NCmax_c)*DIN/(DIN+k_din_c)`; Phaeo with `_p`. `nfix_*` (S:2982-2995) is small phytoplankton only.
- Chl synthesis (S:3116-3133): `if (PARave >= tiny .and. not NaN) ChlSynth_cocco = N_assim_cocco*Chl2N_max_c*
  min(1, Cphot_cocco/(alfa_c*Chl2C_cocco*PARave))`.
- Degradation (S:2777-2780, 2855-2890): `KOchl_cocco = deg_Chl_c`; with `use_photodamage`: `deg_Chl_c*0.1` if pMax <
  tiny or NaN, else `max(deg_Chl_c*0.1, deg_Chl_c*(1 - exp(-alfa_c*CHL2C_plast_cocco*PARave/pMax_cocco)))`.
- Respiration (S:3154-3175): `res_phy_const ? res_phy_c + biosynth*N_assim_cocco : res_phy_c*limitFacN_cocco +
  biosynth*N_assim_cocco` (res_phy_c 0.0075, res_phy_p 0.008).
- Overflow (S:4294-4318, applied S:6545-6579): `f_ovf > 0`: `ovfRate_X = f_ovf*(1 - recom_limiter(NMinSlope, NCmin_X,
  quota_X))*Cphot_X`; `res_ovf_X_eff = res_ovf_X` or `res_ovf` if < 0; `ovrRate_X = res_ovf_X_eff*(1-limiter)*Cphot_X`.
  `ovfFlux += ovfRate_cocco*CoccoC + ovfRate_phaeo*PhaeoC`, `sms(icocc/iphac) -= ovfRate*C*dt_b`, `sms(idoc) +=
  ovfFrac*ovfFlux*dt_b` (ovfFrac 1; spectral+CDOM+ovf_to_cdom: 1-fcdom and `sms(icdom) += fcdom*ovfFlux*dt_b`);
  `ovrFlux`: `sms(idic) += ovrFlux*dt_b`, `sms(ioxy) -= ovrFlux*redO2C*dt_b`. PhyCalc untouched.

## 4. Calcification and calcite
- Calcification (S:4390-4481): with coccos `PICPOCtemp = max(tiny, T < 10.6 ? 0.104*T - 0.108 : 1)`; `PICPOCCO2 =
  a_co2_calc*HCO3*Cunits/(b_co2_calc + HCO3*Cunits) - exp(-c_co2_calc*CO2*Cunits) - d_co2_calc*10**(-pH)` clamped
  [0, 3]; `PICPOCN = max(tiny, -0.31*DIN/(DIN + (k_din_calc >= 0 ? k_din_calc : k_din_c)) + 1.31)`;
  `calcification = Cphot_cocco*CoccoC*PICPOCtemp*PICPOCN`, `if (CO2lim) *= PICPOCCO2`. Without coccos
  `calc_prod_ratio*Cphot*PhyC` (S:4479). `k_din_calc` default -1 (M:568).
- Losses (S:4511-4556): `calc_loss_agg = aggregationrate*PhyCalc` (the shared rate); with coccos `aux =
  recipQuota_Cocco/(CoccoC+tiny)*PhyCalc`, `calc_loss_gra(2,3) = grazingFlux_Cocco(2,3)*aux`; without,
  `recipQuota/(PhyC+tiny)` with grazingFlux_phy(2,3).
- PhyCalc sms (S:6350-6388): `+calcification - lossC_c*limitFacN_cocco*PhyCalc - phyRespRate_cocco*PhyCalc -
  calc_loss_agg - calc_loss_gra - (calc_loss_gra2 + calc_loss_gra3)*is_3zoo2det`; 2-PFT with lossC/limitFacN/phyRespRate.
- DetCalc (S:6402-6420, coccos + 3zoo): `+lossC_c*limitFacN_cocco*PhyCalc + phyRespRate_cocco*PhyCalc + calc_loss_agg
  + calc_loss_gra3 - calc_loss_gra3*calc_diss_guts - calc_diss*DetCalc`; 2-PFT S:6445-6458.
- DetZ2Calc (S:5704-5714), DIC (S:4696-4739) and Alk (S:4768-4799) formulas unchanged, inputs differ; dissolution
  (S:2260-2340) independent of coccos.

## 5. Grazing
- Mesozoo (S:3227-3443): variable preference `aux += pzCocco*CoccoN + pzPhaeo*PhaeoN`, `varpzCocco = pzCocco*CoccoN/aux`,
  `fCoccoN = varpzCocco*CoccoN` (fixed: `pzCocco*CoccoN`); `food += fCoccoN + fPhaeoN` (S:3344-3346); `grazingFlux_Cocco =
  grazingFlux*fCoccoN/food`; `grazEff = gfin + 1/(0.2*food+2)` includes them (S:3419); `grazingFluxcarbon_mes +=
  grazingFlux_Cocco*recipQuota_Cocco*grazEff + Phaeo...` (S:3439-3443) -> `mesfecalloss_c` (S:4163).
- Macrozoo (S:3520-3695): pzCocco2/pzPhaeo2 -> grazingFlux_Cocco2/Phaeo2, `grazingFluxcarbonzoo2 += ...*grazEff2`
  -> `Zoo2fecalloss_c` (S:4155). Microzoo (S:3775-3876): pzCocco3 (0), pzPhaeo3 (1.0) -> _Cocco3/_Phaeo3, food3.
  Defaults pzCocco 0.5, pzCocco2 0.2, pzPhaeo2 0.5 (M:364, 386-389, 410-414).
- Sinks/receivers (Grazing_detritus, 3zoo branch): CoccoN (S:5988-6007) `+N_assim_cocco*CoccoC - lossN_c*limitFacN_cocco*
  CoccoN - aggregationRate*CoccoN - grazingFlux_Cocco - (Cocco2 + Cocco3)*is_3zoo2det`; CoccoC (S:6021-6044)
  `+Cphot_cocco*CoccoC - lossC_c*limitFacN_cocco*CoccoC - phyRespRate_cocco*CoccoC - aggregationRate*CoccoC -
  grazingFlux_cocco*recipQuota_cocco - (Cocco2+Cocco3)*recipQuota_cocco*is_3zoo2det` (abort if sms > 100, S:6049-6067);
  CoccoChl (S:6080-6099) `+ChlSynth_cocco*CoccoC - KOchl_cocco*CoccoChl - aggregationRate*CoccoChl - grazingFlux_cocco*
  Chl2N_cocco - (2,3)*Chl2N_cocco*is_3zoo2det`; PhaeoN/C/Chl (S:6117-6205) identical with `aggregationrate_phaeo`.
  HetC (S:5281-5282) `+grazingFlux_Cocco*recipQuota_Cocco*grazEff*is_coccos + Phaeo`; Zoo2C (S:5372-5373) grazEff2;
  MicZooC (S:5459-5460) grazEff3. DetN (S:4932-4940) `+(grazingFlux_Cocco3 - Cocco3*grazEff3)*is_coccos + Phaeo3 +
  aggregationRate*CoccoN*is_coccos + aggregationrate_phaeo*PhaeoN*is_coccos`; DetC (S:5107-5115) with recipQuota and
  (1-grazEff3); DetZ2N (S:5503-5506) `+grazingFlux_Cocco*(1-grazEff) + Cocco2*(1-grazEff2) + Phaeo...`; DetZ2C
  (S:5598-5601) times recipQuota. Other configurations: DetN 4965-5042, DetC 5141-5214, DetZ2N 5546-5549, DetZ2C 5641-5644.

## 6. Aggregation, mortality, other tracers
- Aggregation (S:4241-4279): shared kernel `aggregationrate += agg_PP*CoccoN + agg_PP*PhaeoN` (S:4248-4252);
  `aggregationrate_phaeo = aggregationrate`; `if (agg_fac_phaeo_det /= 1) aggregationrate_det = agg_PD*DetN (+agg_PD*
  DetZ2N); aggregationrate_phaeo = aggregationrate - (1-agg_fac_phaeo_det)*aggregationrate_det`; `*= agg_fac_phaeo`.
- Mortality (S:4284-4288, 6212-6218): `mort_phaeo > 0: mortRate = mort_phaeo (Fe < k_Fe_p: *mort_fac_fe_phaeo)`;
  `sms(iphan/iphac/iphachl) -= mortRate*X*dt_b; sms(idetn) += mortRate*PhaeoN*dt_b; sms(idetc) += ...PhaeoC`.
- `*is_coccos` terms: DIN (S:4658-4659) `-N_assim_Cocco*CoccoC - N_assim_Phaeo*PhaeoC`; DIC (S:4703-4717)
  `-Cphot*C + phyRespRate*C` for both; Alk (S:4775-4776) `+1.0625*N_assim*C`; DON (S:5740-5741) `+lossN_c*limitFacN_
  Cocco*CoccoN + lossN_p*...`; DOC (S:5783-5784) `+lossC_c*limitFacN_cocco*CoccoC + lossC_p*...`; Fe (S:6308-6316)
  `Fe2N*(-N_assim*C + lossN*limitFacN*N)`; O2 (S:6602-6610) `(+Cphot*C - phyRespRate*C)*redO2C`. lossN_c 0.05, lossC_c 0.10.

## 7. CDOM (spectral only, lane W)
S:6493-6538: `cdom_photo_rate = phot_cdom*min(PARave/kphot_CDOM, 1)`; `sms(icdom) = fcdom*(lossC*limitFacN*phyC +
lossC_d*limitFacN_dia*DiaC + lossC_c*...*is_coccos + lossC_p*...*is_coccos + reminC*arrFunc*O2Func*DetC +
reminC*arrFunc*O2Func*DetZ2C*is_3zoo2det + lossC_z*HetC + lossC_z2*Zoo2C*is_3zoo2det + lossC_z3*MicZooC*is_3zoo2det)*dt_b
+ (-rho_cdom*arrFunc*cdomC - cdom_photo_rate*cdomC)*dt_b`; DOC `sms(idoc) = (1-fcdom)*sms(idoc)` before the rho_c1
sink (S:5798-5803). Quirk: both reassign the whole accumulated `sms` (exact only for biostep 1, which is asserted).
Constants fixed in `module REcoM_spectral` (M:2678-2681): fcdom 0.02, rho_cdom 0.003, phot_cdom 0.167,
kphot_cdom 13.51. Clip F:1029-1033 `max(tiny, .)`. `cdomC` S:553-555. Overflow split S:6556-6559.

## 8. Sinking (K)
- `ver_sinking_recom` (K:416-700): `Vsink = VCocco` for IDs 1029-1031 (literal, K:504-508); `VPhaeo` for
  `recom_phaeo_tracer_id` (K:512-513); PhyCalc `VCocco` with coccos else VPhy (K:517-522); VCocco = VPhaeo = 0 default.
  Sinks only if `Vsink > 0.1` (K:551). `allow_var_sinking`: `-(Vdet_a*|zbar| + Vsink)/86400`. Phaeo override (K:595-598):
  `fe_up = max(DFe(max(nzmin, nz-1), n), 0); Wvel = -VPhaeo*(1 - fe_up/(fe_up+k_Fe_p))/86400`. Ballasting only for
  1007/1008/1017/1021 and 1025-1028 (precedence bug K:601-605: `enable_3zoo2det .and. ID==1025 .or. ...`).
- `ver_sinking_recom_benthos` (K:49-265): Vben from det/phy/dia ID lists only (M:69-71): cocco gets Vben 0 (+Vdet_a
  term), and cocco N/C are not accumulated (bottom outflux lost; flag). PhyCalc uses **VPhy** here (VCocco in the
  water column; flag), into pool 4. Phaeo (`enable_coccos .and. VPhaeo > 0.1`, K:97-98): `Vben(nz) = VPhaeo*(1 -
  fe_loc/(fe_loc+k_Fe_p))` with the LOCAL layer's Fe, no depth term; `aux = -trarr*Vben*(area(nz)-area(nz+1))`;
  PhaeoN -> pool 1 (K:170), PhaeoC -> pool 2 (K:179), PhaeoChl not accumulated. `recom_sinking_tracer_id` (M:63-68)
  includes 1029-1034, not CDOM.

## 9. Diagnostics, update, clip
- Per layer (S:7163-7676, `Diags`, `*recipbiostep`): vertNPPc `+= Cphot_cocco*CoccoC - PhyRespRate_cocco*CoccoC`;
  vertGPPc `+= Cphot_cocco*CoccoC`; vertNNAc `+= N_assim_cocco*CoccoC - lossN*limitFacN_cocco*CoccoN` (quirk: `lossN`,
  as NNAd); vertChldegc `+= KOchl_cocco`; vertaggc `+= aggregationrate*CoccoC` (aggp with _phaeo); vertdocexc `+=
  (lossC_c*limitFacN_cocco + ovfRate_cocco)*CoccoC`; vertrespc `+= PhyRespRate_cocco*CoccoC + ovrRate_cocco*CoccoC`;
  vertcalcif `+= calcification`. Copy-out MAIN:470-479, 507-537; column integrals F:1122-1132.
- Missing in the fork: grazmeso_c and the grazing diagnostics are never computed; VTqlimitFac_phyto/diatoms are
  written unconditionally (S:2400, 2426) though allocated only with coccos (MAIN:338-381).
- Update and clip (F:1004-1022): `state = max(tiny, state+sms)`; with coccos `icchl/iphachl max(tiny_chl)`, `icocn
  max(tiny_N_c)`, `icocc max(tiny_C_c)`, phaeo `_p`; once per physics step. In-loop floors S:624-639 per biostep.

## 10. Where `enable_coccos` changes the 2-PFT terms (keep the coccos-off path byte-identical)
1. pMax: `enable_coccos ? qlimitFac*Temp_phyto : P_cm*qlimitFac*arrFunc` (S:2393-2399); diatoms `qlimitFac*
   Temp_diatoms : P_cm_d*qlimitFac*arrFunc` (S:2421-2425).
2. Si uptake: `.not. enable_coccos: V_cm_fact_d*P_cm_d*arrFunc*SiCUptakeRatio*limitFacSi*Si/(Si+k_si)` else
   `V_cm_fact_d*Temp_diatoms*...` (S:3035-3044).
3. Light: the attenuating Chl adds CoccoChl+PhaeoChl.
4. Grazing: cocco/phaeo enter all food sums and variable-preference denominators.
5. Aggregation: `aggregationrate += agg_PP*(CoccoN+PhaeoN)`.
6. Calcification switches from `calc_prod_ratio*Cphot*PhyC` to the cocco forms (S:4479, 4513-4556, 6350-6478).
7. PhyCalc sinking speed VCocco vs VPhy (K:517-522).
8. The `*is_coccos` terms are exact zeros with coccos off (uninitialised locals are 0 under -init=zero): drop them.
9. The 14C path always uses `calc_prod_ratio*Cphot*PhyC_14` (S:4619).
10. Unaffected: carbonate refresh, calc_diss, zoo respiration/losses, DetSi, Si.
Also: `par_layer_mean` defaults to .true. in the fork (M:283): the cell-mean exponential, not the interface light
of sms.py (692f804). Both forms must exist, selected by the switch.
