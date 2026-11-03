# Lane L-SFC: jaxcpl's ocean surface and physics settings in fesom_jax

Plan: `~/awicm_jax/docs/plans/20261103-jaxcpl-core3.md` (lane L-SFC, tasks 1-2 and the config of task 3).
Branch `coupled/core3_sfc` from `5b88770` (tag `awicm-v0-tco95-core2`). Outputs: `/work/ba1583/a270088/awicm/core3/sfc/`.

## Status (2026-11-03, session 1)

- Task 1 (diff + ports): diff done (table below). Ported behind switches: hemispheric GM ceiling, `opt_visc = 5`
  (visc_filt_bcksct), FESOM main's mEVP mass factor and sigma reset, the coupled albedo ramp (`alb_tramp`) and
  snow blend (`h_snowscale`), configurable `snowmelt_tgate` and melt-pond parameters, the GM/Redi ceilings from the
  config. Defaults unchanged.
- Task 2 (melt ponds): the coupled path (mode F, `fesom_jax.coupled.flux`) already had FESOM2's melt ponds
  (ported from dbf28153; `ice_meltponds.F90` and the pond albedo are unchanged from dbf28153 to 0aaa2d73, re-read
  line by line here). What was missing for jaxcpl: the switch and the `&meltpond` values (jaxcpl rfracmax 0.6,
  pndaspect 1.6, the module defaults are 1.0, 0.8), now `ice.cpl_meltponds`, `ice.pnd_*`. The standalone forced
  path has no ponds and needs none (FESOM2's `ice_thermo_oce.F90` has none either).
- Task 3: `configs/core2_cpl_jaxcpl.yaml` (CORE2, zstar, single precision). The CORE3 config waits for lane L-OCN.

## Method

fesom_jax was verified against the port base `9271ae92` (FESOM2 branch `llmport-fortran-reference`, base PR #867
`d729beaa`), its coupled flux path against `dbf28153` (AWI-ESM3 v3.5.1; 38 commits before 0aaa2d73, none of them in
the ice, the coupling or GM). jaxcpl runs `0aaa2d73` (1,198 commits after `d729beaa`). For each routine jaxcpl's
namelists switch on: a semantic diff base -> main (comments, OpenMP/OpenACC directives, `use` lines and the
`mesh%`/`partit%` prefixes stripped: `/work/ba1583/a270088/awicm/core3/sfc/tools/{sdiff,rdiff}.sh` on exported
trees `src_base/`, `src_main/`), then each remaining difference read against the fesom_jax module (some upstream
fixes were already taken into fesom_jax, e.g. the K33 weights and the shortwave depth, with `legacy_*` flags).

## Diff table (jaxcpl namelists, FESOM main 0aaa2d73 vs fesom_jax 5b88770)

| option (jaxcpl value) | Fortran (0aaa2d73) | fesom_jax | verdict | done / to do |
|---|---|---|---|---|
| fer_gm, scaling_resolution (K_GM_resscalorder 2), K_GM_min 2, K_GM_cm 3, K_GM_cmin 0.1, K_GM_rampmax -1 | oce_fer_gm.F90:192-485 init_Redi_GM | gm.init_redi_gm | same | - |
| scaling_gmzexp (500 m, smin 0.6), redi_ktaper, Redi_Kmin 100 | init_Redi_GM:398-475 | gm.init_redi_gm | same | - |
| scaling_odm95 (Scr 2e-3, Sd 1e-3), neutral slopes, sigma_xy | oce_ale_pressure_bv.F90 compute_neutral_slope / compute_sigma_xy | gm.compute_neutral_slope / compute_sigma_xy | same (main: one exchange of both components, OpenACC copies) | - |
| K_GM_max 2500, Redi_Kmax 1000 | namelist values | Params.k_gm / redi_kmax (1000 / 1000), no config key | value; no way to set it from a config | **done**: `gm.k_gm_max`, `gm.redi_kmax_default` feed `Params.defaults(gm_cfg)` when a run passes no params |
| **K_GM_max_NH 1000, K_GM_hemi_trans 10** (the handover's "hemispheric GM") | oce_fer_gm.F90:295-306 (and the _acc twin :608-616), oce_modules.F90:75-76 | absent | **missing** | **done**: `gm.k_gm_max_nh`, `gm.k_gm_hemi_trans` (only fer_K's ceiling; Ki keeps Redi_Kmax, as the Fortran) |
| redi (Redi terms, K33) | oce_ale_tracer.F90 diff_part_hor_redi, diff_ver_part_redi_expl, diff_ver_part_impl_ale | gm_redi.* | same (main restructured to gathers; the K33 interface-weight fix of main is already in fesom_jax, `legacy_k33_weights` off) | - |
| GM template level with cavities | init_Redi_GM writes fer_k(nzmin_cons) and reads fer_k(nzmin_reg) | top value broadcast | differs only where ulevels differ around a node (cavity edge); Fortran then reads a stale value | note for L-OCN / cavities |
| mix_scheme cvmix_TKE, param_tke (c_k 0.1, c_eps 0.7, alpha 30, mxl_min 1e-8, mxl_choice 2, kappaM 0/100, cd 3.75, surf_min 1e-4, tke_min 1e-6, no Langmuir) | cvmix_driver/gen_modules_cvmix_tke.F90, cvmix_tke.F90 | tke.py, cvmix_tke.py, config.TKE_* | same (cvmix_tke.F90 unchanged; main casts the column to real(8) in an SP build, which fesom_jax's single precision already does) | - |
| N² horizontal smoothing (N2smth_h default .true.) | oce_ale_pressure_bv.F90:354 | eos.smooth_nod3D (TkeConfig.n2_source "smoothed") | same | - |
| use_momix false, use_instabmix true (0.1), use_windmix false | oce_mo_conv.F90 mo_convect | pp.mo_convect | same (main adds an _acc twin only) | - |
| whichEVP 1, evp_rheol_steps 120, pstar 3e4, ellipse 2, c_pressure 20, delta_min 1e-11, cd_oce_ice 5.5e-3, theta_io 0 | ice_maEVP.F90 EVPdynamics_m / stress_tensor_m | ice_mevp | same except the two rows below | - |
| alpha_evp 300, beta_evp 300 | namelist | IceConfig 250 / 250 | value | config |
| **mEVP node mass factor** | ice_maEVP.F90:393-394: `1/(max(M,9)·area)` | `M/((1+M²)·area)` (the base's, "dimensionally inconsistent" in main's comment) | **changed upstream** (thin ice: base 0.5/area at M = 1, main 0.11/area) | **done**: `ice.mevp_mass_floor` |
| **mEVP sigma on ice-free elements** | stress_tensor_m: set to 0 (mean m_ice <= 0.01) | carried | **changed upstream** | **done**: `ice.mevp_zero_sigma` |
| mEVP m_ice_old | `m_ice_old = m_ice` (base: `= a_ice`) | - | no effect (thermodynamics overwrites it before any read) | - |
| ice FCT (ice_gamma_fct 0.5), ice_diff 0.0 | ice_fct.F90 | ice_adv | same (main: gathers instead of locked scatters; mass-matrix check relative); ice_diff value (fesom_jax default 10) | config |
| ice_temp advected as a 4th tracer | ice_setup_step.F90 / ice_fct | not advected | known difference (flux.py docstring; with use_atm_ice_tskin read only where a_ice <= 0.001); main sets 271.35 K where a_ice <= 1e-6 after advection | as before |
| use_atm_ice_tskin | ice_thermo_cpl.F90:193-203 | coupled.flux.thermo_cpl | same (only this branch is ported) | - |
| albsn 0.82, albsnm 0.65, albi 0.70, albim 0.68, albw 0.1 | namelist | IceConfig 0.81 / 0.77 / 0.70 / 0.68 / 0.1 | values | config |
| **alb_tramp 1** | ice_thermo_cpl.F90:731-735 ice_albedo | absent (step at 273.15 K) | **missing** | **done**: `ice.alb_tramp` |
| h_snowscale 0 | ice_albedo:744-747 | absent (1 mm step) | 0 = the step = same | **done** anyway: `ice.h_snowscale` |
| **snowmelt_tgate .false.** | ice_thermo_cpl.F90:410 | FluxConfig.snowmelt_tgate (True), not reachable from a config | **differs** | **done**: `ice.snowmelt_tgate` |
| **use_meltponds .true., &meltpond** (hi_min 0.1, hs1 0.03, pndaspect 1.6, rfracmin 0.15, rfracmax 0.6, albpnd 0.2, albpnd_frz 0.36) | ice_meltponds.F90 meltpond_area / meltpond_albedo, ice_thermo_cpl.F90:526-542 | coupled.flux._pond_area / _pond_albedo, FluxConfig fixed (module defaults rfracmax 1.0, pndaspect 0.8) | code same (re-read line by line); **values differ, no config** | **done**: `ice.cpl_meltponds`, `ice.pnd_*` |
| open_water_albedo 0 | oce_shortwave_pene.F90 open_water_albedo_at | constant albw | same | - |
| use_sw_pene | cal_shortwave_rad (depth from the surface level: already in fesom_jax, `legacy_sw_depth` off) | forcing.cal_shortwave_rad | same | - |
| **opt_visc 5**, visc_gamma0/1/2 0.003/0.1/0.285, **visc_easybsreturn 1.5** | oce_dyn.F90:318-460 visc_filt_bcksct | only opt_visc 7 (visc_filt_bidiff) | **missing** | **done**: `visc.opt_visc`, `visc.easybsreturn` (momentum.visc_filt_bcksct, dispatcher momentum.horizontal_viscosity) |
| se_visc (gammas 10/19500/0) | oce_ale_ssh_splitexpl_subcycl.F90:668-720 | ssh_se (split-explicit, a different scheme) | **inactive in jaxcpl**: use_ssh_se_subcycl .false. (the CG runs) | nothing to do |
| momadv_opt 2, use_ivertvisc, ab_order 2, A_ver 1e-4, C_d 0.0025 | oce_ale_vel_rhs.F90 | momentum | same (main: gathers instead of locked scatters) | - |
| soltol 1e-5, maxiter 2000, precond_variant 1 | solver.F90 | ssh (MAXITER 500) | maxiter only bounds the CG | - |
| which_ale linfs | oce_ale.F90 | `ale: null` (linfs) exists in double precision; single precision is ported for zstar only (RunConfig.validate) | **SP linfs missing** | L-OCN (ALE, linfs with cavities) |
| use_cavity, cavity_gamma_scale 0.6 | cavity_param.F90 and ~90 branches | absent | missing | L-OCN |
| icebergs | icb_*.F90 | absent | out of plan | - |
| k_ver 1e-5, kv0_const, double_diffusion false, surf_relax_s 0, balance_salt_water, ref_sss_local | namelist.tra | config defaults; SSS restoring off in the coupled ocean | same | - |
| tracer scheme MFCT/QR4C/FCT (0, 1) | oce_adv_tra_* | tracer_adv (TracerConfig) | same | - |

Summary: of the options jaxcpl switches on in my scope, 6 code differences (hemispheric GM, opt_visc 5, two mEVP
changes, alb_tramp, the pond/tgate switches that were fixed in code) and 8 value-only differences (K_GM_max,
Redi_Kmax, alpha/beta_evp, ice_diff, four albedos); everything else is the same. Outside my scope: linfs in single
precision and cavities (L-OCN), icebergs (later).

## New switches (all defaults reproduce 5b88770)

| block | field | default | jaxcpl | Fortran |
|---|---|---|---|---|
| gm | k_gm_max / redi_kmax_default | 1000 / 1000 | 2500 / 1000 | namelist K_GM_max, Redi_Kmax (now used by `Params.defaults(gm_cfg)` when no params are passed) |
| gm | k_gm_max_nh / k_gm_hemi_trans | -1 (off) / 10 | 1000 / 10 | oce_fer_gm.F90:295-306 |
| visc | opt_visc / easybsreturn | 7 / 1.0 | 5 / 1.5 | oce_dyn.F90 viscosity_filter, visc_filt_bcksct |
| ice | mevp_mass_floor / mevp_zero_sigma | False / False | True / True | ice_maEVP.F90:393-394, stress_tensor_m |
| ice | cpl_meltponds / snowmelt_tgate | True / True | True / False | ice_thermo_cpl.F90 (mode F only) |
| ice | alb_tramp / h_snowscale | 0 / 0 | 1 / 0 | ice_thermo_cpl.F90 ice_albedo |
| ice | pnd_hi_min, pnd_hs1, pnd_rfracmin, pnd_rfracmax, pnd_aspect, pnd_albpnd, pnd_albpnd_frz | 0.1, 0.03, 0.15, 1.0, 0.8, 0.2, 0.36 | rfracmax 0.6, aspect 1.6 | ice_meltponds.F90 &meltpond |

The switches sit in the existing sub-configs (`gm:`, `visc:`, `ice:` of a run YAML) in a block of their own at the
end of each NamedTuple (`# --- FESOM main 0aaa2d73 switches for jaxcpl (lane L-SFC)`), so no step signature changed.

## Verdicts

| check | what | result |
|---|---|---|
| kernels (CPU, pi mesh, double) `scripts/coupled/sfc_kernels_check.py` | bcksct vs a loop transcription of oce_dyn.F90; hemispheric fer_K vs the formula; alb_tramp+h_snowscale vs ice_albedo; opt_visc 7 through the dispatcher | bcksct 1.5e-16 relative; fer_K 1.8e-16; albedo 1.1e-16; dispatcher bitwise. OK |
| GPU job 28696484 (A100, 12 min) `scripts/coupled/sfc_gate.py`, snapshots `snap/5b887707aa`, `snap/c0f43b3991` | (1) standalone `core2_cpl.yaml` 96 steps from the 1989-01-01 hindcast restart, branch vs 5b88770 | **BITWISE** (every restart leaf) |
| same job | (1b) determinism control: 5b88770 run a second time | **differs** (T 0.25 K, uv 0.11 m/s max after 96 steps): the standalone GPU path is not reproducible run to run in this job (XLA:GPU autotuning picks kernels by timing, docs/USER_GUIDE.md:210). Branch = base bit for bit is still conclusive (an exact match does not happen by chance once anything differs), but a second base-vs-branch pair would make it airtight |
| same job | (2) coupled mode F (`check_mode_f.py`, offline JRA55 atmosphere) `core2_cpl.yaml` 2 days, branch vs 5b88770 | **BITWISE** (mode F restart incl. ponds, ice_temp, ice_alb) |
| same job | (3) `core2_cpl_jaxcpl.yaml` mode F 10 days (1989-01-01..11) | finite every day; SST 10.469 -> 10.558, ice area NH 14.64 -> 15.05, SH 4.76 -> 3.65 (1e12 m²); restart round trip OK; 48.6 ms/step (default 50.6). End state vs the default's (2 days): ponds on 6,998 nodes, apnd max 0.38 (default 0.65, aspect 1.6 makes them deeper and smaller), hpnd max 1.64 m (capped by the freeboard of thick ice, as the Fortran); ice albedo max 0.82 (= albsn); sigma11 non-zero on 67,805 elements (default 98,846: the ice-free elements are reset) |

## Next

1. On CORE3 (L-OCN's mesh, forced oracle `/work/ba1583/a270088/awicm/core3/fesom_oracle/run_cav`, which runs
   opt_visc 5, GM with K_GM_max_NH, mEVP): the ladder for viscosity, GM coefficients and mEVP against the dumps.
   The forced oracle has no melt ponds (ice_thermo_oce.F90 has none); the coupled pond path can only be checked
   against a coupled (__oifs) FESOM2 build or jaxcpl's output.
2. `configs/core3_cpl_jaxcpl.yaml` from the blocks of `core2_cpl_jaxcpl.yaml` with L-OCN's mesh, linfs, cavities.
3. Single-precision linfs (RunConfig.validate refuses it today): with L-OCN.
