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

## Status (2026-11-03, session 3)

On `merge/core3_ocn` ba2a797 (lane L-OCN's cavities in the step), against L-OCN's double-precision CORE3 cavity
oracle `run_cav_dp` and this lane's instrumented copy of it (`oracle_core2/run_cav_dp_evp`, the same namelists,
restart and source plus sea-ice dumps), split as L-OCN's ladder (cavity / shelf front = open node touching a cavity
cell / open), max |diff| / max |field| (p99):

| rung | job | cavity | front | open | control |
|---|---|---|---|---|---|
| 6 viscosity opt_visc 5 (Δuv_rhs) | 28703336 | 3.0e-14 | 3.6e-14 | 2.3e-13 | opt_visc 7: 0.20 / 0.26 / 2.9 |
| 14 GM fer_uv, jaxcpl | 28703486 | **0.18** (p99 3.3e-2) | **6.0e-2** (p99 7.3e-3) | 1.1e-13 | no hemispheric ceiling: open 1.25 |
| 14 GM fer_uv with FESOM's stale template emulated (below) | 28703486 | 4.0e-14 | 4.1e-14 | 1.1e-13 | |
| 18 mEVP u/v_ice on the oracle's EVPdynamics_m inputs (metric_factor_zero, mass floor, front = coast) | 28703447 | - (cleaned) | bitwise (0 in both) | 2.1e-7 | metric terms: 4.6e-2; front not a coast: 1.33 |
| 18 ice FCT + cut_off + cavity cleaning, a_ice | 28703336 -> 28703447 | bitwise | **4.2e-2 -> 1.0e-14** | 3.0e-3 -> 9.4e-15 | |
| 18 the same, m_ice / m_snow | 28703447 | bitwise | 9.3e-16 / 7.5e-16 | 3.0e-15 / 1.5e-15 | |

**The shelf-front sea-ice difference (L-OCN's whole-step a_ice 0.08 at the front) is the ice FCT limiter.**
FESOM's `ice_fem_fct` bounds each node by the old and low-order values of its `nn_pos` neighbours (ice_fct.F90:518-547),
and `nn_pos` includes the cavity nodes next to a front node (ice 0 there). fesom_jax took the bounds over the node's
open elements only, so a front node could never be drained below its open neighbours' minimum. Fixed in
`fesom_jax/ice_adv._fem_fct`: the bounds come from all elements around the node (identical off the cavities, where
every element is open; pad elements have ulevels 0 and node -1 and were already included). Not in cavity.py.

**GM at the shelf: an upstream FESOM bug that jaxcpl runs.** At the 7,237 nodes with `ulevels_nod2D /=
ulevels_nod2D_max` (1,104 of them the front), FESOM main's `init_Redi_GM` writes the surface template
`fer_k(nzmin,n)` at the conservative level `nzmin = ulevels_nod2D_max` (F1 loop) but builds the profile from
`fer_k(ulevels_nod2D(n),n)` (F2 loop, `nzmin` redefined), which holds what the previous step left there: 500 m²/s at
setup (`oce_setup_step.F90:1042`, not restarted), then times `zscaling(ulevels_nod2D)` every step. So in jaxcpl the
GM coefficient there is `500·zscaling(z)` at the front (zscaling = 1 at the sea surface, so it stays 500 for good) and
`500·zscaling(z_top)^(n-1)·zscaling(z)` under the shelves (dying off within tens of steps of each leg), whatever
K_GM_max, the resolution scaling or the hemispheric ceiling say. The emulation in rung 14 (step 1) reproduces the
oracle to 4e-14. fesom_jax does what the code intends (the template at every node). **Not ported; decision needed**
(reproduce jaxcpl's bug behind a GM switch, or report it to FESOM/Jan and keep the intended behaviour). Reproducing it
exactly needs the step count since the leg's start (fer_K is not in FESOM's restart); its steady state (front 500·zs,
the other mismatched nodes 0) needs none. The fix in FESOM would be one line: write the template at
`mesh%ulevels_nod2D(n)` in the F1 loop (or read `fer_k(nzmin_cons)` in F2).

**One whole step against the DP oracle** (`oracle_core2/jobs/step_core3.sbatch`: `run_from_config.py`, 1 step on
CPU from L-OCN's converted restart, then L-OCN's `scripts/core3/compare_step.py`), max |diff| / max |field| (p99):

| field | `configs/core3_cavity.yaml` (fesom_jax defaults; job 28703616) | `configs/core3_cavity_jaxcpl.yaml` (jaxcpl's settings; job 28703613) |
|---|---|---|
| T | cavity 5.8e-5, front 3.1e-4, open 3.6e-3 | cavity 5.8e-5, front 2.5e-5, open 7.7e-5 (p99 9.4e-8) |
| S | 4.5e-5 / 1.1e-4 / 1.1e-3 | 3.1e-5 / 8.2e-5 / 2.2e-4 |
| u / v | open 1.4e-2 / 2.9e-2 | open 5.5e-5 / 1.3e-4; cavity, front <= 1e-5 |
| eta_n | front 1.1e-3, open 8.8e-3 | front 1.1e-3, open 2.1e-3 (p99 8.3e-5; L-OCN's rung 9: the CG tolerance) |
| a_ice | front 2.9e-3 (p99 6.1e-4), open 4.3e-3 (L-OCN before the FCT fix: front 8.0e-2) | front 2.5e-3 at one point (**p99 1.8e-10**), open 2.3e-4 (p99 1.0e-9) |
| m_ice | 2.4e-4 / 7.3e-4 | 7.3e-6 / 4.9e-6 |

What is left after one step with jaxcpl's settings: a_ice 2.5e-3 at a few points (p99 1e-10), which can only come
from the forced thermodynamics (ice_thermo_oce.F90 changed by 558 lines from the port base; not in jaxcpl's coupled
path, not replayed here); eta_n 2e-3 (the CG solved to soltol, rung 9); the GM shelf template above (it enters the
tracers from step 1; whether it is the 2.5e-5 of T at the front is not measured).

## Status (2026-11-03, session 2)

- Rung 2 of the ladder against FESOM main 0aaa2d73 (lane L-OCN's oracle builds, read only) on **CORE2 without
  cavities**, with jaxcpl's namelists (zstar, forced JRA55-do 1958, from the Fortran CORE2 restart of 1958-02-01):
  viscosity opt_visc 5 and the hemispheric GM agree with the **DP oracle to 1.8e-13 / 8.9e-14** (relative to the
  field maximum), mEVP to 3.5e-7 on the oracle's own inputs; each control (opt_visc 7, no hemispheric ceiling, the
  base mass factor, the metric terms) is off by 2e-4 ... 2, so the checks discriminate.
- The oracle found one jaxcpl switch the session-1 diff missed: **`metric_factor_zero = .true.`** (namelist.config),
  which removes the metric terms of the sea-ice dynamics; without it fesom_jax's ice velocity was 1.4e-2 m/s off in
  compact ice. Ported as `ice.metric_factor_zero`. It also showed that session 1's `ice.mevp_zero_sigma` ported a
  routine main's mEVP never calls; removed.
- Single-precision linfs: allowed (two guards relaxed); a CORE2 SP linfs run of 3 days is finite and its SP - DP
  difference is the size of zstar's (below).
- Melt ponds and alb_tramp (coupled only): field-level checks of session 1 (sfc_kernels_check.py and the line-by-line
  read of ice_meltponds.F90 / ice_albedo) stand; the forced oracle has neither.

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
| mEVP sigma on ice-free elements | stress_tensor_m sets it to 0, but EVPdynamics_m inlines its own sigma update without that and never calls stress_tensor_m (ice_maEVP.F90:433 `!NR call stress_tensor_m`) | carried | **same** (session 1 misread this as a change and ported it as `ice.mevp_zero_sigma`; the oracle's EVPdynamics_m output showed sigma carried; the switch is removed in session 2) | - |
| **metric_factor_zero .true.** (namelist.config &geometry; missed in session 1, found by the oracle) | oce_mesh.F90:2352: `metric_factor = 0`; read only by the sea-ice EVP/mEVP (strain-rate and stress-divergence metric terms) | tan(lat)/R | **missing** | **done** (session 2): `ice.metric_factor_zero` |
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
| which_ale linfs | oce_ale.F90 | `ale: null` (linfs) exists in double precision; single precision was refused by two guards | the guards were a scope limit, not a failure (session 2, below) | **done** (session 2): guards relaxed, d94fd35 |
| use_cavity, cavity_gamma_scale 0.6 | cavity_param.F90 and ~90 branches | absent | missing | L-OCN |
| icebergs | icb_*.F90 | absent | out of plan | - |
| k_ver 1e-5, kv0_const, double_diffusion false, surf_relax_s 0, balance_salt_water, ref_sss_local | namelist.tra | config defaults; SSS restoring off in the coupled ocean | same | - |
| tracer scheme MFCT/QR4C/FCT (0, 1) | oce_adv_tra_* | tracer_adv (TracerConfig) | same | - |

Summary: of the options jaxcpl switches on in my scope, 6 code differences (hemispheric GM, opt_visc 5, the mEVP mass
factor, metric_factor_zero, alb_tramp, the pond/tgate switches that were fixed in code) and 8 value-only differences (K_GM_max,
Redi_Kmax, alpha/beta_evp, ice_diff, four albedos); everything else is the same. Outside my scope: linfs in single
precision and cavities (L-OCN), icebergs (later).

## New switches (all defaults reproduce 5b88770)

| block | field | default | jaxcpl | Fortran |
|---|---|---|---|---|
| gm | k_gm_max / redi_kmax_default | 1000 / 1000 | 2500 / 1000 | namelist K_GM_max, Redi_Kmax (now used by `Params.defaults(gm_cfg)` when no params are passed) |
| gm | k_gm_max_nh / k_gm_hemi_trans | -1 (off) / 10 | 1000 / 10 | oce_fer_gm.F90:295-306 |
| visc | opt_visc / easybsreturn | 7 / 1.0 | 5 / 1.5 | oce_dyn.F90 viscosity_filter, visc_filt_bcksct |
| ice | mevp_mass_floor | False | True | ice_maEVP.F90:393-394 |
| ice | metric_factor_zero | False | True | oce_mesh.F90:2352 (namelist.config &geometry) |
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

### Session 2: the ladder against the FESOM main oracle on CORE2

Oracle runs (`/work/ba1583/a270088/awicm/core3/sfc/oracle_core2/`, job `jobs/run.sbatch`, 128 ranks, 6 steps of
1800 s, probes 17 nodes / 17 elements in `probes/` (7 earlier CORE2 probes, 3 Arctic and 3 Antarctic pack ice, 4 ice
edge), full fields steps 1-2): jaxcpl's namelists with `meshpath` = `fesom_jax_dev/fortran/meshes/core2_prefix`
(the source of fesom_jax's CORE2 mesh, `dist_128`), `use_cavity = .false.`, `which_ale = 'zstar'` (fesom_jax's
single precision was zstar-only when this started), `use_atm_ice_tskin = .false.`, icebergs off, L-OCN's forced
namelist.forcing (JRA55-do v1.4.0, 1958, CORE2 runoff), the restart `fortran/core2_rst/jan/fesom.1958.{oce,ice}.restart`
(read through main's pre-2.8 fallback; no sigma in it, so the mEVP starts from 0 as fesom_jax's converter does).

Replays (`scripts/coupled/sfc_oracle_replay.py`, CPU; each block on the oracle's inputs of the same step, output
against the oracle's dump; max |Δ| / max |ref| over the wet points, rms ratio, the probe columns through
`verify.compare_column` at 1e-5 relative + 1e-6 of the field maximum):

| block | oracle | job | jaxcpl switches | control |
|---|---|---|---|---|
| viscosity (Δuv_rhs, steps 1, 2) | DP `run_dp` (28698559) | 28699395 | **1.8e-13**, rms 8.5e-14, probes 17/17 | opt_visc 7: 2.1, probes 0/17 |
| viscosity | SP `run1` (28697167) | 28699396 | 4.5e-5 / 3.9e-5, probes 17/17 | 2.1 |
| GM fer_uv (step 2) | DP | 28699395 | **8.9e-14**, probes 17/17 | without K_GM_max_NH: 1.5, probes 7/17 |
| GM fer_uv | SP | 28699396 | 3.2e-5, probes 9/17 at the 1e-5 bar | 1.5 |
| mEVP u/v_ice on the oracle's own EVPdynamics_m inputs and outputs (instrumented DP copy, substeps 21/22) | DP `run_dp_evp` (28699044, build 28698871) | 28699292 | step 1 **3.5e-7** (sigma 2.4e-5), step 2 3.7e-8 (sigma 1.5e-6) | base mass factor 3.6e-4 / 2.1e-4; with the metric terms 1.4e-2 / 1.6e-2 (sigma 31 %) |
| mEVP from inputs rebuilt from the restart and stages 17/20 | DP | 28699395 | step 1 6.2e-7, probes 13/13; step 2 3.8e-4 (fesom_jax's own step-1 sigma and the rebuilt atmosphere-ice stress) | metric terms 2.5e-2 |
| same | SP | 28699396 | 5.4e-4 / 7.9e-4 | metric terms 2.5e-2 (the SP floor, FESOM SP's float32 geometry, hides the mass-factor control: 6.4e-4) |

How the missing switch was found: with the metric terms the mEVP was 2.5e-2 off in DP and in SP alike, only in
compact ice (a >= 0.5; free drift agreed to 1e-6), and no mEVP constant explained it (alpha/beta 250 or 500,
240 sweeps, P*, C, delta_min, Cd, ellipse; `tools/mevp_diag.py`, jobs 28697893, 28698681). The instrumented
copy (`oracle_core2/fesom-evpdump`, ice_setup_step.F90 only) took the input reconstruction out of the question; then
jaxcpl's namelist.config showed `metric_factor_zero = .true.`.

### Session 2: single-precision linfs

Why it was refused: `RunConfig.validate` and the guard at the top of `step.step` listed "zstar without w_split" among
the production physics of the single-precision plan (docs/plans/20261020-core2-single-precision.md:266, "Guards: SP
with linfs ... raises"). That is a scope statement of the plan; the plan measured nothing for fesom_jax linfs, and
its own Fortran anchor ran a linfs SP-DP pair without trouble (table Task 1b). Nothing in the single-precision path
depends on zstar: the float64 tracer content and the carry (`step._tracers_advect_sp`, `tracer_adv.ale_reconstruct_carry`),
the float64 implicit-diffusion column (`Z3d_diff` from `hnode_new`, which is `hnode` under linfs) and the WP_full
`hnode`/`hnode_new` read the same fields under linfs.

The fix (d94fd35, 5 lines, no line of ale.py): `fesom_jax/step.py` the guard `if _sp and (...)` drops `ale_cfg is None`
and keeps w_split refused (`ale_cfg is not None and ale_cfg.use_wsplit`); `fesom_jax/run_config.py` `validate` the
same (`self.ale is not None and self.ale.use_wsplit`).

Checks:

| check | job | result |
|---|---|---|
| pi mesh, CPU (`tools/sp_linfs_pi_check.py`): GM/Redi + TKE + mEVP, linfs, single | login node | no implicit f32/f64 promotion in the traced step (the test_precision net); 4 steps finite with the SP dtypes; SP - DP max after 4 steps T 7.7e-2, S 1.4e-2, eta 1.3e-4 for linfs against 7.7e-2, 1.4e-2, 1.1e-4 for zstar |
| CORE2, A100 (`tools/sp_linfs_core2.py`): configs/core2_cpl.yaml physics, PHC cold start 1958-01-01, 144 steps (3 days), {linfs, zstar} x {DP, DP with a float32-rounded IC, SP}, `--budget` | 28697619 | all six finite; surface rms SP - DP: linfs T 1.55e-3, S 3.5e-4, eta 8.2e-5, a_ice 1.4e-4, uv 2.2e-3; zstar T 1.36e-3, S 3.0e-4, eta 8.5e-5, a_ice 6.0e-5, uv 2.2e-3. Content after 3 days, SP - DP: linfs salt -5.8e10 psu m³ (5.8e10 of 4.6e19; the f32-IC envelope -1.9e12), heat +9.5e9 J/(ρc), volume 0 (linfs); zstar salt -1.5e12, heat -1.4e10, volume +6e5 m³; global mean eta SP 6e-10 (linfs) and 1.8e-9 m (zstar) against 1e-17 in DP |

So SP linfs drifts like SP zstar. In both, the surface SP - DP after 3 days is 10-30x the f32-IC envelope (T: linfs
10x, zstar 28x), a 3-day spread; the plan's own gate is a 1-year one and was not repeated here.
(The first submission, 28697407, failed in all six runs: no `--ic-dir`.)

### Session 2: default paths

GPU job 28699578 (A100, 12 min; `sfc_gate.py`, snapshot `snap/77ceaf25e4` against `snap/5b887707aa`):

| check | result |
|---|---|
| coupled mode F, `core2_cpl.yaml`, 2 days, branch vs base | **BITWISE** |
| standalone `core2_cpl.yaml`, 96 steps | branch vs base of this job differs, by exactly the numbers of session 1's base-vs-base control (T 0.25336742 K, Av 0.09997762, ...). The standalone GPU run has two outcomes, A and B, and each one is reproduced bit for bit: gate1 base = gate2 base (A), **gate1 base2 = gate2 branch (B, bitwise)**, gate1 base = gate1 branch (A). Every branch result equals a base result bit for bit; which of the two a process gets is not decided by the code (probably XLA:GPU autotuning or a cache shared between the processes of a job). |
| `core2_cpl_jaxcpl.yaml` mode F, 5 days (now with metric_factor_zero, sigma carried) | finite every day; SST 10.470 -> 10.515; ice area NH 14.64 -> 14.86, SH 4.76 -> 4.34; ponds on 7,787 nodes (apnd max 0.47, hpnd max 0.30 m) |


## Next

0. (session 3) Decide the GM shelf template (reproduce jaxcpl's FESOM bug or report it upstream); replay the forced
   ice thermodynamics against the oracle (the remaining front a_ice points); the coupled ice path with cavities
   (`coupled/flux.ice_surface_step_cpl` has no `cavity_cfg`; L-OCN's open item) uses the same `ice_adv` fix.
1. `configs/core3_cpl_jaxcpl.yaml` (the coupled one) from `configs/core3_cavity_jaxcpl.yaml` (forced, session 3);
   and `core2_cpl_jaxcpl.yaml` itself can switch to linfs from a linfs restart.
2. The same replays on CORE3 against L-OCN's `run_cav` (opt_visc 5, GM with K_GM_max_NH, mEVP with
   metric_factor_zero), separately for cavity, front and open points (the GM template-level note above).
3. Melt ponds / alb_tramp: only a coupled (`__oifs`) FESOM2 build or jaxcpl's own output can check them beyond the
   field-level checks; the forced oracle has neither.
4. Not yet compared against the oracle: the forced ice thermodynamics (ice_thermo_oce.F90 changed by 558 lines from
   the port base; not used by the coupled jaxcpl path, which runs ice_thermo_cpl.F90), and the rest of the step.
