# fesom-jax REcoM: validation against FESOM2-REcoM

The gates of the REcoM port (plan `docs/plans/20261025-recom-port.md`, Testing Strategy) and their numbers. The Fortran
reference is described in `docs/recom/FORTRAN_REFERENCE.md`. Lane gates G1-G4a (the pieces fed with Fortran's own
inputs) are in the plan under Tasks 8-12; this file collects the gates of the connected model (lane W,
`docs/plans/20261025-recom-lane-wiring.md`). All runs: pi, double precision, CPU, `configs/pi_recom_bgc.yaml`, from the
physics-matched twin's step-0 state (`fesom_jax.recom.reference.twin_step0_state`; twin
`recom_ref/pi_30d_jan_jaxphys`, FESOM2 e96a2b13 + REcoM 692f804, with FESOM2 #944/#1116 in fesom-jax).

## Summary

| check | where | result |
|---|---|---|
| pieces with Fortran's own inputs (G1-G4a) | plan Tasks 8-12 | carbonate chemistry, sources and sinks, sinking/benthos, stage 3 at round-off |
| one full step against the twin (G4b) | below | every BGC tracer <= 1.8e-8, below the physics' own T difference 4.1e-8 |
| 30 days against the twin (G5) | below | daily inventories within 1.1e-9, CO2 flux 1.9e-5, NPP 5.0e-6 |
| budget | below | every tracer closes to <= 7e-15 of its inventory (DP) |
| gradient (G6) | below | adjoint against central differences 7e-7 ... 5e-5 |
| sharded against dense | below | BGC tracers <= 4e-13 |
| CORE2 year | below | finite, plausible (no Fortran CORE2 + REcoM run exists to compare with) |
| `bgc_every = 4` | below | inventories hold, biology 5-15 % low after a month: not faithful |
| single precision (opt-in, STOP 4 2026-10-26) | below | annual biology equal to DP to three digits; budget open item (1e-11 in five tracers) |

## G4b: one full JAX step against the twin's step 1

`fesom_jax/tests/test_recom_step.py::test_g4b_one_step_vs_twin` (job 28421905). Relative error = max |JAX - Fortran| /
max |Fortran| over the wet cells.

| field | rel. error |
|---|---|
| physics floor of the pair: T / S after step 1 | 4.1e-8 (1.2e-6 degC) / 1.5e-8 |
| the 31 BGC tracers | 2.2e-10 (DFe) ... 1.8e-8 (HetN, Zoo2N, MicZooN, DICremin); DIC 1.6e-8, Alk 1.7e-8, O2 1.6e-8, DIN 2.5e-9 |
| Benthos (4 pools) | 1.1e-14 ... 3.1e-11 |
| CO2 flux / pCO2surf / surface pH / NPP | 7.5e-9 / 3.6e-14 / 2.5e-15 / 2.2e-10 |

Every BGC tracer is below the relative error of T: the BGC tracers see the physics difference through the transport
(and a_ice, 2.7e-7, in stage 1). Gate: each field <= 100 x the T/S floor.

## Budget (JAX alone)

`test_budget_closed` (4 steps, `bgc_every` 1 and 2) and the G5 run (30 days): for every tracer, the inventory change
sum X hnode areasvol equals the accumulated surface BC + stage 1 (sources, sinks, clip, DICremin reset) + REcoM's
tracer-loop terms (benthic return flux, sinking, deposit) to <= 4.6e-15 of the inventory after 4 steps and <= 6.9e-15
(DFe; median 2.0e-15) after 30 days. Element budgets over 4 steps (water column + benthos pools, changed by the
surface BC and the clip only): N, Si and C close to the rounding of the inventory sums (<= 4e-15 of the inventory).

## G5: 30 days against the twin

`scripts/recom/g5_pi_30d.py` through `run_from_config` (one device, daily chunks, the daily mean stream with the BGC
fields), job 28422570 (0.56 node-h; ~2 s per step on 64 cores).

- Daily inventories (sum of the daily-mean field x nominal thickness x areasvol, the same weighting on both sides):
  DIC, Alk, DIN, DSi, DFe, O2 within 1.1e-9 of the twin on every one of the 30 days (largest: DFe on day 30).
- Global air-sea CO2 flux (daily mean, area-integrated): within 1.9e-5 of the twin (day 19); NPP within 5.0e-6.
- End state (step 960) pointwise, max |J - F| / max |F| over wet cells: T 7.1e-3, S 7.8e-4 (the physics' own
  divergence after 30 days); DIC 1.5e-3, Alk 7.9e-4, DIN 1.5e-2, DSi 4.1e-3, DFe 2.9e-4, O2 1.0e-2; the phyto- and
  zooplankton 1e-4 ... 7e-3; the second detritus class (DetZ2N/C/Si/Calc, concentrated in few cells) 3e-2 ... 3.1e-1;
  Benthos 1.4e-5.

## G6: gradient of a 1-day NPP cost

`fesom_jax/tests/test_recom_grad.py`: J = sum over the 32 steps of day 31 of the area-integrated column NPP, from the
G5 day-30 restart (`FESOM_G6_RESTART`), `jax.grad` through the full step against central differences (h = 1e-3 p),
job 28424311 (J = 2.70e17):

| parameter | adjoint | FD | rel. difference |
|---|---|---|---|
| `P_cm` (nanophytoplankton max. photosynthesis) | 3.18811e16 | 3.18808e16 | 8.8e-6 |
| `rho_N` (DON remineralisation) | 3.74665e15 | 3.74684e15 | 4.9e-5 |
| `alfa_d` (diatom light slope) | 2.35483e17 | 2.35483e17 | 6.9e-7 |
| `graz_max` (mesozooplankton grazing) | -46.1 | 0 | numerically zero both ways: dJ/dlog p = -14 against J = 2.7e17 (the mesozooplankton are still at their tiny initial value after the 30-day cold start) |

Two derivative-only fixes were needed (forward values unchanged, G2/G3/G4a as before): the masked divisions by
`areasvol` below the bottom in the sinking routines, and the photodamage exponent, which overflows where N:C < NCmin
(`exp(1706)` on day 31; the max() takes deg/10, in Fortran too) and gave 0*inf = nan in the reverse pass.

## Sharded against dense

`test_sharded_equals_dense`: two full REcoM steps on FESOM2's pi `dist_4` (4 host devices, the forced sharded runner,
the global coastal mask) against the dense steps, every owned node: BGC tracers worst 4.0e-13 (DetN; median 6.9e-15),
Benthos 1.8e-16; the physics T 2.5e-14, S 6.0e-16.

## CORE2: one year with full REcoM (no reference run)

`configs/core2_gpu_recom.yaml` (DP, 4 GH200) with `dt: 1800` + `dt_ramp: {after_step: 561024, dt: 2700}` from the v3
1990 physics restart, BGC from the 1-degree IC set, job 28431153; `scripts/recom/core2_year_summary.py` on the monthly
means. Plausibility only (no Fortran CORE2 + REcoM run to compare with): the year is finite; annual means CO2 uptake
2.03 Pg C/yr (constant 278 uatm atmosphere; area-mean surface pCO2 263-274 uatm), NPP 47.3 Pg C/yr, POC export at
100 m 5.63 Pg C/yr; the outgassing sits in the equatorial and coastal upwellings, the uptake in the subpolar North
Atlantic and North Pacific and at 40-50 S; NPP is high in the upwellings and low in the subtropical gyres. Global
inventories, December against January mean: DIC -4.0e-5, Alk +6.2e-6, DIN -1.2e-3, DSi -4.6e-4, DFe -3.0e-3,
O2 -5.3e-4 (relative).

## CORE2: the error of `bgc_every = 4` over one month

January 1990 from the v3 restart (ramp config, 4 GH200), `scripts/recom/kerr_compare.py`: k = 1 twice (the GPU's
run-to-run floor), k = 4 against k = 1 on the same code. Relative difference of the global monthly-mean integral:

| field | floor (k = 1 twice) | k = 4 - k = 1 |
|---|---|---|
| DIC / Alk / DIN inventory | -1.6e-12 / -1.6e-12 / 5.2e-11 | 1.5e-6 / -1.2e-7 / 1.2e-5 |
| DFe / O2 inventory | -7.5e-11 / 2.0e-10 | 1.8e-5 / -1.6e-5 |
| Chl | 1.4e-6 | -5.5e-2 (rms 9.7e-2) |
| air-sea CO2 flux | 1.6e-6 | -6.0e-2 |
| NPP | -7.3e-8 | -8.1e-2 |
| POC export at 100 m | 6.3e-6 | -1.4e-1 |
| T (end state, rms) | 2.5e-5 | 2.8e-5 |

The physics of the k = 4 run is the k = 1 run's up to the floor (the BGC tracers do not feed back); the conserved
inventories hold to <= 2e-5, but the biological rates are 5-15 % low after a month. `bgc_every = 4` is therefore a
transport shortcut that does not reproduce the biology; the source of the error (the 3 h source/sink step, or the
light of one step standing for the interval) is not yet separated.

## Single precision with full REcoM (an opt-in deviation; plan `docs/plans/20261026-recom-lane-sp.md`)

STOP 4 (user, 2026-10-26): accepted as an opt-in for spin-ups, ensembles and sensitivity runs; double precision
stays the default and the production mode.

FESOM2 refuses `USE_SINGLE_PRECISION` with `RECOM_COUPLED`, so this mode has no Fortran counterpart; it is defined in
the plan's S1 design (the BGC tracers in FESOM2 #1207's tracer scheme with their own carry `State.bgc_carry`, REcoM's
code in float64, every REcoM increment as float64 content into the one carried rounding; deviations R0-R6) and is
never a default (`precision: single` with full REcoM and `recom.bgc_every: 1`). Tests:
`fesom_jax/tests/test_recom_sp.py` (pi, CPU; numbers from jobs 28438265, 28439024, 28439735, 28439897).

| check | result |
|---|---|
| stage 3 alone, the G4a transport fields (steps 1, 224) in SP and DP | every tracer <= 2.5e-7 of its max (median 1e-7, a few float32 ulps); Benthos identical |
| one model step from the twin's state, SP - DP (max / rms relative) | T 7.7e-3 / 1.2e-3, S 3.3e-4 / 6.5e-6; BGC from 6e-7 (zooplankton) to 4.0e-2 (DIN; DetC, DetSi, O2 ~3.6e-2), every tracer <= 100 x the T/S value. Not a float32 floor: DP from the float32-rounded state moves T by only 7e-6 / 2e-7; the step pair is dominated by the physics' own SP - DP (first-step `Kv` differs by ~36 % rms, `l_mslp` on or off) |
| budget, carry counted as content | median tracer 8e-14 (4 steps) / 2e-14 (30 steps), worst 1.7e-11 (DetCalc) / 8.6e-12 (DOC); DP closes to <= 7e-15 |
| budget, the carry dropped before every step | median 6e-9 / 4e-8, worst 1e-7 / 7e-7: the carry buys 3-6 orders of magnitude |
| sharded `dist_4` SP vs dense SP, 2 steps | DOC 1.2e-4 (1015 eps32), DON 7e-5, the rest <= 6e-6 (median 1.4e-7); the DP pair: DOC 2.5e-14 (114 eps64), worst 4e-13. DOC amplifies its rounding ~9x more in SP than in DP |
| restart through `run_from_config` | `bgc_carry` on disk (float64) next to `carry_T`/`carry_S`, back to WP bitwise; an SP run continues from it; a DP restart starts SP (carry from zero), an SP restart starts DP (carries dropped) |
| DP unchanged | the full and passive pi REcoM steps lower to the same StableHLO before and after (`scripts/recom/dp_hash_recom.py`, jobs 28437900, 28438264); `test_dp_lock.py` and the REcoM DP tests green |

### CORE2: one year SP against DP

`configs/core2_gpu_recom_v3_1990.yaml` (4 GH200, dt ramp, v3 1990 restart, BGC from the 1-degree IC), the same commit
73e2171 for all three runs: DP (job 28439582), SP (`--precision single`, 28439583), DP again (28439750, the year's
run-to-run floor); `scripts/recom/kerr_compare.py runs/recom/sp_twin_1990 sp:sp:dp floor:dp2:dp`. Relative difference of
the global integral (d_int) and the pointwise rms over the wet cells (relative to the field's rms), December 1990
monthly mean and end state:

| field | SP - DP: d_int | SP - DP: rms | DP twice: d_int | DP twice: rms |
|---|---|---|---|---|
| DIC / Alk inventory | 1.1e-8 / -1.3e-9 | 7.2e-5 / 3.7e-5 | 4e-15 / 2e-13 | 1.5e-9 / 7.2e-10 |
| DIN / DSi / DFe / O2 inventory | 4.5e-8 / 7.4e-8 / -4.4e-7 / -4.0e-7 | 5.7e-4 / 4.5e-4 / 7.4e-4 / 5.4e-4 | <= 2.7e-11 | <= 4e-8 |
| Chl | -4.6e-5 | 1.0e-2 | -2.0e-9 | 1.9e-6 |
| air-sea CO2 flux / surface pCO2 | 8.5e-5 / 8.4e-6 | 8.0e-3 / 1.0e-3 | 1.3e-9 / -1.8e-10 | 1.3e-7 / 3.2e-8 |
| NPP / POC export at 100 m | 4.3e-5 / -2.9e-4 | 7.4e-3 / 7.6e-3 | -2.4e-8 / 5.6e-7 | 2.0e-6 / 3.6e-4 |
| T / S (the physics) | 6.9e-7 / -2.9e-10 | 1.6e-3 / 4.0e-5 | 0 / 0 (bitwise) | 0 / 0 |
| end state DIN / DIC / O2 / DFe | | 9.5e-4 / 1.3e-4 / 8.2e-4 / 1.5e-3 | | <= 8.4e-8 |
| end state PhyC / DiaC / DetC / DetZ2C | | 1.5e-2 / 2.1e-2 / 7.0e-3 / 5.6e-2 | | 7.9e-6 / 6.3e-6 / 1.8e-5 / 3.6e-2 |
| end state T / S | 3.5e-7 / 5.2e-11 | 3.2e-3 / 8.0e-5 | 0 / 0 | 0 / 0 |

Annual means: CO2 uptake 2.032 / 2.032 Pg C/yr, NPP 47.363 / 47.365, POC export 5.63 / 5.63 (DP / SP); the
inventory changes over the year (December against January mean) agree to the printed digits (DIC -4.03e-5, Alk
+6.17e-6, DIN -1.19e-3, DSi -4.64e-4, DFe -2.97e-3, O2 -5.34e-4). Reading: the DP run is reproducible (the physics
bitwise; the BGC to GPU summation noise, inventories <= 3e-11, DetZ2C noisy even so), so every SP - DP number is the
SP physics' divergence (T rms 1.6e-3 in December) carried into the biology. The conserved inventories differ like the
physics' own salt (-2.9e-10): Alk -1.3e-9, DIC 1.1e-8 (consistent with the 8.5e-5 CO2 flux difference times the
flux-to-inventory ratio ~5e-5 a year); no leak signature. Pointwise the dissolved tracers stay below T's rms
(<= 1.5e-3 against 3.2e-3); plankton and detritus amplify it (1-6e-2), as the biology does any perturbation. The largest integral difference of a
non-conserved pool is the second zooplankton's carbon at the end of the year (Zoo2C -3.5e-3; PhyC 8e-5, DiaC
-1.3e-4, DetC -1.8e-4). For scale:
`bgc_every = 4` misses NPP by -8 % and export by -14 % after one month; SP misses them by 4e-5 and -3e-4 after a year.

Open: the SP budget residual above 1e-12 sits in the five tracers that grow by orders of magnitude from REcoM's tiny
initial value in the first steps (DetCalc, DICremin, PhyCalc, DOC, DON), always a small loss (~1e-11 of the step's
content change) that does not grow from 4 to 30 steps; stage 3 alone conserves to <= 3e-13 on the G4a inputs (with
GM off ~1e-14: the WP Redi increments, as upstream's), so it enters where stage 1 meets the tracer step. Not isolated.
