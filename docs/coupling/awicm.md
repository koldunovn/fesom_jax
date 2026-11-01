# fesom_jax as the ocean of the coupled AWI-CM JAX system

fesom_jax (this repository, branch `coupled/awicm`) is the ocean and sea-ice component of awicm_jax: the JAX
atmosphere ifs_jax (TCo95) coupled to fesom_jax (CORE2, 126,858 nodes), modelled on AWI-ESM3 v3.5.1 TCo95-CORE2
(OpenIFS + FESOM2 + OASIS). Plan: `~/awicm_jax/docs/plans/20261101-awicm-jax-coupling.md`. Everything here is opt-in:
the standalone, file-forced model does not import `fesom_jax.coupled`, and the changes to shared modules are dead
branches unless the coupled code turns them on (see "Changes to shared modules").

## API (`fesom_jax.coupled`)

```python
from fesom_jax.coupled import init_ocean, ocean_interval, ocean_to_atm, write_ocean_restart, flush_monthly

state, static = init_ocean("configs/core2_cpl.yaml", mode="bulk", monthly_out="/work/.../monthly")
for interval in ...:                                   # every 7200 s
    o2a = ocean_to_atm(state, static)                  # {sst_feom, ...}: (nod2D,) arrays, INSTANT
    ...                                                # the atmosphere runs its 2 steps, the coupler remaps
    state = ocean_interval(state, static, forcing)     # 4 ocean steps of 1800 s
flush_monthly(static)
write_ocean_restart(state, static, "/work/.../restart")
```

* `init_ocean(config, *, restart_in="config", mesh_dir=None, ic_dir=None, start=None, year0=1958, nsteps=4,
  mode="bulk", sss_restoring=False, monthly_out=None) -> (state, static)`. `restart_in`: a restart store
  (`write_ocean_restart` or any fesom_jax restart), `"config"` = the config's `restart_in`, `None` = the PHC cold
  start with FESOM's initial sea ice at `start=(y, m, d)`. The calendar counts steps from `year0`-01-01 (the CORE2
  hindcast restarts are stamped so). `mode`: `"bulk"` (B) or `"flux"` (F). `sss_restoring` off by default
  (AWI-ESM3's `surf_relax_s = 0`).
* `ocean_interval(state, static, forcing, nsteps=None, mode=None) -> state`: one coupling interval. `forcing` is a
  dict of `(nod2D,)` arrays (held over the interval, as OASIS fields are) or `(nsteps, nod2D)` (one row a step) on
  the FESOM nodes in fesom_jax's global node order, float32 or float64, numpy or device arrays. Advances
  `static.step` (the calendar), adds the interval to the monthly means and writes a month when the interval starts
  the next one.
* `ocean_to_atm(state, static) -> dict` of `(nod2D,)` device arrays (working precision), global node order.
* `write_ocean_restart(state, static, path)`: the full State (float64 on disk, `zarr_output.write_restart`, global
  layout); restart with `init_ocean(..., restart_in=path)`. A mode F restart carries the mode F fields.
* `flush_monthly(static)`: write the month accumulated so far (call at the end of a run).

What is on the device and what is not: `state` (a `State` pytree) stays on the GPU between intervals; the interval
is one call of a compiled program (`static.executable(nsteps, bootstrap, state)` returns it: a pure jitted
function `(mesh, state, operator, stress, halo, forcing stack, *constants) -> (state, sums)`). `ocean_interval`
keeps the calendar, the month of the climatologies and the monthly sums on the host side of that call (no device
to host copy). The forcing stack is built from the dict with eager element-wise device operations.

Speed (one A100 80 GB, `configs/core2_cpl.yaml`, single precision, dt 1800): see the lane report / the check
outputs (`check_mode_b.json`, `check_mode_f.json`: `ms_per_step_ocean_interval`).

## Configuration (`configs/core2_cpl.yaml`)

`configs/core2_gpu.yaml`'s physics (zstar, GM/Redi, classical TKE, mEVP sea ice every step) and its speed options
that work at dt 1800 on one device (Chebyshev-preconditioned CG with the gathered solve, ELL SSH matvecs, CG
batching, mEVP unroll, Pallas column kernels, column unroll), plus single precision with ordered sums
(`core2_gpu_fast.yaml` pairs them). Partition `serial` (one device; its lanes are the global node order, so no
permutation anywhere). Not taken: dt 2700 and what comes with it (split-explicit SSH, ice every second step), the
multi-GPU options (no-ops on one device), forcing on device (the coupled path feeds arrays).

Initial state: `restart_in` = the fesom_jax CORE2 hindcast v3 (JRA55-do from PHC at rest on 1958-01-01, dt 1800,
double precision; `/work/ab0995/a270088/port_jax/runs/core2_hindcast_v3`) on 1989-01-01 (`fesom.1989.001.00000`,
step 543504), the only spun-up 1 January state of the coupled start year (the OMIP cycles of
`fesom_jax_dev/runs/omip` ran at dt 2400/2700 and have no 1989 restart).

## Mode B (bulk): the atmosphere's near-surface state -> fesom_jax's bulk formulae

`forcing` keys (interval means on the FESOM nodes): `u10, v10` [m s-1, geographic east/north], `t2m` [K], `q2m`
[kg kg-1], `swdn, lwdn` [W m-2, downward at the surface], `rain, snow` [kg m-2 s-1], `msl` [Pa, optional, unused:
FESOM2's `l_mslp` is off in AWI-ESM3 and in the CORE2 configuration].

Conversions (`ocean.bulk_step_forcing`, the operations of `jra55.JRA55Reader.step` in its order): the wind is
rotated into the model's rotated frame (FESOM2 `vector_g2r`), `Tair = t2m - 273.15`, `prec = rain / 1000`,
`snow / 1000` [m s-1]. Then the standalone path unchanged: L&Y09 (ncar) bulk at the current SST and surface current,
the standalone sea-ice thermodynamics, oce_fluxes, the shortwave penetration (Sweeney chlorophyll climatology),
runoff from the CORE2 runoff climatology (plan decision 1; the atmosphere's runoff is not used).

Notes: the bulk formulae take T and q at 10 m (`ncar_bulk_z_tair = 10`, JRA55-do's height); the atmosphere sends
2 m values. SSS restoring off by default (on in the bitwise check, which must equal the standalone run).

Check (`scripts/coupled/check_mode_b.py`): arrays built from the JRA55-do files as the atmosphere would send them
(the reader's interpolated values before rotation and unit conversion) reproduce the file-driven chunk call; see
the lane report for the result and its reason.

## Mode F (flux): FESOM2's coupled path (`fesom_jax.coupled.flux`)

`forcing` keys = the FESOM2 names of the namcouple fields (what OASIS delivers to `cpl_oasis3mct_recv`), in the
units and signs FESOM2 uses them without conversion (the ECE put side, `ece_fesom_set_ocean_fluxes.F90`, divides
the water fluxes by RHOH2O):

| key | namcouple source | units, sign | FESOM2 array |
|---|---|---|---|
| `taux_oce, tauy_oce` | A_TauX/Y_oce | N m-2, geographic | stress_atmoce_x/y (rotated g2r) |
| `taux_ico, tauy_ico` | A_TauX/Y_ice | N m-2, geographic | stress_atmice_x/y (rotated g2r) |
| `prec_oce`, `snow_oce` | A_Precip_liquid/solid | m s-1 fresh water, + down | prec_rain, prec_snow |
| `evap_oce` | A_Evap | m s-1, + down (negative = evaporation), per open-water area | evap_no_ifrac |
| `subl_oce` | A_Subl | m s-1, + down, per ice area | sublimation |
| `heat_oce` | A_Qns_oce | W m-2, + down, non-solar over open water | oce_heat_flux |
| `heat_swo` | A_Qs_all | W m-2, net shortwave into open water | shortwave |
| `heat_ico` | A_Q_ice | W m-2, + down, into the ice | ice_heat_flux (a2ihf) |
| `tsk_ico` | A_Ice_tskin | K, the ice-tile skin | ist_ref (use_atm_ice_tskin) |
| `hydr_oce, calv_oce` | R_Runoff/Calving_oce | (accepted, not used: the CORE2 runoff climatology) | runoff |
| `u10w_oce, v10w_oce` | A_WindX/Y | (accepted, not used: FESOM2 reads them only for icebergs) | u_wind, v_wind |

`oasis_to_fesom(fields) -> dict` (`fesom_jax.coupled`) is FESOM2's receive step for the coupler: it takes the remapped
fields by their namcouple OASIS names (`A_Qns_oce`, `A_Qs_all`, `A_Precip_liquid`, `A_Precip_solid`, `A_Evap`, `A_Subl`,
`A_Q_ice`, `A_Ice_tskin`, `A_TauX/Y_oce`, `A_TauX/Y_ice`, `A_WindX/Y`, `A_Runoff`, `A_Calving`) or the FESOM2 names above,
and returns the keys of this table. It renames and does nothing else, because FESOM2 does nothing else on receive
(`prec_rain(:) = exchange(:)` ...): the units are those the atmosphere puts, and the fork's put
(`ece_fesom_set_ocean_fluxes.F90`) already divides the water fluxes by RHOH2O (m s-1). A water flux in kg m-2 s-1 at
the OASIS level is therefore a put-side conversion the atmosphere lane owes, not one FESOM2 makes. Weighting by the
ice/open-water fraction happens in the thermodynamics (`thermo_cpl`), the g2r stress rotation in `ocean_interval`.

No conservation correction on the FESOM side: under `__oifs` FESOM2's `force_flux_consv` returns at once (the
OASIS remapping is conservative, GLBPOS/GSSMAR; the coupler lane applies them).

Per ocean step (`flux.ice_surface_step_cpl`, called by `fesom_jax.step` when the forcing is a `CplForcing`):
ocean2ice -> mEVP with the received ice stress -> FCT -> cut_off -> `thermo_cpl` (ice_thermo_cpl.F90: the ice
surface temperature = the received skin clamped to 273.15 K where a_ice > 0.001 and the freezing point elsewhere; the
albedo of that temperature with melt ponds; the growth driven by `-heat_ico` over ice and
`-(heat_oce + heat_swo)` over open water, ocean-ice heat `(T - Tf) 10 m/day cc`; evaporation and sublimation
weighted by 1 - A and A; snow melt only above 273 K; melt ponds; flooding) -> oce_fluxes with the coupled zstar
freshwater balance (`+ residualifwflx`, `x rhofwt/rhowat`) -> oce_fluxes_mom with the received ocean stress ->
the penetration of `0.54 (1 - 0.1) heat_swo`.

State fields mode F adds (`State.CPL_FIELDS`, None otherwise): `cpl_ice_temp` [K], `cpl_ice_alb`, `cpl_apnd`,
`cpl_hpnd`, `cpl_ipnd`; FESOM2's initial values (265.15 K, 0.6, 0) when the restart has none.

Differences from FESOM2 (all deliberate, each its reason):

* `ice_temp` is not advected (FESOM2 advects `ice_temp * a_ice` as a 4th FCT tracer). With use_atm_ice_tskin the
  thermodynamics overwrite it wherever a_ice > 0.001, so only nodes with a_ice <= 0.001 read the advected value.
* The atmosphere's runoff and calving are dropped (plan decision 1); enthalpyoffuse = 0.
* The ocean physics is fesom_jax's CORE2 configuration (zstar, classical TKE, mEVP, GM/Redi, CG SSH), AWI-ESM3's FESOM2
  runs linfs, KPP, standard EVP; the coupled branches ported are those of zstar.
* Mode F differs from mode B (FESOM2 coupled vs standalone): the atmosphere computes the fluxes at its own surface
  (its own albedo, roughness, skin temperature), FESOM's ice thermodynamics is the zero-layer coupled-slab scheme
  instead of the standalone 7-class growth with its own skin-temperature Newton, sublimation and evaporation come
  from the atmosphere, the shortwave penetration takes the atmosphere's net shortwave, and the EVP takes the
  atmosphere's ice stress (which does not see the ice velocity).

Check (`scripts/coupled/check_mode_f.py`): one month driven by an offline atmosphere (CORE2 bulk at the current
ocean surface for open water, fesom_jax's standalone ice-surface budget for the ice), compared with the standalone
month; see the lane report.

## Ocean -> atmosphere (`ocean_to_atm`), as FESOM2 sends them (`gen_forcing_couple.F90`, `__oifs`)

| key | value | units |
|---|---|---|
| `sst_feom` | T(1) + 273.15 | K |
| `sie_feom` | a_ice | 0..1 |
| `snt_feom` | m_snow (grid-cell mean) | m |
| `ist_feom` | ice_temp * a_ice (concentration-weighted, the atmosphere divides by the ice fraction) | K x fraction |
| `sia_feom` | ice albedo | 0..1 |
| `u_feom, v_feom` | surface node velocity UVnode(:,1), rotated r2g to geographic east/north | m s-1 |
| `sit_feom` | m_ice (grid-cell mean = effective thickness) | m |

INSTANT at the call. The node velocity is recomputed from the carried element velocity (what FESOM2's
`compute_vel_nodes` gives at the top of the next step). Mode B's ice temperature is the standalone skin
(`t_skin + 273.15` where a_ice > 0.001, the freezing point of the surface salinity elsewhere) and its albedo
ice_thermo_cpl's `ice_albedo` of it without ponds; mode F sends the coupled thermodynamics' own.

## Changes to shared modules (all dead branches for the standalone model)

* `sss_runoff.sss_runoff_fluxes`: `FESOM_SURF_RELAX_S` (environment, read at trace time) replaces the SSS restoring
  rate; unset = the constant (the coupled ocean sets "0" unless `sss_restoring=True`).
* `state.State`: the optional `CPL_FIELDS` (None by default: no pytree leaves); `zarr_output` writes and reads them
  when present.
* `ice_step.IceStepOut.cpl` (None by default) and `step.step`: the coupled ice step when the forcing is a
  `CplForcing` (a trace-time type check), and the CPL fields carried when `cpl` is set.
