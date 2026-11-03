The pi namelist set for current upstream FESOM2 (2.8, `fesom2.8_recom_spectral`,
`docs/plans/completed/20261102-fortran-spectral-on-fesom28.md`) on Albedo: `../namelists_jaxphys/` (the upstream-format,
physics-matched set of the 692f804 twin: FESOM2 e96a2b13, `bin_2p3z2d`) with the Albedo inputs. Changes from
`../namelists_jaxphys/`:

| file | change |
|---|---|
| forcing | JRA55-do 1958 `uas vas huss rsds rlds tas prra prsn psl` from `fesom_jax_data/forcing_1958_mslp/`; `CORE2_runoff.nc`, `PHC2_salx.nc`, `Sweeney_2005.nc` from `fesom_jax_data/core2_mesh_ic/` |

`nm_ocean_mask_file` keeps its Levante path: `use_ocean_only_forcing = .false.`, so it is never opened.
`namelist.cvmix` keeps the IDEMIX paths (`/albedo/work/user/pscholz/...`, `/pool/...`); IDEMIX is off.

Step 1 of the plan (vanilla upstream with its REcoM submodule) runs it as is; the spectral set for steps 5.3-5.4
will add the fork's `namelist.recom` (`&spectral`) and optics on top.

Run: `FESOM2=/albedo/work/user/sloza/SPECTRAL/fesom2.8_recom_spectral NML=<this dir> RUN=<name> DAYS=1
sbatch scripts/albedo/run_pi_recom_fork.sbatch`.

`namelist.recom`: the nine keys REcoM `259e70e3` no longer reads are deleted (`HetRespFlux_plus` (now always the
`.true.` form), `UseFeDust`, `UseDustClim`, `UseDustClimAlbani`, `REcoMDataPath` (dust now from `ClimateDataPath`),
`cPhyN`, `cHetN`, `cZoo2N`, `P_cm_c`); all were marked unused in `../namelists_jaxphys/`. The first run with them
aborted reading `&pavariables` (job 48698080).
