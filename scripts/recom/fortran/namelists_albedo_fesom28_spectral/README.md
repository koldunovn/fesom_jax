The pi namelist set of the spectral REcoM on FESOM 2.8 (`fesom2.8_recom_spectral`,
`docs/plans/20261102-fortran-spectral-on-fesom28.md`, steps 3-5): the physics of `../namelists_albedo_fesom28/`
(upstream-format, = the 692f804 twin's `namelists_jaxphys` with the Albedo inputs) with the REcoM setup of
`../namelists_albedo_spectral_jaxphys/` (the fork's physics-matched spectral set).

| file | source |
|---|---|
| config, cvmix, dyn, forcing, ice, oce | `../namelists_albedo_fesom28/` unchanged |
| recom | `../namelists_albedo_spectral_jaxphys/namelist.recom` unchanged (4 PFT + CDOM, `&spectral`, OASIM v2, `DIC_PI`) |
| tra | `../namelists_albedo_fesom28/namelist.tra` with `nml_tracer_list` = T, S + the fork's 37 BGC tracers 1001-1037 (same advection settings) and `&tracer_init3d` = the fork's (its ID order: entry 5 = DIC, the entry `DIC_PI` overrides in the fork) |
| io | `../namelists_albedo_fesom28/namelist.io` without `xCO2atm` (no such stream in the fork) plus `NPPp GPPp NNAp ed0 es0 eu0` |

Run: `FESOM2=/albedo/work/user/sloza/SPECTRAL/fesom2.8_recom_spectral NML=<this dir> RUN=<name> DAYS=1
sbatch scripts/albedo/run_pi_recom_fork.sbatch` (the job copies the optics tables from `recom_pi/optics/`).
