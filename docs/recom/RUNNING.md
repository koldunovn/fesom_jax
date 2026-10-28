# Running fesom-jax with REcoM on Levante

For a colleague with a Levante account who wants to run the CORE2 + REcoM setup of the results page (the 2000–2019
hindcast) or a variant of it. Everything below except the code is already on Levante and readable by every user.

## What you need

- A Levante account.
- GPU time, one of:
  - **dolpung** (one node = 4 × NVIDIA GH200): the tested route. It needs membership in project `mh1571`, ask
    Nikolay. The partition is small and shared, so tell him before you start a run.
  - **gpu** (one node = 4 × NVIDIA A100 80 GB) with your own project's `<project>_gpu` account. It runs the same setup;
    it has not been timed with the current code.
- About 85 GB on `/work` for a 20-year run: 34 GB monthly means, 51 GB yearly restarts.

## Get the code

```bash
git clone https://github.com/koldunovn/fesom_jax.git fesom_jax_recom
cd fesom_jax_recom
ln -s /work/ab0995/a270088/port_jax/data data      # the CORE2 mesh the configs read as data/mesh_core2
```

No installation: the run scripts use the existing Python environments on `/work` (an aarch64 one for dolpung, an
x86 one for the A100 nodes).

## Run the 2000–2019 hindcast

From the clone, on a login node (this only submits the job):

```bash
WT=$PWD
OUT=/work/<project>/$USER/recom/core2_2000_2019          # must not exist yet: the script refuses to overwrite
LOGS=/work/<project>/$USER/recom/logs; mkdir -p $LOGS
sbatch -A mh1571 -o $LOGS/%x.%j.out --time=08:00:00 -J recom_2000_2019 \
  --export=ALL,WT=$WT,OUT=$OUT,CONFIG=$WT/configs/core2_gpu_recom_v3_2000_co2.yaml,CMDBUF=on \
  scripts/perf/run_core2_gpu_gh200.sbatch \
  --restart-in /work/ab0995/a270088/port_jax/runs/core2_hindcast_v3/restart_archive/fesom.2000.001.00000 \
  --steps 233760 --progress --diagnostics \
  --monthly-out $OUT/monthly --restart-out $OUT/restart \
  --restart-archive-out $OUT/restart_archive --restart-archive-period year
```

On the A100 nodes: `-A <project>_gpu` and `scripts/perf/run_core2_gpu.sbatch` instead of the `_gh200` script,
everything else the same.

- **Time**: 4 hours on one GH200 node in double precision (123 model years a day). The setup and the compilation
  take the first minute; after that the log prints one progress line every three model days.
- **Steps**: the time step is 45 minutes, 32 steps a day. One year is 11680 steps (11712 in a leap year); 233760 is
  2000–2019 including the five leap days.
- **Start**: ocean and sea ice from 1 January 2000 of our 1958–2019 JRA55-do hindcast. REcoM cold-starts from WOA13
  nitrate and silicate, WOA18 oxygen, GLODAPv2 present-day DIC and alkalinity, and the PISCES iron field, as in
  FESOM2's `bin_2p3z2d` configuration. Atmospheric CO2 is the monthly Global Carbon Budget 2023 series (1750–2023).

## Common changes

All of them are a copy of the config with a few lines changed (or one extra flag), then the same `sbatch`.

| change | how |
|---|---|
| REcoM parameters | copy `scripts/recom/fortran/namelists_jaxphys_co2/namelist.recom`, edit it, point `recom.namelist` at the copy. It is read as FESOM2 reads it; a switch the port does not support stops the run at start-up instead of being ignored. |
| single precision | add `--precision single`: 227 instead of 123 model years a day (about 2 hours for 20 years). FESOM2 does not allow single precision with REcoM, so this is our deviation; one CORE2 year gave the same global NPP, export and CO2 flux as double precision to three digits. |
| constant CO2 | `recom.namelist: scripts/recom/fortran/namelists_jaxphys/namelist.recom` (constant_CO2 on, CO2_for_spinup from the namelist), `recom.ic_set: core2` (GLODAPv2 preindustrial DIC); `forcing.co2_file` is then not read. |
| what is written | `recom.output`, any of DIC, Alk, DIN, DSi, DFe, O2, Chl, co2flux, pco2surf, npp, export100, ph, o2flux. The 3-D fields take most of the disk. |
| another start year | `--restart-in .../restart_archive/fesom.<YEAR>.001.00000` (1959–2020 exist) and, in the config, `dt_ramp.after_step` = 48 × the days from 1958-01-01 to 1 January of that year: the hindcast ran at 30 minutes, the REcoM runs switch to 45 there. `python -c "import datetime as d; print(48 * (d.date(1990,1,1) - d.date(1958,1,1)).days)"` |
| continue a run | `--restart-in $OUT/restart_archive/fesom.<YEAR>.001.00000` (BGC included) into a new `OUT`, the same config. |

## What comes out

- `$OUT/monthly/<YYYY_MM>/`: one zarr store a month with the monthly means of `recom.output` and the physics
  (temp, salt, u, v, ssh, a_ice, m_ice), on the CORE2 mesh nodes in FESOM2's node order (`nod2`, 48 levels for 3-D
  fields), with `lon` and `lat`. Units as REcoM: tracers in mmol m⁻³ (Chl in mg m⁻³), npp, export100 and co2flux in
  mmol C m⁻² d⁻¹ (co2flux positive into the ocean), pco2surf in µatm. Read it with
  `xarray.open_zarr("$OUT/monthly/2010_01", consolidated=False)`.
- `$OUT/restart_archive/fesom.<YYYY>.001.00000`: the full state on 1 January of every year, BGC included.
- The log in `$LOGS`: the setup, then progress lines.

The results page's figures come from `scripts/recom/page/make_figures.py` (run it with
`/work/ab0995/a270088/mambaforge/envs/nereus/bin/python`, on a compute node, not the login node); the observations on
the CORE2 mesh it needs are in `/work/ab0995/a270088/fesom_jax_dev/runs/recom/page/obs/obs_on_mesh.npz`:

```bash
python scripts/recom/page/make_figures.py $OUT/monthly --obs <that npz> --out <figure dir> --label "my run"
```

## What is not ported

Coccolithophores and Phaeocystis, rivers, erosion, aeolian nitrogen, carbon isotopes, the atmospheric CO2 box, the
Geider limiter, MEDUSA sediments. The rest of `bin_2p3z2d` (31 tracers, four benthic pools, mocsy) is, and on the pi
mesh it reproduces FESOM2-REcoM to round-off: [VALIDATION.md](VALIDATION.md).
