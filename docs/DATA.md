# Input data: where it lives, how to point the model at it

> **The `pi` mesh needs none of this.** It ships inside the package, so
> `pip install fesom-jax` is enough to run the model and take gradients — that is
> [`examples/01_pi_quickstart.ipynb`](../examples/01_pi_quickstart.ipynb). Read on only if you
> want the **realistic** setup ([`examples/02_core2_realistic.ipynb`](../examples/02_core2_realistic.ipynb)).

## Quick start: get one year of CORE2

Everything needed to run the CORE2 (~1°) global ocean for one year is published as a single Zenodo
record, in two archives:

| Archive | Size | Contents |
|---|---|---|
| `core2_mesh_ic.zip` | ~370 MB | the mesh (dense `.npy` + raw FESOM text), the cached PHC initial state, the PHC source file, SSS restoring, river runoff, chlorophyll |
| `core2_forcing_1958.zip` | ~10.4 GB | the eight JRA55-do fields for 1958 |

```bash
python scripts/fetch_data.py --dest ~/fesom-data --record <ZENODO_RECORD_ID>
eval "$(python scripts/fetch_data.py --dest ~/fesom-data --print-env)"   # set the env vars
```

Add `--mesh-only` to skip the 10.4 GB forcing (enough to load the mesh and plot the initial state,
not enough to run). The script verifies checksums and can resume.

### Running more than one year

The Zenodo archive carries **1958 only**, to keep the download reasonable. For a longer run, get the
remaining years of **JRA55-do v1.4.0** from the original source — it is distributed through **ESGF**
as part of the CMIP6 `input4MIPs` collection (source id `MRI-JRA55-do-1-4-0`); the DKRZ copy used by
this project lives at `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0`.

Whatever route you use, the reader wants **one NetCDF per field per year**, named `{var}.{year}.nc`,
in a single directory pointed at by `$FESOM_JRA_DIR`, with these eight variables:

```
uas  vas   huss  rsds  rlds  tas   prra  prsn
(the two 10 m wind components, humidity, shortwave, longwave, air temperature, rain, snow)
```

ESGF ships input4MIPs files under a longer CMIP-style name, so a rename (and, if the files are split
within a year, a concatenation) is needed to match the `{var}.{year}.nc` convention. Nothing else about
the files is special — same grid, same units.

Please cite JRA55-do if you use it: Tsujino et al. (2018),
[doi:10.1016/j.ocemod.2018.07.002](https://doi.org/10.1016/j.ocemod.2018.07.002).

### ⚠️ A note on the CORE2 mesh version

The vertical level files of the CORE2 mesh (`nlvls.out`, `elvls.out`) were regenerated upstream on
2026-07-03. **The Zenodo package ships the earlier version**, because that is what every `fesom-jax`
result was produced with. The two differ at exactly **2 nodes and 4 elements**, all in the Ross Sea
(~154 °W, 77 °S), where the shipped version is deeper (580 m vs 280 m at the largest difference).
Both are structurally valid. If you need to match a *current* FESOM2 run instead, take
`nlvls.out`/`elvls.out` from [the upstream mesh](https://gitlab.awi.de/fesom/core2) and rebuild the
dense bundle with `scripts/prepare_mesh.py <raw_dir> <out_dir>`.

---

## The full reference

Only **CORE2-and-larger** runs (core2, farc, dars, forca20, ng5) read external data: JRA55-do
atmospheric forcing, the PHC initial condition, the SSS-restoring and runoff files, the chlorophyll
climatology, and the mesh/partition directories.

Every one of those paths is resolved through `fesom_jax/paths.py`, with one precedence rule:

```
explicit argument / run-YAML key   >   environment variable   >   Levante default
```

The environment is read at call time, so exporting a variable in a job script (or in a notebook
after `import fesom_jax`) always takes effect. On Levante, setting nothing reproduces the old
hardcoded behaviour exactly.

## The six inputs

| Input | What it is | Env var | `forcing:` YAML key | Levante default |
|---|---|---|---|---|
| JRA55-do forcing | Per-year daily/3-h atmospheric forcing (`uas,vas,huss,rsds,rlds,tas,prra,prsn`), one NetCDF per field per year. The **only** forcing the model uses. | `FESOM_JRA_DIR` | `jra_dir` | `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0` |
| PHC initial condition | `phc3.0_winter.nc` — the T/S climatology interpolated onto the mesh for a cold start. | `FESOM_PHC_PATH` | — (`load_phc_ic(path=…)`) | `/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc` |
| SSS restoring | `PHC2_salx.nc` — monthly surface-salinity climatology (the restoring target). | `FESOM_SSS_PATH` | `sss_path` | `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/PHC2_salx.nc` |
| Runoff | `CORE2_runoff.nc` — the constant river-runoff field. | `FESOM_RUNOFF_PATH` | `runoff_path` | `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/CORE2_runoff.nc` |
| Chlorophyll | `Sweeney_2005.nc` — monthly chl climatology for the shortwave penetration. (`chl_const` in code replaces it with a constant.) | `FESOM_CHL_PATH` | `chl_path` | `/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc` |
| Mesh / partition root | Directory holding the mesh dirs, each with its `dist_<N>/` domain decompositions. Usually given in full as the YAML `mesh:` key / `--mesh-dir`; the root is only used to *compose* one (`partit.default_mesh_dir`). | `FESOM_MESH_ROOT` | — (YAML `mesh:` / `--mesh-dir`) | `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1` |

One extra, repo-local (not a `/pool` path): the **cached PHC IC** directory holding the
interpolated `T_ic.npy` / `S_ic.npy` — default `data/ic_core2` (repo-relative), overridable with
`FESOM_IC_DIR` or `--ic-dir`. Build it once with `phc_ic.build_and_cache_ic(mesh)`.

## Setting them

Environment (a whole machine / job script at once):

```bash
export FESOM_JRA_DIR=/data/JRA55-do-v1.4.0
export FESOM_PHC_PATH=/data/phc3.0/phc3.0_winter.nc
export FESOM_SSS_PATH=/data/JRA55-do-v1.4.0/PHC2_salx.nc
export FESOM_RUNOFF_PATH=/data/JRA55-do-v1.4.0/CORE2_runoff.nc
export FESOM_CHL_PATH=/data/Sweeney/Sweeney_2005.nc
export FESOM_MESH_ROOT=/data/MESHES_FESOM2.1
export FESOM_IC_DIR=/scratch/me/ic_core2        # optional: IC cache off-repo
```

Run YAML (per run; wins over the environment):

```yaml
forcing:
  kind: core2
  start_year: 1958
  jra_dir: /data/JRA55-do-v1.4.0        # optional
  sss_path: /data/JRA55-do-v1.4.0/PHC2_salx.nc
  runoff_path: /data/JRA55-do-v1.4.0/CORE2_runoff.nc
  chl_path: /data/Sweeney/Sweeney_2005.nc
mesh: /data/MESHES_FESOM2.1/core2
```

The `forcing:` block is strict-keyed: only `kind`, `start_year`, `jra_dir`, `sss_path`,
`runoff_path`, `chl_path` are accepted; anything else raises at config load.

Python (explicit argument; wins over everything):

```python
from fesom_jax import core2_forcing
forcing = core2_forcing.build_core_forcing(mesh, 1958, sst_ic=sst0,
                                           jra_dir="/data/JRA55-do-v1.4.0")
```

## Checking what will be used

```python
from fesom_jax import paths
for row in paths.describe():
    print(f"{row['name']:12s} {row['resolved']}  exists={row['exists']}")
```

A missing file raises at read time with the env var, the YAML key, and this document named:

```
JRA55-do forcing not found: /pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0 —
set $FESOM_JRA_DIR, or the `forcing.jra_dir` key in the run YAML. On Levante the
default should work; elsewhere see docs/DATA.md.
```
