# Surface forcing across meshes (CORE2 / farc / dars / NG5)

**The forcing is interpolated AT RUNTIME, exactly like FESOM** — `JRA55Reader.__init__(mesh, year)`
builds the bilinear interpolation weights ONCE at setup (FESOM's `sbc_ini`), then `.step(day, sec)`
reads the JRA slices off disk and bilinear-interps to the mesh nodes + time-interps **every step**.
SSS-restoring / runoff / Sweeney-chl readers work the same way. **Nothing is pre-interpolated to
disk.** So "staging" a mesh's forcing means only: (1) the SOURCE files resolve (they are
mesh-independent, native-grid), and (2) the runtime interpolation *setup* initializes for that mesh
(no pole / dateline / coast out-of-bounds at scale). Verify with `scripts/debug/check_forcing.py MESH_DIR`.

## Source files (mesh-independent, native grid — all resolve on Levante)

| field | reader | path |
|---|---|---|
| JRA55-do atmosphere (u/v wind, shum, sw, lw, Tair, rain, snow) | `jra55.JRA55Reader` | `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0` |
| SSS restoring target | `sss_runoff` | `…/JRA55-do-v1.4.0/PHC2_salx.nc` |
| runoff | `sss_runoff` | `…/JRA55-do-v1.4.0/CORE2_runoff.nc` |
| Sweeney chlorophyll climatology | `sss_runoff` | `/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc` |

These are the same inputs that drove the CORE2 5-yr spin-up + 10-yr reference; no per-mesh copies.

## Per-mesh status

| mesh | raw FESOM mesh (Levante) | JAX-exported? | forcing-init smoke |
|---|---|---|---|
| CORE2 | `/pool/.../MESHES_FESOM2.1/core2` | yes (`data/mesh_core2`) | ✅ `NG5_FORCING_OK` (all 10 fields finite; `scripts/debug/check_forcing.py data/mesh_core2`) |
| farc | `/pool/.../MESHES_FESOM2.1/farc` | ⏳ needs export (B0) | run `check_forcing.py` after export |
| dars | `/pool/.../MESHES_FESOM2.1/dars` | ⏳ needs export (B0) | run `check_forcing.py` after export |
| NG5 | `/pool/.../MESHES_FESOM2.1/ng5` | ⏳ needs export (B0) | run `check_forcing.py` after export |

**The only per-mesh step is the JAX mesh preparation** (`load_mesh` needs the `.npy` layout, not the raw
FESOM `nod2d.out`/`elem2d.out`). This is now a **pure-Python, offline** step —
`python scripts/prepare_mesh.py RAW_MESH_DIR OUT_DIR` (Task A8, no C build needed; verified byte-faithful
vs the C export, `docs/MESH_EXPORT_LAYOUT.md`). Once a mesh is prepared, the runtime forcing pipeline is
mesh-agnostic — `check_forcing.py` confirms the bilinear-weight setup + a finite 1-step interpolation.
**De-risk on farc/dars before NG5** (cost + the 7.4 M-node out-of-bounds edge cases).

## What was verified (CORE2)

`scripts/debug/check_forcing.py data/mesh_core2` (126 858 nodes): all 10 `StepForcing` fields finite with
physical ranges — Tair −42…+31 °C, shortwave 0…1122 W/m², shum 2e-5…0.023, SSS 3.8…39.3, chl
6.4e-5…14.5 — ⇒ `NG5_FORCING_OK`. The runtime interpolation + the bulk inputs are sound; farc/dars/NG5
re-run the same one-line check after the mesh export.
