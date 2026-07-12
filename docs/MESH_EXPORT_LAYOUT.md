# Mesh export layout (Task 0.3)

> **Using your own FESOM2 mesh?** The practical recipe — the two preparation steps, the
> sanity gates, picking `dt` — is [`NEW_MESH.md`](NEW_MESH.md). This document is the *format
> contract* that a prepared bundle must satisfy.

The C port (`fesom2_port`, branch `jax-mesh-export`) writes its **static
mesh/geometry** arrays so the JAX port consumes *exactly* what the C kernels
compute (gradients, areas, Coriolis, edge geometry) rather than re-deriving them.

> **Standalone preparation (no C build needed) — `scripts/prepare_mesh.py` (Task A8).** A pure-numpy
> port of `fesom_mesh.c`'s mesh setup produces **this exact layout** from the raw FESOM ASCII
> (`nod2d/elem2d/aux3d.out` + `nlvls/elvls/edges/edge_tri.out`), run **once, offline**:
> `python scripts/prepare_mesh.py RAW_MESH_DIR OUT_DIR`. It is verified byte-faithful against the C
> export below (all 32 arrays on CORE2: ints exact, floats rel ≤1e-13 — `prepare_mesh.py --verify
> C_EXPORT_DIR`). So a shipped model can prepare a mesh in Python; the C exporter is no longer
> required (it remains the original provenance / cross-check). The format below is the contract both
> producers honour.

## How it is produced

`src/fesom_mesh_export.c` adds `fesom_mesh_export(mesh, partit, dir)`, called
from `fesom_main.c` right after `fesom_mesh_compute_metrics` +
`fesom_mesh_alloc_state`, **env-gated**:

```bash
FESOM_MESH_EXPORT=/path/to/export_dir  fesom_port <mesh_dir> ...
```

Run with **npes == 1** so the exported arrays are the *global* mesh (in serial
each local array == the global array: `myDim == global`, `eDim == 0`). The
exporter aborts the export (with a warning) if `npes != 1`.

## Format

One `<name>.npy` per array — standard NumPy v1.0 format, so dtype and shape are
**self-describing** and `np.load` reads each natively (no custom parser). Plus a
`meta.txt` of scalars/counts (`key value` per line). `real_t` is `double` →
`<f8`; C `int` → `<i4`. Arrays are C order (row-major); a `[n*k]` C array maps to
shape `(n, k)`. All indices are **0-based** (converted from the 1-based mesh
files on read). Counts for pi: `nod2D=3140`, `elem2D=5839`, `nl≈23`.

## meta.txt (scalars)

| key | meaning |
|---|---|
| `nod2D` | total nodes (== global for npes==1) |
| `elem2D` | total elements |
| `edge2D` | total edges |
| `nl` | max vertical levels |
| `edge2D_in` | # interior edges; IDs `[0, edge2D_in)` interior, `[edge2D_in, edge2D)` boundary |
| `myDim_edge2D` | owned edges (== edge2D for npes==1); length of `edge_up_dn_tri` |
| `npes` | rank count at export (must be 1) |
| `ocean_area` | total open-ocean surface area Σ areasvol(surface), m² |

## Arrays

### Nodes (shape leading dim `nod2D`)
| file | shape | dtype | units / meaning | notes |
|---|---|---|---|---|
| `coord_nod2D` | (nod2D,2) | f8 | (lon,lat) **rotated** radians | computations are in rotated frame |
| `geo_coord_nod2D` | (nod2D,2) | f8 | (lon,lat) **geographic** radians | IC/forcing interpolation |
| `coast_flag` | (nod2D,) | i4 | 0=interior, 1=coast | |
| `nlevels_nod2D` | (nod2D,) | i4 | K_v⁺ (MAX over cells) | levels per node (tracers/W) |
| `nlevels_nod2D_min` | (nod2D,) | i4 | K_v⁻ (MIN over cells) | ALE deformation limit |
| `ulevels_nod2D` | (nod2D,) | i4 | upper level (=1 no cavity) | |
| `ulevels_nod2D_max` | (nod2D,) | i4 | MAX of ulevels over cells | GM |
| `depth` | (nod2D,) | f8 | bathymetry, m | input metadata only |
| `mesh_resolution` | (nod2D,) | f8 | Voronoi diameter, m (3-pass smooth) | GM/Redi |
| `coriolis_node` | (nod2D,) | f8 | 2Ω sin(geo lat), s⁻¹ | |
| `area` | (nod2D,nl) | f8 | upper-edge scalar CV area, m² | |
| `areasvol` | (nod2D,nl) | f8 | "mid" CV area, m² | == area without cavity |
| `zbar_3d_n` | (nod2D,nl) | f8 | per-node interface depths, m (≤0) | |
| `nod_in_elem2D_offsets` | (nod2D+1,) | i4 | CSR offsets into `nod_in_elem2D` | |

### Elements (leading dim `elem2D`)
| file | shape | dtype | units / meaning | notes |
|---|---|---|---|---|
| `elem_nodes` | (elem2D,3) | i4 | node ids, 0-based, **CW** | orientation-normalized |
| `nlevels` | (elem2D,) | i4 | K_c per-cell level count | |
| `ulevels` | (elem2D,) | i4 | upper level per cell | |
| `elem_area` | (elem2D,) | f8 | cell area, m² | |
| `elem_cos` | (elem2D,) | f8 | cos(rotated lat at centroid) | |
| `metric_factor` | (elem2D,) | f8 | tan(rot lat)/R_earth, m⁻¹ | |
| `coriolis` | (elem2D,) | f8 | 2Ω sin(geo lat at centroid), s⁻¹ | |
| `elem_center_x` | (elem2D,) | f8 | centroid x, rotated radians | cyclic-aware |
| `elem_center_y` | (elem2D,) | f8 | centroid y, rotated radians | |
| `gradient_sca` | (elem2D,6) | f8 | ∂N_i/∂x (cols 0..2), ∂N_i/∂y (cols 3..5), 1/m | linear shape-fn gradients |

### Edges (leading dim `edge2D`, except `edge_up_dn_tri`)
| file | shape | dtype | units / meaning | notes |
|---|---|---|---|---|
| `edges` | (edge2D,2) | i4 | node ids, 0-based | |
| `edge_tri` | (edge2D,2) | i4 | adjacent elem ids, 0-based; **−1** = boundary/halo-absent | |
| `edge_dxdy` | (edge2D,2) | f8 | node2−node1, rotated radians | |
| `edge_cross_dxdy` | (edge2D,4) | f8 | (cell1c−edgemid, cell2c−edgemid), **meters** (lon×elem_cos) | cols 2,3 zero on boundary |
| `edge_up_dn_tri` | (myDim_edge2D,2) | i4 | MFCT up/down-wind triangle per edge, 0-based; −1 absent | length `myDim_edge2D` |

### Flat / vertical
| file | shape | dtype | units / meaning | notes |
|---|---|---|---|---|
| `nod_in_elem2D` | (Σcounts,) | i4 | CSR flat; node n's cells = `[off[n]:off[n+1]]`, 0-based | |
| `zbar` | (nl,) | f8 | interface depths, m (≤0, downward) | |
| `Z` | (nl-1,) | f8 | mid-layer depths = ½(zbar[k]+zbar[k+1]) | |

**Not exported** (these are evolving *state*, not static geometry; the JAX
`State` pytree owns them): `hnode`, `hnode_new`, `helem`, `hbar`, `hbar_old`,
`bc_index_nod2D`. **`elem_edges` is intentionally omitted** — it is not used
anywhere in the C port (verified by grep), so the JAX port does not need it
either (edge→element scatters use `edge_tri`).
