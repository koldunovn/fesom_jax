# Running fesom-jax on your own FESOM2 mesh

This assumes you already have a **standard FESOM2 mesh with partitioning run through FESOM** — i.e.
you have used FESOM2's own mesh/partitioning tooling and it produced the `dist_<N>/` directories.
That is the starting point; everything below is what fesom-jax needs on top of it.

There are **two preparation steps**, both one-off, both pure Python (no C or Fortran build):

1. **`prepare_mesh.py`** — FESOM's ASCII mesh → the dense `.npy` bundle the model loads.
2. **`cache_phc_ic.py`** — the PHC climatology → a temperature/salinity initial state on *your* mesh.

Neither runs per-timestep. Do them once, point a config at the result, and you are running.

> You do **not** need any of this for the shipped meshes: `pi` ships inside the package, and the
> CORE2 package on Zenodo ships both the prepared bundle and the raw files
> (see [`DATA.md`](DATA.md)).

---

## 0. What you should already have

fesom-jax reads **seven** files from a FESOM2 mesh directory:

| file | what it is | where it comes from |
|---|---|---|
| `nod2d.out` | node coordinates | **the mesh itself** |
| `elem2d.out` | triangles | **the mesh itself** |
| `aux3d.out` | vertical axis + node depth | **the mesh itself** |
| `nlvls.out` | levels per node | *derived* — FESOM's mesh setup |
| `elvls.out` | levels per element | *derived* — FESOM's mesh setup |
| `edges.out` | edge list | *derived* — FESOM's mesh setup |
| `edge_tri.out` | edge → adjacent triangles | *derived* — FESOM's mesh setup |

Only the first three come with a bare mesh; the other four are produced by FESOM (`fvom_init`)
when you set the mesh up and partition it. If you have run FESOM on this mesh, they are already
sitting in the mesh directory next to the `dist_<N>/` folders. (`edgenum.out` is not used.)

⚠️ **The derived files are not immutable.** `nlvls.out`/`elvls.out` encode the bathymetry as level
counts, and re-running FESOM's mesh setup can change them. That is not hypothetical: the community
CORE2 mesh had its level files regenerated on 2026-07-03, shifting 2 nodes and 4 elements in the Ross
Sea. If you re-prepare a mesh and results move, check these two files first.

Throughout, keep two directories straight:

```
MESH_RAW   # FESOM's directory: the 7 files above + dist_2/ dist_4/ ... — you already have this
MESH_JAX   # what prepare_mesh.py writes: the .npy bundle fesom-jax loads
```

The `dist_<N>/` partitions **stay in `MESH_RAW`**; the model reads them from there directly.

---

## 1. Prepare the mesh

```bash
python scripts/prepare_mesh.py  MESH_RAW  MESH_JAX
```

Pure numpy, and about **15 s for a 127 k-node mesh** (minutes at multi-million-node scale). It writes
32 `.npy` arrays plus a `meta.txt` — the format in [`MESH_EXPORT_LAYOUT.md`](MESH_EXPORT_LAYOUT.md).

**Why a conversion step at all?** FESOM derives ~32 geometric arrays at startup — cell areas,
gradient operators, edge geometry, Coriolis, metric factors. fesom-jax does that derivation **once,
offline**, so the arrays the kernels use are bit-identical to the ones the C/Fortran model uses.
(Association order in the reductions is load-bearing; re-deriving per run would be both slower and a
source of drift.) `prepare_mesh.py` is a numpy port of FESOM's `fesom_mesh_compute_metrics`, checked
array-by-array against the C exporter: integers exact, floats to ≤1e-13 relative.

It also **orients every triangle clockwise** (FESOM's `orient_cw`) before deriving geometry, which is
what `load_mesh` later asserts. If you ever see a CW-orientation failure, the mesh did not come
through `prepare_mesh.py`.

Check it loads:

```python
from fesom_jax.mesh import load_mesh
mesh = load_mesh("MESH_JAX")            # runs the CW-orientation check on the way in
print(mesh.nod2D, mesh.elem2D, mesh.nl)
```

---

## 2. Build an initial state for the mesh

The model needs temperature and salinity interpolated onto *your* nodes. This is the PHC 3.0 winter
climatology (see [`DATA.md`](DATA.md) for the file and how to point at it):

```bash
python scripts/cache_phc_ic.py --mesh-dir MESH_JAX --out-dir IC_MYMESH
```

That writes `T_ic.npy` / `S_ic.npy`. It is the slow step at large mesh sizes — but you only do it
once per mesh.

---

## 3. Check the mesh before you trust a run

Three gates, cheap, in increasing strength.

**a. Does the forcing interpolation set up on this mesh?** New meshes break on poles, the dateline,
and coastlines — this catches all three:

```bash
python scripts/debug/check_forcing.py MESH_JAX --year 1958 --ic-dir IC_MYMESH
```

It builds the runtime interpolation weights and asserts one step of forcing is finite everywhere.
(Forcing is interpolated at runtime, exactly as FESOM does it — nothing is pre-interpolated to disk;
see [`FORCING_MESHES.md`](FORCING_MESHES.md).)

**b. Does a short forward run stay finite?**

```python
import jax.numpy as jnp
from fesom_jax.mesh import load_mesh
from fesom_jax.ssh import build_ssh_operator
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.integrate import integrate
from fesom_jax.diagnostics import state_diagnostics, format_diagnostics

mesh   = load_mesh("MESH_JAX")
state0 = core2_initial_state(mesh, "IC_MYMESH")      # mesh-agnostic despite the name
op     = build_ssh_operator(mesh, dt=DT)
stress = jnp.zeros((mesh.elem2D, 2))

state = integrate(state0, mesh, op, stress, n_steps=20, dt=DT)
print(format_diagnostics(state_diagnostics(state), label="new mesh"))
```

You want `FINITE`, temperatures inside roughly −2…31 °C, and velocities that are not climbing. A
`NaN`, or `max|uv|` growing every step, means the timestep is too long — see below.

**c. Compare against a FESOM run on the same mesh**, if you have one. Expect *climate-close, not
bit-identical*: floating-point reassociation in the scatters/reductions makes the two diverge at the
~1e-12 floor and then chaotically. That is the same behaviour as the C and Kokkos ports.

---

## 4. Choose the timestep

`dt` is **CFL-bound and mesh-specific** — it is the one number you cannot copy from another mesh. The
values known to work here:

| mesh | nodes | levels | dt (cold start) |
|---|---|---|---|
| pi | 3.1 k | 48 | 100 s |
| CORE2 | 127 k | 48 | 1800 s |
| farc | 638 k | 48 | 900 s |
| dars | 3.16 M | 57 | 180 s |
| NG5 | 7.4 M | 70 | 180 s |

Scale from the mesh with the closest *minimum* resolution, and start conservative: a too-long `dt`
shows up as growing `max|uv|` and then `NaN` within a few hundred steps. If a mesh is stable at a
short `dt` but you want a longer production one, `dt_ramp` in the run YAML switches over after a set
step ([`USER_GUIDE.md`](USER_GUIDE.md)).

---

## 5. Write a config and run it

```yaml
# configs/mymesh.yaml
ale:  {}                 # zstar vertical coordinate
gm:   {}                 # eddy parameterization -- keep ON for coarse (~1 deg) meshes, OFF if eddy-resolving
kpp:  {}                 # vertical mixing (kpp XOR tke -- exactly one)
tke:  null
ice:  {whichEVP: 1}      # mEVP sea ice
visc: {}                 # horizontal viscosity; the finest meshes may want {gamma1: 0.2}
dt:   900.0              # YOUR mesh's CFL-safe timestep (section 4)

mesh:      MESH_JAX
partition: dist_8        # or `serial` / null for one device
forcing:   {kind: core2, start_year: 1958}
output_dir: runs/mymesh
duration:   1yr
```

```bash
# single device (CPU or one GPU): no partition needed
python scripts/run_from_config.py configs/mymesh.yaml \
    --mesh-dir MESH_JAX --ic-dir IC_MYMESH \
    --partition serial --steps 480 --restart-out runs/mymesh/seg0

# multi-GPU: --dist-dir points at the RAW mesh dir, where FESOM left dist_<N>/
python scripts/run_from_config.py configs/mymesh.yaml \
    --mesh-dir MESH_JAX --dist-dir MESH_RAW --ic-dir IC_MYMESH \
    --partition dist_8 --steps 480 --restart-out runs/mymesh/seg0
```

The partition name must match a directory that exists: `--partition dist_8` reads
`MESH_RAW/dist_8/`. `serial` (or `dist_1`) synthesises a single-device partition and reads nothing.

Multi-job campaigns are a SLURM dependency chain — `scripts/chain_submit.sh`, and
[`USER_GUIDE.md`](USER_GUIDE.md) for the full YAML schema, restarts and output.

---

## Gotchas, in the order people hit them

- **`FileNotFoundError` on `nlvls.out` / `edges.out`** — the mesh was never set up through FESOM.
  Those four derived files come from FESOM's own tooling, not from the mesh repository.
- **CW-orientation assertion in `load_mesh`** — you pointed it at a raw FESOM directory, or at a
  bundle some other tool wrote. Run it through `prepare_mesh.py`.
- **`dist_<N>/` not found** — the partitions live in `MESH_RAW`, not in the prepared `MESH_JAX`.
  Pass `--dist-dir MESH_RAW`.
- **Blows up in a few hundred steps** — `dt` too long for this mesh (section 4).
- **Out of memory on a big mesh** — the compiled step's working set is heavier per node than
  FESOM's hand-managed memory, so the largest meshes need a minimum device count (dars ≥ 2 nodes,
  NG5 ≥ 8 nodes on A100s). Spread it wider rather than shrinking the timestep.
- **Bit-comparing against a C dump** — the PHC initial condition is *partition-dependent* (the
  land-fill is an order-dependent Gauss–Seidel sweep run per rank), so an IC meant to match a C run
  bit-for-bit must be built with that run's partition. For ordinary science runs this does not
  matter.
