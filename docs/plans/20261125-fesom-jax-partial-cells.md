# fesom-jax partial bottom cells (FESOM2 `use_partial_cell`)

## Overview

Port FESOM2's partial bottom cells into fesom-jax as a run option, following the Fortran FESOM2 code
(GitHub issue koldunovn/fesom_jax#3). With partial cells, the bottom layer of every column ends at the real
topographic depth instead of the nearest model interface. The adjustment is bounded: it can thin the bottom
layer down to half its nominal thickness, or deepen it by up to half the layer below. This matters for meshes
whose bathymetry varies a lot within one model layer. The issue's case is vertically coarsened data from another
model.

- Reference: Fortran FESOM2 upstream (`/work/ab0995/a270088/fesom2_upstream`, dbf28153). The geometry is
  `oce_ale.F90:244-460` (`init_bottom_elem_thickness`, `init_bottom_node_thickness`) and `init_ale`
  (`:202-237`), and `init_thickness_ale` / `update_thickness_ale` / `restart_thickness_ale` (`:586-1211`) use
  it. FESOM3 (`~/fesom3`) has the namelist flag but no implementation (its drivers always build full cells), so
  FESOM2 is the only reference.
- Namelist (FESOM2 `&ale_def`): `use_partial_cell` (default `.false.`) and `partial_cell_thresh` (default
  0 m). FESOM2 itself labels the option "not recommended" in `config/namelist.config`.
- Scope: zstar and linfs (user, 2026-11-25). Not in scope, each raising a clear error: cavity partial cells
  (`use_cavity_partial_cell`), REcoM with partial cells, cavities with partial bottom cells, single precision
  with partial cells. FESOM2's `use_depthonelem` has no JAX key, so nothing to guard there.
- `use_partial_cell: false` (the default) must trace the identical program and stay byte-identical. The
  existing suite checks this.

## Context (from discovery)

- **Level counts do not change.** The partitioner's `find_levels` (`mod_mesh_utils.F90:530`) sets
  `nlevels` from the depth against the mid-depths `Z`, independently of partial cells. The exported meshes
  (`nlvls.out`/`elvls.out` → `.npy`) therefore stay valid, and `mesh.depth` (node, ≤ 0) already holds the
  bathymetry the option needs.
- **Depth clamp.** FESOM2 clamps node depth to `zbar(thers_zbar_lev)` = `zbar(5)` = −30 m on read
  (`oce_mesh.F90:619`). Our `depth.npy` is not clamped. On CORE2, 13,825 nodes lie between −20 and −30 m, and
  for them the clamp decides the partial bottom. The builder applies the clamp to its own copy and leaves
  `mesh.depth` unchanged. The only other readers of `mesh.depth` are NN-only inputs (`gm.gm_nn_features`,
  `tke_nn`), which have no FESOM2 counterpart.
- **What reads the partial geometry in FESOM2.** The bottom interfaces `zbar_e_bot` (element) and
  `zbar_n_bot` (node), the bottom thicknesses, and the bottom entries of `zbar_3d_n`/`Z_3d_n`/`hnode`/`helem`
  are read by: the PGF (`oce_ale_pressure_bv.F90`, every scheme anchors `zbar_n(nle+1)=zbar_e_bot`), the SSH
  stiffness (`oce_ale.F90:1362`), implicit vertical viscosity (`oce_ale.F90:3380`), vertical tracer advection
  (`oce_adv_tra_ver.F90:65`), implicit diffusion and Redi (`oce_ale_tracer.F90:1280,1705`), GM
  (`oce_fer_gm.F90:60,723`), and the IC interpolation onto `Z_3d_n` (`gen_ic3d.F90:500`). The IC conversion
  to potential temperature deliberately stays at the nominal `|Z|` (`oce_ale_pressure_bv.F90:3346-3353`: the
  FESOM2 comment says `Z_3d_n` there made partial-cell runs unstable after init).
- **Why the JAX change stays contained.** zstar stretches only the layers above `nlevels_nod2D_min−2`
  (`ale._stretch_mask`). The partial bottom (node interface `nlevels_nod2D−1`, 0-based) always lies below
  them, because `nlevels_nod2D_min ≤ nlevels_nod2D`, so the moving part is unaffected and only the static
  fill changes. The kernels take per-node and per-element geometry through optional `Z3d`/`zbar3`/`elem_geo`
  arguments (eos, pp, kpp, tke, gm, gm_redi, tracer_adv, tracer_diff, momentum, forcing). These branches are
  purely geometric (plan review, 2026-11-25), so static partial-cell geometry can be passed under linfs too.
- **Places in JAX that build full-cell geometry** (to change under partial cells):
  `ale.init_thickness_zstar`, `ale.commit_thickness` (vertex-mean `helem`, also at the bottom),
  `ale.live_geometry` (nominal fill below the stretch range), `ale.live_geometry_elem` and
  `pgf.pressure_force_shchepetkin` (anchor `zbar[nlevels−1]`), `ssh.py:578` (stiffness depth),
  `ssh_se.cfl_omega_max` (`ssh_se.py:718`, the `helem=None` fallback only; the split-explicit dynamics use
  `He = Σhelem` and follow `helem`), `state.State.rest` (linfs `helem` nominal), the IC (cached
  `T_ic.npy`/`S_ic.npy`, interpolated on `mesh.Z` with one bracket per layer, `phc_ic.py:376,476`), the
  restart load, and `fortran_restart.py:210` (vertex-mean `helem`).
- **Setup order.** `scripts/run_from_config.py:219-255` builds the mesh, the partition, the sharded mesh, the
  SSH operator and the cold state before `fesom_jax.run.run_from_config` is called. The partial-cell geometry
  must therefore be applied right after `load_mesh` there, not inside `run.py`.
- **linfs PGF.** FESOM2 does not allow the full-cell PGF together with partial cells
  (`pressure_force_4_linfs`, `oce_ale_pressure_bv.F90:525-576`). The default `which_pgf='shchepetkin'`
  selects `pressure_force_4_linfs_shchepetkin` (`:796-1040`). It differs from the zstar version only by
  leaving out the slope term (`dz_dx`) in the top layer (when `ule==1`) and the interior layers. With static
  geometry, `Z_3d_n` is equal at the three vertices on every element layer above the bottom, so that term is
  zero up to rounding there. Our zstar kernel with a flag that drops the term off the bottom layer is
  therefore a faithful transcription.
- **Sharding.** `shard_mesh.build_sharded_mesh` partitions the `NODE_FIELDS`/`ELEM_FIELDS` lists, and
  `_default_pad` pads floats with 1.0 and `nlevels_nod2D` with 0. The two new arrays need entries there, and
  any in-trace indexing by level count must be clipped on pad lanes.
- **Fortran reference setup.** `/work/ab0995/a270088/fesom2_icefw_runs/A_main_ias2_year` is a CORE2 zstar
  setup matched to our standard physics (namelists annotated with the JAX equivalents). It was run for 1958
  with the clean upstream build `build_base` (installed from `fesom2_icefw_base`, 4413769a, no local
  changes): 998 s on 4 nodes, monthly output, `use_partial_cell=.false.`. It is the full-cell Fortran year.
  It reads the `/pool` core2 levels, which were regenerated on 2026-07-03. They differ from ours (the pre-fix
  levels) at 2 nodes and 4 elements in the Ross Sea; see the memory note core2-mesh-prefix-levels.

## Design (user-chosen, option A)

- New module `fesom_jax/partial_cells.py`. `apply(mesh, *, thresh=0.0)` returns a `Mesh` with two new
  optional fields, `zbar_e_bot (elem2D,)` and `zbar_n_bot (nod2D,)`. It also writes `zbar_n_bot` into the
  bottom slot of `mesh.zbar_3d_n`, as `init_ale` does. It is a host-numpy transcription of
  `init_bottom_elem_thickness` / `init_bottom_node_thickness`, run once at setup in
  `scripts/run_from_config.py`, right after `load_mesh`.
- `None` on those fields means full cells, the same pattern as the existing optional `inv_*`/`se_*` fields.
  Kernels branch on `mesh.zbar_e_bot is not None`, which is pytree structure and therefore fixed at trace
  time. No flag is threaded through the step/integrate/adjoint signatures.
- Small helpers in `partial_cells.py` build the rest from the two arrays:
  `bottom_elem_thickness`/`bottom_node_thickness` (`zbar[nlev−2] − zbar_*_bot`),
  `static_node_geometry(mesh)` → `(zbar3, Z3d)` (the `init_ale` formulas, with the strictly decreasing
  nominal fill below the bottom required by the AD rule D5), and
  `static_elem_geometry(mesh, helem)` → `(zbar_n, Z_n)` anchored at `zbar_e_bot`.
- RunConfig: top-level `use_partial_cell` (bool, default false) and `partial_cell_thresh` (m, default 0.0),
  named as in FESOM2 `&ale_def`. `thers_zbar_lev` stays a module constant (5, FESOM2's default).
  `fesom_jax.run.run_from_config` asserts `cfg.use_partial_cell == (mesh.zbar_e_bot is not None)`.
- linfs + partial cells: the step passes the static partial-cell geometry wherever it passes the live
  geometry under zstar, and the PGF goes through the Shchepetkin kernel with the linfs flag.

## Development Approach

- **Testing approach: Regular** (port each FESOM2 routine, then its test). The Fortran defines the expected
  result, and each test checks a JAX kernel against a literal per-element/per-node loop transcribed from the
  Fortran.
- Complete each task fully before moving to the next; small focused commits on `feature/partial-cells`
  (worktree `/work/ab0995/a270088/fesom_jax_dev/wt_partial`).
- **Every task includes new or updated tests; all tests pass before the next task starts.**
- Tests run on a compute node (sbatch, `-p compute --time=30:00`), never on the login node, with
  `PYTHONPATH=$WT` (the editable install points to `~/port_jax`).
- Keep `use_partial_cell: false` byte-identical: no change to a full-cell code path's arithmetic.
- Append a lesson per task to `docs/PORTING_LESSONS.md`.

## Testing Strategy

- **Unit tests** in `fesom_jax/tests/test_partial_cells.py`, run on pi and CORE2 meshes:
  - transcription checks (bit-identical to a literal loop over the Fortran routine);
  - degeneracy: `partial_cell_thresh` larger than every layer ⇒ `zbar_e_bot = zbar[nlevels−1]`, and one zstar
    step equals the full-cell step to `max|Δ| ≤ 1e-12·max|field_FC|` per prognostic field (same PGF scheme).
    For linfs this check does not apply, because partial cells change the PGF scheme;
  - invariants: bounds `Z(nle) ≤ zbar_e_bot ≤ Z(nle−1)`, the `nle==nl` branch, strictly decreasing geometry
    on real lanes, finite values on pad lanes, finite gradients (masked-NaN probe).
- **Full-cell regression:** the existing CPU suite (`sbatch scripts/runs/run_suite.sbatch`) unchanged.
- **Fortran gates.** G1 (geometry, against `fesom.mesh.diag.nc`) runs early, right after Task 1. G2 (the
  response of the model to partial cells, JAX against Fortran) runs in Task 8. The pass bars are fixed now,
  before any run (see Task 8).
- No UI/e2e tests in this project.

## Progress Tracking

- Mark completed items with `[x]` immediately when done.
- Add newly discovered tasks with ➕ prefix; document blockers with ⚠️ prefix.
- Update this plan if the implementation deviates from scope.

## What Goes Where

- **Implementation Steps** (`[ ]`): code, tests, runs and docs in this repository.
- **Post-Completion** (no checkboxes): the issue reply, the push, and follow-ups that need the user.

## Implementation Steps

### Task 1: Partial-cell geometry builder and Mesh fields

**Files:**
- Create: `fesom_jax/partial_cells.py`
- Modify: `fesom_jax/mesh.py`
- Create: `fesom_jax/tests/test_partial_cells.py`

- [x] add optional `zbar_e_bot` / `zbar_n_bot` fields to `Mesh` (default `None`). Keep them out of
      `load_mesh`'s file loading and `mesh_field_files()`, like the `inv_`/`se_` prefixes
- [x] `partial_cells.apply(mesh, *, thresh=0.0)`: clamp depth to `zbar[THERS_ZBAR_LEV−1]`
      (`oce_mesh.F90:619`); element depth `dd = sum(depth[elnodes])/3` in the order of `mesh.elem_nodes`;
      the threshold branch; the `dd < zbar(nle)` branch with the `nle == nl` case; the
      `min(Z(nle−1), dd)` branch (`oce_ale.F90:258-345`); `zbar_n_bot` = minimum over the node's elements
      (`:439-445`, CSR `nod_in_elem2D`); the `zbar_3d_n` bottom slot (`:222`). Refuse a cavity mesh
      (`ulevels > 1` anywhere)
- [x] helpers `bottom_elem_thickness`, `bottom_node_thickness`, `static_node_geometry`
      (`Z_3d_n(nzmax−1) = zbar(nzmax−1) + (zbar_n_bot − zbar(nzmax−1))/2`, `:236`; nominal fill below the
      bottom), `static_elem_geometry`. Clip every level-count index so pad lanes (`nlevels_nod2D = 0`) stay
      in range
- [x] tests: `apply` bit-identical to a literal Python loop transcription on pi and CORE2; threshold → ∞ ⇒
      full cells; bounds; the `nle == nl` branch on a synthetic deep column; the clamp; static geometry
      strictly decreasing on every real lane (below the bottom only where the lane exists: for
      `nln == nl` there is none); cavity mesh refused
- [x] run tests on a compute node — must pass before Task 2

### Task 1b: Early Fortran check (G1 and a day-1 reference)

**Files:**
- Create: `scripts/partial_cells/` (Fortran run-dir notes, `g1_geometry.py`)

- [x] copy `A_main_ias2_year` → a partial-cell run dir (`use_partial_cell=.true.`), cold start, 1 day with
      daily output of the comparison fields; run on 4 nodes (a few minutes; cost stated before submitting).
      It writes `fesom.mesh.diag.nc` at init
- [x] G1: JAX `zbar_e_bot`/`zbar_n_bot` against `zbar_e_bottom`/`zbar_n_bottom` (written as `−zbar_*_bot`).
      Pass: bit-identical everywhere except (a) elements and nodes whose levels differ between our mesh and
      the `/pool` levels, plus nodes that take their minimum from such an element, listed explicitly, and
      (b) at most 1 ulp where the `dd` sum is the selected value (summation order). Any other difference is a
      transcription bug
- [x] record the job id, the counts in (a) and (b), and the result in this plan
      ✅ 2026-11-25, Fortran job 29278125 (zstar partial-cell month), G1 job 29278291: **G1_GEOMETRY_OK**.
      (a) 4 elements / 2 nodes with other /pool levels (6 nodes excluded). (b) 65,112 of 244,655 element
      bottoms and 27,641 of 126,852 node bottoms differ by exactly 1 ulp (max 9.1e-13 m), all where the
      element depth is the selected value. The cause is not the summation order (the element node order is
      identical) but the division: the Intel build computes `sum(depth)/3.0` as `sum * (1/3)`; recomputed
      that way, every element and node outside (a) is bit-identical. Kept `/3.0` (the Fortran source).
      Partial cells move 94.5 % of the CORE2 element bottoms, by 49 m on average.

### Task 2: Thickness init, commit and restart

**Files:**
- Modify: `fesom_jax/ale.py`, `fesom_jax/state.py`, the restart loader in `fesom_jax/run.py`,
  `fesom_jax/fortran_restart.py`
- Modify: `fesom_jax/tests/test_partial_cells.py`

- [x] `ale.init_thickness_zstar`: bottom `hnode` = `bottom_node_thickness`, bottom `helem` =
      `bottom_elem_thickness` under partial cells (`oce_ale.F90:745-813`)
- [x] `ale.commit_thickness`: under partial cells keep `helem` at the element bottom layer =
      `bottom_elem_thickness`, vertex mean only above it (`update_thickness_ale`, `:1031-1043`). The linfs
      commit follows the same rule (`init_thickness_ale` linfs, `:618-660`)
- [x] `state.State.rest` (linfs, jnp and host-numpy paths): `hnode` already follows `mesh.zbar_3d_n`; set
      the bottom `helem` = `bottom_elem_thickness`
- [x] restart load under partial cells: mirror `restart_thickness_ale` (`:1104-1211`): bottom `hnode` from
      the mesh, `helem` recomputed with the bottom from the mesh. The restart arrays are in the folded
      `[P·Lmax]` layout, so use the sharded-mesh fields. A full-cell restart can then start a partial-cell
      run, as in Fortran
- [x] restart metadata records `use_partial_cell`; a partial-cell restart read by a full-cell run is
      refused (the full-cell path is unchanged, so it would otherwise keep the partial `hnode`, while
      Fortran resets it, `:1142,1172`)
- [x] `fortran_restart.py`: under partial cells set the bottom `helem` from the mesh; otherwise unchanged
- [x] tests: `hnode`/`helem` against the Fortran loops (zstar with a nonzero synthetic `hbar`, linfs);
      bottom `helem` survives a commit; restart of a full-cell state into partial cells; the refusal;
      full-cell outputs byte-identical to `main`
- [x] run tests — must pass before Task 3

### Task 3: zstar live geometry and PGF anchor

**Files:**
- Modify: `fesom_jax/ale.py`, `fesom_jax/pgf.py`
- Modify: `fesom_jax/tests/test_partial_cells.py`

- [x] `ale.live_geometry`: below the stretch range take `static_node_geometry` instead of nominal
      `zbar`/`Z` (FESOM2 keeps the init values there, `update_thickness_ale` `:1001-1018`)
- [x] `ale.live_geometry_elem`: anchor at `zbar_e_bot` (`impl_vert_visc_ale` `:3380`)
- [x] `pgf.pressure_force_shchepetkin`: `zbot = zbar_e_bot` (`:2075`)
- [x] `AleConfig.validate` message (`ale.py:95-98`) no longer lists partial cells as out of scope
- [x] tests: cold-start live geometry == static partial geometry; after a synthetic SSH change the bottom
      interface stays `zbar_n_bot` and the column sum of `hnode` equals `hbar − zbar_n_bot`; the index
      assertion `nlevels_nod2D_min ≤ nlevels_nod2D` (so `vert_vel_zstar_distribute`'s anchor is never the
      partial interface); shchepetkin PGF against a literal per-element transcription of
      `pressure_force_4_zxxxx_shchepetkin` on a pi slice with partial cells; density linear in z ⇒
      `max|pgf| ≤ 1e-12·max|g·Δρ·h/ρ0|`; full-cell byte-identical
- [x] run tests — must pass before Task 4

### Task 4: SSH operators

**Files:**
- Modify: `fesom_jax/ssh.py`, `fesom_jax/ssh_se.py`
- Modify: `fesom_jax/tests/test_partial_cells.py`

- [x] `ssh.py` stiffness assembly: element depth `zbar_e_bot − zbar[ulevels−1]` (`oce_ale.F90:1362`;
      `zbar_e_srf = zbar(ulevels)` without cavity partial cells)
- [x] `ssh_se.cfl_omega_max`: the `helem=None` fallback uses `H = −zbar_e_bot` under partial cells (the SE
      dynamics already follow `helem`; the JAX SE is the Danilov scheme, a stated deviation, so the FESOM2
      `splitexpl_subcycl` code is not its reference)
- [x] tests: stiffness entries against a literal transcription of `init_stiff_mat_ale` on pi with partial
      cells; matrix symmetry kept; at rest the SE `He = −zbar_e_bot`; full-cell byte-identical
- [x] run tests — must pass before Task 5

### Task 5: linfs: Shchepetkin PGF and static geometry in the step

**Files:**
- Modify: `fesom_jax/pgf.py`, `fesom_jax/step.py`
- Modify: `fesom_jax/tests/test_partial_cells.py`

- [x] `pgf._shchepetkin_elems`: a static `linfs` flag that drops the slope term off the element's bottom
      layer (`jnp.where` selecting `drho_dx`, not a multiply by 0, so the values are exact), giving
      `pressure_force_4_linfs_shchepetkin` (`oce_ale_pressure_bv.F90:796-1040`)
- [x] `step.py`: linfs + partial cells ⇒ `zbar3_live, Z3d_live = static_node_geometry`,
      `elem_geo_live = static_elem_geometry(st.helem)`, `zbar3_new/Z3d_new` = the same (linfs `hnode_new =
      hnode`); the PGF goes to the Shchepetkin kernel with the linfs flag
- [x] spot-check while wiring that each consumer's `Z3d`/`zbar3`/`elem_geo` branch is purely geometric
      (the plan review found no zstar-only logic); note anything unexpected here
- [x] tests: linfs PGF against a literal per-element transcription on a pi slice; density linear in z ⇒
      PGF ≈ 0 (same bar as Task 3); one linfs + partial-cell CORE2 step finite; full-cell linfs
      byte-identical
- [x] run tests — must pass before Task 6

### Task 6: Setup wiring, IC, config, guards and sharding

**Files:**
- Modify: `scripts/run_from_config.py`, `fesom_jax/run.py`, `fesom_jax/run_config.py`,
  `fesom_jax/phc_ic.py`, `fesom_jax/shard_mesh.py`, `fesom_jax/step.py`
- Create: `scripts/partial_cells/build_ic.py`
- Modify: `fesom_jax/tests/test_partial_cells.py`

- [x] RunConfig keys `use_partial_cell`, `partial_cell_thresh`; `scripts/run_from_config.py` calls
      `partial_cells.apply` right after `load_mesh`, before the partition, `build_sharded_mesh`,
      `build_ssh_operator` and `cold_start_state`; `run.py` asserts that the mesh matches the config
- [x] IC: a partial-cell IC cache. Each node's bottom layer is interpolated at its own `Z_3d_n` with its
      own depth bracket (`gen_ic3d.F90:500`); all other layers are the existing cache values; the
      potential-temperature conversion stays at nominal `|Z|` (`oce_ale_pressure_bv.F90:3346-3353`). The
      cache carries a marker (`partial_cells.json`: thresh, source cache); `cold_start_state` refuses a
      cache whose marker does not match the mesh
- [x] guards: partial cells + cavity, + REcoM, + single precision ⇒ `NotImplementedError` with the reason
- [x] `shard_mesh`: partition `zbar_e_bot`/`zbar_n_bot` when present; pad lanes finite (clipped indices);
      single-device `npes=1` invariant kept
- [x] tests: YAML parse and defaults; the mesh/config assertion; guard errors; IC bottom layer against a
      per-node interpolation; the marker refusal; sharded N-vs-1 (CPU fake devices, 2 and 4) one zstar and
      one linfs partial-cell step
- [x] run tests — must pass before Task 7

➕ Tests of Tasks 1-6 (all green): jobs 29277995 (Task 1, 41 passed), 29278103 (Tasks 2-3 with test_ale,
test_ale_zstar, test_restart, test_fortran_restart, test_run_config: 134 passed, 1 over-strict own test fixed),
29278292 (Tasks 4-6 with test_phc_ic, test_pgf, test_state, test_ssh: 110 passed, 3 over-strict own tests
fixed), 29278293 (test_partial_cells_sharded on 2 fake devices: 5 passed), 29278402 (test_partial_cells
complete: 41 passed). One commit for Tasks 1-7 instead of one per task: ale.py and step.py carry changes of
several tasks.

### Task 7: Gradient and stability

**Files:**
- Modify: `fesom_jax/tests/test_partial_cells.py`
- Create: run configs under `experiments/partial_cells/`

- [x] masked-NaN gradient probe through a CORE2 partial-cell step (zstar and linfs): gradient finite
      everywhere, zero on masked lanes ➕ done on pi (2 steps, d(mean bottom T)/d(T0)); the same code paths,
      far cheaper than CORE2
- [x] 10-day CORE2 zstar + partial cells on one A100: stable, min `hnode` > 0 ➕ covered by the G2 year run
      (job 29278409, 1958 complete); `scripts/partial_cells/check_year_restart.py` on its end-of-year restart
      (job 29279249): min wet `hnode` 4.56 m, min wet `helem` 4.56 m, T/S/SSH finite, bottom layers still equal
      to the bottom thicknesses exactly. **YEAR_RESTART_OK**
- [x] run tests — must pass before Task 8 (test_partial_cells, job 29278402: 41 passed)

### Task 8: Fortran FESOM2 reference and comparison (G2)

**Files:**
- Modify: `scripts/partial_cells/` (comparison script)
- Create: `docs/PARTIAL_CELLS_VALIDATION.md`

- [x] Fortran runs (cost stated before submitting; ≈ 1.5 node-hours in total): zstar partial-cell 1 year
      (monthly output, the full-cell year is `A_main_ias2_year`); zstar full-cell and partial-cell 1 month
      with daily output; linfs full-cell and partial-cell 1 month with daily output (`which_ALE='linfs'`,
      default `which_pgf`). All on the `/pool` levels, so the Fortran pairs differ only in
      `use_partial_cell`
      ✅ jobs 29278124/125 (zstar fc/pc month), 29278199/200 (linfs fc/pc month), 29278204 (zstar pc year);
      2.8 node-hours allocated (above the estimate: 4 nodes per run including startup)
- [x] JAX runs: the same set from the matched JAX config, ≈ 1–2 A100-hours. IC: a cache built for the
      Fortran runs' partition (`dist_512`; FESOM2's IC extrapolation depends on the partition, see the
      memory note zstar-forcing-dump-config-gap) and its partial-cell variant. If that cache is not cheap
      to build, use the existing one and add the nodes where the two IC fields differ to the mask
      ✅ IC caches `fc_dist512`, `pc_dist512` (`scripts/partial_cells/build_ic.py`); day 1 (29278405/408/410/
      412), zstar year fc/pc (29278406/409, 78 min each), linfs month fc/pc (29278411/413). 3.0 A100-hours,
      above the 1–2 estimated: 0.23 s/step makes a year 78 min per run
- [x] G2 comparison. Fields: bottom-layer temperature, bottom-layer speed and SSH, as daily means (day 1),
      the month-1 mean and the year mean. Mask: wet nodes, minus the level-mismatch points from Task 1b.
      Δ_X = X_pc − X_fc for model X. Pass bars (fixed now, before the runs):
      - direct: `rms(JAX_pc − F_pc) ≤ 1.2·rms(JAX_fc − F_fc)` at day 1 and month 1, and both
        `≤ 0.5·rms(Δ_F)`;
      - response: correlation(Δ_JAX, Δ_F) ≥ 0.9 at day 1 and ≥ 0.7 for the year mean; amplitude ratio
        `std(Δ_JAX)/std(Δ_F)` within 0.7–1.4.
      A miss is investigated, not explained away
      ✅ job 29279308: **G2_RESPONSE_OK**, every bar in all five cases (zstar day 1 / January / 1958, linfs
      day 1 / January); corr ≥ 0.999, amplitude 1.00, JAX–Fortran difference 1–2 % of the response over the
      first month. ⚠️ fixed on the way: the first full run (29279243) took the mean of the JAX monthly speeds but
      the speed of the Fortran year-mean velocity (year bottom speed corr 0.94); both now average u, v first
- [x] write `docs/PARTIAL_CELLS_VALIDATION.md` (numbers, figures, job ids) ➕ figure
      `docs/figures/partial_cells_response.png` (nereus, `scripts/partial_cells/fig_response.py`, job 29279275)

### Task 9: Verify acceptance criteria

- [x] zstar and linfs partial cells work end to end through `scripts/run_from_config.py` (every G2 JAX run)
- [ ] full-cell default byte-identical: full CPU suite `sbatch scripts/runs/run_suite.sbatch` green (known
      pre-existing failure `test_dp_lock` noted, not re-pinned)
- [x] guards for the out-of-scope combinations raise clearly (test_partial_cells: cavity, REcoM, single
      precision, config/mesh mismatch, restart and IC-cache kind)

### Task 10: Documentation

- [x] README: the option, its FESOM2 names, the scope and the "not recommended" note from FESOM2 (README
      "Can" list; the schema keys and the full paragraph in `docs/USER_GUIDE.md`)
- [x] `docs/PORTING_LESSONS.md`: lessons from each task
- [ ] move this plan to `docs/plans/completed/`

## Technical Details

Indices: Fortran 1-based `nle = nlevels(elem)` (bottom interface), `nln = nlevels_nod2D(n)`. 0-based in JAX:
the bottom interface is `nlevels−1`, the bottom layer `nlevels−2`.

```
element (use_depthonelem = .false.):
  dd = (depth[n1] + depth[n2] + depth[n3]) / 3          depth clamped to ≤ zbar(5)
  if zbar(nle−1) − zbar(nle) ≤ thresh:  zbar_e_bot = zbar(nle)
  elif dd < zbar(nle):                   zbar_e_bot = max(dd, zbar(nle)+(zbar(nle)−Z(nle−1)))  if nle == nl
                                         zbar_e_bot = max(Z(nle), dd)                           otherwise
  else:                                  zbar_e_bot = min(Z(nle−1), dd)
  bottom_elem_thickness = zbar(nle−1) − zbar_e_bot
node:
  zbar_n_bot = min over the node's elements of zbar_e_bot
  bottom_node_thickness = zbar(nln−1) − zbar_n_bot
  zbar_3d_n(nln) = zbar_n_bot;  Z_3d_n(nln−1) = zbar(nln−1) + (zbar_n_bot − zbar(nln−1))/2
```

`zbar_n_bot` lies in `[Z(nln), Z(nln−1)]` (deeper only when `nln == nl`), so the nominal interface below the
bottom stays strictly deeper and the D5 fill keeps positive spacing.

Layers set from the bottom thickness in FESOM2: `hnode(nlevels_nod2D−1)` and `helem(nlevels−1)` (1-based) in
init (zstar and linfs) and restart; the zstar update never touches them (nodes: `nz ≤ nlevels_nod2D_min−2`;
elements: `nz ≤ nlevels−2`).

## Revision Log

1. 2026-11-25, plan review (one read-only reviewer agent). The formulas, branches, index mappings and the
   stretch-range argument were confirmed against the Fortran. Changes made: the partial-cell geometry is
   applied in `scripts/run_from_config.py`, not `run.py`; a partial-cell IC cache with a marker (the cold
   start reads cached IC files), with the potential-temperature conversion kept at nominal `|Z|`; G1 moved
   to an early Fortran 1-day run (Task 1b), with the `/pool`-levels mismatch and the 1-ulp `dd` sum handled
   explicitly; direct day-1 gates and defined G2 metrics; SE reduced to the `cfl_omega_max` fallback; the
   linfs PGF as a flag on the zstar kernel; guards for single precision; restart metadata and refusal;
   `fortran_restart.py` bottom `helem`; pad-lane index clipping. The wrong claim that `mesh.depth` causes a
   pre-existing GM/TKE deviation was dropped (those readers are NN-only).

## Post-Completion

*Items requiring the user or external systems.*

- Reply on issue #3 with what was ported, the scope and the validation numbers. The exact text is shown
  to the user first.
- The push to GitHub is done by the user (squash via `git commit-tree` as in earlier pushes; grep the tree
  for session ids and scratchpad paths first).
- Possible follow-ups if asked: REcoM with partial cells, single precision with partial cells, cavity
  partial cells (`use_cavity_partial_cell`), `use_depthonelem`, the other linfs PGF schemes (`nemo`,
  `cubicspline`, `easypgf`).
