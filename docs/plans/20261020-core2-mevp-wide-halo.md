# mEVP wide halo: one exchange every K subcycles (CORE2 GPU speed plan, Tier A)

## Overview
- The sharded mEVP (`fesom_jax/ice_mevp.py:231-237`) refreshes the `u_aux`/`v_aux` node halo after EVERY one of
  its 120 subcycles: 120 collectives per step, the largest share of the ~345 collectives of a CORE2 step on 4 GPUs
  (ledger `docs/PERFORMANCE_CORE2_GPU.md`; fesom3-jax runs the same physics with 72 collectives a step).
- A subcycle updates an element's stresses from its vertices' velocities and a node's velocity from its elements'
  stresses, so the region a shard computes correctly shrinks by one ring per subcycle. Give each shard K rings of
  "ice lanes", run K subcycles between refreshes and recompute the outer rings redundantly
  (communication-avoiding; fesom3-jax `shard/evp_wide.py` + `ice/dyn.py _subcycles_wide`, -6 % at K = 8 on 4 A100;
  the Kokkos port's EVPWIDE, byte-certified at K = 2/4/8).
- Owned lanes get the same values from the same operations in the same order ⇒ **bitwise the current sharded
  mEVP on CPU** on every owned lane (and on the standard halo lanes after the final standard exchange). This is a
  performance change, not a physics change (the project rule: fesom-jax is the shadow of FESOM2).
- Opt-in: a `gpu:` key `mevp_wide: K` in `configs/core2_gpu.yaml` (0/absent ⇒ today's per-subcycle exchange, the
  default everywhere; `core2_full.yaml` and the adjoint untouched). Forward-only first: the gradient path keeps the
  one-ring loop unless the AD test below passes.

## Design (verified against fesom3-jax's working implementation)
- Rings of device r: R_0 = owned nodes; E_j = elements with a vertex in R_{j-1}; R_j = R_{j-1} ∪ vertices(E_j).
  The ice lanes are R_K nodes and E_K elements. After j subcycles from a full refresh, stresses are current on
  E_{K-j+1} and velocities on R_{K-j}; after K, on E_1 and R_0 (owned). All mEVP precomputes (`mevp_setup`:
  ssh-tilt scatter, pressure_fac, masks) are correct on R_{K-1}/E_K, enough for every subcycle.
- **Lane order (load-bearing for bitwise sums):** the ice lanes of device r are its STANDARD local lanes first, in
  the standard order (owned + one-ring halo, `shard_mesh` local order), then the extra ring lanes in ascending global
  id, then pads. An owned node's elements are all standard local elements, so `ops.scatter_add` adds its
  contributions in exactly today's order. Standard lane i == ice lane i for i < n_local, so an owner's lane index for
  an owned entity is the same in both layouts.
- Maps: for every ice lane, `(src_dev, src_lane)` = (its owner, the owner's standard local lane) from the existing
  owner tables (`shard_mesh._exchange_map` inputs: `owner`, `owner_local`); own-owned lanes read themselves. These
  are ordinary exchange maps in the ice-lane index space, so `halo.compact_exchange_maps` / `halo_exchange_compact`
  do the refresh (generalise them with the source lane count, since the entry transfer reads a standard-layout
  field into ice lanes). Elements: owner as in the standard elem exchange map.
- Schedule per ice step: (1) entry: move the mEVP inputs (u_ice, v_ice, σ11/12/22, a_ice, m_ice, m_snow,
  elevation, srfoce_u/v, stress_ax/ay; node and element fields batched) onto the ice lanes in ONE exchange per kind;
  (2) blocks of K subcycles with a refresh of (u, v | σ11, σ12, σ22) between blocks (120/K - 1 refreshes; K must
  divide 120 or handle the remainder block); (3) exit: owned lanes back into the standard layout
  (`where(valid, ice[:Lmax_std], std)`), then one standard exchange of u, v (nod) and σ (elem) so the standard halo
  lanes match today's. Collectives per step at K = 8 with nod+elem per refresh: ~2 + 14·2 + 2 ≈ 32 (today 120).
- Geometry on the ice lanes = the GLOBAL mesh's values gathered by global id (`elem_nodes` remapped to ice-lane
  indices, `gradient_sca`, `metric_factor`, `elem_area`, `area[:, 0]`, `coriolis_node`, `ulevels`,
  `ulevels_nod2D`, the GLOBAL coastal mask for `bc_index` — never recomputed locally, T9 of `ice_mevp.py`). Check
  `strain_rates`/`stress_div_scatter` in `fesom_jax/ice_evp.py:76-115` and `mevp_setup` for every field read.
- Plumbing: host-built ice-lane arrays and maps enter `run_steps_sharded_forced` as extra sharded inputs (the
  `ssh_gather` precedent: `integrate_sharded.py` extras + cache key), reach `ice_step` through `HaloCtx` (a new
  optional slot), and `mevp_dynamics` takes the wide path when that slot is set and K ≥ 2. One device or K < 2 ⇒
  unchanged code path.

## Development Approach
- Testing: gate-first (tests written with each task), CPU with 4 fake devices
  (`XLA_FLAGS=--xla_force_host_platform_device_count=4`), never on the login node.
- Work in your own worktree/branch `perf/core2-mevp-wide` from `perf/core2-gpu`; GPU jobs only from a frozen
  snapshot worktree of a committed HEAD.

## Implementation Steps

### Task 1: ice-lane rings and maps (host numpy)
**Files:** Create `fesom_jax/ice_wide.py`; Create `fesom_jax/tests/test_mevp_wide_sharded.py` (planned as `test_ice_wide.py`; `test_suite_groups.py` puts every `test_ice_*` file in the `ice` group, and this one needs the 4 fake devices of `sharding`); Modify
`scripts/runs/suite_groups.txt` (the new test file joins the `sharding` group).
- [x] rings R_K/E_K per device, lane order as above, ice-lane mesh arrays, `(src_dev, src_lane)` per kind, the
  entry/exit index maps
- [x] tests: rings on a small synthetic mesh and CORE2 dist_4 (R_0 ⊂ R_1 = standard local nodes, E_1 = owned
  elements, every E_K vertex in R_K); standard lanes are a prefix of the ice lanes; every owned node's elements are
  standard local elements in the same order
- [x] run tests

### Task 2: the wide mEVP loop
**Files:** Modify `fesom_jax/ice_mevp.py`, `fesom_jax/halo.py` (HaloCtx slot; compact maps with a source lane
count), `fesom_jax/ice_step.py` if needed; tests in `fesom_jax/tests/test_mevp_wide_sharded.py`.
- [x] entry transfer, K-subcycle blocks with refreshes, exit + standard exchange; K = 0/1 and one device keep today's
  path byte-identical
- [x] test (CPU, 4 fake devices, CORE2 active ice, e.g. the `test_mevp_sharded_step_owned_matches` setup in
  `fesom_jax/tests/test_step_sharded.py`): wide K = 2, 4, 8 == today's sharded mEVP BITWISE on owned lanes and on
  standard halo lanes; **negative control**: a deliberately skipped refresh must FAIL at K = 2 (a stale outer ring's
  influence decays per ring, so only small K exposes a wrong schedule — the fesom3 lesson)
- [x] gradient: forward-only unless `jax.grad` of a scalar of the owned u_ice matches today's sharded gradient
  (rel 1e-12); if it does not, refuse the wide path under `return_grad_fn` with a clear error
- [x] run tests, plus `test_halo.py` and `test_step_sharded.py -k "mevp or compact"`

### Task 3: wiring and config
**Files:** Modify `fesom_jax/integrate_sharded.py`, `fesom_jax/run.py`, `fesom_jax/run_config.py` (`gpu.mevp_wide`,
int ≥ 0), `configs/core2_gpu.yaml` (leave it OFF until Task 4 measures it), tests in
`fesom_jax/tests/test_run_config.py` and a sharded forced-run test (wide vs not, a few steps, bitwise owned).
- [x] extras into `run_steps_sharded_forced` (cache key includes them); built once per run
- [x] config key + validation; env/CLI override only if needed for A/B
- [x] run tests

### Task 4: GPU measurement
- [x] 4 A100 (`scripts/perf/gpu_profile.sbatch`, `FESOM_HALO_COMPACT=1 FESOM_COLUMN_UNROLL=1`, `--cheb-degree 3
  --forcing-on-device`, the wide K via the config or an env override): K = 0 vs K = 4, 8, 12; steady ms/step,
  collectives per step, noise compare (`NOISE=1`, `BASE_NOISE` = the K = 0 run's noise dir, factor 2). ~2 GPU-h.
- [x] ledger rows in `docs/PERFORMANCE_CORE2_GPU.md`, a lesson in `docs/PORTING_LESSONS.md`
- [x] report back; the default stays off until the user decides

## Implementation notes (2026-10-20, branch perf/core2-mevp-wide)
- **Node sums are gathered, not scattered, on the ice lanes.** The lane-order rule above makes an OWNED node's
  `scatter_add` order match, but an outer-ring node computed redundantly on device r must also get its OWNER's
  bits, and r's lane order is not the owner's. So the wide subcycle sums each node's `[elem, vertex]` terms from a
  per-node slot list (`ice_wide` `csr`) in the owner's standard element-lane order, from zero, one add at a time
  (`MevpWide.node_sum`; a `+0.0` for a pad slot leaves a sum unchanged). fesom3-jax does the same (ascending
  `nod_in_elem2D` gather).
- **The precompute stays on the standard lanes.** `mevp_setup` runs unchanged on the standard layout; its outputs
  (inv_thickness, mass, ssh tilt, masks, bc_index, pressure_fac) move to the ice lanes in the entry exchange with
  the inputs, so the ssh-tilt scatter is never redone in another order and the ice lanes need only `elem_nodes`,
  `gradient_sca`, `metric_factor`, `elem_area`, `coriolis_node`.
- **One collective per exchange point:** node and element fields travel in one compact `all_gather` (entry, each
  refresh, exit). Collectives per ice step: 1 + (120/K - 1) + 1, i.e. 16 at K = 8 (today 120).
- **No 1-subcycle block.** XLA removes a trip-count-1 loop and inlines its body; the inlined subcycle rounded
  differently (K = 7, remainder 1: 1-ulp differences on 8.6 % of the owned velocities). A remainder of 1 shortens
  the first block by one instead.
- **Standard halo σ:** the exit exchange gives every standard halo element its owner's σ. On halo elements whose
  three vertices are local this is bitwise today's value; on the extended-halo (eXDim) elements today's value is
  computed from a sentinel vertex and is not meaningful, so those lanes are not compared.
- Gradient: `jax.grad` through the wide path matches the per-subcycle gradient at rel 4.7e-15 to 4.9e-15
  (K = 2, 4, 8) ⇒ the wide path is allowed under `return_grad_fn`; the `adjoint_subcycles` truncation keeps the
  per-subcycle path.

- **Task 4 (4 A100, snapshot f997148):** K = 0/4/8/12: 57.11/51.96/51.77/51.22 ms/step (jobs 28289821-24),
  344/255/240/235 collectives per step; timing repeat at K = 8 51.60 (28290233). CPU, 96 steps of the full
  bench physics, K = 8 vs K = 0: every owned lane of every State field bitwise (28290236/37). GPU noise judge
  (owned lanes, 2x the 28282453 floor): 5 of 7 single draws above (worst 10x density/S, Hudson Bay under ice),
  2 within 1.8x; none at a partition seam. Default stays off; the user decides.

## Post-Completion
- GH200 measurement on dolpung is done by the main session (node limit).
