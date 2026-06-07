# FESOM2 → JAX Port: A Differentiable Ocean Model

## Overview

Port the FESOM2 unstructured-grid finite-volume ocean model to **JAX**, producing a
**differentiable** ocean model whose primary purpose is **hybrid ML**: embedding
trainable neural-network parameterizations (initially vertical mixing and mesoscale
eddy fluxes) and training them end-to-end through the model.

- **What it solves:** gives us an ocean model that is a clean, differentiable forward
  function callable inside a training loop, so an embedded NN parameterization can be
  optimized against data/objectives via gradients that flow through the dynamics.
- **Baseline / source of truth:** the existing **C port** at
  `/home/a/a270088/port2/fesom2_port/` (~23K lines). We match it kernel-by-kernel.
- **Reference for parallelization & fidelity lessons:** the **Kokkos port** at
  `/home/a/a270088/port_kokkos/`.
- **Integration:** stands alone in `/home/a/a270088/port_jax/`; reuses the C port's
  reference-dump mechanism and the Kokkos port's comparison scripts for verification.

This is a **multi-week, session-by-session** effort. Phases 0–3 below are detailed to
task granularity (the immediate sessions); Phases 5–8 are sketched and will each be
expanded into their own `docs/plans/` sub-plan when reached.

## Locked Decisions (from the brainstorm — do not re-litigate)

1. **Use case:** Hybrid ML parameterizations. The model is a differentiable forward
   function; performance matters; the ML swap-out points are **vertical mixing
   (PP/KPP)** and **mesoscale eddy fluxes (GM/Redi)**.
2. **Strategy:** **Full-fidelity first** — port the *complete* model, built bottom-up in
   phases (minimal step → pi → CORE2 → ice/KPP/GM), not a stripped toy.
3. **AD discipline:** **AD-safe by construction** (idiomatic functional JAX everywhere)
   + an **early end-to-end gradient smoke test** right after the minimal forward step
   works, then re-run gradient checks at **every** milestone gate. AD is never deferred.
4. **Mesh representation:** **index gather/scatter mirroring the C loops** — fancy-index
   gather, `jax.ops.segment_sum`/`.at[].add` for scatter, dense `[n_entity, nl]` arrays
   + boolean masks for ragged vertical extent, connectivity arrays as static index
   arrays. Sparse-operator matrices are an *opportunistic later* optimization for hot
   linear kernels only.
5. **Parallelism:** **single-device + data-parallel-over-batch** now (`vmap` + device
   shard over training samples); CORE2 fits one 40–80 GB GPU. **Mesh sharding**
   (`shard_map` + `ppermute` halo, reusing `dist_N/` files, AD-through-collectives) is a
   **committed later milestone (Phase 8)**.

## The Golden Rule (adapted for JAX)

The C/Kokkos rule was *"copy line-by-line, don't simplify, trust the Fortran."* For JAX
it becomes:

> **Preserve the exact computation — the math, and the load-bearing association order —
> but express it as vectorized array ops. Do NOT do a literal loop-by-loop translation,
> and do NOT simplify the physics.** When in doubt, dump the C port's intermediate value
> at a probe node and match it.

## Context (from discovery)

- **C source to port (per module):** `fesom_mesh`, `fesom_eos`, `fesom_momentum`,
  `fesom_ssh`, `fesom_ale`, `fesom_tracer_adv`, `fesom_tracer_diff`, `fesom_pp`,
  `fesom_kpp`, `fesom_gm`, `fesom_ice*`, `fesom_jra55`, `fesom_forcing`, `fesom_halo`,
  `fesom_partit`, `fesom_step`, `fesom_main` (all in `port2/fesom2_port/src/`).
- **Authoritative algorithm/physics reference:** `port2/FRESH_START.md` — timestep
  sequence (§5), ALE (§6), EOS (§8), mixing (§10), SSH CG solver (§11), FCT (§12),
  gotchas (§14), CORE2 default params (§14.7).
- **Verification backbone (Fortran-sourced — corrected after plan review):** the
  **Fortran** model writes the binary reference dumps via
  `port2/fesom2/src/fesom_dump_shim.F90` (called from `oce_ale.F90`). The C port itself
  was validated against these. So the **C port is the *algorithmic* source-of-truth to
  mirror**, and the **Fortran dumps are the *numerical* reference**. C↔Fortran is already
  only climate-close, so JAX↔Fortran inherits the same tolerance class — not a tighter
  one. Record layout (little-endian, stream, no header): `int32 step, int32 substep_id,
  int32 probe_global_id (1-based), int32 nlevels, char[24] field_name, real64
  values[nlevels]`. Reader: `port2/inspect_dump.py`.
- **The shim records NODE fields only** (per-node columns, truncated to
  `nlevels_nod2D(node)`). Element-based fields (`pgf_x/y`, `uv_rhs`, `uv`, `Av`) are **not**
  in the existing shim, and `Av` is explicitly deferred in the Fortran (`oce_ale.F90:3561`
  "Av is at elements, defer"). Verifying the element kernels therefore requires either a
  Fortran-side element-dump extension or indirect verification — see Task 0.3 and Phase 2.
- **The 16 substep IDs** (the porting + verification granularity for one ocean step):
  `0 init, 1 pressure_bv, 2 sw_alpha_beta, 3 pressure_force, 4 mixing, 5 vel_rhs,
  6 viscosity_filter, 7 impl_vert_visc, 8 ssh_rhs, 9 ssh_solve, 10 update_vel,
  11 compute_hbar, 12 eta_n, 13 ale_step, 14 gm_bolus, 15 solve_tracers,
  16 update_thickness`. **Notes** (from review): the C `compute_vel_rhs` *bundles* Coriolis
  + PGF + **momentum advection** (`momadv_opt=2`, `fesom_momentum.c:156`) into substep 5 —
  momentum advection is a real edge→node scatter, not a separate substep, and must not be
  dropped. Substep 2 (`sw_alpha_beta`) is consumed only by GM/KPP → deferred to Phase 6.
  Substeps 3/5/6/7/10 produce element-based fields → not in the node-only shim.
- **Reusable comparison scripts** (`port_kokkos/scripts/`): `diff_snap.py`,
  `eps_climate_compare.py`, `eps_climate_compare_2yr.py`, `eps_vertical_profiles.py`,
  `gpu_fidelity_check.py`, `kpp_dump_diff.py`.
- **Computational pattern split** (drives effort estimate): ~60% clean maps/gathers
  (EOS, PGF, Coriolis, velocity update, advection fluxes) — easy; ~30% mechanical
  (per-column TDMA for vertical visc [per-element, u&v] and tracer diff [per-node];
  biharmonic viscosity edge→element scatter); ~10% AD-hard (CG solver, FCT Zalesak
  limiter, KPP lookup tables).
- **`port_jax/` is empty** — clean slate. Not yet a git repo.

## Why bit-identity is not the target

JAX cannot be bit-identical to the C port for the same reason the Kokkos GPU/OpenMP
backends weren't: **edge→node scatters and global reductions reassociate the
floating-point sums**, and JAX's `segment_sum` does not preserve C's edge ordering. The
target is **climate-close**, exactly the Kokkos GPU acceptance:

- **map/gather kernels:** `max|Δ|` ~1e-15 (FMA/transcendental differences only)
- **scatter/reduction kernels:** `max|Δ|` ~1e-12 per step
- Scatter-add is **fully differentiable** (its gradient is a gather), so this loss of
  bit-identity costs us nothing for AD.
- **Guardrail (from `port_kokkos/docs/SCATTER_STRATEGY.md`):** a scatter's `max|Δ|` is
  *association-order-dependent*. `segment_sum`, `.at[].add`, and a per-receiver
  gather-reformulation give **different** residuals — all acceptable (all ≤ ~1e-12), but
  do **not** chase a discrepancy below that floor; it is reassociation, not a bug.

## Development Approach

- **Testing approach: reference-comparison-driven (TDD-flavored).** The Fortran reference
  dump (see Context) is the known-good expected output and exists *before* the JAX kernel
  does. For each ported kernel: (1) write the comparison gate against the Fortran dump
  first, (2) port the JAX kernel, (3) make `max|Δ|` fall within the per-kind tolerance
  before moving on. (Element kernels need the Task 0.3 dump extension or indirect
  verification.)
- **AD-safe by construction:** pure functional JAX, no in-place mutation, `.at[].set/add`
  only, no Python control flow on traced values (use `lax.scan`/`lax.cond`/`lax.while`).
- Complete each task fully (kernel + its verification gate) before the next.
- **Every task includes its verification gate as a required deliverable**, listed as
  separate checklist items — never bundled into the implementation line.
- **All gates must pass before starting the next task.** No exceptions.
- **Update this plan file when scope changes**; expand Phases 5–8 into sub-plans when
  reached.
- **STANDING RULE — keep a running lessons log (`docs/PORTING_LESSONS.md`).** Append to
  it *as you go* (not only at phase end) whenever a task surfaces something non-obvious:
  a config that differs from the docs, a sign/index/association-order trap, an AD
  subtlety, a fidelity surprise, or a "this cost an hour" fact. One entry = one lesson,
  cite the C `file:line` or dump probe that proves it. This is the project's externalized
  memory across sessions — treat it as a required deliverable of every task, on par with
  the verification gate.

## Verification Strategy (the ladder)

This replaces the generic "unit/e2e tests" of a normal project. Four rungs:

1. **Substep probe-column diff** (the per-kernel gate): run the **Fortran** model with the
   dump shim enabled at a probe node (node fields) — or the C port once the shim is
   extended for element fields (Task 0.3); run the JAX kernel on the same input; assert
   `max|Δ|` over the column within tolerance (1e-15 map/gather, 1e-12 scatter/reduction).
   **Truncate the JAX column to `nlevels_nod2D(node)` before diffing** (the shim drops the
   below-bottom padding; a full-length compare spuriously fails on the tail). Harness:
   `fesom_jax/verify.py` (a reusable port of `inspect_dump.py` + tolerance compare).
2. **Full-field snapshot diff:** compare JAX vs C NetCDF snapshots
   (`fesom_io_write_snapshot`) via adapted `diff_snap.py` / `eps_climate_compare.py`.
3. **Multi-year climate stats:** correlation / bias / RMS per field vs the C↔Fortran
   budget, via `eps_climate_compare_2yr.py` and `eps_vertical_profiles.py`.
4. **Gradient checks (NEW):** finite-difference vs reverse-mode AD for a scalar loss
   w.r.t. a chosen parameter; `|grad_AD − grad_FD| / |grad_FD|` within tol. **Re-run at
   every gate from Phase 3 onward.**

Notional test command (provided by the harness): `pytest fesom_jax/tests/ -k verify`
(reads cached C dumps under a fixtures dir) and `pytest -k gradient` for the AD checks.

## Progress Tracking

- mark completed items `[x]` immediately when done
- add newly discovered tasks with ➕ prefix
- document blockers with ⚠️ prefix
- keep this plan in sync with actual work

## What Goes Where

- **Implementation Steps** (`[ ]`): code, kernels, verification harness, comparison
  scripts — everything achievable in this repo (+ small additive helpers in the C port,
  e.g. a mesh-array exporter, clearly scoped).
- **Post-Completion** (no checkboxes): SLURM/GPU runtime tuning, large multi-year
  reference runs, scientific evaluation of trained parameterizations.

---

## Implementation Steps

### Phase 0 — Foundations

#### Task 0.1: JAX environment on Levante

**Files:**
- Create: `fesom_jax/__init__.py`
- Create: `pyproject.toml` (or `environment.yml`)
- Create: `README.md`
- Create: `docs/ENV.md`

- [x] confirm/create a Python env with JAX + a GPU backend available on Levante; record exact versions in `docs/ENV.md` — mamba env `fesom-jax`, Python 3.12.13, jax/jaxlib 0.10.1, jax-cuda12 0.10.1 (CUDA 12.9 pip wheels)
- [x] verify `jax.config.update("jax_enable_x64", True)` works and `jnp.ones(1).dtype == float64` — confirmed CPU + GPU
- [x] verify GPU is visible (`jax.devices()`); record device memory; note CPU fallback for CI — A100-40 `[CudaDevice(id=0)]`, ~31.8 GB usable (job 25374974); login node falls back to CPU (benign cuInit warning)
- [x] minimal smoke: jit a float64 function on GPU, confirm it runs — `scripts/gpu_smoke.py` float64 matmul on CudaDevice, rc=0
- [x] write `fesom_jax/config.py` (global float64 enable, constants from FRESH_START §17: PI, RAD, DENSITY_0=1030, G=9.81, R_EARTH=6367500, OMEGA, VCPW=4.2e6) — mirrors `fesom_constants.h` incl. truncated π + Phase-1 namelist defaults
- [x] verification: a tiny `tests/test_config.py` asserting x64 + constants match FRESH_START §17 — 4 passed

#### Task 0.2: Verification harness (the backbone)

**Files:**
- Create: `fesom_jax/verify.py`
- Create: `fesom_jax/io_dump.py`
- Create: `fesom_jax/tests/__init__.py`
- Create: `fesom_jax/tests/conftest.py`

- [x] port `port2/inspect_dump.py`'s binary reader into `io_dump.py` (record layout + the 16 substep-name map) — `DumpRecord` + `read_records`/`load_records`/`find_record`
- [x] implement `verify.py`: `compare_column(jax_vals, c_record, kind)` returning `max|Δ|`, with per-kind tolerance (`map`/`gather`=1e-15, `scatter`/`reduction`=1e-12) and a pass/fail + pretty report; **truncate `jax_vals` to `c_record.nlevels` before diffing** (drop below-bottom padding) — implemented as `|Δ|≤atol+rtol·|c|` (rtol=per-kind; atol calibratable floor) since field magnitudes span eta~O(1)…pressure~1e7; reports abs+rel
- [x] add a snapshot comparison shim that calls the Kokkos `eps_climate_compare.py` (copy into `fesom_jax/scripts/`) — copied; JAX-snapshot adaptation (rung 2) deferred to first snapshot output (Task 2.11)
- [x] `conftest.py`: fixtures that locate cached C dump files under `fesom_jax/tests/fixtures/` — `load_dump` (skips until fixtures exist) + pinned `probe_gid=1001`
- [x] write `tests/test_verify.py`: feed a known record through `compare_column`, assert correct pass/fail behavior at both tolerances — + binary round-trip
- [x] run tests — must pass before Phase 1 — **14 passed**

#### Task 0.3: Mesh/geometry exporter (C-side; gating dependency for all of Phase 1)

> Review finding (Important #5): this routine does **not** exist in the C port today and
> is the single largest hidden C-side task; everything in Phase 1 depends on it. The export
> layout MUST match exactly what the C kernels consume, or every Phase-1/2 gate fails for a
> reason unrelated to JAX. Coordinate this edit in the `port2/` repo.

**Files:**
- Modify (C port): a `fesom_mesh_export` routine (binary/NPZ of all mesh/geometry arrays), env-gated
- Create: `docs/MESH_EXPORT_LAYOUT.md` (the field-by-field spec)

- [x] write `docs/MESH_EXPORT_LAYOUT.md`: for EVERY array, fix shape, dtype, units, 0-vs-1-based, and packing order — cross-referenced to the C macros. Critical packings: `gradient_sca` is `[6*elem]` with the [1..3]=dNi/dx, [4..6]=dNi/dy split (`fesom_ssh.c:140`); `edge_cross_dxdy` is `[4*ed]` packed (dx1,dy1,dx2,dy2) in **meters** (`fesom_ssh.c:290-306`); `nod_in_elem2D` as CSR (offsets+flat); `edge_tri` uses −1 for boundary — done; **`elem_edges` dropped** (unused in the C port, verified by grep)
- [x] add `fesom_mesh_export` to the C port: after mesh init, write coords, `elem_nodes`, `edges`, `edge_tri`, `nod_in_elem2D` CSR, `nlevels*`, `ulevels*`, `zbar`, `Z`, `elem_area`, `area`, `areasvol`, `gradient_sca`, `edge_dxdy`, `edge_cross_dxdy`, `coriolis*`, `metric_factor`, `elem_cos` to one file matching the spec — `src/fesom_mesh_export.c` on branch `jax-mesh-export`; writes 31 `.npy` + `meta.txt`, env-gated `FESOM_MESH_EXPORT`, npes==1
- [x] verification: load the exported file in Python, assert counts (pi: nod2D=3140, elem2D=5839, **nl=48** — NOT ~23; "~23" is the per-node count, global nl=len(zbar)=48), index ranges (`elem_nodes ∈ [0,nod2D)`, `edge_tri` ≤ 2 nonneg), and value ranges (areas, gradients) per FRESH_START §20 — `scripts/verify_mesh_export.py` → PASS
- [x] run — must pass before Phase 1 — export cached at `port_jax/data/mesh_pi/` (job 25375272, 6 s)

#### Task 0.4: Reference-dump enablement + element-dump extension

> Review finding (Critical #1, #2): node dumps come from the **Fortran** shim
> (`fesom_dump_shim.F90`); element fields (`pgf`, `uv_rhs`, `uv`, `Av`) are **not** dumped
> by it. The five element-kernel gates in Phase 2 (Tasks 2.2/2.4/2.5/2.6/2.8) need either
> this extension or indirect verification.

**Files:**
- Modify (Fortran): add a `dump_shim_record_elem` routine + wire it at substeps 3/5/6/7/10
- Create: `docs/REFERENCE_RUNS.md` (how to produce dumps for pi)

> **Path A chosen (user, 2026-06-05): a C-port dump WRITER, not the Fortran shim.**
> The C port is the algorithmic SoT and holds all node+element fields at the exact
> JAX config — so JAX↔C diffs are pure FP reassociation (tightest gate), config is
> auto-matched, and no Fortran namelist/forcing/IC archaeology is needed. The
> Fortran dump stays a climate-level secondary cross-check.

- [x] per-substep oracle at the pinned `DUMP_PROBE_GIDS=[1001,1500,2000,2500,3000]`; step-1..10 NODE dumps captured into `tests/fixtures/pi_cdump.00000` (generated by the C writer)
- [x] element dump (probe element = first cell incident to the node probe → elem gids 1757/2656/3688/4604/5575) wiring `pgf_x/y`(3), `Av`(4), `uv_rhs_u/v`(5,6,7), `uv_u/v`(10) + all node fields — `fesom_dump.c`/`.h` on branch `jax-mesh-export`, PP+linfs+opt_visc7, constant IC, dt=100
- [x] document exact build + run commands in `docs/REFERENCE_RUNS.md`; note the JAX↔C↔Fortran chain — done (incl. why the existing Fortran dump is NOT per-substep comparable: realistic stratified IC + KPP/opt_visc5)
- [x] run — must pass before Phase 1 — fixture validated by `tests/test_reference_dump.py`; full suite **17 passed**

**GATE 0 — ✅ MET (2026-06-05):** env reproducible (x64+GPU); harness reads the
per-substep dumps and compares with tolerance (with `nlevels` truncation); pi mesh
arrays exported + verified; node AND element step-1..10 dumps captured as fixtures
(`pi_cdump.00000`) at the pinned probes — generated by the **C-port dump writer**
(Path A), the algorithmic source of truth. **Phase 0 complete.**

---

### Phase 1 — Mesh & State

#### Task 1.1: Load & verify mesh/geometry

**Files:**
- Create: `fesom_jax/mesh.py`
- Create: `fesom_jax/tests/test_mesh.py`

- [x] `mesh.py`: load the C-exported mesh NPZ into a frozen `Mesh` pytree/dataclass (registered via `register_dataclass`: arrays = leaves, scalar counts = static meta); keep connectivity as `int32`. **No 1→0 conversion needed** — the export is already 0-based (`edge_tri`/`edge_up_dn_tri` use −1 for boundary), verified.
- [x] build derived static arrays — four ragged-level masks (`node_/elem_ × layer/iface`) from `(ulevels,nlevels)`: **layer** valid `[ulevels-1, nlevels-1)` (T,S,ρ,p,u,v), **interface** valid `[ulevels-1, nlevels-1]` (bvfreq,w,Kv,Av), per `fesom_eos.c:93-208`. CSR (`nod_in_elem2D` offsets+flat) consumed as-is.
- [x] `tests/test_mesh.py`: every array == export bit-for-bit (31 arrays); indices 0-based + in range (`elem_nodes∈[0,nod2D)`, `edge_tri∈{−1}∪[0,elem2D)`, CSR consistent w/ `elem_nodes`); 8531 interior + 455 boundary edges; masks match level counts; no-cavity (`ulevels==1`); pytree round-trip — **12 passed**
- [x] run — must pass before Task 1.2 — green

#### Task 1.2: State pytree & gather/scatter/mask primitives

**Files:**
- Create: `fesom_jax/state.py`
- Create: `fesom_jax/ops.py`
- Create: `fesom_jax/tests/test_ops.py`

- [x] `state.py`: a `State` pytree (registered dataclass) holding all evolving fields (T,S,T_old,S_old,del_ttf; uv,uv_rhs,uv_rhsAB,uvnode,uvnode_rhs; w,w_e,w_i,cfl_z; eta_n,d_eta,ssh_rhs,ssh_rhs_old; hnode,hnode_new,helem,hbar,hbar_old; density,hpressure,bvfreq,Kv,Av,pgf_x,pgf_y) as `[n,nl]`/`[e,nl]`/`[·,nl,2]` dense arrays, each annotated w/ its C owner (`fesom_dyn`/`fesom_aux`) + layer-vs-interface. `State.zeros`/`State.rest(mesh,T0,S0)` factories
- [x] `ops.py`: `gather`/`gather_nodes_to_elem`/`gather_to_edges`; `scatter_add(vals,seg,n)` + `…_edges_to_nodes`/`…_edges_to_elems` (masked `segment_sum`; −1 sentinel contributes 0 fwd **and** in grad); `mask_below_bottom(field,mask)` (`where`, broadcasts over component axis); `tdma(a,b,c,d)` two-`lax.scan` (fwd elim + reverse back-sub), vectorized over the entity axis
- [x] `tests/test_ops.py` (+`test_state.py`): gather/scatter round-trip = degree-weighted; `scatter_add` vs reference loop; −1 masking; **scatter transpose == gather** (analytic vjp); `tdma` vs dense `linalg.solve` + **grad vs central FD** (d & b, ≤1e-6); mask zeros below-bottom & grad passes valid-only; State scan+grad — **17 passed**
- [x] run — must pass before Phase 2 — full suite **46 passed**

**GATE 1 — ✅ MET (2026-06-05):** mesh arrays match the C export **bit-for-bit**
(stronger than FP tol); `ops` primitives verified forward **and** under autodiff
(scatter transpose == gather; TDMA grad == finite-diff). **Phase 1 complete.**

---

### Phase 2 — Minimal Forward Step on pi

Configuration: **linfs** ALE, **PP** mixing, **upwind** tracer advection, **CG** SSH
solver, no GM/KPP/ice, zero or analytical forcing. Port the 16 substeps **in order**,
each verified against the C dump for that substep. C reference: `fesom_step.c` (driver),
files noted per task.

#### Task 2.1: EOS, pressure, N² (substep 1)

**Files:**
- Create: `fesom_jax/eos.py`
- Create: `fesom_jax/tests/test_eos.py`

- [x] port full Jackett-McDougall EOS (`densityJM_components`), in-situ density, hydrostatic pressure, Brunt-Väisälä N² — ref `fesom_eos.c`, FRESH_START §8 — `fesom_jax/eos.py` (`jm_components`, `pressure_bv`); `density` matches the dump **bit-for-bit** (max|Δ|=0, pointwise map), `pressure` ~1e-11/rel 1e-16 (cumsum integration)
- [x] **port the `fesom_smooth_nod3D(bvfreq)` pass** (`fesom_step.c:92`, `fesom_eos.c:226`, N2smth_h=.true.) — **CORRECTION: it is `n_smooth=1` (a SINGLE sweep)**, not "3-pass"; the "3" is the 3-vertex normalization `1/(3·Σ_patch area_el)`. Per owned node: `arr[n,nz] ← (Σ_{el∈patch(n)} area_el·(arr[v0]+arr[v1]+arr[v2])) / (3·Σ area_el)` over `nz∈[ulevels-1, nlevels-1]` (an element→node area-weighted patch average via `nod_in_elem2D` CSR → scatter/reduction class, ~1e-12). The substep-1 `bvfreq` dump is POST-smooth, so without this the gate fails (review Minor #13) — `eos.smooth_nod3D` (element→node `scatter_add`); verified **load-bearing** (raw bvfreq FAILS, smoothed PASSES @1e-16)
- [x] (EOS/pressure are map/gather → ~1e-15; the smoother is a node-patch **scatter** → ~1e-12) — confirmed
- [x] write `tests/test_eos.py`: compare `density`, `pressure`, `bvfreq` (post-smooth) probe columns vs **C** dump substep 1 at **all 5 probes**; assert per-kind tol. Note: substep 2 (`sw_alpha_beta`) is deferred to Phase 6 (GM/KPP-only) — `fesom_jax/ic.py` (constant + **T-blob** IC) added as the EOS input; node 1001 in-blob (bvfreq≠0), 3000 out (=0)
- [x] gradient check: `d(mean density)/d(T at a node)` AD vs finite-diff — central-FD step sweep, rel err <1e-6
- [x] run — must pass before next task — **test_eos.py 18 passed; full suite 64 passed**

#### Task 2.2: Pressure-gradient force (substep 3)

**Files:**
- Create: `fesom_jax/pgf.py` (or fold into `momentum.py`)
- Create: `fesom_jax/tests/test_pgf.py`

- [x] port PGF at elements: `fesom_pressure_force_linfs_fullcell` (`fesom_step.c:104`; Fortran `pressure_force_4_linfs`, `oce_ale.F90:3461`) — `fesom_jax/pgf.py` (`pressure_force_linfs`): gather hpressure→elem + `gradient_sca` contraction in C association order, /ρ0, masked to `elem_layer_mask`
- [x] write `tests/test_pgf.py`: compare `pgf_x/pgf_y` (element field) vs the Task-0.4 element dump substep 3 — element dumps ARE present, verified **directly** at all 5 element probes (max|Δ|~1e-20, gather class) + below-bottom-zero + EOS→PGF gradient flow
- [x] run — must pass before next task — **test_pgf.py 12 passed**

#### Task 2.3: PP vertical mixing (substep 4)

**Files:**
- Create: `fesom_jax/pp.py`
- Create: `fesom_jax/tests/test_pp.py`

- [x] port PP scheme (shear/N² factor, background, convective adjustment min) — ref `fesom_pp.c`, FRESH_START §10; keep the `max(N²,0)` clamp and the convective `max` exactly — `fesom_jax/pp.py` (`compute_vel_nodes`, `pp_mixing`, `mo_convect`, `mixing_pp`). 3-loop order preserved (Av reads factor² before Kv→factor³). Outputs on **interior** interfaces `[nzmin+1,nzmax)` only (surface/bottom 0)
- [x] write `tests/test_pp.py`: compare `Kv` (node) vs C dump substep 4 — `Av` **element dump present → verified DIRECTLY** at all 5 element probes (not indirect). Step-1 is at-rest (uv=0 → Kv=K_ver, Av=A_ver); the shear/N²/factor path + convective bump checked against an **independent loop-based numpy reference** of `fesom_pp.c` (synthetic uvnode/N²)
- [x] gradient check on `Kv(shear, N²)` away from the `max(N²,0)` / convective-`max` kinks — `d(ΣKv)/d(uvnode)` AD vs central FD, rel <1e-6
- [x] run — must pass before next task — **test_pp.py 14 passed; full suite 90 passed**

#### Task 2.4: Momentum RHS — Coriolis(AB2) + PGF + SSH grad (substep 5)

**Files:**
- Create: `fesom_jax/momentum.py`
- Create: `fesom_jax/tests/test_momentum.py`

- [x] port `compute_vel_rhs`: AB2 Coriolis (single-slot history, AB_order=2, ε=0.1 offset), PGF, SSH gradient; ref `fesom_momentum.c:49` — `fesom_jax/momentum.py` (`compute_vel_rhs`). AB-slot order preserved (OLD `uv_rhsAB` drives the shift, NEW Coriolis overwrites it, advection adds to NEW)
- [x] **port `momentum_adv_scalar` (momadv_opt=2)** — edge→node scalar-CV advection (`fesom_momentum.c:156`); element→node vertical-flux scatter + **antisymmetric edge→node** horizontal scatter, /areasvol, vertex→element. NOT omitted — verified nonzero & matched
- [x] write `tests/test_momentum.py::test_vel_rhs`: compare `uv_rhs` (element field) vs element dump substep 5 — **directly** at all 5 element probes (step-1 rest → `uv_rhs=−dt·pgf`). Coriolis/SSH/advection exercised by a synthetic test vs an **independent loop numpy reference** (both `is_first_step`) + AD gate
- [x] run — must pass before next task — **test_momentum.py 14 passed; full suite 104 passed**

#### Task 2.5: Horizontal (biharmonic) viscosity (substep 6)

**Files:**
- Modify: `fesom_jax/momentum.py`
- Modify: `fesom_jax/tests/test_momentum.py`

- [x] port `visc_filt_bidiff` (**biharmonic**, opt_visc=7, `fesom_momentum.c:654`) two edge→element antisymmetric scatter stages — `momentum.py` (`visc_filt_bidiff`, `_bidiff_edge_terms`). Interior edges only (el1≥0 AND el2≥0); per-edge overlap level range `[max(ulevels)-1, min(nlevels)-1)`. Flow-aware `sqrt(|∇u|²)` uses an **AD-safe double-`where` sqrt** (forward-identical, finite grad at the |∇u|=0 kink)
- [x] write `test_momentum.py::test_visc_filter`: compare `uv_rhs` vs element dump substep 6 — **directly** at all 5 element probes (rest → substep6==substep5). Synthetic vs **numpy reference** + AD gate + a **no-NaN-grad-at-rest** test for the safe-sqrt
- [x] run — must pass before next task — **test_momentum.py 27 passed; full suite 117 passed**

#### Task 2.6: Implicit vertical viscosity TDMA (substep 7)

**Files:**
- Modify: `fesom_jax/momentum.py`
- Modify: `fesom_jax/tests/test_momentum.py`

- [x] port per-element TDMA (2 unknowns u,v), wind stress surface BC, quadratic bottom drag — ref `fesom_momentum.c:291`; vectorize over elements using `ops.tdma` — `momentum.py` (`impl_vert_visc`). Phase-2 simplifications: `w_i=0` (advective tridiag terms drop), no partial cells (`zbar_n=zbar`, `Z_n=Z`). Bottom drag `|u|` uses `_safe_sqrt`. ➕ **`forcing.py`** added (analytical wind, **double-averaged** elem→node→elem per `oce_fluxes_mom`)
- [x] write `test_momentum.py::test_impl_vert_visc`: compare `uv_rhs` vs element dump substep 7 — **directly** at all 5 element probes (wind stress active even at rest → real TDMA solve). Synthetic vs numpy reference + re-averaged-stress unit test
- [x] gradient check through the TDMA solve — `d(Σdu)/d(uv_rhs)` (linear) and `d(Σdu)/d(Av)` (nonlinear via matrix) AD vs FD
- [x] run — must pass before next task — **test_momentum.py 40 passed; full suite 130 passed**

#### Task 2.7: SSH RHS + CG solve (substeps 8–9) — the AD-critical solver

**Files:**
- Create: `fesom_jax/ssh.py`
- Create: `fesom_jax/tests/test_ssh.py`

- [x] port `compute_ssh_rhs` linfs branch (edge→node scatter; the α / (1−α) blend) — ref `fesom_ssh.c:261`. This IS a node field → dumped at substep 8 — `ssh.compute_ssh_rhs`; antisymmetric edge→node scatter of `α·((v+vr)·dx−(u+ur)·dy)·helem`. **`α=1` ⇒ the `(1−α)·ssh_rhs_old` blend vanishes.** At step-1 rest `uv=0` but `uv_rhs=du` (wind-forced from substep 7) → the field is non-trivial (driven by the wind increment)
- [x] port the stiffness-matrix assembly (element Galerkin, NEGATIVE factor −g·dt·α·hbar) — ref `init_stiff_mat_ale` / `fesom_ssh.c`, FRESH_START §11. **In linfs the operator is built ONCE and reused every step** (`fesom_ssh.c:9-12`; `update_stiff_mat_ale` gated off) — so it is a *static* operator, not a per-step closure. (Per-step rebuild is a Phase-5/zlevel concern.) Represent it as a precomputed matvec (element contributions + `segment_sum`) — `ssh.build_ssh_operator` (host scipy assemble → static COO `segment_sum` matvec). The NEGATIVE comes from `depth=zbar_bot−zbar_srf<0` (= −hbar) × positive `factor=g·dt·α·θ`. Static (uses `zbar`, never the evolving `hbar`)
- [x] port the **MITgcm-style symmetric preconditioner** (`solver.F90:77-86` / `fesom_ssh.c:239-253`): `pr[diag]=1/diag(row)`, `pr[off,node]=-0.5*(off/diag_row)/(diag_row+diag(node))` — it has **off-diagonal terms** and is applied as a sparse **matvec**, NOT a diagonal/Jacobi scaling. Getting this wrong changes the Krylov path and the `d_eta` residual structure — `ssh.ssh_precond` (19336 off-diag entries). **Verified load-bearing**: a Jacobi variant gives a different early-stopped `d_eta` (off by 2.9e-10 @ probe 1001 → fails the dump)
- [x] solve with **`jax.lax.custom_linear_solve`** (symmetric; the preconditioner is part of `solve`); global dot = plain sum on single device (`psum` under shard_map in Phase 8) — `ssh.solve_ssh`. **⚠️ KEY FINDING:** the C stops at a *loose* `soltol=1e-5` (≈3 iters, `cond(S)≈800`), so the dumped `d_eta` is the **early-stopped** iterate — it matches the 3-iter PCG to ~1e-18 but the *exact* solve only to ~2e-10. So the forward `solve` **replicates the C PCG exactly** (early-stop), while `transpose_solve` converges *tight* → the gradient is the clean implicit-diff `S⁻¹` regardless
- [x] write `tests/test_ssh.py`: compare `ssh_rhs` (substep 8) and `d_eta` (substep 9) vs Fortran dump (≤1e-12; CG residual reassociates) — `d_eta` matches **~1e-18** at all 5 probes; `ssh_rhs` matches at **atol 1e-7** (transport divergence with cancellation → abs floor set by upstream `du`’s ~1e-12 rel amplified by `dx·helem~1e7`, NOT the scatter). + synthetic-vs-numpy-reference (nonzero `uv` exercises the dormant `(u+ur)` part), operator-symmetric, residual<soltol
- [x] **gradient check:** `d(d_eta)/d(rhs)` from `custom_linear_solve` vs finite-diff / vs an unrolled fixed-iter reference on a small case. Note: with a static linfs operator, the AD story is simpler — the operator does not depend on the evolving `hbar` — AD cotangent == tight `S⁻¹·w` (rel 2e-14) and == central-FD; finite; flows back through `compute_ssh_rhs` to `du`
- [x] run — must pass before next task — **test_ssh.py 18 passed; full suite 148 passed**

#### Task 2.8: Velocity update + hbar + eta_n (substeps 10–12)

**Files:**
- Modify: `fesom_jax/momentum.py`, `fesom_jax/ssh.py`
- Modify: tests

- [x] port `update_vel` (gather SSH correction to elements), `compute_hbar` (transport-divergence edge→node scatter, save `ssh_rhs_old`), `eta_n` blend — ref `fesom_momentum.c:474/779`, `fesom_ale.c` — `momentum.update_vel` (barotropic `∇N·d_eta` correction, `uv += du + F`), `ssh.compute_hbar` (= `compute_ssh_rhs` with `uv_rhs=0`,`α=1`, then `hbar += ssh_rhs_old·dt/areasvol`), `ssh.eta_n_update` (`α=1` ⇒ `eta_n = hbar`)
- [x] write tests: `hbar` (11) and `eta_n` (12) are **node** fields → compare vs dumps directly (≤1e-12). `uv` (10) is **element** → **element dump directly** at all 5 probes. **uv ~2e-17, hbar/eta_n ~1e-17** (gather/`÷area`-suppressed, far tighter than `ssh_rhs`'s 1e-7); + `update_vel`/`compute_hbar` synthetic-vs-numpy refs + AD (linear) + end-to-end `d(Σeta_n)/d(du)` through `custom_linear_solve`
- [x] run — must pass before next task — **test_ssh.py +18 / test_momentum.py +2 (= 175 full suite)**

#### Task 2.9: ALE step (linfs) (substep 13)

**Files:**
- Create: `fesom_jax/ale.py`
- Create: `fesom_jax/tests/test_ale.py`

- [x] port linfs ALE: `hnode_new = hnode` (static memcpy, `fesom_ale.c:10`); compute `w` (vertical velocity) — `fesom_jax/ale.py` (`thickness_linfs`, `compute_w`). `w` = the **per-level** antisymmetric edge→node `(v·dx−u·dy)·helem` transport divergence (the ssh_rhs/hbar scatter kept per-level, new `uv`, α=1), then a **reverse bottom→top cumsum** (`lax.cumsum(reverse=True)`; the masked scatter ⇒ no-flux `w[nzmax]=0` falls out), then **÷`mesh.area`** (⚠️ NOT `areasvol`), safe-divide guarded. helem recompute is substep 16 (Task 2.10), not here.
- [x] write `tests/test_ale.py`: compare `w`, `hnode_new` vs C dump substep 13 — `w` matches **~4e-20** (tight, hbar-class: ÷area crushes the cancellation floor), `hnode_new` **bit-for-bit** (max|Δ|=0) at all 5 node probes; step-1 `w` is a REAL gate (post-`update_vel` wind-driven `uv`). + synthetic-vs-numpy-loop-ref (~1e-18), `w[nzmax]==0` BC, AD==central-FD (linear in `uv`, finite at rest), end-to-end `d(Σw)/d(du)` through `custom_linear_solve`
- [x] run — must pass before next task — **test_ale.py 17 passed; full suite 192 passed**

#### Task 2.10: Tracer advection (upwind) + diffusion + commit (substeps 15–16)

**Files:**
- Create: `fesom_jax/tracer_adv.py`
- Create: `fesom_jax/tracer_diff.py`
- Create: `fesom_jax/tests/test_tracers.py`

- [x] port **upwind** horizontal+vertical advection (no FCT yet) — `fesom_jax/tracer_adv.py` (`adv_flux_hor`/`adv_flux_ver`/`flux2dtracer`/`ale_reconstruct`/`advect_one`). The 5 C level-zones collapse to a masked per-element `vflux` sum; upwind face `-½(T₁(v+|v|)+T₂(v−|v|))`; vertical uses `w_e=w`; AB2 `ttfAB` drives the flux, `values` the reconstruction. **Matches an independent numpy upwind loop ref bit-for-bit.** `edge_vflux` sign verified via constant-tracer-stays-constant (exact).
- [x] port `diff_tracers_ale`: ALE reconstruction `T_new=(T·hnode+del_ttf)/hnode_new` (linfs: thickness term 0), implicit vertical TDMA (per-node, 1 unknown) — `fesom_jax/tracer_diff.py` (`impl_vert_diff`). `impl_vert_visc`'s per-NODE sibling + `area/areasvol` ratio + `hnode_new` mass diagonal; `gm=NULL`/`do_wimpl=0`/`bc_surface=0`/`sw_3d=0`. ⚠️ `where(dZ==0,1,dZ)` to keep `d/d(Kv)` finite (0·inf NaN trap). Conserves `Σ areasvol·hnode·T` ~1e-16.
- [x] port thickness commit `hnode = hnode_new` + `helem=⅓Σ_vertices hnode` (substep 16) — `fesom_ale.c:18`, `ale.commit_thickness`
- [x] write `tests/test_tracers.py`: ⚠️ **dump runs FCT, port runs upwind** → `S=35` (constant) matches the dump **bit-for-bit** (clean gate), `T` (blob) differs ~3e-7 = upwind−FCT antidiffusive gap (bounded `<1e-5`; tight `T` match is Phase 4). + numpy-upwind-ref (bit-exact), constant-tracer (exact), diffusion conservation+smoothing, AD (linear in T, finite in uv kink, `d/d(Kv)` vs FD, end-to-end `d(ΣT)/d(du)`). `hnode` (16) bit-for-bit; `helem`==`State.rest`
- [x] run — must pass before next task — **test_tracers.py 20 passed; full suite 212 passed**

#### Task 2.11: Assemble `step()` and run pi forward

**Files:**
- Create: `fesom_jax/step.py`
- Create: `fesom_jax/tests/test_step_pi.py`
- Create: `fesom_jax/ic.py` (constant T=10,S=35 init for now)

- [x] wire the substeps into a single jitted `step(state, mesh, op, stress) -> state` — `fesom_jax/step.py` (`step` eager + `step_jit` `static_argnames=(dt,is_first_step)` + `run` loop). Mirrors `fesom_step.c` order; threads the warm-start `d_eta`, AB2 slots (`uv_rhsAB`,`T_old`,`S_old`), `hbar_old`, lagged `eta_n`/`w_e`. ⚠️ **`ssh.solve_ssh` fix:** the warm-start early-stop threshold must use the ORIGINAL `‖ssh_rhs‖` (`rtol_abs`), not `‖b_eff‖` — load-bearing for step-≥2 `d_eta`.
- [x] rest-state test: constant T/S (no blob), eta=0, uv=0, **zero** wind → stays at rest to machine precision — `max|uv|`~2e-16, T/S exactly constant (FRESH_START §20)
- [x] run 100 steps on pi at dt=100; assert stable — `max|uv|`=0.075, `|eta|`=0.35 m, no NaN, `S` exactly 35, T∈[10,15] (jitted `run`, ~20 s)
- [~] full-field snapshot diff vs a C 100-step pi run — **deferred** (rung 2; needs a C pi snapshot. Multi-step verified instead by: step-1 tight integration gate, exact-`S` invariant, climate-close SSH/velocity to the dump at step 2, warm-start load-bearing.) ⚠️ tight multi-step `T` match is a Phase-4 FCT gate (upwind≠FCT cascades via density).
- [x] write `tests/test_step_pi.py` — step-1 integration (all per-kernel gates via `step()`) + rest-state + exact-`S` + step-2 climate-close + warm-start load-bearing + 100-step stability + jit==eager (62 tests)
- [x] run — must pass before Phase 3 — **test_step_pi.py 62 passed; full suite 274 passed**

**GATE 2 — ✅ MET (2026-06-05):** pi 100 steps stable (no NaN, bounded uv/eta, `S` exactly
constant); step 1 reproduces **every** per-kernel substep dump gate through the integrated
`step()`; rest state machine-precision; CG warm-start load-bearing. Snapshot rung-2 deferred
(no C pi snapshot); the upwind→FCT tight multi-step `T` match is Phase 4. **Phase 2 complete.**

---

### Phase 3 — AD Smoke Test (de-risking gate)

#### Task 3.1: Scan + checkpoint time loop

**Files:**
- Create: `fesom_jax/integrate.py`
- Create: `fesom_jax/tests/test_integrate.py`

- [x] wrap `step` in `jax.lax.scan` over N steps; apply `jax.checkpoint` (rematerialization) to the step fn — `fesom_jax/integrate.py` (`integrate`/`integrate_jit`). **Step 1 runs eagerly (`is_first_step=True`) OUTSIDE the scan; steps 2..N scan with `is_first_step=False` baked in** (uniform body, no traced bool). Loop-invariant `mesh`/`op`/`stress_surf`/`params` closed over (carry = just `State`).
- [x] confirm forward result of the scan == the Phase-2 manual loop (climate-close) — **BIT-IDENTICAL** (`integrate`==`run`: uv ~4e-19, all else 0.0); checkpoint on==off forward exactly 0.0
- [x] memory sanity: N=200 pi steps backward pass fits in device memory with checkpointing — `scripts/phase3_grad_memory.py` + `.sbatch` (GPU job 25378918, A100-40). **Checkpointed: grad finite, peak 4.23 GB / 31.8 GB (13%), 26 s. Un-checkpointed: OOM at 48.7 GB** (XLA couldn't remat below 28 GiB) ⇒ **checkpointing is load-bearing** for the backward pass.
- [x] write `tests/test_integrate.py` (forward-equivalence) — scan==run (N=1,2,5,12); checkpoint forward-transparent; small-N backward finite + checkpoint-invariant
- [x] run — must pass before next task — **test_integrate.py + test_gradient.py 12 passed (CPU)**

#### Task 3.2: End-to-end gradient check

**Files:**
- Create: `fesom_jax/tests/test_gradient.py`

- [x] define a scalar loss (e.g. mean SST after N steps); choose the param/loss to stay in a **smooth regime** — verify the probe column never goes convective so the PP `max(Kv,0.1)` / `max(N²,0)` kinks don't bite (review Important #8) — `loss = mean SST` over wet surface nodes; `test_smooth_regime` certifies the blob probe column stays **stratified** (bvfreq>0), `S`=35 (≫0.5 floor), no genuine convection (bvfreq only dips to ~-2e-14 FP-noise in the dead constant-T region, carrying no gradient)
- [x] reverse-mode `jax.grad` of loss w.r.t. a scalar parameter (PP `K_ver` background diffusivity) — threaded as a traced **`Params` leaf** (`params.py`, the ML-hook seam) `step→pp.mixing_pp`; `params=None ⇒ defaults` keeps the 274-test suite bit-identical
- [x] **finite-difference check with a step-size SWEEP:** compute FD at `h ∈ {1e-4…1e-7}` (relative, central, float64), report the FD-convergence plateau, and assert `|grad_AD − grad_FD|/|grad_FD| < 1e-4` at the plateau — not at a single `h` (chaos floor below, truncation error above) — swept `h∈{1e-3..1e-7}`; **plateau 5.9e-7 at `k_ver=1e-4`** (lifted off the ~eps·10 round-off floor); physical `k_ver=1e-5` gives 4.5e-5 + correct sign. ⚠️ plateau is at LARGE `h` (loss near-linear in k_ver ⇒ round-off dominates small `h`)
- [x] keep N modest for the smoke test (the forward model is mildly chaotic via scatter reassociation — see `GPU_FIDELITY.md` M5.8/M5.9; long windows amplify it). Note this as a known long-window gradient-stability risk — **N=20, dt=100** (the validated 100-step-stable config); noted in the test docstring
- [x] confirm gradient flows through the CG `custom_linear_solve` (perturb a param affecting the stiffness/RHS) — `test_grad_flows_through_cg`: `d/d(a_ver)` routes through the CG *within* a step (a_ver→Av→impl_vert_visc→du→ssh_rhs→CG), FD-confirmed; `k_ver` routes through it *across* steps
- [x] grad w.r.t. an initial-condition field (vector-valued) sanity check — `d(loss)/d(T₀)` finite **everywhere** (incl. masked lanes — this exposed + now guards the eos `zdiff` backward-NaN trap), nonzero on wet, exactly 0 on below-bottom
- [x] write `tests/test_gradient.py` as the permanent AD gate — 5 tests (k_ver sweep, physical point, CG-flow, IC-field, smooth-regime)
- [x] run — must pass before Phase 4 — **test_gradient.py green (CPU)**

**GATE 3 — ✅ MET (2026-06-06):** end-to-end gradient passes on the assembled pi model.
`integrate` (checkpointed `lax.scan`) reproduces the Phase-2 `run` loop **bit-identical**;
`d(mean SST)/d(k_ver)` AD↔FD plateau **5.9e-7** (≪1e-4) with the gradient flowing through
the CG `custom_linear_solve`; `d(loss)/d(T₀)` finite (after fixing the eos `bvfreq` `1/zdiff`
backward-NaN trap — the masked-NaN hunt the gate was meant to surface); the N=200 backward
fits device memory **only** with checkpointing (4.23 GB vs 48.7 GB OOM). The hard AD patterns
(scan+checkpoint, `custom_linear_solve`, upwind/PP kinks) are **proven on the real model** —
*the project's biggest risk is retired.* **Phase 3 complete; full suite 286 passing.** Then
Phase 4 (FCT + opt_visc7 completion + pi 1000-step) — at which the tight multi-step `T/S` dump
match becomes available.

---

### Phase 4 — pi Fully Stable

#### Task 4.1: FCT (Zalesak) advection + the limiter-gradient decision

**Files:**
- Modify: `fesom_jax/tracer_adv.py`
- Create: `docs/LIMITER_GRADIENTS.md`
- Modify: `fesom_jax/tests/test_tracers.py`

- [x] port high-order + Zalesak limiter (`fct_plus/minus`, local min/max bounds, sign-dependent flux selection) — ref `fesom_tracer_adv.c:814+`, FRESH_START §12 — `tracer_adv.advect_one_fct` (MFCT 3rd-order hor + QR4C 4th-order ver + `compute_fct_lo` + `fill_up_dn_grad` + `zalesak_limit` + `flux2dtracer_fct`); wired into `step.py`. **⚠️ Found+fixed THE bug**: the C step-1 `T_old` = pre-blob base T=10 (NOT the blob), mis-attributed 2 phases as the "upwind−FCT gap" — `ic.py`.
- [x] **RESEARCH ITEM:** decide & document the limiter-gradient strategy in `docs/LIMITER_GRADIENTS.md` — **(a) subgradient as-is** chosen: NaN-safe (the C `flux_eps=1e-16` floors every limiter ratio), FD-consistent where smooth, forward unchanged; (b) smooth-relax + (c) stop_gradient rejected (b changes the forward / fails the dump gate; c biases the gradient). Plain `jnp` VJP.
- [x] forward verification: FCT `T/S` vs C dump substep 15 — **TIGHT: `T` matches 1.8e-15** (was 3.4e-7 under the `T_old` bug), `S` bit-for-bit; + an independent numpy FCT reference (smooth **and** a synthetic limiter-ACTIVE case)
- [x] gradient check with the chosen limiter strategy — `test_gradient.py` green with FCT live: `d(SST)/d(k_ver)` plateau **5.7e-7** (≪1e-4), flows through the CG **and** the limiter; `d/d(T₀)` finite everywhere (no limiter/qr4c masked-NaN)
- [x] run — must pass before next task — **test_tracers.py 35 / test_step_pi.py 67 / test_gradient.py 5; full suite green**

#### Task 4.2: Complete opt_visc=7 (flow-aware biharmonic) + wsplit

> Review finding (Important #6): the C port runs **opt_visc=7** (biharmonic, flow-aware;
> `fesom_step.c:134-137`, required for dt=1800 stability), NOT opt_visc=5 harmonic+backscatter.
> Task 2.5 already ports the biharmonic `visc_filt_bidiff` scatter — so this task COMPLETES
> the flow-aware terms of opt_visc=7 (if Task 2.5 did a basic version) and adds wsplit; it
> does not re-port a different scheme.

**Files:**
- Modify: `fesom_jax/ale.py`, `fesom_jax/step.py`
- Modify: `fesom_jax/tests/test_momentum.py`, `test_ale.py`, `test_step_pi.py`

- [x] flow-aware terms of `visc_filt_bidiff` (opt_visc=7, γ0/γ1/γ2 = 0.003/0.1/0.285) —
  **were already fully ported in Task 2.5** (the `max(γ0, max(γ1·|du|, γ2·|du|²))` flow-aware
  `max` is present + matches the C). The gap was **verification, not code**: the dump can't
  reach the flow-aware regime (max edge-velocity-diff ≤8e-4 over all 10 steps ≪ the |du|>0.03
  γ1-onset / |du|>0.351 γ2-onset; **flow-aware-active = 0 edges** every step), and the moderate
  `_synthetic` test (amp 0.1) only reaches the γ1 branch (51% of edges). Added
  `test_visc_filter_flow_aware_branches_vs_reference` (strong synthetic flow ~2 m/s exercising
  BOTH γ1 and the quadratic γ2 branch vs the numpy ref, with a load-bearing branch-selection assert).
- [x] `use_wsplit` vertical-velocity splitting (wsplit_maxcfl=1.0) — `ale.compute_cfl_z` +
  `ale.compute_wvel_split` (faithful to `fesom_ale.c:204,241`), wired into `step.py` (computes
  `cfl_z` every step → `State.cfl_z`). **`use_wsplit=0` in the reference config** (`fesom_constants.h:56`)
  ⇒ the split is the **identity** (`w_e=w, w_i=0`) on the dump-matching path (numerically
  transparent — all gates + the gradient plateau 5.70e-7 unchanged); pi's CFL never nears
  maxcfl (max `cfl_z`~1e-4) so it's inactive even if turned on. The active branch is verified
  vs the numpy ref with a synthetic super-critical CFL. ⚠️ `impl_vert_visc`'s `w_i` advective
  tridiagonal terms stay dropped (correct at `w_i=0`) — re-enabling them is a Phase-5 item.
- [x] forward verification vs dumps: substep-13 `w` (node, already green); **substep-6 `uv_rhs`
  (element) now gated at step 2** — `test_step2_uv_rhs_visc_matches_dump` (real wind-driven flow,
  FCT made the trajectory tight) matches the dump **~1e-17** (the step-1 gate was rest-trivial).
- [x] run — **full suite green** (test_ale 21, the 2 new viscosity gates, gradient plateau 5.70e-7)

#### Task 4.3: pi 1000-step stability + AD re-check

**Files:**
- Modify: `fesom_jax/tests/test_step_pi.py`, `fesom_jax/tests/test_gradient.py`

- [x] run pi 1000 steps at dt=100; assert stable — `test_1000_step_stability` (~48 s,
  jitted `run`): no NaN, max|uv|=0.17, max|eta|=0.63 m, **S exactly 35** over the whole
  window (bit-exact threading), T∈[10.0, 14.98] (blob ~+5°C, bounded). Max `cfl_z`=2.8e-3
  ≪ maxcfl=1.0 ⇒ wsplit would stay inactive even at 1000 steps (use_wsplit=0 self-consistent).
  ⚠️ "snapshot climate-close to C" stays **indirect** (no C 1000-step pi snapshot exists —
  same as the Phase-2 deferral): verified instead by the tight step-1..10 dump match (FCT),
  the S-exact invariant, bounded/stable fields, and the step-2 element gate.
- [x] re-run the Phase-3 gradient gate with FCT + opt_visc7 active — `test_gradient.py`
  green with the full pi physics (FCT limiter + opt_visc7 + the wsplit machinery live):
  `d(SST)/d(k_ver)` plateau **5.70e-7** (≪1e-4), flows through the CG, `d/d(T₀)` finite.
  ("backscatter" in the old wording was a misnomer — the C runs opt_visc=7 biharmonic.)
- [x] run — full suite **313 passed**

**GATE 4 — ✅ MET (2026-06-06):** pi 1000 steps stable (no NaN; bounded |uv|/|eta|; S exactly
35; T bounded; CFL ≪ maxcfl); the gradient gate still passes with the full pi physics (FCT +
opt_visc7) live — plateau 5.70e-7, flowing through the CG. Climate-close-to-C stays indirect
(no C 1000-step snapshot). **Phase 4 complete.** Next: Phase 5 (CORE2 single-device) — expand
into `docs/plans/<date>-fesom-jax-core2.md`.

---

### Phase 5 — CORE2 Single-Device

*(To be expanded into `docs/plans/<date>-fesom-jax-core2.md` when reached.)* Outline:

- CORE2 mesh specifics: rotation auto-detect, CW element orientation (`test_tri`),
  partial cells, `nlevels_nod2D_min` (K_v⁻) — FRESH_START §2/§4/§14.
- **zlevel** ALE (surface-layer thickness change; local-zstar fallback) — `fesom_ale.c`.
- PHC initial conditions (bilinear interp + extrap + vertical fill) — `fesom_phc.c`.
- JRA55 forcing reader (bilinear→mesh, time interp, L&Y09 bulk formulae) — `fesom_jra55.c`,
  `fesom_bulk.c`, FRESH_START §9.
- SSS restoring + runoff (additive virtual freshwater flux) — `fesom_sss_runoff.c`.
- **GATE 5:** CORE2 1-day (172 steps, dt=500) and 10-day climate-close to C; gradient
  check on a CORE2 slice.

### Phase 6 — Full Physics

Split into focused sub-plans (one per subsystem), per the Phase-5 discipline.

- **Sea ice — ✅ COMPLETE (2026-06-06/07, GATE 6 met).** Sub-plan
  `docs/plans/20260606-fesom-jax-phase6-seaice.md`. Thermodynamics + EVP dynamics (120 fixed
  subcycles → checkpointed `lax.scan`) + FCT advection + coupling, all per-kernel bit-exact vs
  the C ice-ON dumps; assembled prognostic-ice CORE2 step matches the C at step 1; **10 days
  stable with the high-lat supercooling CAPPED at −1.91 °C + runoff active** (the two Phase-5
  findings resolved); gradient-gated (FD↔AD plateau 4.5e-10 + masked-NaN clean at scale).
  New modules: `ice.py`/`ice_thermo.py`/`ice_coupling.py`/`ice_evp.py`/`ice_adv.py`/`ice_step.py`.
- **GM/Redi — ✅ COMPLETE (2026-06-07, GATE 6B met).** Sub-plan
  `docs/plans/20260607-fesom-jax-gmredi.md`. The mesoscale eddy parameterization = bolus
  advection (GM) + neutral diffusion (Redi); the **second ML-hook seam** (eddy fluxes,
  `k_gm`/`redi_kmax` threaded through `params.py`). ⚠️ NOT one substep-14 kernel — it threads
  the step at **6 points** (`sw_alpha_beta` + the coefficient/bolus block + `fer_w` + the bolus
  advection wrap + the Redi G7a/G7b explicit terms + the K33 implicit augmentation).
  **Stateless** (recomputed each step from T/S/N² — no new State fields, unlike ice σ). 7-task
  ladder (G.1-G.7), each kernel dump-gated bit-exact; **assembled GM step bit-exact (7e-15) vs
  the C GM-ON dump** (GM is deterministic — no ice-EVP floor, so it closes K33's gate too);
  **10 days stable + GM smooths fronts** (front|∇T| ~6% lower than GM-OFF); **gradient-gated —
  the 2nd ML-hook `d/d(k_gm)` plateau 3.5e-6 (well-conditioned) + masked-NaN clean.** New modules
  `gm.py`/`gm_redi.py` + `eos.compute_sw_alpha_beta`. Wired behind `gm_cfg=None` (bit-identical).
- **KPP (6C):** (FESOM1.4, lookup-table, mix_scheme_nmb=1) — `fesom_kpp.c`, FRESH_START §10.
  **Forward-only** (the NN-replacement target); verify via `kpp_dump_diff.py`-style probes.
  Own sub-plan; the **first ML-hook's alternative** (`pp.py` ↔ a `kpp.py` behind the seam).
- **GATE 6 (sea ice) ✅ met;** the climate-stats GATE (CORE2 multi-year vs C/Fortran,
  `eps_climate_compare_2yr.py`) is the cross-phase acceptance after GM/Redi + KPP land.

### Phase 7a — Differentiable Parameter Tuning / Calibration (the on-ramp; user-requested 2026-06-07)

*(To be expanded into its own sub-plan.)* **Use the differentiable port to CALIBRATE physics
parameters (in GM, sea ice, mixing, …) against a target, and push the optimum back into the
operational Fortran model.** This is the *same `params.py` ML-hook seam* — but the trainable leaves
are the existing physical constants, not NN weights — so it is a **small extension, not a restructure**.
The machinery is already proven: the `Params` pytree (`params=None ⇒ bit-identical`), the checkpointed
differentiable `integrate`, and **FD-verified gradients** (`d/d(k_ver)` 5.8e-10, `d/d(k_gm)` 3.5e-6,
masked-field `d/d(T0)` clean). What's missing is only an objective + an optimizer loop + a target.

**Three Fortran-transfer tiers (increasing work):**
1. **Scalar / depth-profile → Fortran `namelist.oce` — ZERO Fortran code (the killer app).** Tune
   `K_GM_max`/`Redi_Kmax`/`K_GM_min`/`GMzexp_*`/background diffusivities in JAX → write the optimum
   into the namelist → run operational Fortran. The port becomes an **offline calibration engine**.
2. **2D/3D parameter FIELD → Fortran reads a netCDF.** Make the `Params` leaf a `[nod2D]`/`[nod2D,nl]`
   array (JAX: ~0 structural change — array leaves already differentiate); Fortran: a modest
   field-read + use-per-cell (~tens of lines/param; FESOM already reads many 2D fields).
3. **NN parameterization → Fortran inference** (= the Phase-7 goal). Export weights; hand-code the
   small-MLP forward in Fortran (~150 lines) or wire FTorch/SmartSim (~1-2 wk incl. build system).

**⚠️ The decades-spin-up problem (do NOT backprop through it).** The fully-developed GM/eddy state
takes years-to-decades; a multi-decade adjoint is infeasible on BOTH counts — memory (a decade ≈ 630k
steps; even O(√N) checkpointing → >150 GB) AND, decisively, **chaotic gradient blow-up** (adjoint
sensitivity grows exponentially past the Lyapunov time; the MITgcm-adjoint reason long-time adjoints
are run over *months*, not decades). The architecture instead **decouples spin-up from gradient**:
- **spin up FORWARD, no AD** (the stability run already is this; or use the Fortran/C model) to
  (near-)equilibrium — cheap, no memory/chaos issue;
- **tune with SHORT-window adjoint anchored at the equilibrated state** (days-to-months — memory +
  chaos OK), minimizing the short-window drift / obs-misfit;
- **for the slow equilibrated mean, use gradient-free Ensemble Kalman Inversion (EKI)** over forward
  runs (no adjoint, no chaos, embarrassingly parallel via `vmap`). The diff. port still serves the
  fast/short-window adjoint + the cheap batched forwards EKI needs.

**Well-conditioned first targets** (clean gradients shown): `k_gm`/`redi_kmax` (GM eddy — sets ACC /
stratification), `k_ver`/`a_ver` (mixing — SST/MLD), the GM depth/resolution scalings (`GMzexp_zref`,
`refscalresol`). Ice *thermo* params tunable; ⚠️ EVP *rheology* params are stiff (`1/delta_min` ~1e16,
Task 6.7) — `stop_gradient` the EVP or use EKI for those.

**Task ladder (draft):** (7a.1) **perfect-model twin** — inject `k_gm=1500`, recover from 800 by
gradient descent over a SHORT window (proves the optaxc/loss loop end-to-end; ~1 day). (7a.2) the
`Params`-expansion pattern for N tunables + a misfit (reference-run first, then real obs SST/SSS/ice).
(7a.3) short-window-adjoint-at-equilibrium tuning + an EKI baseline for the slow mean. (7a.4) export
the optimum to `namelist.oce` and confirm the **Fortran** run reproduces the JAX-predicted improvement
(the config-match caveat: tune for the config you'll run — the port is the reduced default-namelist
physics). **GATE 7a:** a tuned scalar (e.g. `k_gm`) measurably reduces a defined misfit in JAX AND,
written to the namelist, in Fortran; the perfect-model twin recovers the injected value.

### Phase 7 — ML Hooks + Batch-Parallel Training

*(To be expanded into its own sub-plan.)* Outline:

- Refactor **mixing** (`Kv/Av`) and **eddy flux** (GM bolus / Redi) behind a clean
  swappable interface: `param_fn(state, mesh, static_params) -> diffusivities/fluxes`.
- NN-backed implementations (pick **flax** or **equinox** — decide at phase start);
  keep the physics versions as baselines behind the same interface.
- **Batch/ensemble parallelism:** `vmap` over samples + device shard over the batch
  (data-parallel; no mesh sharding).
- Demonstrate end-to-end training of an embedded NN parameterization on a toy objective.
- **GATE 7:** a trained embedded NN parameterization measurably improves a toy objective
  end-to-end; physics-baseline path unchanged when the NN is swapped out.

### Phase 8 — Mesh Sharding (deferred)

*(To be expanded into its own sub-plan.)* Outline:

- `jax.experimental.shard_map` + `jax.lax.ppermute` halo exchange, **reusing the
  `dist_N/` partition files** (read `my_list`/`com_info`, build send/recv index maps).
- Replace global dots with `jax.lax.psum`; replace the host halo with `ppermute`.
- **AD through collectives:** verify `ppermute`/`psum` transposes give correct
  distributed gradients (gradient check, multi-device).
- Validate on a large mesh (`farc`, ~638K nodes).
- **GATE 8:** distributed run climate-close to single-device AND gradient-correct.

---

### Task N−1: Verify acceptance criteria (per phase)

- [ ] all substep gates for the phase pass within tolerance
- [ ] phase stability run completes (no NaN/blowup)
- [ ] gradient gate passes (Phase 3+)
- [ ] full verification suite green: `pytest fesom_jax/tests/`

### Task N: Documentation & plan hygiene

- [ ] update `README.md` and `docs/` with what the phase delivered
- [ ] record any new gotchas in a `docs/PORTING_LESSONS.md` (mirror the C/Kokkos lesson logs)
- [ ] move this plan to `docs/plans/completed/` only when **all** phases are done; until
      then, check off completed phases here and spawn sub-plans for Phases 5–8

## Technical Details

- **Array layout:** node-3D `[n_nod, nl]`, elem-3D `[n_elem, nl]`, elem-vector
  `[n_elem, nl, 2]`; matches C macros `FESOM_NODE3D/ELEM3D/ELEMVEC`. Ragged depth via a
  boolean `[n_entity, nl]` mask (zeros below bottom).
- **Connectivity (static):** `elem_nodes [n_elem,3]`, `edges [n_edge,2]`,
  `edge_tri [n_edge,2]` (−1 for boundary → handle with masked scatter),
  `elem_edges [n_elem,3]`, `nod_in_elem2D` as CSR (offsets + flat).
- **Scatter:** `jax.ops.segment_sum(contrib, segment_ids, num_segments=n)`; boundary
  (−1) entries masked or routed to a dump slot. Non-deterministic order on GPU →
  climate-close, differentiable.
- **CG:** `jax.lax.custom_linear_solve(matvec, b, solve, symmetric=True)`. In **linfs**
  the stiffness operator is **static** (built once — `fesom_ssh.c:9-12`), so `matvec` uses
  a precomputed operator, not a per-step closure (per-step rebuild is Phase-5/zlevel). The
  preconditioner is **MITgcm-style symmetric** (off-diagonal terms, `fesom_ssh.c:239-253`),
  applied as a matvec inside `solve` — **not** Jacobi/diagonal.
- **Time loop:** `jax.lax.scan(checkpointed_step, state0, xs)`; for very long windows use
  nested/`policy`-based checkpointing.
- **Precision:** float64 everywhere (`jax_enable_x64`). float32/mixed is a Phase-7+
  training-perf lever, gated behind a config flag, never on the verification path.
- **Determinism note:** to *debug* a discrepancy, a CPU single-thread deterministic
  scatter can recover near-bit-identity for map/gather kernels; production stays
  vectorized/climate-close.

## Post-Completion

*Items requiring runtime/external action — informational, no checkboxes.*

**Reference runs & evaluation:**
- generating multi-year CORE2 C/Fortran reference output for the Phase-6 climate gate
  (SLURM, `port_kokkos/docs/REFERENCE_RUNS.md` as template)
- GPU memory/throughput tuning for training (Phase 7), incl. float32/mixed experiments
- scientific evaluation of trained NN parameterizations vs physics baselines

**External:**
- the small additive C-port routines (mesh exporter; any extra probe dumps) live in the
  C port repo, not here — coordinate those edits there

## Revision Log

- **2026-06-05 — created** from the brainstorm (5 locked decisions, 8-phase roadmap).
- **2026-06-05 — revised after plan-review** (agent cross-checked the C/Fortran source).
  Fixes applied:
  - **Reference provenance corrected:** the per-substep dumps come from the **Fortran**
    model (`fesom_dump_shim.F90`), not the C port; C is the algorithmic source-of-truth,
    Fortran is the numerical reference (verified via `fesom_ssh.c` header + `inspect_dump.py`).
  - **Node-only dumps:** the shim records node fields only; element fields (`pgf`, `uv_rhs`,
    `uv`, `Av`) need a Fortran element-dump extension (new Task 0.4) or indirect verification.
    Element-kernel gates (2.2/2.3/2.4/2.5/2.6/2.8) updated accordingly.
  - **SSH solver corrected** (verified in `fesom_ssh.c:9-12,239-253`): linfs builds the
    stiffness matrix **once** (static operator, not per-step); preconditioner is
    **MITgcm-style symmetric** (off-diagonal, matvec), not Jacobi. Task 2.7 + Technical Details.
  - **Momentum advection** (`momadv_opt=2`, `fesom_momentum.c:156`) was missing — folded into
    Task 2.4; substep list annotated.
  - **opt_visc 5↔7 inconsistency** fixed (C runs opt_visc=7 biharmonic); Task 4.2 reconciled
    with Task 2.5.
  - **Mesh exporter** promoted to a specced Task 0.3 (the Phase-1 gating dependency).
  - Added: `bvfreq` smoothing pass (Task 2.1), probe-GID pinning + `nlevels` truncation
    (Tasks 0.2/0.4), gradient-check step-size sweep (Task 3.2), scatter-reassociation
    guardrail.
- **2026-06-05 — execution session 1** (Phase 0). Tasks 0.1, 0.2, 0.3 complete.
  - Env (0.1): mamba env `fesom-jax`, Python 3.12.13, **jax/jaxlib 0.10.1** + jax-cuda12
    (CUDA 12.9 pip wheels); x64 verified CPU+GPU; A100-40 verified (job 25374974).
  - Verify harness (0.2): `io_dump.py` + `verify.py` (`compare_column`, isclose-form
    `|Δ|≤atol+rtol·|c|`); 14 tests green.
  - Mesh exporter (0.3): added to `port2/fesom2_port` (branch `jax-mesh-export`); pi mesh
    exported to `data/mesh_pi/` (31 arrays) + verified.
  - **CORRECTION: the pi mesh is nl=48 globally** (`nlevels_nod2D ∈ [5,46]`); FRESH_START's
    "nl≈23" is the per-node level count. **Size all JAX node/elem columns to nl=48.** Other
    "~23" mentions in this plan (Task 1.1, Gate 0) should read nl=48.
  - Cross-repo policy (user): Fortran (and C) edits live on separate `port2` branches;
    I drive builds + pi SLURM runs (acct ab0995, `shared`/`compute`).
  - Task 0.4 done via **Path A** (user-chosen): a per-substep dump WRITER added to
    the C port (`fesom_dump.c`/`.h` + 17 hooks in `fesom_step.c`, branch
    `jax-mesh-export`) — node + element fields, config auto-matched (PP/linfs/
    opt_visc7, constant T=10/S=35 IC, dt=100). pi reference `fixtures/pi_cdump.00000`
    (10 steps, 1200 records) captured + validated (`test_reference_dump.py`; suite 17
    passed). The Fortran shim is NOT extended; the existing Fortran dump is a
    climate-level cross-check only (realistic IC + KPP/opt_visc5 → not per-substep
    comparable). **GATE 0 MET — Phase 0 complete.** Next: Phase 1 (mesh load + ops,
    pure JAX).
- **2026-06-05 — execution session 2** (Phase 1). Tasks 1.1, 1.2 complete; **GATE 1 MET**.
  - Mesh (1.1): `fesom_jax/mesh.py` — frozen `Mesh` registered as a JAX pytree
    (`register_dataclass`: 31 arrays = leaves, 7 scalar counts = static meta), `load_mesh`.
    **Export confirmed already 0-based** (no 1→0 conversion); −1 = boundary in
    `edge_tri`/`edge_up_dn_tri`. Four ragged-level masks derived from `(ulevels,nlevels)`:
    **layer** `[ulevels-1,nlevels-1)` (T,S,ρ,p,u,v) vs **interface** `[ulevels-1,nlevels-1]`
    (bvfreq,w,Kv,Av) — read off `fesom_eos.c:93-208` (density loop `nz<nzmax`; bvfreq
    padded at nzmin/nzmax). `test_mesh.py`: bit-for-bit vs export + index/CSR/mask/level
    consistency, 12 passed.
  - State+ops (1.2): `state.py` (`State` pytree mirroring `fesom_dyn`+`fesom_aux`+thickness,
    `zeros`/`rest` factories) and `ops.py` (gather; masked `scatter_add` via `segment_sum`,
    −1→0 fwd+grad; `mask_below_bottom`; vectorized `tdma` = two `lax.scan` sweeps).
    `test_ops.py`+`test_state.py`: forward correctness + **AD gates** (scatter transpose ==
    gather; TDMA grad == central FD to 1e-6; State through scan+grad), 17 passed.
  - **Full suite 46 passed** (CPU; login-node `cuInit 303` warning is the documented benign
    GPU-absent fallback — run with `JAX_PLATFORMS=cpu` to silence). Convention note for
    Phase 2: 3D field layout is row-major `[n_entity, nl]` == C `FESOM_NODE3D`; vectors
    `[·, nl, 2]` == `FESOM_ELEMVEC`. **Next: Phase 2 (Task 2.1, EOS/pressure/N² substep 1).**
- **2026-06-05 — execution session 3** (Phase 2 start, Task 2.1).
  - **Added a STANDING RULE + `docs/PORTING_LESSONS.md`** (living lessons log; see
    Development Approach). Append per-task, cite source.
  - **⚠️ IC CORRECTION (load-bearing for ALL of Phase 2):** the pi reference dump is
    **NOT** a bare constant T=10/S=35 IC. `fesom_main.c:744-753` adds
    `fesom_ic_tracer_T_blob` (Gaussian +5 °C T-blob, centre (−45°,40°) geo, σ_h=10°,
    σ_z=300 m, 4σ horizontal cutoff, S unchanged) on top of the constant whenever no PHC
    path is given — and the dump run gives none. Probe 1001 is inside the blob
    (stratified, bvfreq≠0), 3000 outside (T=10, bvfreq=0). Every T/S-dependent gate
    (EOS→pressure→PGF→momentum→…) must reproduce the blob. `REFERENCE_RUNS.md` IC row
    updated. T/S are effectively frozen over the 10 dumped steps (weak flow), so
    substep-1 EOS fields are step-independent here. Detail in PORTING_LESSONS.md.
- **2026-06-05 — execution session 4** (Phase 2, Task 2.7: SSH RHS + CG solve,
  substeps 8–9). `fesom_jax/ssh.py` + `tests/test_ssh.py` (18 tests; **full suite
  148 passed**). Phase-2 config unchanged (linfs, PP, opt_visc7, analytical wind,
  dt=100).
  - **ssh_rhs (8):** `compute_ssh_rhs` — antisymmetric edge→node scatter of
    `α·((v+vr)·dx−(u+ur)·dy)·helem`. **`SSH_ALPHA=1` ⇒ the `(1−α)·ssh_rhs_old` blend
    term is identically 0.** Step-1 is at rest (`uv=0`) but `uv_rhs=du` (the
    wind-forced increment overwritten into `uv_rhs` at substep 7), so ssh_rhs is
    non-trivial. Matches the dump at **atol 1e-7** (not 1e-12): ssh_rhs is a
    transport divergence with heavy cancellation; its abs floor (~5e-9 @ probe 1500)
    is the upstream `du`’s ~1e-12 *relative* error **amplified by `dx·helem ~ 1e7`**,
    not the ssh_rhs scatter (a numpy-sequential ref and `segment_sum` both land ~5e-9
    vs the dump — the floor is shared upstream `du`, confirming the diagnosis).
  - **Stiffness operator:** **static in linfs** — `build_ssh_operator` assembles the
    element-Galerkin `S` once (host scipy COO→CSR), stored as a `segment_sum` matvec.
    The "NEGATIVE factor −g·dt·α·hbar" = positive `factor=g·dt·α·θ` × `depth=zbar_bot−
    zbar_srf < 0` (= −hbar in linfs, the *static* full depth). `cond(S)≈800`,
    symmetric to FP.
  - **⚠️ KEY FINDING — the C stops the CG at a *loose* `soltol=1e-5`** (≈**3
    iterations** on pi: residuals `[65, 1.0, 0.015]` vs `rtol=0.197`), so the dumped
    `d_eta` is the **early-stopped iterate**, which matches the 3-iter PCG to ~1e-18
    but the *exact* solve only to ~2e-10. ⇒ we **replicate the C PCG exactly**
    (static `S` + MITgcm preconditioner + same stop) for the forward value, and use
    `custom_linear_solve`’s **tight `transpose_solve`** for the gradient ⇒ forward =
    dump-matching early-stop, backward = clean implicit-diff `S⁻¹`. The huge residual
    margin (5× above / 13× below the threshold between iters 2–3) makes the 3-iter
    stop robust to `segment_sum` reassociation. **d_eta matches the dump ~1e-18 at
    all 5 probes.**
  - **MITgcm symmetric preconditioner** (19336 off-diag entries) verified
    **load-bearing**: a Jacobi/diagonal variant gives a different early-stopped
    `d_eta` (off 2.9e-10 @ probe 1001 → fails the dump).
  - **AD:** `d(d_eta)/d(ssh_rhs)` from `custom_linear_solve` == tight `S⁻¹·w`
    (rel 2e-14) == central-FD; finite; flows through `compute_ssh_rhs` to `du`. The
    static linfs operator makes the AD clean (operator independent of evolving state).
  - **Warm start:** the C does NOT zero `d_eta` between steps (`fesom_main.c` only
    inits it) → step ≥2 warm-starts from the previous `d_eta`. `solve_ssh` takes an
    `x0` (stop_gradient’d, folded into the rhs so the inner solve stays *linear* for
    `custom_linear_solve`); step-1 `x0=0`. **Exact warm-start dump-matching at step
    ≥2 (the stop threshold uses the original ‖b‖) is finalized with the full
    `step()` in Task 2.11.** Next: Task 2.8 (update_vel / compute_hbar / eta_n).
- **2026-06-05 — execution session 5** (Phase 2, Task 2.8: velocity update + hbar +
  eta_n, substeps 10–12). `momentum.update_vel` + `ssh.compute_hbar`/`eta_n_update`
  (+20 tests; **full suite 175 passed**). Phase-2 config unchanged. **Substeps 1–12
  (the full momentum + SSH + free-surface chain) now ported.**
  - **update_vel (10):** `uv += du + (Fx,Fy)`, `(Fx,Fy)=∇N·(−g·θ·dt·d_eta)` at the
    element (gather d_eta to 3 vertices, contract `gradient_sca`). The correction is
    **barotropic** (one per-element scalar broadcast over all layers), unlike the
    per-level `du`; `uv` **accumulates**. At step-1 `uv=0` ⇒ first wind-driven uv
    (~1e-3 surface). ELEM dump at all 5 probes, **max|Δ| ~2e-17** (gather class — both
    `du` ~1e-17 and the replicated early-stop `d_eta` ~1e-18 are near-exact). `d_eta`
    is read, not consumed (→ next step's CG `x0`).
  - **compute_hbar (11):** `ssh_rhs_old` = the substep-8 antisymmetric edge→node
    transport scatter **reused verbatim** (`compute_ssh_rhs` with `uv_rhs=0`, `α=1`,
    bare new `uv`), then `hbar = hbar_old + ssh_rhs_old·dt/areasvol[n,0]`. ⚠️ Although
    `ssh_rhs_old` carries the *same* near-cancelling ~1e-7 floor as `ssh_rhs`, the
    `÷areasvol` (1e9–1e12 m²) divides it back down ⇒ **hbar matches the dump ~1e-17
    absolute**. Edge range `myDim_edge2D == edge2D` single-rank → all-edges scatter
    exact.
  - **eta_n (12):** `eta_n = α·hbar + (1−α)·hbar_old`; **`α=1` ⇒ `eta_n = hbar`
    exactly** (dump confirms `eta_n == hbar` at every probe). Non-cavity nodes only
    (all of pi).
  - **AD:** `update_vel`/`compute_hbar` are linear ⇒ AD == central FD (exact); the
    **end-to-end `d(Σeta_n)/d(du)`** flows `compute_ssh_rhs → custom_linear_solve →
    update_vel (du term + d_eta gather) → compute_hbar → eta_n`, finite & nonzero —
    the implicit-diff chain now spans substeps 8–12. Next: Task 2.9 (ALE step:
    `w`/`hnode_new`, substep 13).
- **2026-06-05 — execution session 6** (Phase 2, Task 2.9: ALE step linfs, substep
  13). `fesom_jax/ale.py` + `tests/test_ale.py` (+17 tests; **full suite 192
  passed**). Phase-2 config unchanged. **Substep 13 (`w` + `hnode_new`) ported —
  the full momentum + SSH + free-surface + ALE-vertical-velocity chain (substeps
  1–13) now runs.**
  - **`compute_w` (13):** the SAME antisymmetric edge→node `(v·dx−u·dy)·helem`
    transport-divergence scatter as `compute_ssh_rhs`/`compute_hbar`, but kept
    **per-level** (not column-summed), driven by the **new** post-`update_vel`
    `uv` (α=1, no AB-velocity). Then (3) a **reverse bottom→top cumsum**
    (`lax.cumsum(div, axis=1, reverse=True)`) — the masked scatter is already 0 at
    and below each node's bottom interface (element layer range ⊆ node range), so
    the suffix-sum == the C's bounded `for nz=nzmax-1..nzmin` loop and the no-flux
    `w[nzmax]=0` BC falls out for free (verified `w[nzmax]==0` exactly). Then (4)
    **÷ `mesh.area`** — ⚠️ the *upper-edge scalar CV area*, **NOT `areasvol`**
    (which `compute_hbar` used) — safe-divide guarded (`where(area>0,area,1)`)
    mirroring the C's `if(a>0)`. Final `node_iface_mask`.
  - **Fidelity:** like `hbar`, the ÷area (1e9–1e12 m²) crushes the near-cancelling
    divergence's amplified floor ⇒ **`w` matches the dump ~4e-20 on CPU** (TIGHT,
    hbar-class — not the loose ssh_rhs ~1e-7). **Step-1 `w` is a REAL gate**
    (post-`update_vel` `uv` ~1e-3 wind-driven ⇒ `w` ~1e-6). Gated at `W_ATOL=1e-12`
    (hbar precedent, GPU-safe). Synthetic O(0.1)-uv vs an independent numpy loop ref
    agrees ~1e-18 (rel 3e-16).
  - **`hnode_new` (13):** `= hnode` **bit-for-bit** (linfs memcpy, `fesom_ale.c:10`)
    — confirms `State.rest().hnode` (the `zbar_3d_n` differences) equals the C's
    static `hnode` exactly (max|Δ|=0 at all 5 probes). The `helem` recompute +
    `hnode=hnode_new` commit is `commit_thickness` = substep **16** (Task 2.10).
  - **Config note:** `use_wsplit=0` in Phase 2 ⇒ `w_e=w`, `w_i=0`; the substep-13
    `w` IS `w_e` for tracer advection and `w_i=0` confirms the Task-2.6
    `impl_vert_visc` simplification. `cflz`/`wvel_split` (no substep-13 dump) →
    ported when consumed (Task 2.10/2.11).
  - **AD:** `w` is **linear** in `uv` ⇒ AD == central FD exactly (~6e-15), finite at
    `uv=0`; end-to-end `d(Σw)/d(du)` flows `compute_ssh_rhs → custom_linear_solve →
    update_vel → compute_w`, finite & nonzero. Next: Task 2.10 (upwind tracers +
    diffusion + thickness commit, substeps 15–16).
- **2026-06-05 — execution session 7** (Phase 2, Task 2.10: upwind tracers +
  diffusion + thickness commit, substeps 15–16). `fesom_jax/tracer_adv.py` +
  `tracer_diff.py` + `ale.commit_thickness` + `tests/test_tracers.py` (+20 tests;
  **full suite 212 passed**). **The full single-step ocean chain (substeps 1–16) is
  now ported** — only `step()` assembly (2.11) remains in Phase 2.
  - **Upwind advection (15):** `tracer_adv.advect_one` — AB2 `ttfAB` drives the
    horizontal upwind edge flux (the C's 5 level-zones collapse to a masked
    per-element `vflux` sum; the per-cell term is the **negation** of `compute_w`'s)
    + the vertical `w_e`-flux (edge-replicated `T_above` ⇒ surface `-w·T·area` for
    free), `÷areasvol` assembly, then ALE reconstruction (`hnode_new=hnode` ⇒
    `T += del_ttf/hnode`). **Matches a numpy upwind loop ref bit-for-bit.**
  - **⚠️ Dump gate (FCT vs upwind):** the dump runs **FCT**, this runs **upwind**.
    **`S=35` (constant) matches the dump bit-for-bit** (the transport divergence
    cancels by discrete continuity — a constant tracer is preserved *exactly*); **`T`
    (the blob) differs ~3e-7** = the limited antidiffusive flux. So `S` is the clean
    step-1 gate, `T` is bounded (`<1e-5`) + verified vs the numpy ref; the tight `T`
    match is a Phase-4 (FCT) gate. **Corrected** the REFERENCE_RUNS "step-1
    horizontally constant" claim (only `S` is; the T-blob isn't).
  - **Vertical diffusion (15):** `tracer_diff.impl_vert_diff` — `impl_vert_visc`'s
    per-NODE 1-unknown sibling (+`area/areasvol` ratio, `hnode_new` mass diagonal).
    `gm=NULL`/`do_wimpl=0`/`bc_surface=0` (no heat/water flux)/`sw_3d=0` (`USE_SW_PENE`
    gated on `use_jra`, off for analytical). ⚠️ **`where(dZ==0,1,dZ)`** to stop the
    Z-padding's exact 0 from poisoning `d/d(Kv)` (0·inf NaN, the eos-class trap).
    Conserves `Σ areasvol·hnode·T` ~1e-16; smooths a vertical gradient.
  - **Commit thickness (16):** `ale.commit_thickness` — `hnode:=hnode_new` +
    `helem=⅓Σ_vertices hnode`. `hnode` dump **bit-for-bit**; `helem`==`State.rest`.
  - **AD:** advection linear in T (AD==FD), kink-safe in `uv` (the `|vflux|` upwind
    kink); diffusion `d/d(Kv)` matches FD; end-to-end `d(ΣT)/d(du)` through the whole
    chain finite & nonzero. Next: Task 2.11 (assemble `step()`, rest-state +
    100-step stability + snapshot → GATE 2).
- **2026-06-05 — execution session 8** (Phase 2, Task 2.11: assemble `step()` →
  **GATE 2, Phase 2 COMPLETE**). `fesom_jax/step.py` + `tests/test_step_pi.py` (+62
  tests; **full suite 274 passed**). The full single-step ocean model now runs forward.
  - **`step.py`:** `step` (eager) + `step_jit` (`jax.jit`, `static_argnames=(dt,
    is_first_step)`) + `run` (loop). Wires substeps 1–16 in `fesom_step.c` order,
    threading the warm-start `d_eta`, AB2 slots (`uv_rhsAB`, `T_old`, `S_old`),
    `hbar_old` (saved before `compute_hbar`), and the lagged `eta_n`/`w_e`.
  - **⚠️ `ssh.solve_ssh` warm-start fix (the deferred step-≥2 fidelity):** the C's CG
    early-stop threshold uses the ORIGINAL `‖ssh_rhs‖`, not the deflated `‖b_eff‖`
    (the inner residual equals the full residual). Added `rtol_abs`. **Load-bearing:**
    step-2 `d_eta` matches the dump 3–3000× better warm-started than from zero.
  - **Step-1 integration gate (tight):** one `step()` reproduces EVERY per-kernel
    substep dump gate at the probes (density/bvfreq/ssh_rhs/d_eta/uv/hbar/eta_n/w/hnode
    /S; `T` is the upwind−FCT gap). Confirms order + step-1 threading.
  - **⚠️ Multi-step:** a tight dump match is impossible (upwind `T` diverges ~3e-7 →
    cascades via density at step ≥2). Gated by invariants: **`S` exactly 35** through N
    steps (constant-tracer preservation, a sensitive threading check); **rest state**
    (constant T/S + zero wind) to machine precision (`max|uv|`~2e-16); **100 steps
    stable** (`max|uv|`=0.075, `|eta|`=0.35 m, no NaN); step-2 SSH/velocity climate-close.
  - **jit:** XLA FMA-contracts the EOS polynomial ⇒ jitted `density` shifts ~1e-13 (past
    the bit-exact `map` gate) ⇒ tight gates use **eager** `step()`; `step_jit` matches
    eager ~1e-12 (the loose multi-step/stability gates + Phase-3 scan use it).
  - **GATE 2 MET; Phase 2 complete.** Next: **Phase 3** (Task 3.1 `lax.scan`+checkpoint
    time loop; Task 3.2 end-to-end gradient smoke test — the project's biggest-risk gate).
- **2026-06-06 — execution session 9** (Phase 3, Tasks 3.1 + 3.2: the AD de-risking gate →
  **GATE 3, Phase 3 COMPLETE**). `fesom_jax/params.py` + `integrate.py` +
  `tests/test_integrate.py` + `tests/test_gradient.py` + the eos `zdiff` fix (+12 tests;
  **full suite 286 passing**). **The model is now proven differentiable end-to-end — the
  project's biggest risk is retired.**
  - **ML-hook seam (`params.py`):** a registered `Params` pytree (`k_ver`, `a_ver`) threaded
    `step(...,params) → pp.mixing_pp(...,k_ver,a_ver)`; `params=None ⇒ Params.defaults()` (the
    config constants). **Numerically transparent** — the pre-existing 274 tests stay
    bit-identical. This is the first concrete swap-point (Phase 7 puts an NN here).
  - **`integrate.py` (Task 3.1):** `step` wrapped in `lax.scan` + per-step `jax.checkpoint`.
    **Step 1 runs eagerly (`is_first_step=True`) OUTSIDE the scan**, steps 2..N scan with
    `is_first_step=False` baked in (uniform body, no traced bool); `mesh`/`op`/`stress_surf`/
    `params` closed over (carry = just `State`). Forward == the Phase-2 `run` loop
    **BIT-IDENTICAL** (uv ~4e-19, else 0.0); `checkpoint` on==off forward exactly 0.0.
  - **⚠️ eos `bvfreq` backward-NaN trap (found + fixed):** `zdiff = Zd − Zp` is exactly 0 at
    the **bottom-padding** lane (`Zp` tail duplicates `Z[-1]`), not just the surface (`k=0`).
    `1/zdiff=inf` ⇒ `bv[:,bottom]=inf` forward, **clipped out of the output** (so all
    Phase-0..2 forward gates passed) but `0·inf=NaN` in the *backward* pass → `d(loss)/d(T)`
    NaN at 6280 masked lanes. Fix: `where(zdiff==0,1,zdiff)`. **`d/d(scalar k_ver)` was finite
    while `d/d(T₀ field)` was NaN** — the IC-field gradient is the strictly stronger masked-NaN
    probe (k_ver enters additively downstream of the trap). Lesson logged.
  - **Gradient gate (Task 3.2, `test_gradient.py`):** loss = mean SST after N=20 (dt=100).
    `d/d(k_ver)` AD↔FD step-size **sweep** → plateau **5.9e-7** at `k_ver=1e-4` (lifted off
    the ~eps·10 round-off floor; the plateau is at LARGE `h` since the loss is near-linear in
    k_ver); physical `k_ver=1e-5` → 4.5e-5 + correct sign. Gradient **flows through the CG**
    (`d/d(a_ver)` within-step, `d/d(k_ver)` across-step, both FD-confirmed). `d(loss)/d(T₀)`
    finite everywhere + 0 on masked lanes. Smooth regime certified (probe column stratified,
    `S`=35, no genuine convection).
  - **Memory sanity (GPU, job 25378918, A100-40):** checkpointed N=200 backward = **4.23 GB
    (13%)**, finite grad; un-checkpointed **OOMs at 48.7 GB** ⇒ checkpointing load-bearing.
  - **GATE 3 MET; Phase 3 complete.** Next: **Phase 4** (Task 4.1 FCT/Zalesak +
    limiter-gradient decision; Task 4.2 opt_visc7 flow-aware + wsplit; Task 4.3 pi 1000-step +
    AD re-check) — at which the tight multi-step `T/S` dump match becomes available.
- **2026-06-06 — execution session 10** (Phase 4, **Task 4.1: FCT (Zalesak) advection +
  the limiter-gradient decision**). `tracer_adv.advect_one_fct` + `zalesak_limit` (+ MFCT/QR4C
  HO fluxes, `compute_fct_lo`, `fill_up_dn_grad`) wired into `step.py`; `ic.py` `T_old` fix;
  `docs/LIMITER_GRADIENTS.md`; +15 FCT tests. **Full suite green** (the only Phase-4-changed
  gate, `test_step1_T_is_upwind_fct_gap`, was rewritten to the now-tight `T` gate).
  - **FCT port:** the dump's live scheme — `T_new = LO + limited(HO − LO)`. LO = upwind ALE
    (from `values`); HO = MFCT 3rd-order horizontal (`num_ord=0`, element + up/dn-triangle
    gradient from `values`) + QR4C 4th-order vertical (`num_ord=1`, `ttfAB`); the **Zalesak
    limiter** clips the antidiffusive `HO − LO` to introduce no new extrema. Verified
    stage-by-stage against a literal numpy-C reference (`fct_LO`/`tr_xy`/`eud`/`adf_h`/`adf_v`
    all ≤1e-10), and the qr4c against BOTH the C and the Fortran `oce_adv_tra_ver.F90`.
  - **⚠️⚠️ THE bug (cost the session): the C's step-1 `T_old` (AB2 `valuesold`) is the pre-blob
    base T=10, NOT the blob field.** `valuesold` is `calloc`'d 0; a rest-state `advect_one(T)`
    *sanity check* (`fesom_main.c:721`) saves `valuesold = values = 10` BEFORE the blob is added
    to `values` (`:748`). So `ttfAB = -(0.5+ε)·10 + (1.5+ε)·(10+blob)`, not `ttfAB = values`.
    Our `ic.py` set `T_old=T` (blob) — wrong, and **mis-attributed for two phases as the
    "upwind−FCT gap" (~3e-7)**. Fixing it: FCT `T` vs the dump jumps **3.4e-7 → 1.8e-15**. `S`
    (constant) is insensitive to `S_old`, which hid it. The decisive diagnostic was dumping the
    C's `adf_v` intermediate (`-5.6e4` at the surface where ours was 0 ⇒ back-solve `T_old=10`).
  - **Limiter-gradient decision — (a) subgradient as-is** (`docs/LIMITER_GRADIENTS.md`): the C
    `flux_eps=1e-16` floors every limiter ratio finite ⇒ the plain `jnp` min/max/where/`segment_*`
    VJP is NaN-safe (no `0·inf`), forward-faithful, and FD-consistent where smooth — unlike (b)
    smooth-relax (changes the forward) or (c) stop_gradient (biases the gradient). The qr4c
    `Z`-stencil bottom-pad divide needs the `where(d==0,1,d)` guard (the recurring masked-divide
    rule, 4th time).
  - **Gates:** FCT `T/S` vs dump **TIGHT** (`T` 1.8e-15, `S` bit-exact) at all 5 probes; an
    independent numpy FCT ref (smooth **and** a synthetic limiter-ACTIVE case — the dump's smooth
    blob leaves the limiter inactive); `test_gradient.py` green with FCT live (`d(SST)/d(k_ver)`
    plateau **5.7e-7**, flows through CG + limiter, `d/d(T₀)` finite). **Bonus:** the `T_old` fix
    closed the deferred Phase-2 tight multi-step `T` gate (`step()` substep-15 `T` 1.8e-15 vs the
    committed dump) AND dump-verified the blob vertical diffusion + the step-2 SSH cascade
    (now <1e-11, was gated 1e-7).
  - **GATE-4 forward (tracers) MET.** Next: **Task 4.2** (complete opt_visc=7 flow-aware terms +
    `use_wsplit`), then **Task 4.3** (pi 1000-step stability + AD re-check) → GATE 4.
- **2026-06-06 — execution session 11** (Phase 4, **Task 4.2: complete opt_visc=7 flow-aware
  biharmonic + `use_wsplit`**). `ale.compute_cfl_z`/`compute_wvel_split` + `step.py` wiring +
  6 new tests (test_momentum +1, test_step_pi +1, test_ale +4). **Full suite green; gradient
  plateau 5.70e-7 unchanged.** No new pi physics was actually needed — the work was *closing
  verification gaps* and *adding CORE2-ready machinery*.
  - **opt_visc=7 flow-aware viscosity was ALREADY fully ported in Task 2.5** (γ0/γ1/γ2 =
    0.003/0.1/0.285, the `max(γ0, max(γ1·|du|, γ2·|du|²))` flow-aware `max` present + matching
    the C line-for-line). The review-finding's "if Task 2.5 did a basic version" hedge resolves
    to "it did the full version." The real gap was **coverage**: a diagnostic showed
    **flow-aware-active = 0 edges** at every one of the 10 dump steps (max edge-velocity-diff
    grows 8e-5→8e-4 ≪ the |du|>0.03 γ1-onset), and the existing `_synthetic` test (amp 0.1)
    only reaches the γ1 branch (51% of edges, **never** the quadratic γ2, which needs |du|>0.351).
    Added `test_visc_filter_flow_aware_branches_vs_reference` (strong synthetic flow ~2 m/s,
    BOTH branches bind, vs the numpy ref + a branch-selection load-bearing assert) and a step-2
    substep-6 `uv_rhs` dump gate (`test_step2_uv_rhs_visc_matches_dump`, ~1e-17 — the step-1 gate
    was rest-trivial; FCT made step-2 tight, unlocking it).
  - **`use_wsplit` ported but OFF (the reference config).** `fesom_constants.h:56` sets
    `use_wsplit=0` (the split seeded a Fortran day-92 blow-up), so `w_e=w, w_i=0` IS the
    dump-matching path — and the step-1 substep-15 `T` 1.8e-15 match already confirmed it. Ported
    `compute_cfl_z` (`fesom_ale.c:204`) + `compute_wvel_split` (`:241`) faithfully (CORE2-ready),
    wired into `step.py` (populates `State.cfl_z`); the split is the **identity** at use_wsplit=0
    (numerically transparent — every gate + the gradient unchanged), and pi's max `cfl_z`~1e-4 ≪
    maxcfl=1.0 means it'd be inactive even if on. Active branch verified vs the numpy ref with a
    synthetic super-critical CFL + an AD-finite gate. ⚠️ `impl_vert_visc`'s `w_i` advective
    tridiagonal terms remain dropped (correct at `w_i=0`) — a Phase-5/CORE2 item.
  - **Task 4.2 DONE.**
  - **Task 4.3 (pi 1000-step stability + AD re-check) → GATE 4 MET.** `test_1000_step_stability`:
    pi 1000 steps at dt=100 (full physics — FCT + opt_visc7 + wsplit machinery) is stable in
    ~48 s — no NaN, max|uv|=0.17, max|eta|=0.63 m, **S exactly 35** over the window, T∈[10.0,14.98],
    and max `cfl_z`=2.8e-3 ≪ maxcfl=1.0 (wsplit inactive even long-window). AD re-check: the
    permanent `test_gradient.py` runs with the full pi physics live and stays green (plateau
    5.70e-7, flows through the CG). Climate-close-to-C stays indirect (no C 1000-step pi snapshot;
    the tight step-1..10 dump match + S-exact + boundedness stand in). **Phase 4 complete; full
    suite 313 passed.** Next: **Phase 5 (CORE2 single-device)** — expand into a
    `docs/plans/<date>-fesom-jax-core2.md` sub-plan (zlevel ALE, PHC IC, JRA55 forcing, partial
    cells, the `w_i` advective terms in `impl_vert_visc` when use_wsplit turns on).
- **2026-06-06 — Phase 5 (CORE2 single-device) COMPLETE, GATE 5 met.** Sub-plan
  `docs/plans/20260606-fesom-jax-core2.md` (Tasks 5.1-5.8). The pi physics on the CORE2 mesh +
  PHC IC + JRA55/SSS/runoff, per-substep dump-gated, stable days 1-7, gradient-gated. Two
  standing findings: no-ice supercooling (physical) + inert runoff — both deferred to Phase 6.
- **2026-06-06/07 — Phase 6 SEA ICE COMPLETE, GATE 6 met.** Sub-plan
  `docs/plans/20260606-fesom-jax-phase6-seaice.md` (Tasks 6.1-6.7). Read the C `fesom_ice*.c`
  (the spec); ported thermo + EVP (120-subcycle `lax.scan`) + FCT + coupling, each bit-exact vs
  ice-ON C dumps; assembled prognostic-ice CORE2 step. **The two Phase-5 findings RESOLVED:**
  10 days stable with the supercooling **capped at −1.91 °C** (sea ice) + runoff active (the
  ice freshwater handoff). Gradient-gated (FD↔AD 4.5e-10 + masked-NaN clean at scale; the EVP
  IC-gradient is stiff-but-finite — trainable gradients flow through the mixing seam). Suite
  ocean 376 + ice 47. **Next big phases: GM/Redi (6B, the 2nd ML-hook) + KPP (6C) — own
  sub-plans, scoped by reading `fesom_gm.c`/`fesom_kpp.c` first.**
- **2026-06-07 — Phase 6B (GM/Redi) sub-plan DRAFTED.**
  `docs/plans/20260607-fesom-jax-gmredi.md` (Tasks G.1-G.7). Scoped from a first-hand read of
  `fesom_gm.c` (+ `.h`) and the `fesom_step.c`/`fesom_ale.c`/`fesom_tracer_diff.c`/`fesom_eos.c`
  integration seams. **User-confirmed decisions:** (1) thread the GM eddy diffusivities
  (`k_gm`/`redi_kmax`) through `params.py` **now** (the 2nd ML-hook seam, default=config ⇒
  bit-identical, like `k_ver`/`a_ver`); (2) 7-task data-flow ladder, each kernel dump-gated.
  **Key findings:** GM/Redi is **stateless** (no new State fields); threads the step at **6
  points** (not just substep 14); **mixing-scheme-independent** (dump GM-ON, ice-OFF, PP); a
  NEW all-node `fesom_gm_dump` C hook gates the intermediates; the active namelist is a small
  subset (ODM95 taper + GMzexp depth scaling + resolution scaling + Redi=GM sync + Redi_Ktaper);
  AD hazards are all established patterns. Wired behind a static `gm_cfg=None` (⇒ pi/Phase-5/ice
  bit-identical, the `ice_cfg` precedent). **Execution begins at Task G.1** (`sw_alpha_beta` +
  the `params.py` seam + `GMConfig` + the dump hook).
- **2026-06-07 — Phase 6B GM/Redi Tasks G.1–G.6 COMPLETE (the whole physics), committed on `main`**
  (`c587b83` G.1-3, `2e43ed9` G.4, `c3886e9` G.5, `dbf40ca` G.6). Every GM kernel dump-verified vs
  a new GM-ON all-node CORE2 dump (`fesom_gm_dump`+`fesom_redi_blob` on port2 `jax-mesh-export`):
  `sw_alpha_beta` bit-exact; neutral slopes map-class; `init_redi_gm` (`d/d(k_gm)=2.03e6` — the 2nd
  ML-hook gradient LIVE); the streamfunction TDMA 8.9e-15 + bolus velocity 1.1e-16; the
  `gm_diagnostics` driver `fer_uv` end-to-end 2.2e-16; the Redi terms — G7a 1.78e-15, **G7b's
  5-branch edge loop 1.07e-14** (the 5→3-case collapse), K33 ("augment Kv"). New modules
  `gm.py`/`gm_redi.py` + `eos.compute_sw_alpha_beta` + the `params.py` 2nd-ML-hook seam
  (`k_gm`/`redi_kmax`). Compute-node test runner `scripts/run_suite.sbatch` adopted (the CORE2
  backprop tests hang on the login node). **Only Task G.7 remains** (assemble into `step.py` behind
  `gm_cfg=None` + multi-day GPU stability + gradient re-check with `d/d(k_gm)` = GATE 6B). Handoff
  in `docs/NEXT_SESSION_PROMPT.md`.
- **2026-06-07 — Phase 6B GM/Redi COMPLETE (Task G.7, GATE 6B met).** GM/Redi wired into `step.py`/
  `integrate.py` behind a static `gm_cfg=None` arg (the `ice_cfg` precedent — bit-identical when
  None; the **453-test suite is green**: 406 ocean + 47 ice). **Assembled GM step bit-exact (T
  7.1e-15 / S 2.1e-14** vs the GM-ON substep dump — GM is deterministic, no ice-EVP reassociation
  floor, so the assembly is bit-exact-class not climate-close, and this CLOSES K33's tight gate).
  **10-day GM+ice stability** (max|vel| 2.84, SST capped −1.91, no NaN) + **GM smooths fronts**
  (front|∇T| 7.42e-6 ON vs 7.89e-6 OFF, growing). **Gradient `GM_GRAD_GATE_OK`:** the 2nd ML-hook
  `d/d(k_gm)` plateau **3.5e-6 (well-conditioned — not stiff)**, `d/d(k_ver)` 5.8e-10, masked-NaN
  `d/d(T0)` clean; backward 37 GB/64 GB @ N=4 CORE2. ⚠️ Redi reads the **pre-step `st.T`/`st.S`**
  (the returned `T_old`), not `st.T_old`. New: `test_gm_step.py`, `scripts/core2_gm_stability_run.py`
  +`_gpu.sh`, `scripts/core2_gm_grad_gate.py`+`.sbatch`; 4 lessons. **Both ML hooks now training-ready.**
- **2026-06-07 — Phase 7a (Differentiable Parameter Tuning) added to the plan (user-requested).**
  Use the diff. port to calibrate physics params (GM/ice/mixing) against a target and push the
  optimum back to the Fortran `namelist.oce` (scalar = zero Fortran code) — the SAME `params.py`
  seam, a small extension. ⚠️ Do NOT backprop through the multi-decade spin-up (memory + chaotic
  gradient blow-up): spin up forward (no AD) → short-window adjoint at equilibrium + gradient-free
  EKI for the slow mean. First experiment = the perfect-model `k_gm` twin. **Next phases: 7a
  (tuning) and/or 6C (KPP).**
