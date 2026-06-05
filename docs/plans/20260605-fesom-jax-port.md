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

- [ ] port `update_vel` (gather SSH correction to elements), `compute_hbar` (transport-divergence edge→node scatter, save `ssh_rhs_old`), `eta_n` blend — ref `fesom_momentum.c:474/779`, `fesom_ale.c`
- [ ] write tests: `hbar` (11) and `eta_n` (12) are **node** fields → compare vs Fortran dumps directly (≤1e-12). `uv` (10) is **element** → element dump or indirect via `hbar` @11
- [ ] run — must pass before next task

#### Task 2.9: ALE step (linfs) (substep 13)

**Files:**
- Create: `fesom_jax/ale.py`
- Create: `fesom_jax/tests/test_ale.py`

- [ ] port linfs ALE: `hnode_new = hnode` everywhere; compute `w` (vertical velocity, edge→node scatter divergence) and `helem` — ref `fesom_ale.c`, FRESH_START §6
- [ ] write `tests/test_ale.py`: compare `w`, `hnode_new` vs C dump substep 13
- [ ] run — must pass before next task

#### Task 2.10: Tracer advection (upwind) + diffusion + commit (substeps 15–16)

**Files:**
- Create: `fesom_jax/tracer_adv.py`
- Create: `fesom_jax/tracer_diff.py`
- Create: `fesom_jax/tests/test_tracers.py`

- [ ] port **upwind** horizontal+vertical advection first (no FCT yet) — ref `fesom_tracer_adv.c`; watch the `edge_vflux` sign (FRESH_START §12 / §14.5) via a constant-advection test
- [ ] port `diff_tracers_ale`: accumulate `del_ttf`, ALE reconstruction `T_new=(T·hnode+del_ttf)/hnode_new`, implicit vertical TDMA (per-node, 1 unknown) — ref `fesom_tracer_diff.c`, FRESH_START §5/§14.1
- [ ] port thickness commit `hnode = hnode_new` (substep 16) — ref `fesom_ale.c:18`
- [ ] write `tests/test_tracers.py`: compare `T`, `S` vs C dump substep 15/16; plus a **constant-tracer-stays-constant** advection test and a **pure-diffusion** smoothing test
- [ ] run — must pass before next task

#### Task 2.11: Assemble `step()` and run pi forward

**Files:**
- Create: `fesom_jax/step.py`
- Create: `fesom_jax/tests/test_step_pi.py`
- Create: `fesom_jax/ic.py` (constant T=10,S=35 init for now)

- [ ] wire the 16 substeps into a single jitted `step(state, mesh, params) -> state`
- [ ] rest-state test: constant T/S, eta=0, uv=0, zero forcing → stays at rest to machine precision (FRESH_START §20)
- [ ] run 100 steps on pi at dt=100; assert stable (max|uv|~0.3, |eta|<5m, no NaN)
- [ ] full-field snapshot diff vs a C 100-step pi run (climate-close via `eps_climate_compare`)
- [ ] write `tests/test_step_pi.py` capturing the rest-state + 100-step stability gates
- [ ] run — must pass before Phase 3

**GATE 2:** pi 100 steps stable; each substep matches C within tolerance; full-field
snapshot climate-close to C.

---

### Phase 3 — AD Smoke Test (de-risking gate)

#### Task 3.1: Scan + checkpoint time loop

**Files:**
- Create: `fesom_jax/integrate.py`
- Create: `fesom_jax/tests/test_integrate.py`

- [ ] wrap `step` in `jax.lax.scan` over N steps; apply `jax.checkpoint` (rematerialization) to the step fn
- [ ] confirm forward result of the scan == the Phase-2 manual loop (climate-close)
- [ ] memory sanity: N=200 pi steps backward pass fits in device memory with checkpointing
- [ ] write `tests/test_integrate.py` (forward-equivalence)
- [ ] run — must pass before next task

#### Task 3.2: End-to-end gradient check

**Files:**
- Create: `fesom_jax/tests/test_gradient.py`

- [ ] define a scalar loss (e.g. mean SST after N steps); choose the param/loss to stay in a **smooth regime** — verify the probe column never goes convective so the PP `max(Kv,0.1)` / `max(N²,0)` kinks don't bite (review Important #8)
- [ ] reverse-mode `jax.grad` of loss w.r.t. a scalar parameter (PP `K_ver` background diffusivity)
- [ ] **finite-difference check with a step-size SWEEP:** compute FD at `h ∈ {1e-4…1e-7}` (relative, central, float64), report the FD-convergence plateau, and assert `|grad_AD − grad_FD|/|grad_FD| < 1e-4` at the plateau — not at a single `h` (chaos floor below, truncation error above)
- [ ] keep N modest for the smoke test (the forward model is mildly chaotic via scatter reassociation — see `GPU_FIDELITY.md` M5.8/M5.9; long windows amplify it). Note this as a known long-window gradient-stability risk
- [ ] confirm gradient flows through the CG `custom_linear_solve` (perturb a param affecting the stiffness/RHS)
- [ ] grad w.r.t. an initial-condition field (vector-valued) sanity check
- [ ] write `tests/test_gradient.py` as the permanent AD gate
- [ ] run — must pass before Phase 4

**GATE 3 (DE-RISKING):** end-to-end gradient check passes; the hard AD patterns
(scan+checkpoint, `custom_linear_solve`) are proven on the real model. *This is the gate
that retires the project's biggest risk.*

---

### Phase 4 — pi Fully Stable

#### Task 4.1: FCT (Zalesak) advection + the limiter-gradient decision

**Files:**
- Modify: `fesom_jax/tracer_adv.py`
- Create: `docs/LIMITER_GRADIENTS.md`
- Modify: `fesom_jax/tests/test_tracers.py`

- [ ] port high-order + Zalesak limiter (`fct_plus/minus`, local min/max bounds, sign-dependent flux selection) — ref `fesom_tracer_adv.c:814+`, FRESH_START §12
- [ ] **RESEARCH ITEM:** decide & document the limiter-gradient strategy in `docs/LIMITER_GRADIENTS.md` — options: (a) subgradient as-is, (b) smooth min/max relaxation, (c) `stop_gradient` on the limiter coefficients (treat the limited flux mask as fixed in the backward pass). Implement the chosen one.
- [ ] forward verification: FCT `T/S` vs C dump substep 15 (≤1e-12)
- [ ] gradient check with the chosen limiter strategy (must be finite + finite-diff-consistent where smooth)
- [ ] run — must pass before next task

#### Task 4.2: Complete opt_visc=7 (flow-aware biharmonic) + wsplit

> Review finding (Important #6): the C port runs **opt_visc=7** (biharmonic, flow-aware;
> `fesom_step.c:134-137`, required for dt=1800 stability), NOT opt_visc=5 harmonic+backscatter.
> Task 2.5 already ports the biharmonic `visc_filt_bidiff` scatter — so this task COMPLETES
> the flow-aware terms of opt_visc=7 (if Task 2.5 did a basic version) and adds wsplit; it
> does not re-port a different scheme.

**Files:**
- Modify: `fesom_jax/momentum.py`, `fesom_jax/ale.py`
- Modify: tests

- [ ] complete the flow-aware terms of `visc_filt_bidiff` (opt_visc=7, visc_gamma0=0.003 — FRESH_START §14.7) to match the live C kernel
- [ ] port `use_wsplit` vertical-velocity splitting (wsplit_maxcfl=1.0)
- [ ] forward verification vs Fortran dumps (substep 13 `w` is a node field → direct; substep 6 `uv_rhs` is element → element dump / indirect)
- [ ] run — must pass before next task

#### Task 4.3: pi 1000-step stability + AD re-check

**Files:**
- Modify: `fesom_jax/tests/test_step_pi.py`, `fesom_jax/tests/test_gradient.py`

- [ ] run pi 1000 steps at dt=100; assert stable; snapshot climate-close to C
- [ ] re-run the Phase-3 gradient gate with FCT + backscatter active
- [ ] run — must pass before Phase 5

**GATE 4:** pi 1000 steps stable & climate-close; gradient check still passes with full pi
physics.

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

*(To be expanded into its own sub-plan.)* Outline:

- **GM/Redi:** neutral slopes, tapering, bolus velocity (substep 14) — `fesom_gm.c`.
- **KPP** (FESOM1.4, lookup-table version, mix_scheme_nmb=1) — `fesom_kpp.c`,
  FRESH_START §10. **Forward-only** (no AD requirement; it's the NN-replacement target):
  verify via `kpp_dump_diff.py`-style probe dumps.
- **Sea ice:** EVP dynamics (data-dependent subcycle loop → `lax.fori_loop`/`scan`; the
  coastal-BC + scatters), FCT, thermodynamics — `fesom_ice*.c`.
- **GATE 6:** CORE2 multi-year climate stats vs C/Fortran within the C↔Fortran budget
  (`eps_climate_compare_2yr.py`).

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
