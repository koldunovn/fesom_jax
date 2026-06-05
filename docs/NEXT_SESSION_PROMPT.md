# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. It assumes Phases 0 **and 1**
are complete (committed) and **Phase 2 (minimal forward step on pi, substep by
substep)** is next.

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean
model for hybrid ML (embedding trainable NN parameterizations for vertical mixing
and mesoscale eddy fluxes, trained end-to-end). This continues a multi-session
effort. Work from `/home/a/a270088/port_jax`. Use max effort.

## START HERE, in order
1. **Read the plan (source of truth):**
   `/home/a/a270088/port_jax/docs/plans/20260605-fesom-jax-port.md` — decisions,
   verification ladder, 8 phases, per-task gates, Revision Log. Keep its checkboxes
   in sync as you go. Phase 2 = Tasks 2.1–2.11.
2. **Read the project memory:**
   `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/fesom-jax-port.md`
3. **Read** `/home/a/a270088/port_jax/docs/REFERENCE_RUNS.md` (the oracle: the C
   per-substep dump + the JAX↔C↔Fortran chain) and `docs/MESH_EXPORT_LAYOUT.md`
   (mesh arrays). The Phase-1 modules you build on: `fesom_jax/mesh.py`,
   `fesom_jax/state.py`, `fesom_jax/ops.py`, `fesom_jax/verify.py`, `io_dump.py`.
4. Skim `/home/a/a270088/port2/FRESH_START.md` (physics/algorithm reference; EOS §8,
   timestep §5, mixing §10) and the C module for the substep you start
   (`/home/a/a270088/port2/fesom2_port/src/`).

## STATUS — Phases 0 & 1 COMPLETE (GATE 0 + GATE 1 met), committed
- **Phase 0:** env (`fesom-jax`, jax 0.10.1 x64, A100), verify harness
  (`io_dump.py` + `verify.py`, isclose-form `|Δ|≤atol+rtol·|c|`), pi mesh export
  (`data/mesh_pi/`, 31 arrays), and the **C-port per-substep dump oracle**
  (`fesom_jax/tests/fixtures/pi_cdump.00000`, 10 steps, node + element fields).
- **Phase 1 (pure JAX):**
  - `fesom_jax/mesh.py` — frozen `Mesh` registered as a JAX pytree
    (`register_dataclass`: 31 arrays = leaves, 7 scalar counts = static meta).
    `load_mesh()`. **Export is already 0-based** (no conversion); `−1` = boundary.
    Four ragged-level masks: **layer** valid `[ulevels-1, nlevels-1)` (T,S,ρ,p,u,v)
    vs **interface** valid `[ulevels-1, nlevels-1]` (bvfreq,w,Kv,Av) — per
    `fesom_eos.c:93-208`.
  - `fesom_jax/state.py` — `State` pytree mirroring `fesom_dyn`+`fesom_aux`+thickness
    (every field tagged with C owner + layer/interface). `State.zeros`/`State.rest`.
  - `fesom_jax/ops.py` — `gather*`; masked `scatter_add` (`segment_sum`; `−1`→0
    fwd **and** in grad); `mask_below_bottom`; `tdma` (two `lax.scan` sweeps,
    vectorized over the entity axis).
  - **AD gates pass:** scatter transpose == gather; TDMA grad == central FD (1e-6).
- **Full test suite: 46 passing** (`JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`).

## KEY PATHS
- Working repo (git, branch `main`, local-only — no remote): `/home/a/a270088/port_jax`
- **Env python (use for ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → tests: `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`
  (the `cuInit 303` warning on the login node is the documented benign GPU-absent
  fallback; `JAX_PLATFORMS=cpu` silences it. Phase 2 is pure JAX → CPU is fine.)
- Phase-1 JAX modules: `fesom_jax/{mesh,state,ops,verify,io_dump,config}.py`
- Exported pi mesh (gitignored): `data/mesh_pi/*.npy` + `meta.txt`
- Reference dump fixture: `fesom_jax/tests/fixtures/pi_cdump.00000`
- C port (ALGORITHMIC source of truth, mirror it): `/home/a/a270088/port2/fesom2_port/src/`
  — JAX-port C additions on branch **`jax-mesh-export`** (mesh exporter + dump writer).
- Fortran model (numerical cross-check only): `/home/a/a270088/port2/fesom2/src/`
- Kokkos port (parallelization + fidelity lessons + compare scripts):
  `/home/a/a270088/port_kokkos/`

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi).
2. Full-fidelity, bottom-up (minimal step → pi → CORE2 → ice/KPP/GM), not a toy.
3. AD-safe by construction (idiomatic functional JAX) + an EARLY end-to-end gradient
   smoke test (Phase 3), re-run at every gate. AD is never deferred.
4. Mesh = index gather/scatter mirroring the C loops (`segment_sum`/`.at[].add` scatter,
   dense `[n_entity,nl]` + boolean masks for ragged depth, connectivity as static
   index arrays). **Already built in `ops.py`/`mesh.py` — reuse, don't re-derive.**
5. Single-device + data-parallel-over-batch now; mesh sharding is Phase 8.

## THE VERIFICATION ORACLE — exact (substep → field) map in the fixture
The C dump matches at probes — node gids **1001,1500,2000,2500,3000**; element gids
(first incident cell) **1757,2656,3688,4604,5575**. Element records carry the
*element* gid; node records the node gid. **All element fields ARE dumped**, so
element kernels verify **directly** (ignore the plan's older "indirect verification"
hedges). **Always truncate the JAX column to the record's `nlevels` before diffing**
(node → `nlevels_nod2D`; element → `nlevels`). Use `verify.assert_close(jax_col,
record, kind=...)` with kind ∈ {`map`,`gather`}=1e-15, {`scatter`,`reduction`}=1e-12.

| substep | C fields (entity) | port task |
|---|---|---|
| 1 pressure_bv | `bvfreq`,`density`,`pressure` (NODE) | 2.1 |
| 3 pressure_force | `pgf_x`,`pgf_y` (ELEM) | 2.2 |
| 4 mixing | `Kv` (NODE), `Av` (ELEM) | 2.3 |
| 5 vel_rhs | `uv_rhs_u`,`uv_rhs_v` (ELEM) | 2.4 |
| 6 viscosity_filter | `uv_rhs_u`,`uv_rhs_v` (ELEM) | 2.5 |
| 7 impl_vert_visc | `uv_rhs_u`,`uv_rhs_v` (ELEM) | 2.6 |
| 8 ssh_rhs | `ssh_rhs` (NODE) | 2.7 |
| 9 ssh_solve | `d_eta` (NODE) | 2.7 |
| 10 update_vel | `uv_u`,`uv_v` (ELEM) | 2.8 |
| 11 compute_hbar | `hbar` (NODE) | 2.8 |
| 12 eta_n | `eta_n` (NODE) | 2.8 |
| 13 ale_step | `w`,`hnode_new` (NODE) | 2.9 |
| 15 solve_tracers | `T`,`S` (NODE) | 2.10 |
| 16 update_thickness | `hnode` (NODE) | 2.10 |

(`density` = `density_m_rho0` = in-situ ρ − ρ0; `pressure` = `hpressure`. Substeps
1–13,16 match at **every** step; substep 15 (T,S) matches at **step 1 only** until
Phase 4, because the C port runs FCT while Phase-2 JAX runs upwind — at step 1 the
field is horizontally constant so upwind==FCT==dump.)

## IMMEDIATE WORK — Phase 2, start at Task 2.1
Port the 16 substeps **in order**, each gated against the dump above. Config: pi,
linfs, PP, upwind tracers (Phase 2), CG SSH, no GM/KPP/ice, analytical wind, dt=100,
constant T=10/S=35 IC, opt_visc=7. nl=48.

- **Task 2.1 — EOS / pressure / N² (substep 1).** Create `fesom_jax/eos.py` +
  `tests/test_eos.py`. Port `fesom_pressure_bv` (`fesom_step.c:77`, `fesom_eos.c:78`):
  the Jackett-McDougall in-situ density (`densityJM_components`, literal port of
  `oce_ale_pressure_bv.F90:2605-2669`), hydrostatic `hpressure`, and `bvfreq` (N²,
  interior `nz∈[nzmin+1,nzmax)` then padded at `nzmin`/`nzmax`). Loop bounds:
  `nzmin=ulevels-1`, `nzmax=nlevels-1` (layers run `nz<nzmax`). Then port the
  **`fesom_smooth_nod3D(bvfreq, nl, n_smooth=1)`** pass (`fesom_step.c:92`,
  `fesom_eos.c:226`) — a **SINGLE-sweep** area-weighted node-patch average
  `arr[n,nz] ← (Σ_{el∈patch} area_el·(arr[v0]+arr[v1]+arr[v2])) / (3·Σ area_el)`
  over `nz∈[ulevels-1, nlevels-1]` (use `nod_in_elem2D` CSR or an element→node
  area-weighted scatter via `ops.scatter_add`; ~1e-12). The dump's `bvfreq` is
  **POST-smooth** — without the smoother the gate fails. EOS/pressure are map/gather
  (~1e-15). Verify `density`,`pressure`,`bvfreq` vs substep 1 at all probes; add a
  gradient check `d(mean density)/d(T)` AD vs FD. (Substep 2 `sw_alpha_beta` is
  GM/KPP-only → deferred to Phase 6.)
- **Task 2.2 — PGF (substep 3).** `fesom_pressure_force_linfs_fullcell`
  (`fesom_eos.c:285`): per element/layer `pgf_x[e,nz]=Σ_i gradient_sca[6e+i]·
  hpressure[V_i(e),nz]/ρ0` (i=0..2), `pgf_y` uses cols 3..5. A clean
  gather(node→elem via `elem_nodes`) + `gradient_sca` contraction (map/gather ~1e-15).
  Verify `pgf_x/pgf_y` vs substep 3 (ELEM probes).
- Continue 2.3 … 2.11 per the plan. **GATE 2:** pi 100 steps stable; each substep
  matches C within tolerance; full-field snapshot climate-close.

## GOLDEN RULE (JAX-adapted)
Preserve the EXACT computation — the math and the load-bearing association order — but
express it as vectorized array ops over `ops.py` primitives. Do NOT do a literal
loop-by-loop translation, and do NOT simplify the physics. When in doubt, dump the C
value at a probe and match it.

## FIDELITY TARGET
Bit-identity to C is NOT achievable (scatters + reductions reassociate FP sums).
Target: ~1e-15 map/gather, ~1e-12 scatter/reduction. Does not hurt AD.

## CRITICAL GOTCHAS (verified against source)
- **Truncate every JAX probe column to the record's `nlevels` before diffing.**
  Node → `nlevels_nod2D`; element → `nlevels`. (`verify.compare_column` does this.)
- **Layer vs interface masks** (in `mesh.py`): use `node_layer_mask`/`elem_layer_mask`
  for T,S,ρ,p,u,v,pgf; `node_iface_mask`/`elem_iface_mask` for bvfreq,w,Kv,Av.
- 3D field layout is row-major `[n_entity, nl]` (== C `FESOM_NODE3D`); vectors
  `[·, nl, 2]` (== `FESOM_ELEMVEC`, u then v).
- Mesh indices already 0-based; `edge_tri`/`edge_up_dn_tri` use `−1` (masked by
  `ops.scatter_add`). `gradient_sca` is `[elem,6]` (∂N/∂x cols 0–2, ∂N/∂y cols 3–5).
  `edge_cross_dxdy` is `[edge,4]` in METERS.
- SSH solver (Task 2.7): stiffness built ONCE in linfs (static operator);
  preconditioner is MITgcm-style SYMMETRIC (off-diagonal matvec — NOT Jacobi). Use
  `jax.lax.custom_linear_solve`.
- `compute_vel_rhs` (2.4) bundles momentum advection (momadv_opt=2) into substep 5 —
  an edge→node scatter; don't omit it. Live viscosity is opt_visc=7 (biharmonic
  `visc_filt_bidiff`, 2.5).

## WORKFLOW NOTES
- The plan is authoritative; mark checkboxes `[x]` as you finish; keep the Revision
  Log current. Commit only when asked (Phases 0,1 are committed on `main`).
- C/Fortran edits go on a **separate `port2` branch** (currently `jax-mesh-export`);
  never touch the port2 main branches. Regenerate the dump:
  `sbatch /home/a/a270088/port2/fesom2_port/jobs/jax_cdump_pi.sh` (build first:
  `source env.sh && cmake --build build -j`).
- All python/pytest via the env python above (NOT base conda). GPU work = SLURM
  `shared`/`gpu`, account `ab0995`/`ab0995_gpu`.
- ⚠️ Loose end: `/home/a/a270088/port2/fesom2` (Fortran) was switched to an empty
  branch `jax-elem-dump` last session. Restore with
  `git -C /home/a/a270088/port2/fesom2 checkout fix-coldstart-wind-rotation` if desired.

Confirm you've absorbed this, tell me which task you're starting, then proceed.
