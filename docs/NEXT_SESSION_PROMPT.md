# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. Phases 0 & 1 are complete and
**Phase 2 substeps 1–9 (Tasks 2.1–2.7: the momentum chain + the SSH RHS/CG solver)
are done**. Tasks 2.1–2.6 are committed (`0b9d9d1`); **Task 2.7 (`ssh.py`) is done
and tested but NOT yet committed.** Next is **Task 2.8 — velocity update + hbar +
eta_n (substeps 10–12)**.

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean
model for hybrid ML (trainable NN parameterizations for vertical mixing and
mesoscale eddy fluxes, trained end-to-end). This continues a multi-session effort.
Work from `/home/a/a270088/port_jax`. Use max effort.

## START HERE, in order
1. **Read the plan (source of truth):**
   `/home/a/a270088/port_jax/docs/plans/20260605-fesom-jax-port.md` — decisions,
   verification ladder, 8 phases, per-task gates, Revision Log. Keep its checkboxes
   in sync as you go. Phase 2 = Tasks 2.1–2.11; **2.1–2.7 are `[x]`**, start at 2.8.
2. **Read the lessons log (NEW, read every session):**
   `/home/a/a270088/port_jax/docs/PORTING_LESSONS.md` — the running gotcha/lesson
   log. **STANDING RULE: append one entry per task as you go** (config traps,
   sign/index/association gotchas, AD subtleties), citing the C `file:line`.
3. **Read the project memory:**
   `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/` (MEMORY.md +
   `fesom-jax-port.md` + `porting-lessons-log.md`).
4. **Read** `docs/REFERENCE_RUNS.md` (the oracle + the ⚠️ T-blob IC) and
   `docs/MESH_EXPORT_LAYOUT.md` (mesh arrays). Skim the Phase-2 modules you build on
   (all committed): `fesom_jax/{eos,ic,pgf,pp,momentum,forcing}.py` and their tests,
   plus the Phase-1 base `{mesh,state,ops,verify,io_dump,config}.py`.
5. Skim the C modules for substeps 10–12: `update_vel` + `compute_hbar`
   (`/home/a/a270088/port2/fesom2_port/src/fesom_momentum.c:474,779`), the eta_n blend
   (`fesom_ale.c`), and the Phase-2 SSH module you build on (`fesom_jax/ssh.py`, done).

## STATUS — Phases 0 & 1 + Phase 2 substeps 1–9 COMPLETE (2.7 uncommitted)
- **Phase 0 (GATE 0):** env (`fesom-jax`, jax 0.10.1 x64, A100), verify harness
  (`io_dump.py`+`verify.py`), pi mesh export (`data/mesh_pi/`, 31 arrays), and the
  **C-port per-substep dump oracle** (`fesom_jax/tests/fixtures/pi_cdump.00000`).
- **Phase 1 (GATE 1):** `mesh.py` (frozen `Mesh` pytree, 4 ragged-level masks),
  `state.py` (`State` pytree), `ops.py` (`gather*`, masked `scatter_add`,
  `mask_below_bottom`, vectorized `tdma`). AD gates pass.
- **Phase 2 substeps 1–7 (Tasks 2.1–2.6), committed `0b9d9d1`:**
  - 2.1 `eos.py` (+`ic.py`): JM-EOS density (**bit-exact** vs dump), hydrostatic
    pressure, N² + the single-sweep area-weighted N² smoother. **IC = constant
    T=10/S=35 + a Gaussian T-blob** (`ic.initial_state`) — the dump's real IC.
  - 2.2 `pgf.py`: pressure-gradient force at elements.
  - 2.3 `pp.py`: PP mixing + convective adjustment. Outputs on **interior**
    interfaces `[nzmin+1,nzmax)` only.
  - 2.4 `momentum.py::compute_vel_rhs`: AB2 Coriolis + SSH grad + PGF +
    `momentum_adv_scalar` (momadv_opt=2 edge→node scatter).
  - 2.5 `momentum.py::visc_filt_bidiff`: biharmonic flow-aware viscosity (opt_visc=7),
    two edge→element scatters; AD-safe `_safe_sqrt`.
  - 2.6 `momentum.py::impl_vert_visc`: per-element TDMA (`ops.tdma`), wind stress +
    quadratic bottom drag. **`forcing.py`**: analytical wind, **double-averaged**
    elem→node→elem (the stress `impl_vert_visc` actually reads).
- **Phase 2 substeps 8–9 (Task 2.7), `ssh.py` — DONE, NOT committed:**
  - `compute_ssh_rhs` (8): antisymmetric edge→node transport scatter; `SSH_ALPHA=1`
    ⇒ the `(1−α)·ssh_rhs_old` blend term is 0. Matches dump at **atol 1e-7**
    (cancelling transport divergence; abs floor = upstream `du` ×`dx·helem~1e7`).
  - `build_ssh_operator`: **static** linfs Galerkin stiffness `S` (host scipy
    assemble → `segment_sum` matvec) + **MITgcm symmetric** preconditioner (verified
    load-bearing vs Jacobi).
  - `solve_ssh` (9): **⚠️ the C CG stops at a LOOSE `soltol=1e-5` (≈3 iters), so the
    dump's `d_eta` is the EARLY-STOPPED iterate** — replicated exactly (forward) ⇒
    matches dump **~1e-18**; `custom_linear_solve`’s tight `transpose_solve` gives the
    clean implicit-diff `S⁻¹` gradient (== tight solve rel 2e-14, == FD).
- **Full suite: 148 passing** (`JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`).

## THE PROVEN VERIFICATION RECIPE (follow it for every substep)
This worked cleanly for 2.1–2.6 — reuse it:
1. **Dump gate** at the pinned probes (node fields → node gids `1001,1500,2000,2500,
   3000`; element fields → first-incident-cell gids `1757,2656,3688,4604,5575`).
   Truncate the JAX column to the record's `nlevels`; use
   `verify.assert_close(col, rec, kind=…)` (`map`/`gather`=1e-15, `scatter`/
   `reduction`=1e-12). For tiny-valued fields the `atol` floor is what gates.
2. **⚠️ Step 1 is at REST (uv=η=uv_rhsAB=0).** Velocity-dependent kernels collapse to
   trivial values, so the dump gate is WEAK. Exercise the dormant path with a
   **synthetic-input unit test vs an independent loop-based numpy reference** of the C
   (a different code path). Re-verify the full multi-step field once `step()` exists
   (2.11).
3. **AD check** every kernel (`jax.grad` vs central-FD sweep) in a smooth regime,
   away from kinks (`max(N²,0)`, convection, `|∇u|=0`). Use the double-`where`
   `_safe_sqrt` for any `sqrt(x)` that can hit 0.
4. **Build step-1 inputs by chaining the committed modules:** `ic.initial_state` →
   `eos.compute_pressure_bv` → `pgf.pressure_force_linfs` → `pp.mixing_pp` →
   `momentum.compute_vel_rhs` → `visc_filt_bidiff` → `impl_vert_visc`
   (see `test_momentum.py::_chain_step1`).
5. Append the lesson(s) to `PORTING_LESSONS.md`; tick the plan checkbox; do NOT commit
   unless asked.

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q` (silences the benign login-node
  `cuInit 303` GPU-absent warning; Phase 2 is pure JAX → CPU is fine).
- Exported pi mesh (gitignored): `data/mesh_pi/*.npy` + `meta.txt`
- Reference dump fixture (committed): `fesom_jax/tests/fixtures/pi_cdump.00000`
- C port (ALGORITHMIC source of truth, mirror it): `/home/a/a270088/port2/fesom2_port/src/`
  — JAX-port C additions on branch **`jax-mesh-export`**.
- Fortran model (numerical cross-check only): `/home/a/a270088/port2/fesom2/src/`
- Kokkos port (parallelization + fidelity lessons): `/home/a/a270088/port_kokkos/`

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi).
2. Full-fidelity, bottom-up (minimal step → pi → CORE2 → ice/KPP/GM), not a toy.
3. AD-safe by construction (idiomatic functional JAX) + an EARLY end-to-end gradient
   smoke test (Phase 3), re-run at every gate. AD is never deferred.
4. Mesh = index gather/scatter mirroring the C loops (reuse `ops.py`/`mesh.py`).
5. Single-device + data-parallel-over-batch now; mesh sharding is Phase 8.

## THE VERIFICATION ORACLE — substep → field map (fixture `pi_cdump.00000`)
`density`=`density_m_rho0`=in-situ ρ−ρ0; `pressure`=`hpressure`. Substeps 1–13,16 match
at **every** step; substep 15 (T,S) matches at **step 1 only** until Phase 4 (C runs FCT,
Phase-2 JAX runs upwind; at step 1 the field is horizontally constant so upwind==FCT).

| substep | C fields (entity) | port task | status |
|---|---|---|---|
| 1 pressure_bv | `bvfreq`,`density`,`pressure` (NODE) | 2.1 | ✅ |
| 3 pressure_force | `pgf_x`,`pgf_y` (ELEM) | 2.2 | ✅ |
| 4 mixing | `Kv` (NODE), `Av` (ELEM) | 2.3 | ✅ |
| 5 vel_rhs | `uv_rhs_u`,`uv_rhs_v` (ELEM) | 2.4 | ✅ |
| 6 viscosity_filter | `uv_rhs_u`,`uv_rhs_v` (ELEM) | 2.5 | ✅ |
| 7 impl_vert_visc | `uv_rhs_u`,`uv_rhs_v` (ELEM) | 2.6 | ✅ |
| 8 ssh_rhs | `ssh_rhs` (NODE) | 2.7 | ✅ |
| 9 ssh_solve | `d_eta` (NODE) | 2.7 | ✅ |
| 10 update_vel | `uv_u`,`uv_v` (ELEM) | **2.8** | ⏳ next |
| 11 compute_hbar | `hbar` (NODE) | **2.8** | ⏳ next |
| 12 eta_n | `eta_n` (NODE) | **2.8** | ⏳ next |
| 13 ale_step | `w`,`hnode_new` (NODE) | 2.9 | |
| 15 solve_tracers | `T`,`S` (NODE) | 2.10 | |
| 16 update_thickness | `hnode` (NODE) | 2.10 | |

## IMMEDIATE WORK — Task 2.8 (velocity update + hbar + eta_n, substeps 10–12)
Add to `fesom_jax/momentum.py` + `fesom_jax/ssh.py` (+ tests). Config unchanged: pi,
linfs, PP, upwind tracers, CG SSH, no GM/KPP/ice, analytical wind, dt=100, nl=48. Build
on the done `ssh.py` (`compute_ssh_rhs`, `build_ssh_operator`, `solve_ssh`) and the
momentum chain. **First read `fesom_step.c:176-280`** (the substep 10–12 wiring).
- **`update_vel`** (substep 10, `fesom_momentum.c:474`): `uv += du` (the increment from
  `impl_vert_visc`, currently stored in `uv_rhs`) **plus** the SSH-gradient correction
  `coef=-g·theta·dt`, `uv += coef·∇N·d_eta` gathered to elements (`fesom_momentum.c:468-491`).
  ELEM field (`uv_u`,`uv_v`) → element dump at substep 10 (probes 1757…5575). At step 1
  this is the first nonzero `uv` (wind-driven).
- **`compute_hbar`** (substep 11, `fesom_momentum.c:779` / `fesom_ale.c`): transport-
  divergence edge→node scatter; **save `ssh_rhs_old = ssh_rhs`** (the AB history the linfs
  ssh_rhs blend reads next step), then update `hbar`. NODE field (`hbar`) at substep 11.
- **`eta_n`** (substep 12): the eta_n blend (`eta_n += ...·d_eta`/hbar — check the C). NODE
  field (`eta_n`) at substep 12.
- **Gate:** `hbar` (11) and `eta_n` (12) are NODE → compare vs dump directly (≤1e-12).
  `uv` (10) is ELEM → element dump directly. AD: gradient flows through the gather of the
  CG output `d_eta` into `uv`/`eta_n` (continues the `custom_linear_solve` chain). Watch the
  **warm-start**: `d_eta` is consumed here but NOT zeroed → it is the next step's CG `x0`
  (see the ssh/warmstart lesson; `solve_ssh` already takes `x0`).
- Then 2.9 (ALE `w`/`hnode_new`) → 2.10 (upwind tracers + diffusion) → 2.11 (assemble
  `step()`, rest-state + 100-step stability + snapshot). **GATE 2:** pi 100 steps stable;
  each substep matches C. **Commit Task 2.7 (`ssh.py`) when the user asks.**

## GOLDEN RULE (JAX-adapted)
Preserve the EXACT computation — the math and the load-bearing association order — but
express it as vectorized array ops over `ops.py` primitives. Do NOT do a literal
loop-by-loop translation, and do NOT simplify the physics. When in doubt, dump the C
value at a probe and match it. Fidelity target: ~1e-15 map/gather, ~1e-12 scatter/reduction.

## CRITICAL GOTCHAS (verified; full list in PORTING_LESSONS.md)
- **IC is constant + a Gaussian T-blob** — use `ic.initial_state` (not bare constant).
- **Wind stress is double-averaged** elem→node→elem (`forcing.surface_stress`), not raw.
- **Three level-range classes:** layer `[nzmin,nzmax)` (T,S,ρ,p,u,v,pgf); interface
  `[nzmin,nzmax]` (bvfreq,w); **interior interface** `[nzmin+1,nzmax)` (Kv,Av).
- **Truncate every JAX probe column to the record's `nlevels` before diffing.**
- Mesh indices 0-based; `edge_tri`/`edge_up_dn_tri` use −1 (masked by `ops.scatter_add`).
  `gradient_sca` `[elem,6]` (∂N/∂x cols 0–2, ∂N/∂y cols 3–5); `edge_cross_dxdy` `[edge,4]` m.
- **AD-safe `sqrt`:** double-`where` (`momentum._safe_sqrt`) for any `sqrt(x)` hitting 0.

## WORKFLOW NOTES
- Plan is authoritative; tick `[x]` as you finish; keep the Revision Log + lessons current.
- Commit only when asked (Phase 2 substeps 1–7 are committed `0b9d9d1` on `main`).
- C/Fortran edits go on a separate `port2` branch (`jax-mesh-export`); never touch port2
  main branches. Regenerate the dump: `sbatch /home/a/a270088/port2/fesom2_port/jobs/jax_cdump_pi.sh`.
- All python/pytest via the env python above. GPU work = SLURM `shared`/`gpu`, acct `ab0995`.

Confirm you've absorbed this, tell me which task you're starting, then proceed.
