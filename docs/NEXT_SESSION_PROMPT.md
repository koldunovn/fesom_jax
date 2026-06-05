# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. Phases 0 & 1 are complete and
**Phase 2 substeps 1–13 (Tasks 2.1–2.9: the momentum chain + SSH RHS/CG solver +
velocity update / hbar / eta_n + the ALE vertical velocity `w` / `hnode_new`) are
done**. Tasks 2.1–2.7 are committed (`0b9d9d1`, `a9166e0`, `096ebb6`); **Tasks 2.8
(`update_vel`/`compute_hbar`/`eta_n`) and 2.9 (`ale.py`: `w`/`hnode_new`) are done
and tested but NOT yet committed.** Next is **Task 2.10 — upwind tracer advection +
vertical diffusion + thickness commit (substeps 15–16)**.

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean
model for hybrid ML (trainable NN parameterizations for vertical mixing and
mesoscale eddy fluxes, trained end-to-end). This continues a multi-session effort.
Work from `/home/a/a270088/port_jax`. Use max effort.

## START HERE, in order
1. **Read the plan (source of truth):**
   `/home/a/a270088/port_jax/docs/plans/20260605-fesom-jax-port.md` — decisions,
   verification ladder, 8 phases, per-task gates, Revision Log. Keep its checkboxes
   in sync as you go. Phase 2 = Tasks 2.1–2.11; **2.1–2.9 are `[x]`**, start at 2.10.
2. **Read the lessons log (NEW, read every session):**
   `/home/a/a270088/port_jax/docs/PORTING_LESSONS.md` — the running gotcha/lesson
   log. **STANDING RULE: append one entry per task as you go** (config traps,
   sign/index/association gotchas, AD subtleties), citing the C `file:line`.
3. **Read the project memory:**
   `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/` (MEMORY.md +
   `fesom-jax-port.md` + `porting-lessons-log.md`).
4. **Read** `docs/REFERENCE_RUNS.md` (the oracle + the ⚠️ T-blob IC + the ⚠️
   "substep 15 upwind-vs-FCT" caveat) and `docs/MESH_EXPORT_LAYOUT.md` (mesh arrays).
   Skim the Phase-2 modules you build on (all done): `fesom_jax/{eos,ic,pgf,pp,
   momentum,forcing,ssh,ale}.py` and their tests, plus the Phase-1 base
   `{mesh,state,ops,verify,io_dump,config}.py`.
5. Skim the C for substeps 15–16: **upwind** horizontal advection `adv_tra_hor_upw1`
   (`fesom_tracer_adv.c:212`) + **upwind** vertical `adv_tra_ver_upw1`
   (`fesom_tracer_adv.c:701`) / `flux2dtracer_upwind` (`:740`); the ALE reconstruction
   `ale_reconstruct` (`:792`); the non-FCT driver `fesom_tracer_advect_one` (`:1269`);
   vertical implicit diffusion `fesom_impl_vert_diff_tracers` (`fesom_tracer_diff.c:338`)
   + `diff_ver_part_impl_ale` (`:85`); thickness commit `fesom_ale_commit_thickness`
   (`fesom_ale.c:18`). Wired at `fesom_step.c:316-353` (tracers) and `:420-423`
   (thickness). **NOTE the C dump runs FCT (`fesom_tracer_advect_one_fct`,
   `fesom_step.c:323/332`); JAX Phase 2 runs UPWIND** — see the step-1 caveat below.

## STATUS — Phases 0 & 1 + Phase 2 substeps 1–13 COMPLETE (2.8, 2.9 uncommitted)
- **Phase 0 (GATE 0):** env (`fesom-jax`, jax 0.10.1 x64, A100), verify harness
  (`io_dump.py`+`verify.py`), pi mesh export (`data/mesh_pi/`, 31 arrays), and the
  **C-port per-substep dump oracle** (`fesom_jax/tests/fixtures/pi_cdump.00000`).
- **Phase 1 (GATE 1):** `mesh.py` (frozen `Mesh` pytree, 4 ragged-level masks),
  `state.py` (`State` pytree), `ops.py` (`gather*`, masked `scatter_add`,
  `mask_below_bottom`, vectorized `tdma`). AD gates pass.
- **Phase 2 substeps 1–9 (Tasks 2.1–2.7), committed:**
  - 2.1 `eos.py` (+`ic.py`): JM-EOS density (**bit-exact**), pressure, N² + smoother.
    **IC = constant T=10/S=35 + a Gaussian T-blob** (`ic.initial_state`).
  - 2.2 `pgf.py`; 2.3 `pp.py` (interior interfaces `[nzmin+1,nzmax)`); 2.4–2.6
    `momentum.py` (`compute_vel_rhs` AB2+momadv, `visc_filt_bidiff` opt_visc=7,
    `impl_vert_visc` TDMA + `forcing.py` double-averaged wind).
  - 2.7 `ssh.py` (8–9): `compute_ssh_rhs` antisymmetric edge→node scatter (`α=1`,
    dump **atol 1e-7**, cancelling divergence); static linfs `build_ssh_operator` +
    **MITgcm symmetric** precond; `solve_ssh` — **⚠️ C CG stops LOOSE `soltol=1e-5`
    (≈3 iters) ⇒ dump `d_eta` is the EARLY-STOPPED iterate**, replicated exactly,
    `custom_linear_solve` gives the clean implicit-diff gradient.
- **Phase 2 substeps 10–12 (Task 2.8) — DONE, NOT committed:** `momentum.update_vel`
  (10, barotropic `∇N·d_eta`, ELEM dump ~2e-17), `ssh.compute_hbar` (11, reuse
  `compute_ssh_rhs` with `uv_rhs=0,α=1`; `÷areasvol` ⇒ hbar dump ~1e-17),
  `ssh.eta_n_update` (12, `α=1` ⇒ `eta_n=hbar`).
- **Phase 2 substep 13 (Task 2.9) — DONE, NOT committed:** `fesom_jax/ale.py`
  - `thickness_linfs` (`hnode_new = hnode`): linfs `dh/dt=0` ⇒ static memcpy. NODE
    dump **bit-for-bit** (max|Δ|=0) — confirms `State.rest().hnode` == C's `hnode`.
  - `compute_w` (`w`): the **per-level** sibling of the `ssh_rhs`/`hbar` antisymmetric
    edge→node `(v·dx−u·dy)·helem` scatter (new `uv`, α=1, NOT column-summed), then a
    **reverse bottom→top cumsum** (`lax.cumsum(reverse=True)`; masked scatter ⇒ no-flux
    `w[nzmax]=0` falls out), then **÷`mesh.area`** (⚠️ NOT `areasvol`), safe-divide.
    Step-1 `w` is a REAL gate (post-`update_vel` wind-driven `uv` ⇒ `w`~1e-6); NODE
    dump **~4e-20** (hbar-class: ÷area crushes the cancellation floor; gated 1e-12).
  - **Config:** `use_wsplit=0` ⇒ `w_e=w`, `w_i=0` (no split); `cflz`/`wvel_split`
    have no substep-13 dump → port them when consumed (2.10/2.11).
- **Full suite: 192 passing** (`JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`).

## THE PROVEN VERIFICATION RECIPE (follow it for every substep)
1. **Dump gate** at the pinned probes (node fields → node gids `1001,1500,2000,2500,
   3000`; element fields → first-incident-cell gids `1757,2656,3688,4604,5575`).
   Truncate the JAX column to the record's `nlevels`; use
   `verify.assert_close(col, rec, kind=…)` (`map`/`gather`=1e-15, `scatter`/
   `reduction`=1e-12). For tiny-valued / near-cancelling fields the `atol` floor gates
   (calibrate it; a downstream `÷area`/average often recovers map-class fidelity).
2. **⚠️ Step 1 was at REST for the early kernels, but `uv`/`d_eta`/`w` are now NONZERO
   (wind-driven).** Tracer advection at step 1 IS exercised (the wind-driven `uv` moves
   the T-blob), but ⚠️ **upwind ≠ the dump's FCT in general** — see the step-1 caveat.
   Still add a **synthetic-input unit test vs an independent loop-based numpy reference**
   of the C, plus a constant-tracer-stays-constant + a pure-diffusion smoothing test.
3. **AD check** every kernel (`jax.grad` vs central-FD sweep) in a smooth regime. Use
   the double-`where` `_safe_sqrt` for any `sqrt(x)` that can hit 0; safe-divide any
   `÷area`/`÷h`.
4. **Build step-1 inputs by chaining the done modules** (see `test_ale.py::chain` /
   `test_ssh.py::chain` for the full substep-1→13 chain to copy).
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
at **every** step; substep 15 (T,S) — see the ⚠️ caveat below.

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
| 10 update_vel | `uv_u`,`uv_v` (ELEM) | 2.8 | ✅ |
| 11 compute_hbar | `hbar` (NODE) | 2.8 | ✅ |
| 12 eta_n | `eta_n` (NODE) | 2.8 | ✅ |
| 13 ale_step | `w`,`hnode_new` (NODE) | 2.9 | ✅ |
| 15 solve_tracers | `T`,`S` (NODE) | 2.10 | ⏳ next |
| 16 update_thickness | `hnode` (NODE) | 2.10 | ⏳ next |

## IMMEDIATE WORK — Task 2.10 (upwind tracers + diffusion + thickness commit, 15–16)
Create `fesom_jax/tracer_adv.py` + `fesom_jax/tracer_diff.py` + `fesom_jax/tests/
test_tracers.py`; extend `fesom_jax/ale.py` with `commit_thickness` (substep 16). Config
unchanged. Build on the done modules + the new `uv`/`w`/`hnode_new`.
- **Upwind advection (substep 15, no FCT yet):** horizontal `adv_tra_hor_upw1`
  (`fesom_tracer_adv.c:212`) — edge→node upwind flux of `T`/`S` by the volume flux
  `((v·dx−u·dy)·helem)` (same edge geometry as `compute_w`!), pick the upwind node value
  by flux sign; vertical `adv_tra_ver_upw1`/`flux2dtracer_upwind` (`:701/:740`) using
  **`w_e` (= `w` since `use_wsplit=0`)**. Accumulate into `del_ttf`, then `ale_reconstruct`
  (`:792`): `T_new = (T·hnode + del_ttf)/hnode_new` (linfs `hnode_new=hnode` ⇒ `T +
  del_ttf/hnode`). ⚠️ **Watch the `edge_vflux` sign** (FRESH_START §12/§14.5) — verify with
  a constant-tracer-stays-constant test.
- **Vertical diffusion (substep 15):** `fesom_impl_vert_diff_tracers`
  (`fesom_tracer_diff.c:338`) + `diff_ver_part_impl_ale` (`:85`) — per-NODE implicit TDMA
  (1 unknown, vs `impl_vert_visc`'s per-element 2-unknown), `Kv` diffusivity, surface
  heat/salt flux BC (none in this config). Reuse `ops.tdma`. Pad below-bottom rows
  `(b=1,a=c=0,d=0)` as in `impl_vert_visc`.
- **Thickness commit (substep 16):** `fesom_ale_commit_thickness` (`fesom_ale.c:18`):
  `hnode = hnode_new` (linfs no-op) + recompute `helem` = vertex-mean of `hnode_new`
  over each element's 3 nodes (`/3`), masked to `elem_layer_mask`. NODE dump `hnode` at
  substep 16 = static `hnode` (bit-for-bit, like `hnode_new`).
- **⚠️ STEP-1 GATE CAVEAT (verify carefully, update `REFERENCE_RUNS.md` if needed):** the
  C dump runs **FCT**, JAX runs **upwind**. **S=35 is horizontally constant ⇒ S advects
  trivially ⇒ upwind==FCT==dump at step 1 (clean S gate).** But **T has the Gaussian blob
  (horizontal gradients), so upwind T may NOT equal the dump's FCT T at step 1** — the
  REFERENCE_RUNS "horizontally constant" note predates the T-blob. So: gate **S** at step 1
  directly; for **T**, first CHECK vs the dump — if it matches (limiter inactive at this
  tiny `dt·uv`), gate it; if not, defer T's full match to Phase 4 (FCT) and validate the
  upwind T via the constant-tracer + pure-diffusion synthetic tests (the plan's Task 2.10
  already lists these). AD check the diffusion TDMA (`d(ΣT)/d(Kv)`).
- Then 2.11 (assemble `step()`, rest-state + 100-step stability + snapshot). **GATE 2:** pi
  100 steps stable; each substep matches C. **Commit Tasks 2.8/2.9/2.10 when the user asks.**

## GOLDEN RULE (JAX-adapted)
Preserve the EXACT computation — the math and the load-bearing association order — but
express it as vectorized array ops over `ops.py` primitives. Do NOT do a literal
loop-by-loop translation, and do NOT simplify the physics. When in doubt, dump the C
value at a probe and match it. Fidelity target: ~1e-15 map/gather, ~1e-12 scatter/reduction.

## CRITICAL GOTCHAS (verified; full list in PORTING_LESSONS.md)
- **IC is constant + a Gaussian T-blob** — use `ic.initial_state` (not bare constant).
  S stays 35 (horizontally constant); T varies (the blob).
- **Wind stress is double-averaged** elem→node→elem (`forcing.surface_stress`), not raw.
- **Three level-range classes:** layer `[nzmin,nzmax)` (T,S,ρ,p,u,v,pgf); interface
  `[nzmin,nzmax]` (bvfreq,w); **interior interface** `[nzmin+1,nzmax)` (Kv,Av).
- **`w` ÷`mesh.area`, but `hbar` ÷`areasvol`** — different `[nod2D,nl]` CV-area arrays.
- **`use_wsplit=0` ⇒ `w_e=w`, `w_i=0`** (the substep-13 `w` IS `w_e` for tracer advection).
- **Truncate every JAX probe column to the record's `nlevels` before diffing.**
- Mesh indices 0-based; `edge_tri`/`edge_up_dn_tri` use −1 (masked by `ops.scatter_add`).
  `gradient_sca` `[elem,6]` (∂N/∂x cols 0–2, ∂N/∂y cols 3–5); `edge_cross_dxdy` `[edge,4]` m.
- **AD-safe `sqrt`/divide:** double-`where` (`momentum._safe_sqrt`) for `sqrt(x)→0`;
  `where(area>0, area, 1)` for any `÷area`.

## WORKFLOW NOTES
- Plan is authoritative; tick `[x]` as you finish; keep the Revision Log + lessons current.
- Commit only when asked (Phase 2 substeps 1–7 are committed; 2.8/2.9 are not).
- C/Fortran edits go on a separate `port2` branch (`jax-mesh-export`); never touch port2
  main branches. Regenerate the dump: `sbatch /home/a/a270088/port2/fesom2_port/jobs/jax_cdump_pi.sh`.
- All python/pytest via the env python above. GPU work = SLURM `shared`/`gpu`, acct `ab0995`.

Confirm you've absorbed this, tell me which task you're starting, then proceed.
