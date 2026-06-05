# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. Phases 0 & 1 are complete and
**Phase 2 substeps 1–16 (Tasks 2.1–2.10: the full single-step ocean chain — momentum
+ SSH/CG + free-surface + ALE `w` + upwind tracers + vertical diffusion + thickness
commit) are all ported, committed, and dump-gated.** Next is the **last Phase-2 task,
2.11 — assemble `step()` and run pi forward (rest-state + 100-step stability +
snapshot) → GATE 2.**

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean
model for hybrid ML (trainable NN parameterizations for vertical mixing and
mesoscale eddy fluxes, trained end-to-end). This continues a multi-session effort.
Work from `/home/a/a270088/port_jax`. Use max effort.

## START HERE, in order
1. **Read the plan (source of truth):**
   `/home/a/a270088/port_jax/docs/plans/20260605-fesom-jax-port.md` — decisions,
   verification ladder, 8 phases, per-task gates, Revision Log. Keep its checkboxes
   in sync. Phase 2 = Tasks 2.1–2.11; **2.1–2.10 are `[x]`**, start at **2.11** (GATE 2).
2. **Read the lessons log (read every session):**
   `/home/a/a270088/port_jax/docs/PORTING_LESSONS.md`. **STANDING RULE: append one
   entry per task as you go**, citing the C `file:line`.
3. **Read the project memory:**
   `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/` (MEMORY.md +
   `fesom-jax-port.md` + `porting-lessons-log.md`).
4. **Read** `docs/REFERENCE_RUNS.md` (oracle, T-blob IC, the substep-15 upwind-vs-FCT
   caveat) and skim **all** the done Phase-2 modules + tests:
   `fesom_jax/{eos,ic,pgf,pp,momentum,forcing,ssh,ale,tracer_adv,tracer_diff}.py`
   and `{mesh,state,ops,verify,io_dump,config}.py`. The step-1 chain that wires them
   is in **`test_tracers.py::chain`** / `test_ale.py::chain` — copy it.
5. **Read the C step driver `fesom_step.c` end-to-end** (the substep order + the
   between-substep state bookkeeping) — esp. `:80-279` (momentum→SSH→ALE) and
   `:314-423` (tracers→salinity floor→commit). Note the warm-start / AB2 / `hbar_old`
   bookkeeping called out under IMMEDIATE WORK.

## STATUS — Phases 0 & 1 + Phase 2 substeps 1–16 COMPLETE (all committed)
- **Phase 0 (GATE 0):** env (`fesom-jax`, jax 0.10.1 x64, A100), verify harness, pi
  mesh export (31 arrays), the **C-port per-substep dump oracle** (`pi_cdump.00000`).
- **Phase 1 (GATE 1):** `mesh.py`, `state.py`, `ops.py` (`gather*`, masked
  `scatter_add`, `mask_below_bottom`, vectorized `tdma`). AD gates pass.
- **Phase 2 substeps 1–16 (Tasks 2.1–2.10), all committed (latest `…2.10` commit):**
  - 2.1 `eos.py` (+`ic.py`, IC = constant + **T-blob**), 2.2 `pgf.py`, 2.3 `pp.py`,
    2.4–2.6 `momentum.py` (vel_rhs / visc_filt_bidiff / impl_vert_visc + `forcing.py`).
  - 2.7 `ssh.py` (compute_ssh_rhs atol 1e-7; static linfs operator + MITgcm precond;
    `solve_ssh` — **C CG early-stops at loose `soltol=1e-5`**, replicated, with
    `custom_linear_solve` for the clean gradient).
  - 2.8 `update_vel`/`compute_hbar`/`eta_n` (α=1 ⇒ `eta_n=hbar`).
  - 2.9 `ale.py`: `compute_w` (per-level transport divergence + reverse-cumsum +
    **÷`mesh.area`**, dump ~4e-20) + `thickness_linfs` (`hnode_new=hnode`, bit-exact).
  - 2.10 `tracer_adv.py` (upwind: AB2 `ttfAB`, masked per-element `vflux`, `w_e`-vertical,
    ALE reconstruct) + `tracer_diff.py` (per-node TDMA, `where(dZ==0,1,dZ)` AD-fix) +
    `ale.commit_thickness`. **⚠️ dump runs FCT, port runs upwind**: `S=35` matches
    bit-for-bit (clean gate), `T`-blob differs ~3e-7 (Phase-4 gate); `hnode`(16)
    bit-for-bit. Verified vs a numpy upwind ref + constant-tracer + conservation.
- **Full suite: 212 passing** (`JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`).

## THE VERIFICATION ORACLE — substep → field map (fixture `pi_cdump.00000`)
Substeps 1–13,16 match at **every** step; substep 15 `S` matches every step at the
constant value, `T` is the upwind-vs-FCT Phase-4 gate (see REFERENCE_RUNS).

| substep | C fields (entity) | task | status |
|---|---|---|---|
| 1 pressure_bv | `bvfreq`,`density`,`pressure` (NODE) | 2.1 | ✅ |
| 3 pressure_force | `pgf_x/y` (ELEM) | 2.2 | ✅ |
| 4 mixing | `Kv` (NODE), `Av` (ELEM) | 2.3 | ✅ |
| 5–7 vel_rhs/visc/impl_visc | `uv_rhs_u/v` (ELEM) | 2.4–2.6 | ✅ |
| 8 ssh_rhs / 9 ssh_solve | `ssh_rhs`,`d_eta` (NODE) | 2.7 | ✅ |
| 10 update_vel | `uv_u/v` (ELEM) | 2.8 | ✅ |
| 11 hbar / 12 eta_n | `hbar`,`eta_n` (NODE) | 2.8 | ✅ |
| 13 ale_step | `w`,`hnode_new` (NODE) | 2.9 | ✅ |
| 15 solve_tracers | `T`,`S` (NODE) | 2.10 | ✅ (`S` tight, `T` Phase-4) |
| 16 update_thickness | `hnode` (NODE) | 2.10 | ✅ |

## IMMEDIATE WORK — Task 2.11 (assemble `step()` → GATE 2)
Create `fesom_jax/step.py` + `fesom_jax/tests/test_step_pi.py`. Wire the substeps into
one jitted `step(state, mesh, op, params) -> state`, mirroring `fesom_step.c`'s order
(the per-substep modules are all done; this is **integration + state-threading**, not
new physics). The static SSH `op` (`ssh.build_ssh_operator`) is built once outside the
loop. **Sequence** (Phase 2, no gm/ice):
`eos.compute_pressure_bv → pgf.pressure_force_linfs → pp.mixing_pp → momentum.compute_vel_rhs
→ visc_filt_bidiff → impl_vert_visc(du) → ssh.compute_ssh_rhs → solve_ssh(d_eta) →
update_vel → compute_hbar → eta_n_update → ale.thickness_linfs + compute_w (w_e=w, w_i=0)
→ tracer_adv.advect_one(T) + advect_one(S) → tracer_diff.impl_vert_diff → salinity floor
→ ale.commit_thickness`.
- **⚠️ State-threading / between-step history (get these right — they're the whole point
  of 2.11, and the multi-step dump re-verification is the gate):**
  - **CG warm start:** the C does NOT zero `d_eta` between steps — step ≥2 warm-starts
    `solve_ssh(op, ssh_rhs, x0=d_eta_prev)` (already supported; `x0` is `stop_gradient`’d
    & folded into the rhs). Step 1 `x0=0`. **This is where multi-step `d_eta` dump
    matching is finalized** (the ssh/warmstart lesson flagged it for 2.11).
  - **AB2 momentum:** `compute_vel_rhs` shifts the OLD `uv_rhsAB`; pass `is_first_step`
    only at step 1. Thread the NEW `uv_rhsAB` it returns.
  - **AB2 tracers:** `advect_one` returns `(T_new, T_old_new=T)`; set `state.T_old=T_old_new`.
  - **`hbar_old`:** save `hbar` into `hbar_old` BEFORE `compute_hbar` overwrites it (the
    `eta_n` blend reads `hbar_old`; α=1 ⇒ unused but keep it correct).
  - **`eta_n` / `w_e` feedback:** `compute_vel_rhs` reads the PREVIOUS step's `eta_n`
    (SSH-grad term) and `w_e` (momentum advection vertical flux) — at step 1 both 0.
  - **Salinity floor** (`fesom_step.c:382-393`): `S = max(S, 0.5)` on wet layers
    (`nz<nzmax`). No-op at step 1; needed for stability/CORE2.
- **Gates (write `test_step_pi.py`):**
  1. **Rest state:** constant T/S (no blob), η=0, uv=0, zero wind → stays at rest to
     machine precision (no spurious flow). ⚠️ The analytical wind is nonzero, so use a
     zero-forcing variant for this test (`forcing.surface_stress` → 0).
  2. **Multi-step dump re-verification:** run with the dump config (T-blob + analytical
     wind) for ~10 steps; assert each substep field (1–13,16 + `S`@15) matches
     `pi_cdump.00000` at steps 2–10 at the probes — this is the real test that the
     **history threading is correct** (step-1 alone can't catch an AB2/warm-start slip).
  3. **100-step stability:** dt=100, assert no NaN, `max|uv|` ~O(0.3), `|eta|<5 m`.
  4. (Snapshot climate-close to a C 100-step run — rung 2, `eps_climate_compare`;
     defer if no C snapshot handy, note it.)
- **GATE 2:** pi 100 steps stable; each substep matches C within tolerance. Then Phase 3
  (AD smoke test: `lax.scan`+checkpoint, end-to-end gradient). **Commit when the user asks.**

## THE PROVEN VERIFICATION RECIPE (still applies)
Dump gate at the pinned probes (node gids `1001,1500,2000,2500,3000`; elem gids
`1757,2656,3688,4604,5575`); truncate to `nlevels`; `verify.assert_close(col, rec,
kind=…)`. Per-kind tol: `map`/`gather` 1e-15, `scatter`/`reduction` 1e-12 (calibrate
`atol` for near-cancelling fields). AD-check every kernel; safe-sqrt / safe-divide any
`sqrt(x→0)` / `÷(area→0)`. Append lessons; tick the plan; commit only when asked.

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`.
- Exported pi mesh (gitignored): `data/mesh_pi/*.npy`; reference dump (committed):
  `fesom_jax/tests/fixtures/pi_cdump.00000`.
- C port (ALGORITHMIC source of truth): `/home/a/a270088/port2/fesom2_port/src/`
  (branch `jax-mesh-export`). Fortran: `/home/a/a270088/port2/fesom2/src/` (cross-check
  only). Kokkos: `/home/a/a270088/port_kokkos/`.

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi).
2. Full-fidelity, bottom-up, not a toy. 3. AD-safe by construction + early end-to-end
gradient (Phase 3), re-run at every gate. 4. Mesh = index gather/scatter over `ops.py`.
5. Single-device + data-parallel now; mesh sharding is Phase 8.

## CRITICAL GOTCHAS (verified; full list in PORTING_LESSONS.md)
- **IC = constant + Gaussian T-blob** (`ic.initial_state`); **S=35 constant**, T varies.
- **Wind stress double-averaged** elem→node→elem (`forcing.surface_stress`).
- **`w` ÷`mesh.area`, `hbar` ÷`areasvol`** (different `[nod2D,nl]` arrays).
- **`use_wsplit=0` ⇒ `w_e=w`, `w_i=0`.** **`bc_surface=0`, `sw_3d=0`** (analytical).
- **Three level-range classes:** layer `[nzmin,nzmax)`; interface `[nzmin,nzmax]`;
  interior interface `[nzmin+1,nzmax)` (Kv,Av). Truncate every probe to `nlevels`.
- **AD-safe:** double-`where` `_safe_sqrt` for `sqrt(x→0)`; `where(d==0,1,d)` /
  `where(a>0,a,1)` for any divide whose denominator can hit 0 in a masked lane (the
  forward `where` does NOT stop a 0·inf NaN in the backward pass).
- Mesh indices 0-based; `edge_tri`/`edge_up_dn_tri` −1 = boundary (masked by scatter).
  `gradient_sca [elem,6]` (∂N/∂x 0–2, ∂N/∂y 3–5); `edge_cross_dxdy [edge,4]` meters.

## WORKFLOW NOTES
- Plan is authoritative; tick `[x]`; keep the Revision Log + lessons current.
- Commit only when asked. C/Fortran edits go on the `port2` branch `jax-mesh-export`;
  never touch port2 main. Regenerate the dump:
  `sbatch /home/a/a270088/port2/fesom2_port/jobs/jax_cdump_pi.sh`.
- All python/pytest via the env python above. GPU = SLURM `shared`/`gpu`, acct `ab0995`.

Confirm you've absorbed this, tell me which task you're starting, then proceed.
