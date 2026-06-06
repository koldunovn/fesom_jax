# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0–4 COMPLETE (GATEs 0/1/2/3/4).
Phase 5 (CORE2 single-device) IN PROGRESS — sub-plan created; Tasks 5.1–5.7 DONE;
Task 5.8 (GATE 5 — gradient on a CORE2 slice) is NEXT and is the LAST Phase-5 task.** Full
suite **371 passing** (pi 313 + CORE2 additions: `test_surface_bc.py` +7, `test_core2_step.py`
+6 incl. the new Task-5.7 per-substep dynamics gate, and the 5.1–5.5 module tests).

The pi model is fully ported, dump-gated, jitted, differentiable end-to-end, 1000-step
stable. Phase 5 runs that same physics (PP + **linfs** + FCT + opt_visc7, **no GM/KPP/ice**)
on the **CORE2 mesh** with PHC initial conditions and real JRA55+SSS+runoff forcing,
verified per-substep against a CORE2 C-port dump (**Path A**). DONE: CORE2 mesh exported +
loads zero-code (CW-orientation guard); **PHC IC** ported (numpy, ~1e-14); **JRA55-do
reader** ported (numpy, **bit-exact**); **L&Y09 bulk** ported (AD-safe, ~1e-13);
**SSS-restoring + runoff** ported (numpy + AD-safe); **Task 5.6 — the assembled CORE2 step**
(surface BCs wired, step-1 post-step T/S **bit-exact** vs the C dump); and now **Task 5.7 —
the matched C dump run + CORE2 stability**: per-substep **dynamics** gates added (step-1
bit-exact class), and the full assembled model run **jitted** 1-day + multi-day, cross-checked
against a matched C arbiter. **The single-device CORE2 model now runs, matches the C
per-substep at step 1, and is numerically stable for ~a week** (the only no-ice limitation is
unbounded high-lat supercooling, shared with the C — see the finding). Task 5.8 = GATE 5
(gradient on a CORE2 slice + the new SST→flux / current→stress feedbacks).

**JAX committed on `main`** through Task 5.7 (`50373ef` Task 5.7, `7ca598f` 5.6, `d906750` 5.5,
`c1aa677` 5.4, `5ab28af` 5.3, `d4fcdb2` 5.2, `4b34f6a` 5.1). C-side dump additions on the
port2 `jax-mesh-export` branch through Task 5.6 (`cfb4b5e`). **Task 5.7 added NO C source
changes** — only a new untracked job script `port2/fesom2_port/jobs/jax_core2_stability.sh`
(the stability arbiter; ⚠️ **C job scripts stay untracked** per user; port2 is otherwise the
user's — no housekeeping). **CORE2 data artifacts live on `/work`** — `port_jax/data` is a
**symlink** to `/work/ab0995/a270088/port_jax/data` (gitignored). ⚠️ **Cheap C jobs:
`-p compute --time=00:30:00` (fast debug QOS); a LONG C run (the stability arbiter, ~4.6 s/step
single-rank) needs a non-debug QOS (`--time=02:30:00`). JAX GPU jobs: `-A ab0995_gpu -p gpu
--gres=gpu:1`.**

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for
hybrid ML (trainable NN parameterizations for vertical mixing + mesoscale eddy fluxes,
trained end-to-end). Multi-session effort. Work from `/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Phase-5 sub-plan (source of truth for Phase 5):**
   `docs/plans/20260606-fesom-jax-core2.md` — the scope correction (§0), Path A, the task
   ladder 5.1–5.8 with per-task gates. **Tasks 5.1–5.7 are ticked DONE; 5.8 is next (the
   final Phase-5 task).**
2. **Main plan:** `docs/plans/20260605-fesom-jax-port.md` — decisions, the verification
   ladder, the Revision Log. Phase 5 there is the outline; the sub-plan supersedes it.
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 5** entries
   through **Task 5.7** (the no-ice-supercooling-is-physical finding, the C-arbiter
   3-sig-fig-tracking method, the step-1-bit-exact-vs-steps-2/3-diverge calibration, the
   dump node/element probe layout, the GPU/CPU/eager timings). **STANDING RULE: append a
   lesson per task.**
4. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.
5. For 5.8: the **gradient gate is `tests/test_gradient.py`** (pi: `d(mean SST)/d(k_ver)` &
   `d(.../d(a_ver)` AD↔FD plateau through the CG `custom_linear_solve` + `d(loss)/d(T₀)`
   finite incl. masked lanes). The **assembled CORE2 model** is `core2_forcing.py` +
   `step(..., step_forcing, forcing_static)` / `integrate(..., step_forcings, forcing_static)`
   (checkpointed `lax.scan`, `integrate_jit`). Build the IC with
   `phc_ic.core2_initial_state(mesh, IC_DIR)` (⚠️ `T_old`=const base 10/35) and the forcing
   with `core2_forcing.build_core_forcing(mesh, 1958, sst_ic=state.T[:,0])` +
   `cf.stack(dates_for_steps(1958, 500, N))`. The new Phase-5 differentiable seams are in
   `forcing.bulk_surface_fluxes` (`d(heat_flux)/d(SST)`, `d(stress)/d(current)`) — all
   AD-safe by construction (double-`where` safe-sqrt). A reusable forward driver +
   monitor is `scripts/core2_stability_run.py` (jitted; A100 ~0.06 s/step).

## STATUS
- **Phases 0–4 (GATEs 0–4):** full single-step pi model (`step.py`), checkpointed `lax.scan`
  (`integrate.py`), CG `custom_linear_solve`, FCT + opt_visc7, pi 1000-step stable,
  `test_gradient.py` plateau 5.70e-7. All committed.
- **Phase 5 — scope (user-confirmed, do NOT re-litigate):** the C port (`fesom2_port`) is a
  **deliberately simplified** FESOM — **linfs-only, full-cell, no cavities**. Phase 5 = pi
  physics on CORE2 + PHC IC + JRA55/SSS/runoff. **NOT zlevel/zstar/partial-cells/w_i** (absent
  from the C port; zstar = future work, keep the `which_ale` seam, don't port now). Reference =
  **Path A** (per-substep CORE2 C dump at `FESOM_MIX_SCHEME=PP FESOM_NO_GMREDI=1
  FESOM_NO_ICE_*=1 FESOM_BULK_FIXED_ITERS=1`, dt=500).
- **Tasks 5.1–5.6 DONE** (see the sub-plan Revision Log for the per-task detail): mesh export
  (zero-code load + CW-orientation guard), PHC IC (~1e-14), JRA55 reader (bit-exact), L&Y09
  bulk (AD-safe ~1e-13), SSS/runoff (numpy readers + AD-safe balance), and the assembled CORE2
  step (surface BCs wired; step-1 post-step **T 7.1e-15 / S 2.1e-14 bit-exact**; the static
  `a_ice` mask + `T_old`=base + `FESOM_BULK_FIXED_ITERS` bugs found & fixed).
- **Task 5.7 DONE — matched C dump run + CORE2 stability.** Two parts:
  - **Per-substep dynamics gates** (`test_core2_step.py`, +1 test): step-1 pressure / PGF / Av /
    uv_rhs / ssh_rhs / d_eta / uv / hbar / eta_n / w / hnode are **bit-exact class** (JAX & C
    share the PHC IC; pre-solve ~0..1e-17, CG-derived ~1e-16..8e-15; the big intermediates
    `ssh_rhs`~1e5 / `pressure`~5e5 match ~1e-11 *relative*). Element fields gated at the dump's
    **incident-element** gids (`_emaxabs`). `test_evolution_steps23` extended (uv/d_eta steps
    2-3 ~1e-6 — the discrete CG iter-count + FCT amplify the step-1 ~1e-15; bounded).
  - **Stability run** (`scripts/core2_stability_run.py` + `core2_stability_gpu.sh`, A100 jitted
    ~0.06 s/step): **numerically stable days 1–7** (no NaN; max|vel| ≤ 1.9 < 3; |SSH| ≤ 2.8 < 5;
    Aleutian 94122 calm/warm). The matched **C arbiter** (`jobs/jax_core2_stability.sh`) is
    stable too, and JAX **tracks it to 3 sig figs** on SST_min/max|uv|/max|eta| (step 216:
    −6.60=−6.60, 1.389≈1.39, 2.715≈2.71).
  - ⚠️ **FINDING (anticipated risk #1, NOT a bug):** with no sea ice the SST **supercools
    without bound** (−1.9 IC → −16.5 day 5 → −22.8 day 8); past the JM-EOS-valid range
    (~−20 °C) the spurious density field destabilizes the dynamics at **model day ~8.1**
    (max|vel|>3). **The C supercools + tracks JAX identically through the verified ~day 2.3
    window (step 396; the longer C run was cancelled, so the day-8 figures are JAX's, shared by
    the mechanism)** ⇒ the no-ice run does NOT numerically blow up, ice stays Phase 6, and a
    physically realistic SST simply needs the ice cap. The "C blows up ⇒
    move ice to Phase 5" trigger did **not** fire.

## IMMEDIATE WORK — Task 5.8 (GATE 5; the LAST Phase-5 task)
**5.8 GATE 5 — gradient check on a CORE2 slice** (sub-plan has the full gate). GPU sbatch
(`-A ab0995_gpu -p gpu --gres=gpu:1`; model the job on `scripts/phase3_grad_memory.sbatch` /
`core2_stability_gpu.sh`):
- **Re-run the permanent AD gate on a CORE2 slice (small N):** `d(mean SST)/d(k_ver)` AD↔FD
  plateau (signal-lifted `k_ver`) flowing through the CG `custom_linear_solve`; `d(loss)/d(T₀)`
  finite everywhere incl. masked lanes (the strong masked-NaN probe). Use a **small step count**
  and (if memory needs) a CORE2 sub-slice / O(√N) nested checkpointing.
- **NEW Phase-5 differentiable feedbacks:** `d(heat_flux)/d(SST)` and `d(stress)/d(surface_
  current)` AD vs FD (the bulk's differentiable seam — the whole point of real forcing for
  hybrid ML). Stay clear of the bulk's `where`/safe-sqrt kinks (modest perturbations). ⚠️ keep
  the perturbations in the **EOS-valid SST range** (the no-ice supercooling means a long CORE2
  forward pushes SST out of range; for the gradient gate use a short window / shallow slice).
- **Memory:** CORE2 is ~40× pi nodes → confirm the checkpointed N-step backward fits the
  A100 (the pi N=200 backward was 4.23 GB; expect to drop N or use O(√N) nesting). GPU sbatch.
- **Gate:** full suite green (pi 313 + CORE2 additions); the CORE2-slice AD↔FD plateau passes
  incl. the new SST→flux / current→stress feedbacks. **Lesson:** append.

**GATE 5 (acceptance):** CORE2 reproduces the C per-substep dump at step 1 within tolerance
(✅ done); runs 1-day + multi-day stable (✅ done — numerically, with the documented no-ice
supercooling limitation); **the gradient gate passes on a CORE2 slice incl. the new feedbacks
(← 5.8, remaining)**; full suite green.

**First moves for 5.8:** (1) read `tests/test_gradient.py` (the pi gate to extend). (2) Pick a
small CORE2 slice / short window that keeps SST in the EOS-valid range. (3) Add a CORE2-slice
variant of the AD↔FD plateau through the CG + the two new bulk feedbacks. (4) GPU sbatch for
the memory-bound backward. The pattern is proven (pi GATE 4); 5.8 is mostly scale + the two
new seams.

## THE PROVEN VERIFICATION RECIPE (still applies)
Per-substep dump at pinned probes, truncate to `nlevels`, `verify.assert_close(col, rec,
kind=…)` (`map`/`gather` 1e-15, `scatter`/`reduction` 1e-12; calibrate `atol`; use **relative**
for big intermediate fields like `ssh_rhs`~1e5 / `pressure`~5e5). A per-substep gate is only
"tight" at **step 1** (shared inputs); downstream of the chaotic CG it diverges to ~1e-6 by
step 3 — gate the multi-step trajectory on **robust global reductions** (SST-range, max-speed)
against the matched **C arbiter** instead (tracks to 3 sig figs). AD: any divide/sqrt whose
denominator/arg can vanish in a masked lane must compute a FINITE value (`where(d==0,1,d)` /
double-`where` safe-sqrt) — a forward `where` does NOT stop a 0·inf NaN backward. Re-run
`test_gradient.py` at GATE 5. Phase-5 invariants: **mesh CW-orientation guarded at load**;
**the C port is the spec — port it 1:1 and verify by dump**; **distinguish numerical stability
(bounded vel/SSH/CG, no NaN) from thermodynamic realism (the no-ice run is the former for ~a
week, never the latter at high lat).**

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q` (~9.5 min; CORE2 tests skip if their
  `data/` artifacts are absent). **netCDF4 + scipy installed; jax-cuda12 (GPU) installed.**
- GPU via SLURM: `-A ab0995_gpu -p gpu --gres=gpu:1` (12 h) or `-p gpu-devel` (30 min);
  the env's jitted CORE2 step is **~0.06 s on an A100** (vs ~3 s CPU, ~32 s eager).
- CORE2 mesh: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC IC:
  `/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc`; JRA55:
  `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/`; SSS `…/PHC2_salx.nc`, runoff
  `…/CORE2_runoff.nc`; chl `…/Sweeney/Sweeney_2005.nc`.
- CORE2 data (gitignored): **`data/` is a symlink → `/work/ab0995/a270088/port_jax/data`**.
  Holds `mesh_core2/`, `ic_core2/`, `{phc,jra,bulk,sss,step}_dump_core2/`. C-side jobs
  (untracked in port2): `port2/fesom2_port/jobs/jax_*.sh` (incl. `jax_step_dump_core2.sh`,
  `jax_core2_stability.sh`).
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/`; built binary
  `…/build/fesom_port`. C run-arg order: `<mesh> <out> <dt> <nsteps> <snap> <phc> <jra>`.
  Build: `bash -lc 'source env.sh && make -C build fesom_port'` (incremental, ~30 s).
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu). Cheap C dumps → `-p compute
  --time=00:30:00` (debug). LONG C runs / JAX GPU → a real QOS. See memory
  [[hpc-job-file-conventions]].

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi);
   seam = `fesom_jax/params.py`. 2. Full-fidelity, bottom-up, match the C port 1:1. 3. AD-safe
   by construction + gradient re-run at every gate. 4. Mesh = index gather/scatter over
   `ops.py`. 5. Single-device + data-parallel now; mesh sharding Phase 8. 6. **Phase 5 = linfs
   on CORE2, Path A; zstar/partial-cells/GM/KPP/ice-model are later.** 7. **netCDF4 + scipy +
   jax-cuda12 in the env.** 8. **(2026-06-06) Phase 5 keeps the C's static `a_ice` mask** (=0.9
   where IC SST<0; gates shortwave penetration + the momentum stress blend) — match the C's
   "no-ice" run, NOT a truly ice-free ocean. chl = **Sweeney monthly** (the C default).
9. **(2026-06-06, Task 5.7) The no-ice run's high-lat supercooling is an accepted PHYSICAL
   limitation** (the C supercools identically through the verified ~day 2.3 window; sea ice in
   Phase 6 caps it) — the stability gate is **numerical** (no NaN, bounded vel/SSH), not
   thermodynamic.

## CRITICAL GOTCHAS (full list in PORTING_LESSONS.md)
- **The C port is linfs-only / full-cell / no-cavity — match THAT, not real-FESOM.** zlevel
  lives only in the Fortran; the parent outline's zlevel/partial-cells/w_i are out of scope.
- **⚠️ "No ice" ≠ ice-free:** the C keeps a **static `a_ice=0.9` mask** (IC SST<0, 37089 nodes)
  gating shortwave penetration + blending the wind stress. Replicated in `core2_forcing`
  (`ice_ic_aice`). **AND "no ice" ≠ stable forever:** the no-ice SST supercools without bound
  (−22 by day 8) → max|vel|>3 at day ~8 (EOS-invalid regime). Physical; the C tracks JAX
  identically through the verified ~day 2.3 window.
- **⚠️ CORE2 step-1 `T_old`/`S_old` = const base 10/35, NOT PHC** (`core2_initial_state`
  handles it; don't "fix" to PHC).
- **⚠️ `FESOM_BULK_FIXED_ITERS=1`** must be set on the C reference (the M-O loop is
  non-convergent at calm nodes; JAX runs fixed-5).
- **pi↔CORE2 orientation:** CORE2 raw mesh is ~all CCW; `load_mesh` asserts CW (Aleutian guard).
- **AD masked-NaN rule (bit us 4×):** make masked lanes finite; a forward `where` doesn't stop
  a backward 0·inf. The new bulk feedbacks (`d(heat_flux)/d(SST)`, `d(stress)/d(current)`) are
  the 5.8 AD targets — all double-`where` safe-sqrt-guarded.
- **Eager CORE2 `step()` ≈ 32 s/step; jitted ≈ 3 s CPU / 0.06 s A100** → always jit; GPU for 5.8.
- **The C dump pairs each node probe with its incident ELEMENT gid** — element fields
  (pgf/Av/uv_rhs/uv) are at element gids, node fields at node gids (`_emaxabs` vs `_maxabs`).
- **Runoff is INERT in the no-ice Phase 5 — by the C's design** (folded through ice thermo,
  which is off; the balanced `water_flux` feeds only non-linfs paths). Activates for free in
  Phase 6. **Full spec: sub-plan "Runoff handoff to Phase 6".**
- **netCDF4 import prints a benign `ndarray size changed` ABI warning** — harmless.
- **config = the pi reference physics on CORE2:** linfs, PP, FCT, opt_visc=7, `use_wsplit=0`,
  CG SSH (α=1), dt=500, PHC IC, JRA55+SSS+runoff + the static ice mask.

## WORKFLOW NOTES
- The sub-plan is authoritative for Phase 5; tick `[x]`, keep its Revision Log + the lessons
  current. **Commit only when asked** (per-task commits on `main`). **C edits → port2 branch
  `jax-mesh-export`, NEVER port2 main** (user); job scripts kept untracked there; otherwise leave
  the port2 repo to the user (no housekeeping). **Large generated files → `/work`** (the `data`
  symlink). Cheap C jobs → `-p compute --time=00:30:00`; long C / JAX-GPU → a real QOS. All
  python/pytest via the env python. See memory [[hpc-job-file-conventions]].

Confirm you've absorbed this, tell me which task you're starting (5.8), then proceed.
