# Lane T: REcoM's benthic return flux, ballasting, sinking and benthos (parent plan Task 11)

Parent plan: `docs/plans/completed/20261025-recom-port.md` (read Overview, Context, Task 11 and Task 12, the lane F results
under Task 4 and the lane S results under Task 10). Branch `recom/sinking` off `recom/port`, worktree
`/work/ab0995/a270088/fesom_jax_dev/wt_recom_sink`. REcoM source (read-only): `/work/ab0995/a270088/REcoM_src` at
692f804. Time box: 8 hours wall. Effort: high.

## Goal

`fesom_jax/recom/sinking.py`: the per-tracer REcoM additions inside FESOM2's tracer loop as pure functions, gated
(G3) against the per-tracer stages of the Fortran dumps. Stage 3's step wiring (Task 12) stays with the main session.

## What Fortran does (verify every point in the source; `src/oce_ale_tracer.F90:725-930` in
`/work/ab0995/a270088/fesom2_sp_zstar`, `recom_sinking.F90`)

Per BGC tracer, in tracer order, after advection + explicit Redi and BEFORE the T* update and the implicit diffusion:
1. `diff_ver_recom_expl` for the remin tracers (1001, 1002, 1003, 1018, 1019, 1022): `dtr_bf` from
   `GlodecayBenthos` (stage 1 hands it over in `sfc`, already /86400 — lane S), added to `values`;
2. for the sinking tracers: `get_seawater_viscosity` (T, S after their implicit diffusion, S before its clip),
   `get_particle_density` from the CURRENT detritus tracers (those finished earlier in the loop are final, the others
   still post-stage-1), `ballast` → the scaling factors; `ver_sinking_recom` (TVD-limited sinking flux, CFL cap
   `cfl_safety*dz/dt` — `dt` is an argument: under `bgc_every = k` it is `k·dt`) → `vert_sink`;
   `ver_sinking_recom_benthos` → `str_bf` and the per-tracer `Benthos_tr` buffers (`__usetp` is always compiled with
   REcoM, so `Benthos` is updated after the whole tracer loop, in tracer order);
3. `values += dtr_bf + vert_sink + str_bf`, which enters the tracer as `Δ·hnode/hnode_new` through the T* update.

## Steps

- [ ] T1 read the code above and the dump catalogue (`docs/recom/FORTRAN_REFERENCE.md`: stages `tra_in`, `tra_adv`,
      `tra_redi`, `tra_bf`, `tra_ball`, `tra_sink`, `tra_tstar`, `tra_impl`, `step_end`); write the module docstring
      with the exact sequence and the parameters used (`fesom_jax.recom.params`, `fesom_jax.recom.tracers` from lane S)
- [ ] T2 G3 test first, `fesom_jax/tests/test_recom_sinking.py`: from each dumped tracer's pre-stage state
      (`/work/ab0995/a270088/recom_ref/pi_30d_jan/dumps`, `pi_1d_jul/dumps`, `io_dump.load_recom_dump`), `dtr_bf`,
      the viscosity, the particle densities and scaling factors, `vert_sink`, `SinkingVel`, `str_bf`, `Benthos_tr`
      and the merged `Benthos` match (rel 1e-12 of each field's max, starting value; a looser one needs its reason)
- [ ] T3 `sinking.py`: vectorised over columns (`(nod2D, nl)` arrays, shifts along the level axis, no per-node Python
      loops), `jax.lax` only, jit- and grad-safe; functions per Fortran routine plus one
      `recom_tracer_loop_terms(...)` that runs the whole per-tracer sequence over the stack in Fortran order with the
      density dependence on the detritus tracers made explicit (document what it needs from stage 3: the advected +
      Redi stack, T/S, the geometry, `sfc`). An optional value-preserving batching of the tracers between two
      detritus completions (constant density) only if it is tested against the sequential reference
- [ ] T4 edge cases: one-layer columns, the bottom layer, zero sinking speed, dt (k·dt) large enough to hit the CFL cap
- [ ] T5 tests on a compute node; commits on `recom/sinking`; lesson in `docs/PORTING_LESSONS.md`

## Rules

- New files only (`fesom_jax/recom/sinking.py`, `fesom_jax/tests/test_recom_sinking.py`) plus the lesson; do NOT edit
  `step.py`, `recom/transport.py`, `recom/stage1.py`, the integrators or the config (the main session wires stage 3).
- Mirror REcoM exactly, quirks included; anything you cannot mirror goes into the report and the lesson.
- **Nothing heavy on the login node** (the user's rule after a crash): pytest, XLA compiles, any model run go to a
  compute node (`-A ab0995 -p compute --hint=nomultithread --time=30:00`, `PYTHONPATH=<worktree> JAX_PLATFORMS=cpu`)
  or an interactive allocation; on the login node only git, grep, reading, sacct. Wait on Slurm with a background
  shell `until` loop on `sacct`.
- Never delete, `rm`, `scancel`, push, or touch other worktrees or branches. English commit messages, no tool/AI
  attribution lines, files staged one by one. Commit finished pieces as you go.

## Deliverable

Commits on `recom/sinking`; the G3 table per tracer and dump step; the signature of the loop-terms function and
exactly what stage 3 must hand it and do with its result (where the increments enter relative to T* and the implicit
diffusion, the `Benthos` merge).
