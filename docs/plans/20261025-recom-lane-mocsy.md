# Lane M: REcoM's carbonate chemistry (mocsy) in JAX, gate G1 (parent plan Task 8)

Parent plan: `docs/plans/20261025-recom-port.md` (read its Overview, Context and Task 8 first).
Branch `recom/mocsy` off `recom/port`, worktree `/work/ab0995/a270088/fesom_jax_dev/wt_recom_mocsy`.
REcoM source (read-only): `/work/ab0995/a270088/REcoM_src` at 692f804. Time box: 6 hours wall. Effort: high.

## Goal

`fesom_jax/recom/carbonate.py`: a pure, jit-able, differentiable f64 port of exactly the mocsy paths REcoM
calls, matching Fortran mocsy at round-off, gated by a standalone Fortran driver built from the same
sources.

## Scope (YAGNI: only what REcoM 692f804 calls)

- Find every mocsy entry point REcoM uses and the option strings it passes: `vars_sprac` from
  `recom_carbonate_chemistry.F90` (the interior water-column call) and `flxco2` from `recom_forcing.F90:~234`
  (air-sea CO2 flux, with whatever it calls: K0, Schmidt number, kw660, p2fCO2, ...), plus any other mocsy
  routine REcoM calls (grep). Port only those paths with REcoM's option values; other option values raise.
- The pH solver: mocsy's solver as REcoM configures it (`mocsy/phsolvers.F90`, stop at `|Δ ln h| < 1e-8`,
  lines ~23 and ~612). Port it as a fixed number of iterations (≥ the worst case seen on the G1 grid, stated
  in the code) in which each point FREEZES once converged, so values equal Fortran's per-point stop. No
  data-dependent Python control flow; `jax.lax` only.
- Element-wise over arbitrary array shapes (it will run over `(nod2D, nl)` and the surface `(nod2D,)`).
- The input-range guards REcoM applies before calling mocsy (e.g. the DIC range guard in
  `recom_forcing.F90`) belong to the caller (parent Tasks 9-10); note them in the module docstring, do not
  port them here.
- No 13C/14C (ciso) paths.

## Steps

- [x] M1 read the call sites and write down the exact signature, units and options REcoM uses (module
      docstring of `carbonate.py`)
- [x] M2 standalone Fortran driver `scripts/recom/fortran/mocsy_driver.F90` + `build_mocsy_driver.sh`: compile
      the needed `REcoM_src/mocsy/*.F90` with REcoM's own Intel flags (`REcoM_src/CMakeLists.txt`:
      `-O3 -r8 -i4 -fp-model precise -no-prec-div ...`, env `env/levante.dkrz.de/shell` of
      `/work/ab0995/a270088/fesom2_sp_zstar`); it reads an input grid (T, S, DIC, Alk, Si, PO4, depth/p,
      Patm, lat, wind for flxco2, ...) from a binary/npz-compatible file and writes every output plus the
      solver iteration count per point. Output under `/work/ab0995/a270088/recom_ref/mocsy_g1/`
- [x] M3 the G1 input grid: realistic ocean ranges plus edges (cold/fresh polar water, warm salty surface,
      deep high-pressure, low and high Alk/DIC ratios, near-zero Si/PO4); a few thousand points
- [x] M4 test first: `fesom_jax/tests/test_recom_carbonate.py` (G1: every output rel 1e-12, pH abs 1e-12;
      skips with a clear message if the driver output is absent), plus unit tests that run without the
      driver (shapes, jit, `jax.grad` finite w.r.t. DIC/Alk/T, frozen-convergence invariance: more
      iterations do not change the result)
- [x] M5 `fesom_jax/recom/carbonate.py` until G1 passes; if a field cannot reach 1e-12, find why (XLA
      reassociation, a transcendental's ulp) and state the measured floor in the test
- [x] M6 commit on `recom/mocsy`; lesson in `docs/PORTING_LESSONS.md`

## Rules

- Driver builds and runs fit on the login node or a `-A ab0995 -p compute --time=30:00` job with
  `--hint=nomultithread`; wait on Slurm with a background shell `until` loop on `sacct`, not by polling.
- Only new files plus the lesson entry; do not edit `step.py`, `state.py`, `run_config.py` or other existing
  modules (the parent session is changing them).
- Never delete, `rm`, `scancel`, push, or touch other worktrees or branches. English commit messages, files
  staged one by one.

## Deliverable (final report to the parent session)

Commits on `recom/mocsy`; the G1 result table (max rel error per output, iteration counts); the exact
signature; anything REcoM does around the mocsy calls that the caller must mirror.
