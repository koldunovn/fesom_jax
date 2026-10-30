# Lane I: REcoM's initial conditions (parent plan Task 13, the IC part)

Parent plan: `docs/plans/completed/20261025-recom-port.md` (read Overview, Context, Task 13 and the lane F / lane S results).
Branch `recom/ic` off `recom/port`, worktree `/work/ab0995/a270088/fesom_jax_dev/wt_recom_ic`. Sources (read-only):
FESOM2 `/work/ab0995/a270088/fesom2_sp_zstar/src` (e96a2b13), REcoM `/work/ab0995/a270088/REcoM_src` (692f804).
Time box: 6 hours wall. Effort: high.

## Goal

`fesom_jax/recom/ic.py`: the cold-start BGC state exactly as FESOM2 + REcoM build it — the BGC tracers `(nod2D, nl,
31)` in `fesom_jax.recom.tracers` order (and `bgc_old`), `Alk_surf` (the alkalinity restoring target), the initial
benthos — gated against the Fortran step-1 dump (whose `recom_in` tracers are the IC: `recom()` runs before the first
ocean step).

## What to mirror (verify each point in the source)

- FESOM2 `gen_ic3d.F90` `do_ic3d` for every `tracer_init3d` entry of the BGC IDs: `nc_ic3d_ini`, `getcoeffld`
  (bilinear horizontal + linear vertical), the non-finite → 0, `extrap_nod` (check the current `ic_extrap_det` /
  `ic_extrap_tol` options and their defaults), the cleanup (`> 0.9*dummy` → 0, below `nlevels_nod2D` → 0).
  `fesom_jax/phc_ic.py` ports an earlier version of this path for T/S (`_load_one_variable`, `_extrap_nod3D`
  serial order): reuse what is still identical, port what changed, and say which in the docstring. The Fortran
  reference ran on ONE rank (`dist_1`, identity node order), so the serial extrapolation order is the one to match.
- REcoM `recom_init.F90` (~lines 100-140, 579-750): Fe ×1e9, the treatment of the other tracer IDs (unit
  conversions, `get_tracer_init_value` constants for tracers without an IC file, DICremin), the hydrothermal-vent Fe
  mask/cap, anything else `recom_init` does to the state or allocates with a non-zero initial value (benthos).
- `oce_setup_step.F90:~1272`: `Alk_surf` = the IC surface Alk (before any restart read; never from a restart).
- The IC file list and variable names lane F used: `docs/recom/FORTRAN_REFERENCE.md` and
  `scripts/recom/fortran/namelists/namelist.tra` (`tracer_init3d`); pi files in
  `/work/ab0995/a270088/fesom2_sp_zstar/test/input/recom/`.

## Steps

- [ ] I1 read the code, write down the exact sequence (module docstring)
- [ ] I2 gate first, `fesom_jax/tests/test_recom_ic.py`: JAX IC on the pi mesh with lane F's file list vs the
      step-1 `recom_in` tracers of `/work/ab0995/a270088/recom_ref/pi_30d_jan/dumps` (`io_dump.load_recom_dump`)
      and `Alk_surf`; bitwise or rel 1e-13 (state the reason for anything looser); skips if the dumps are absent
- [ ] I3 `fesom_jax/recom/ic.py`: `bgc_initial_state(mesh, ic_files, switches, params) -> (bgc, alk_surf,
      benthos)` (host numpy is fine: a one-time setup, as `phc_ic`), plus a helper that puts them into a `State`
- [ ] I4 CORE2 smoke (compute node): the same function on our CORE2 mesh (`data/mesh_core2`) with the 1-degree IC
      set in `/work/ab0246/a270092/input/fesom2/recom/` (GLODAPv2 PI TCO2 + TAlk, WOA13 i00/n00, WOA18 o00, PISCES
      Fe) — finite, plausible ranges per tracer (report min/mean/max), node count 126858; and check that
      `/work/ab0246/a270092/input/fesom2/recom/CORE2/DustClimMonthlyAlbani.nc` is in our CORE2 node ORDER (e.g. its
      zero/non-zero pattern or a coastline-dependent field against `data/mesh_core2` coordinates; report how sure)
- [ ] I5 tests on a compute node; commits on `recom/ic`; lesson in `docs/PORTING_LESSONS.md`

## Rules

- New files only (`fesom_jax/recom/ic.py`, `fesom_jax/tests/test_recom_ic.py`) plus the lesson; do not edit
  existing modules (`phc_ic.py` is read and imported, not changed). `/work/ab0246/...` and `/work/bb1469/...` are a
  colleague's directories: read only, copy nothing out of them into the repo.
- **Nothing heavy on the login node** (the user's rule after a crash): pytest, XLA compiles, the CORE2 IC build go
  to a compute node (`-A ab0995 -p compute --hint=nomultithread --time=30:00`, `PYTHONPATH=<worktree>
  JAX_PLATFORMS=cpu`) or an interactive allocation; on the login node only git, grep, reading, sacct. Wait on Slurm
  with a background shell `until` loop on `sacct`.
- Never delete, `rm`, `scancel`, push, or touch other worktrees or branches. English commit messages, no tool/AI
  attribution lines, files staged one by one. Commit finished pieces as you go.

## Deliverable

Commits on `recom/ic`; the gate table per tracer; the function signature; the CORE2 smoke ranges and the dust-order
verdict.
