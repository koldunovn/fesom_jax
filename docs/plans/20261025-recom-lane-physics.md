# Lane P: the physics side of REcoM runs (parent plan Tasks 4b and 5)

Parent plan: `docs/plans/completed/20261025-recom-port.md` (read its Overview, Context, Tasks 4b and 5, and the lane F results
under Task 4). Branch `recom/physics` off `recom/port`, worktree `/work/ab0995/a270088/fesom_jax_dev/wt_recom_phys`.
Time box: 8 hours wall. Effort: high.

## Goal

(1) A pi configuration of our JAX physics for the REcoM verification and its physics-matched Fortran twin with the
REcoM dumps; (2) FESOM2's `l_mslp` (JRA55-do sea-level pressure + the SLP-gradient momentum term) in fesom-jax,
gated against that twin.

## Steps

- [x] P1 `configs/pi_recom.yaml`: the CORE2 production physics of `configs/core2_full.yaml` (zstar, GM/Redi, classical
      TKE, mEVP `whichEVP 1`, `opt_visc 7` with its gammas, CG SSH — no split-explicit, no speed options) on the pi
      mesh, dt 2700, JRA55-do forcing from 1958-01-01, `forcing.mslp: true` (key added in P4), no output by default.
      State every physics choice that differs from `core2_full.yaml` and why.
- [x] P2 (Task 4b) the physics-matched Fortran twin: lane F's instrumented FESOM2+REcoM binary
      (`/work/ab0995/a270088/fesom2_recom`, `docs/recom/FORTRAN_REFERENCE.md`) with lane F's REcoM namelists and
      the physics namelists of our matched CORE2 Fortran setup (`/home/a/a270088/port_jax/docs/CORE2_FORTRAN_SPEC.md`,
      `/work/ab0995/a270088/fesom2_core2/` — read-only) adapted to `configs/pi_recom.yaml` (cvmix TKE, `opt_visc 7`,
      mEVP, zstar, GM/Redi, `l_mslp = .true.`, `step_per_day 32`). 30 days from 1958-01-01 on lane F's `dist_1`, dumps
      at steps 1, 2, 17, 960, into `/work/ab0995/a270088/recom_ref/pi_30d_jan_jaxphys/`. Namelists into
      `scripts/recom/fortran/namelists_jaxphys/`; add the run to `docs/recom/FORTRAN_REFERENCE.md`.
- [x] P3 the physics floor of the pair: JAX `configs/pi_recom.yaml` vs the P2 Fortran run after steps 1 and 2 (T, S,
      uv, eta, hnode, a_ice, Kv) from the same IC (the Fortran step-0 state from the dumps). Report the numbers; do not
      chase a mismatch beyond naming its first field/substep (that decides how G4b is judged).
- [x] P4 (Task 5) `l_mslp`: read JRA55-do `psl.YYYY.nc` exactly like the other fields (time and mesh interpolation as
      FESOM2's `sbc_do` for `l_mslp`, `gen_forcing_couple.F90:680`) through the host and on-device forcing paths
      (`jra55.py`, `surface_forcing.py`, `forcing_local.py`), carried only when `forcing.mslp` is on (no new leaves
      otherwise: a `None` field must not change any pytree or program); the SLP-gradient term of
      `oce_ale_vel_rhs.F90:202` in `momentum.py` (`press_air/1000`, as Fortran — lane F); `run_config.validate`:
      `recom` set ⇒ `forcing.mslp` required (error with the reason).
- [x] P5 tests: `fesom_jax/tests/test_mslp.py` — GP gate (`press_air` on nodes and the momentum term vs the P2 dumps,
      rel 1e-12 starting value), the serial / sharded (pi `dist_4`, 4 host devices) / on-device forcing paths agree,
      `mslp` off ⇒ `test_dp_lock.py` and the identity gates unchanged (compute node: the DP lock needs CORE2).
- [x] P6 commits on `recom/physics`; a lesson in `docs/PORTING_LESSONS.md`.

## Results (2026-10-25)

- P1: `configs/pi_recom.yaml` = core2_full's physics on pi, dt 2700, `forcing.mslp: true`, no output; the
  differences from core2_full are listed in its header (mesh, dt, mslp, outputs). The run driver's pi IC is the
  packaged one; the verification starts from the Fortran step-0 state (`scripts/recom/physics_floor.py`).
- P2: `/work/ab0995/a270088/recom_ref/pi_30d_jan_jaxphys` (job 28420010, 1 node, ~40 min), namelists
  `scripts/recom/fortran/namelists_jaxphys/` (lane F's set + cvmix_TKE, opt_visc 7, no w_split, ice_diff 10, Sweeney
  chl), dumps at mstep 1, 2, 3, 17, 960; `docs/recom/FORTRAN_REFERENCE.md` section "The physics-matched twin".
- P3 (job 28420384): after step 1 forcing 1e-14 rel, press_air 4e-16, a_ice 2.7e-7, m_ice 5e-8, uv 1.1e-6, w 1.8e-7,
  hnode 9e-12, hbar 3e-12, Kv bitwise, T 3.4e-3 degC (1.1e-4 rel), S 1.4e-3 psu (3.7e-5); after step 2 a_ice 1.3e-6,
  uv 2.4e-4, hnode 1.5e-10, Kv 0.1 (a convective switch), T 0.10 degC, S 4.9e-3. The first field off round-off is T
  in the top four layers of ice-covered high-latitude columns (column 203, 77 N) at step 1, entering in the tracer
  step (advection or GM/Redi: Kv is equal and uv/w at 1e-6); not chased further.
- P4/P5: GP passes: the SLP term from Fortran's own press_air at 3.4e-13 rel; the JAX press_air at 4e-16 (steps 1-2)
  and 1.5e-11..4.3e-11 later (the DP evaluation-form difference against FESOM2 e96a2b13's point-slope form, which the
  same bracket reproduces to 8e-16; test tolerance 1e-10, stated in the test). DP lock unchanged; serial = sharded
  (pi dist_4, owned uv 1e-16), on-device tables = host stack (4.6e-16 uv), local forcing bitwise. Job 28420562: 70
  passed (one device, incl. `test_dp_lock`, `test_jra55`, `test_recom_transport`, `test_run_config`,
  `test_partition_state`, `test_forcing_adj`) + 41 passed (four host devices, incl. `test_forcing_sharded`).
- Cost: 1.17 node-hours on `compute` (jobs 28420010, 28420215/295/337/384 floor, 28420224/562 tests).

## Rules

- Edit only what P1-P5 need. `fesom_jax/step.py`: at most the few lines that pass the pressure to the momentum RHS
  (another session edits the REcoM stage there); do not touch `fesom_jax/recom/`.
- Jobs: `-A ab0995 -p compute --hint=nomultithread`, `--time=30:00` where it fits (the 1-rank 30-day Fortran run
  needs ~40 min: `--time=01:00:00`); no GPU. Wait on Slurm with a background shell `until` loop on `sacct`.
- Run worktree code with `PYTHONPATH=<worktree>` (the editable install points at `~/port_jax`); XLA compiles go to a
  compute node (the login node hits the thread limit).
- Never delete, `rm`, `scancel`, overwrite another run's directory, push, or touch other worktrees or branches.
  English commit messages, no tool/AI attribution lines, files staged one by one.

## Deliverable

Commits on `recom/physics`; the P2 run directory; the P3 physics-floor table; the GP result; job ids and node hours.
