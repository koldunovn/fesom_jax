# Investigation G: the step-1 temperature gap under sea ice against FESOM2 e96a2b13 (user-approved 2026-10-25)

Context: `docs/plans/20261025-recom-port.md` Tasks 4b/5 (lane P results) and `docs/recom/FORTRAN_REFERENCE.md`
(the physics-matched twin `pi_30d_jan_jaxphys`). Branch `physics/ice-tgap` off `recom/port`, worktree
`/work/ab0995/a270088/fesom_jax_dev/wt_phys_tgap`. Time box: 6 hours wall. Effort: high.

## The symptom

`scripts/recom/physics_floor.py` (job 28420384): JAX `configs/pi_recom.yaml` (core2_full physics on pi: zstar,
classical TKE, mEVP, GM/Redi, opt_visc 7, CG SSH, no w_split, dt 2700, l_mslp) started from the Fortran twin's
step-0 state, against the Fortran twin (FESOM2 e96a2b13 + REcoM, physics namelists in
`scripts/recom/fortran/namelists_jaxphys/`, dumps at mstep 1, 2, 3, 17, 960 in
`/work/ab0995/a270088/recom_ref/pi_30d_jan_jaxphys/dumps`): Kv bitwise, uv 1.1e-6 rel, w 1.8e-7 rel, a_ice 2.7e-7
rel, but **T differs by 3.4e-3 °C after step 1** (1.4e-3 psu in S), 0.10 °C after step 2, in the top four layers of
ice-covered high-latitude columns (worst column 203, 77°N; 6352 cells above 1e-6, 117 above 1e-3). The gap enters in
the tracer step (advection, GM/Redi, the surface BC under ice: ice-ocean heat/salt flux, shortwave penetration).

## Steps

- [ ] G1 reproduce with `physics_floor.py`; restrict to the worst columns; list which dumped stages the twin has for
      T and S (the dump catalogue lists per-tracer stages `tra_in/tra_ab/tra_adv/tra_redi/tra_tstar/tra_impl/
      tra_out` and the `transport` fields — check whether T/S are included, or only the BGC tracers)
- [ ] G2 localise: compare JAX's T after each part of the tracer step with Fortran's (advection, Redi, T*, the
      implicit diffusion with its surface BC `bc_T` and `sw_3d`); if the twin's dumps lack a stage, extend lane F's
      instrumentation in the Fortran clone `/work/ab0995/a270088/fesom2_recom` (branch `recom-ref-dumps`; the dump
      module + hooks, `scripts/recom/fortran/patches/` — add a patch, keep it env-gated and a no-op without the
      variable) and rerun 1-3 steps of the twin into a NEW run directory under `/work/ab0995/a270088/recom_ref/`
- [ ] G3 name the cause: a JAX deviation from FESOM2 e96a2b13 (file:line on both sides), or an upstream FESOM2 change
      since the version fesom-jax mirrors (name the commit: `git -C /work/ab0995/a270088/fesom2_sp_zstar log`), or an
      input difference of the twin setup (namelist, chlorophyll source, ice IC)
- [ ] G4 if it is a JAX deviation: the fix + a test on `physics/ice-tgap`, the DP-lock impact reported (do not
      regenerate the baseline); if it is an upstream change or a setup difference: no code change, write it up
- [ ] G5 lesson in `docs/PORTING_LESSONS.md`; commits

## Rules

- **Nothing heavy on the login node** (the user's rule): pytest, XLA compiles, model runs and Fortran builds/runs on
  compute nodes (`-A ab0995 -p compute --hint=nomultithread`, `--time=30:00` where it fits) or an interactive
  allocation; login node: git, grep, reading, sacct. Wait on Slurm with a background shell `until` loop on `sacct`.
- `/work/ab0995/a270088/fesom2_sp_zstar` is read-only. Never delete, scancel, push, overwrite another run's directory,
  or touch other worktrees/branches; English commit messages, no attribution lines, files staged one by one.

## Deliverable

The term and substep where T leaves round-off, the cause with file:line, the fix (or the write-up), DP-lock impact,
node-hours.
