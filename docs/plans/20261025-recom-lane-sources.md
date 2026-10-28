# Lane S: REcoM's parameters, sources and sinks, and stage 1 (parent plan Tasks 6, 9, 10)

Parent plan: `docs/plans/20261025-recom-port.md` (read Overview, Context, Tasks 6, 9, 10, the lane F results under
Task 4 and the lane M result under Task 8). Branch `recom/sources` off `recom/port`, worktree
`/work/ab0995/a270088/fesom_jax_dev/wt_recom_src`. REcoM source (read-only): `/work/ab0995/a270088/REcoM_src` at
692f804. Time box: 12 hours wall. Effort: high.

## Goal

The column-local half of REcoM in JAX, gated against the Fortran dumps: the parameter set and tracer map (Task 6),
the sources and sinks (Task 9), and stage 1 = REcoM's per-step forcing + sources/sinks + surface fluxes as one pure
function (Task 10).

## Inputs

- Fortran dumps (bin_2p3z2d REcoM on pi, one rank, global numbering = identity): `/work/ab0995/a270088/recom_ref/
  pi_30d_jan/dumps` (steps 1, 2, 17, 224, 225, 960) and `pi_1d_jul/dumps` (1, 17), read with
  `fesom_jax.io_dump.load_recom_dump`; stages `recom_in` / `recom_out` hold everything `recom()` reads and writes,
  incl. the carbonate cache, the surface fluxes and the 2-D/3-D diagnostics (`docs/recom/FORTRAN_REFERENCE.md`).
  Their physics (KPP, opt_visc 5) does not matter here: every gate starts from dumped inputs.
- The REcoM namelists lane F ran: `scripts/recom/fortran/namelists/namelist.recom` (+ `namelist.tra`).
- `fesom_jax/recom/carbonate.py` (lane M): mocsy, with the caller-side rules in its module docstring.

## Steps

- [ ] S0 (Task 6, commit it FIRST and early — another lane builds on it) `fesom_jax/recom/params.py`: a frozen,
      hashable-static + differentiable-leaves split of every REcoM parameter this configuration uses (defaults from
      `recom_modules.F90` `recom_config`, overridden by the namelist); `fesom_jax/recom/tracers.py`: ID → slot map
      (1001-1030, 1037, slot order = Fortran tracer order), the sinking set, the remin set, the IC-file map, the
      bc_surface IDs; `fesom_jax/recom/config.py`: `RecomConfig` gains the namelist path (keep `passive_tracers` and
      `passive_bc` working: the passive mode stays for tests and benchmarks) and raises `NotImplementedError` for
      every out-of-scope switch (list in parent Task 6). Tests `test_recom_params.py`.
- [ ] S1 (Task 9) `fesom_jax/recom/sms.py`: `sms_initialize_variables`, `REcoM_sms`, `sms_update_state`, vectorised
      over `(nod2D, nl)` (light down the column as a cumulative product/sum, no per-node Python loops), REcoM's f64,
      `biostep == 1` asserted (the `any(abs(sms) <= tiny)` reset is then a no-op, parent plan Context); the clip
      `state = max(tiny, state + sms)` and the per-species floors with the clipped amount returned for the budget.
      Gate G2 first: from each dumped `recom_in`, the updated tracers and the 3-D diagnostics of `recom_out` (rel 1e-11
      of each field's max, starting value; a looser one needs its reason in the test). Use the diagnostics to localise
      a failing process.
- [ ] S2 (Task 10) `fesom_jax/recom/forcing.py` + `fesom_jax/recom/stage1.py`: REcoM_Forcing (surface CO2/O2 fluxes
      through `carbonate.flxco2`/`o2flux`, dust Fe/N, `a_ice` masking), the DICremin surface reset, `bio_fluxes`
      (alkalinity restoring with its global-mean subtraction: write the reduction so the caller can pass a psum —
      take a `reduce_sum` callable, dense default `jnp.sum`), the per-level carbonate-cache refresh rule
      (`recom_carbonate_chemistry.F90:236-238`, strict comparisons, as masks), and
      `recom_sources_sinks(bgc, carb, ..., inputs, dt_bgc) -> (bgc_new, carb_new, sfc)` as ONE pure function of
      arrays + a small per-step input record (calendar quantities, dust field, CO2, `press_air`, winds, shortwave,
      `a_ice`). `dt_bgc` is an argument, never the physics dt from a global: the main session is adding the option to
      step the BGC every k-th physics step (`dt_bgc = k·dt`). Gate G2 on the fluxes, `GlodecayBenthos`, `GloHplus`,
      pCO2surf and the cache at the refresh steps 1, 224, 960.
- [ ] S3 tests on a compute node; commits on `recom/sources` (S0 first, then one per step); lesson in
      `docs/PORTING_LESSONS.md`.

## Rules

- New files under `fesom_jax/recom/` and `fesom_jax/tests/` plus `fesom_jax/recom/config.py`; do NOT edit
  `fesom_jax/step.py`, `fesom_jax/recom/transport.py`, the integrators or `run_config.py` (the main session wires
  stage 1 into the step). Code style: match `fesom_jax/recom/carbonate.py` and `fesom_jax/eos.py` (docstrings citing
  Fortran file:line, `jax.lax` control flow only, jit- and grad-safe).
- Mirror REcoM exactly, quirks included; anything you cannot mirror goes into the report and the lesson, never
  silently changed.
- Jobs: `-A ab0995 -p compute --hint=nomultithread --time=30:00`; pytest with `PYTHONPATH=<worktree>
  JAX_PLATFORMS=cpu` on a compute node (the login node hits the thread limit compiling). Wait with a background
  shell `until` loop on `sacct`.
- Never delete, `rm`, `scancel`, push, or touch other worktrees or branches. English commit messages, no tool/AI
  attribution lines, files staged one by one.

## Deliverable

Commits on `recom/sources`; the G2 table (max error per tracer/diagnostic/flux per dump step); the exact signature of
`recom_sources_sinks` and its input record; everything stage 3 or the step wiring must know (e.g. which fields stage 1
hands to the transport, the order of DICremin reset vs the sources/sinks).
