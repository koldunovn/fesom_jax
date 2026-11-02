# Lane W: full REcoM in the step and the run (parent plan Tasks 10-wiring, 12-wiring, 13, 14)

Parent plan: `docs/plans/completed/20261025-recom-port.md` — read ALL of it (Context; Tasks 2b, 7, 10, 11, 12, 13, 14 and the
lane results under them). Branch `recom/wiring` off `recom/port` (8ac4a0c), worktree
`/work/ab0995/a270088/fesom_jax_dev/wt_recom_wire`. Time box: 12 hours wall. Effort: high.

## What exists (all gated against the Fortran reference, merged on `recom/port`)

- `fesom_jax/recom/stage1.py` `recom_sources_sinks(bgc, carb, benthos, alk_surf, temp, salt, geom, inputs,
  dt_bgc, params, *, reduce_sum, n_iter) -> Stage1Result(bgc, carb, sfc, benthos, sms)` (lane S, G2 at round-off);
- `fesom_jax/recom/transport.py`: `recom_stage3(...)` (full stage 3: batched advection + Redi, lane T's
  `sinking.recom_tracer_loop_terms` with `finish` = T* + implicit diffusion with `surface_bc`), the passive
  `bgc_transport`, and the BGC-every-k machinery `init_every_k` / `bgc_every_k` (passive only so far);
  `test_recom_stage3.py` G4a is xfail until FESOM2 #944 is in (branch `physics/upstream-944-1116`, in progress);
- `fesom_jax/recom/ic.py` `bgc_initial_state(...) -> BgcIC(bgc, bgc_old, alk_surf, benthos, hplus)` (lane I, bitwise);
- `fesom_jax/recom/sinking.py` (lane T, G3 at round-off), `carbonate.py` (lane M), `params.py`/`tracers.py`;
- the per-step inputs (lane P part 2): `step_forcing.fe_dust` `[nod2D]`, `step_forcing.daynew` / `.ndpyr` (int32
  scalars, the clock at the step END), None unless the forcing is built with `bgc_dust_path`; `forcing.mslp` /
  `step_forcing.press_air` (lane P); `cfg.bgc_dust_path()`; `RunConfig.recom.mode == "full"` (namelist);
- `step()` refuses full mode today (`NotImplementedError`), runs the passive stage after the T/S stage.

## Steps

- [x] W1 State (optional `BGC_FIELDS`, entity-leading): `bgc_carb` (the 8 cached carbonate fields), `bgc_benthos`
      `(nod2D, 4)`, `bgc_alk_surf` `(nod2D,)`, `bgc_mstep` `(nod2D,)` int32 (FESOM2's `mstep`: reset to 0 at EVERY
      run start, never taken from a restart; under `bgc_every = k` it counts BGC steps — a stated deviation for k > 1);
      the carbonate cache is not restarted either (Fortran: refreshed at `mstep == 1`), `Alk_surf` = the IC surface
      Alk (Fortran recomputes it from the IC each run; keeping it in the restart is value-identical — say which you do)
- [x] W2 the per-run REcoM constants that stage 1 / stage 3 need and the step cannot build from traced mesh fields
      (lane T's SAAR columns from node lon/lat, the seafloor area, the latitude, `area_top`): carry them through the
      serial AND sharded paths (partitioned like `ForcingStatic`; choose the carrier, keep REcoM-off programs
      byte-identical — the DP lock), built once per run on the host
- [x] W3 `step()` full mode: stage 1 at the Fortran position (after the ice step and the ocean flux coupling,
      before the dynamics; T/S of the step start, `a_ice` after the ice step; `BgcStepInputs` from `step_forcing` +
      State; `relax_alk`'s global mean through a psum over OWNED nodes on the sharded path — `area_top` zero on halo
      copies); stage 3 = `recom_stage3` with the step's transport fields, T_new and S BEFORE `S_FLOOR`; the
      `bgc_every = k` mode for full REcoM (stage 1 + stage 3 on the BGC step with `dt_bgc = k·dt` on the mean transport,
      extend `bgc_every_k`); `recom_cfg=None` and the passive mode byte-identical
- [x] W4 the run: `run_from_config` builds the cold-start BGC state with `ic.bgc_initial_state` (pi:
      `ic.PI_IC_FILES`; CORE2: `ic.CORE2_IC_FILES` resolved in `/work/ab0246/a270092/input/fesom2/recom/` — a
      colleague's directory, read only), the restart with the BGC fields (archival restarts; a physics-only restart
      into a REcoM run starts the BGC from the IC), monthly-mean output (parent Task 13: 3-D DIC, Alk, DIN, DSi,
      DFe, O2, total Chl; 2-D CO2 flux, pCO2surf, integrated NPP, export at 100 m; configurable; the ushow schema);
      `configs/pi_recom_bgc.yaml` (pi_recom.yaml + full REcoM) and `configs/core2_recom.yaml`
- [x] W5 gates (tests first where possible): G4b — one full JAX step on pi from the physics-matched twin's step-0
      state (`/work/ab0995/a270088/recom_ref/pi_30d_jan_jaxphys/dumps`) vs the twin for the BGC tracers, judged
      against the physics floor of the same pair (investigation G: with #944/#1116 T ≤ 2e-6 °C after 1-3 steps;
      without them ~3e-3 — mark it xfail until `physics/upstream-944-1116` is merged, then rerun); G5 — the 30-day pi
      run vs the twin (inventories DIC, Alk, DIN, DSi, DFe, O2; CO2 flux; NPP) + the closed budget in JAX (inventory
      change = surface fluxes + benthos exchange + stage 1's clip term); G6 — `jax.grad` of a 1-day NPP cost w.r.t.
      2-3 REcoM parameters vs central FD; sharded (pi `dist_4`, 4 host devices) == dense for one REcoM step
- [x] W6 tests on compute nodes; `test_dp_lock.py`; commits on `recom/wiring` (small, as you go); a lesson; the
      parent plan's Task 10/12/13/14 items ticked with the numbers

## Rules

- Nothing heavy on the login node (the user's rule after a crash): pytest, compiles, runs on compute nodes
  (`-A ab0995 -p compute --hint=nomultithread`, `--time=30:00` where it fits; `PYTHONPATH=<worktree>`) or an
  interactive allocation; login node: git, grep, reading, sacct. Wait on Slurm with a background shell `until` loop.
- Mirror FESOM2/REcoM exactly; any deviation is stated in code, plan and lesson. Never delete, scancel, push, or touch
  other worktrees/branches; English commit messages, no attribution lines, files staged one by one.
- If `physics/upstream-944-1116` lands on `recom/port` while you work, merge `recom/port` into your branch (do not
  cherry-pick pieces).

## Deliverable

Commits on `recom/wiring`; G4b/G5/G6 numbers (and which wait for #944); the constants carrier design; the restart and
output behaviour; open problems.

## Results (lane W, 2026-10-25/26)

- Carrier (W2): the per-run REcoM node constants (seafloor area under each layer, the SAAR columns) are optional
  `ForcingStatic` fields (`wiring.with_recom_static`, host numpy, once per run), so they are partitioned, folded and
  placed like the static forcing on every path; None adds no leaf (DP lock unchanged). The rest comes from the mesh
  (geographic latitude, `area`, `areasvol`, level counts, the global `ocean_area`); `relax_alk`'s sum is
  `reductions.global_sum` (owned nodes + psum) sharded, `jnp.sum` dense. The SAAR `p_ref` table is a trace constant.
- State (W1): `bgc_benthos`, `bgc_alk_surf` (restarted; Alk_surf kept in the restart: value-identical to Fortran's
  rebuild from the IC), `bgc_carb`, `bgc_mstep`, `bgc_diag`, `bgc_budget` (never restarted, `state.RESTART_SKIP`).
- Step (W3): `wiring.full_bgc_step` at the tracer stage, stage 1 from the values of its Fortran position (stated);
  `bgc_every = k` via `bgc_every_k(full=...)`. Run (W4): `run_from_config` builds the cold BGC IC, resets the segment,
  adds the BGC output fields; `configs/pi_recom_bgc.yaml`, `configs/core2_recom.yaml` (CORE2 dust node order still to
  verify, Task 15).
- Gates (W5): G4b, budget, G5, G6, sharded == dense: numbers in `docs/recom/VALIDATION.md` and the parent plan.
