# Lane P, part 2: REcoM's per-step inputs through the forcing paths (parent plan Task 7)

Parent plan: `docs/plans/20261025-recom-port.md` (Task 7; lane S's results under Task 10, which define the inputs).
Branch `recom/inputs` off `recom/port` (after the lane P merge 1b36b26), worktree
`/work/ab0995/a270088/fesom_jax_dev/wt_recom_inp`. Time box: 6 hours wall. Effort: high.

## Goal

Everything stage 1 (`fesom_jax.recom.stage1.recom_sources_sinks`, its `BgcStepInputs`) needs per step that the
step does not already have, delivered through the host stack, the local (per-rank) stack and the on-device tables
paths, serial and sharded, exactly as FESOM2 computes it at the `call recom` point (`fesom_module.F90:~1203`):

- the calendar of that step: `daynew`, `ndpyr` (and `month`, `day_in_month`, `timenew` where the rules below need
  them) — FESOM2's clock at the moment `recom()` runs, verify against `gen_modules_clock.F90` and the dumps;
- `fe_dust` = `GloFeDust`: the Albani monthly climatology `DustClim(month, nod2d)` held per month with FESOM2's switch
  rule (`gen_surface_forcing.F90:sbc_do_recom`, ~lines 1930-2080): update when `mstep == 1` (month = the current
  month) or on the last day of a month at `timenew == 86400` (month + 1, wrapping 12 → 1); units as read;
- `n_dust` = 0 (`useAeolianN` off — assert it), `co2_atm` = `CO2_for_spinup` with `constant_CO2` (a namelist scalar,
  no per-step input).

NOT in scope here: `mstep` (the main session carries it as a State counter, reset at every run start like
FESOM2's), `press_air`/winds/shortwave/`a_ice` (already in `StepForcing`/`State`), and the wiring of stage 1 into
`step()` (main session).

## Design (keep it like your `press_air` work)

- Optional fields, None unless full REcoM is on (`RunConfig.recom.mode == "full"`), so every other program has
  the leaves it had (DP lock unchanged): `StepForcing.fe_dust` `[nod2D]`, and the per-step calendar either as
  node-broadcast `StepForcing` fields (simplest through every partition/fold/scatter helper) or as `ForcingSched`
  scalars on the tables path — your call, say why. The dust table on the tables path like the sss/chl month tables.
- The dust file path: the REcoM namelist's `nm_fe_data_file`, overridable by a `recom.dust_file` key in the YAML
  (the bin_2p3z2d namelist carries Albedo paths). CORE2: `/work/ab0246/a270092/input/fesom2/recom/CORE2/
  DustClimMonthlyAlbani.nc` (a colleague's, read-only); pi: `/work/ab0995/a270088/fesom2_sp_zstar/test/input/recom/
  DustClimMonthlyAlbani_pimesh.nc`.

## Steps

- [x] N1 read the FESOM2 clock + sbc_do_recom code; write the exact per-step rules into the docstring
- [x] N2 gate first, `fesom_jax/tests/test_recom_inputs.py`: the per-step calendar and `fe_dust` equal the `recom_in`
      values of lane F's dumps (`/work/ab0995/a270088/recom_ref/pi_30d_jan/dumps` steps 1, 2, 17, 224, 225, 960;
      `pi_1d_jul/dumps` steps 1, 17; `pi_30d_split` segment 2 steps 1, 2 — the mstep==1 dust rule) through the host
      stack; host stack == local stack == tables path (bitwise); sharded pi `dist_4` == serial; across a month end
      (synthetic dates) the switch happens on FESOM2's step
- [x] N3 the implementation (`surface_forcing.py`, `forcing_local.py`, `shard_mesh.py`, `integrate_sharded.py`,
      `run.py` chunk driver, `run_config.py`)
- [x] N4 `test_dp_lock.py` unchanged with REcoM off; the passive mode unchanged; tests on compute nodes; commits on
      `recom/inputs`; a lesson in `docs/PORTING_LESSONS.md`

## Results (2026-10-25)

- Where the inputs are inside `step()`: `step_forcing.fe_dust` (`[nod2D]` float64, local lanes on the sharded path,
  = `GloFeDust` of THIS step, i.e. after this step's `sbc_do_recom` update, umol m-2 s-1 as in the file),
  `step_forcing.daynew` and `step_forcing.ndpyr` (int32 scalars, replicated: FESOM2's clock of THIS step after
  `call clock`, i.e. the step END: the last step of a day still has that day's `daynew`). All three are None unless
  `build_surface_forcing(..., bgc_dust_path=...)` / `build_local_forcing(..., bgc_dust_path=...)`; the run driver
  passes `RunConfig.bgc_dust_path()` (full REcoM only). `mstep`, `n_dust` (0) and `co2_atm` (`CO2_for_spinup`)
  are not per-step forcing.
- Choice: the calendar as `StepForcing` scalars (`surface_forcing.STEP_SCALARS`, never partitioned, `[T]`
  replicated scan `xs` on the per-step path) and as `ForcingSched.daynew/ndpyr` on the tables path, whose
  schedule is already a replicated `[T]` structure; a node broadcast would add two `[T, nod2D]` fields (+20 % of
  the host stack on CORE2) to carry two integers. The dust is a node field (`fe_dust`), on the tables path a
  month table `ForcingTables.dust [nm, nod2D]` indexed by `ForcingSched.didx`, like sss/chl.
- The dust file: `forcing.dust_file` in the YAML, else the namelist's `nm_fe_data_file` when absolute. The plan
  said `recom.dust_file`; that needs a `RecomConfig` field, and `fesom_jax/recom/` is read-only for this lane.
- The hold: `BgcForcing` mirrors `sbc_do_recom` (update when the step ends at 24:00 of a month's last day, loading
  month + 1, or at the segment's first step, loading the current month); the segment start is the first step the
  object is asked for (one object per run job, as `run_from_config.py` builds it), later steps replay the rule.
  Limitation: it assumes a constant `dt` (no `dt_ramp`).
- Gate (job 28421337): calendar and `GloFeDust` equal, bitwise, at pi_30d_jan mstep 1, 2, 17, 224, 225, 960,
  pi_1d_jul 1, 17, pi_30d_split segment 2 mstep 1, 2; month-end, year-end, segment-on-last-step and non-dividing-dt
  rules; host = local = tables bitwise; on the sharded pi dist_4 path (host stack and tables) the values reach
  `step()` (a probe around `step`). DP lock unchanged.

## Rules

As before: nothing heavy on the login node (compute nodes or an interactive allocation for every pytest/compile/
run); do not touch `fesom_jax/recom/` except reading, nor `step.py` beyond what threading a new optional
`StepForcing` field strictly needs; never delete, scancel, push, or touch other worktrees/branches; English commit
messages, no attribution lines, files staged one by one; commit finished pieces as you go.

## Deliverable

Commits on `recom/inputs`; the gate result; exactly where the inputs appear inside `step()` (field names, units,
which step's value) so the main session can build `BgcStepInputs` from them.
