# Lane SP: single precision with full REcoM (parent plan Task 16, ends at ⛔ STOP 4)

Parent plan: `docs/plans/20261025-recom-port.md` — read Context, Task 2 (the passive SP estimate), Tasks 10-15 and
their lane results. Physics SP plan: `docs/plans/20261020-core2-single-precision.md` (the mechanism, the f64 list, the
XLA excess-precision guard, the carry, deviations D1-D6). Branch `recom/sp` off `recom/port` (3ea4f49 + 201c101),
worktree `/work/ab0995/a270088/fesom_jax_dev/wt_recom_sp` (`data` is a symlink to `port_jax/data`). Time box: 10 hours
wall. Effort: high. User decision 2026-10-26: production REcoM = `bgc_every` 1 in DP (50-60 SYPD on 4 GH200); the SP
opt-in is the next speed lever, the user decides on it at STOP 4.

## Why

The CORE2 step with full REcoM is transport-bound (nsys job 28430158, `docs/PERFORMANCE_CORE2_GPU.md` "full REcoM on
CORE2"): kernels 122 ms/step, of which REcoM's own kernels ~17 ms; the rest is the FCT/Redi/diffusion of 33 tracers.
SP made the physics-only step ~2x faster (DP 16.7 -> SP 7.4 ms/step); the passive 31-tracer estimate (rounding carry
dropped) was SP 47 vs DP 67 ms/step (Task 2, job 28417702).

## Policy (the user's rules)

- fesom-jax is the shadow of FESOM2: SP = a full repetition of FESOM2 #1207, nothing more (no extra f64 islands).
  Upstream REFUSES `USE_SINGLE_PRECISION` with `RECOM_COUPLED` (FESOM2 `src/CMakeLists.txt:124`), so SP + REcoM is
  itself a deviation: opt-in only, never a default, stated everywhere, the user decides at STOP 4.
- The BGC tracers are tracers of FESOM2's tracer loop: under #1207 every tracer of that loop gets the same scheme as
  T/S (WP values, the WP rounding carry of `MOD_TRACER.F90:42`, content advection with the WP_full `del_ttf`, the
  carried ALE reconstruction, the implicit diffusion with the carry). Mirror `step._tracers_advect_sp` and
  `tracer_diff.impl_vert_diff(..., carry_T=...)` for the stack; reuse their kernels, do not write new ones.
- REcoM's own code stays float64 (the master plan's decision: "stage 1 and mocsy stay f64"): `recom_sources_sinks`,
  mocsy, `recom_tracer_loop_terms` (sinking, benthos, `diff_ver_recom_expl`). Decide and state how their f64 results
  enter the WP stack: the natural choice is to feed every REcoM increment (stage 1's change, the loop terms'
  `dtr_bf`/`vert_sink`/`str_bf`) as f64 content into the single carried rounding, so the SP budget closes like DP's.
  Any other choice is a deviation to list.
- DP stays bitwise what it is: `fesom_jax/tests/test_dp_lock.py` unchanged, every REcoM DP test unchanged.

## Scope

- In: full REcoM, `bgc_every = 1`, the CORE2 production physics (what SP already accepts: zstar without w_split,
  classical TKE, mEVP, GM), serial and sharded, restart (the BGC carry saved like `carry_T`, the physics plan's D6).
- Out: `bgc_every > 1` with full REcoM in SP (refuse it with the reason); SP gradients; the passive mode keeps its
  `sp_estimate` (or gets the real scheme if it falls out for free — say which).
- Remove the SP refusals only for the in-scope combination (`recom.wiring.check_step_inputs`, `RunConfig.validate`).

## Steps

- [x] S1 read: `fesom_jax/precision.py`, `step._tracers_advect_sp`, `tracer_adv.ale_reconstruct_carry`,
      `tracer_adv.advect_one_fct(content=True)`, `gm_redi` `content=True`, `tracer_diff.impl_vert_diff(_one)` carry,
      `recom/transport.py` (`bgc_transport`, `recom_stage3` and its `finish`), `recom/wiring.py` (`full_bgc_step`),
      the SP plan's f64 table; FESOM2 #1207 in `/work/ab0995/a270088/fesom2_sp_zstar/src` (`oce_ale_tracer.F90` the
      tracer loop, `MOD_TRACER.F90`); write the design (where each REcoM increment enters, which arrays are WP/f64)
      into this plan before coding
      ➕ design below (2108bd8; the nested-x64 idea replaced in 73e2171, see the table)
- [x] S2 tests first (CPU, compute node): (a) DP lock + the REcoM DP tests unchanged; (b) one SP step on pi from the
      twin's step-0 state (`fesom_jax.recom.reference.twin_step0_state`, `configs/pi_recom_bgc.yaml` with
      `precision: single`) against the DP step: every BGC tracer's SP - DP relative to the field max, next to T/S's
      SP - DP of the same pair (the f32 floor); (c) the per-tracer SP budget (`recom.budget`) over 4 and 30 steps:
      report the closure (the DP budget closes to <= 7e-15; say what the carry buys); (d) sharded pi `dist_4` == dense
      in SP; (e) SP restart round trip with the BGC carry; (f) the refusals (k > 1, linfs, non-production physics)
      ➕ `fesom_jax/tests/test_recom_sp.py`, 10 tests, numbers in `docs/recom/VALIDATION.md` (SP section). (a) DP lock,
      148 REcoM/precision DP tests, 12 sharded (jobs 28440082/87/89) and equal StableHLO of the DP pi REcoM steps
      (`scripts/recom/dp_hash_recom.py`, 28440099); (b) the step pair is dominated by the physics' own SP - DP (T 7.7e-3
      max, 1.2e-3 rms; DP from the float32-rounded state only 7e-6), BGC <= 4e-2 (DIN), all <= 100 x T/S; ➕ (b2) the
      scheme alone, stage 3 on the G4a transport in SP and DP: <= 2.5e-7 per tracer; (c) carry counted: median 8e-14 /
      2e-14 (4 / 30 steps), worst 1.7e-11 / 8.6e-12; carry dropped every step: 1e-7 / 7e-7 (open item below); (d) DOC
      1.2e-4 = 1015 eps32 vs DP 114 eps64, the rest <= 6e-6; (e) through `run_from_config`, both directions; (f) done
- [x] S3 implement (State `bgc_carry` optional field like `carry_T`, in `precision.CARRY_FIELDS`' handling, the
      restart, the run driver's SP casting); commits as you go
      ➕ 73e2171 (code), 7454287 (tests), 6e65f24 (docs). The restart and driver needed no code: `bgc_carry` is in
      `BGC_FIELDS` (optional on read) and `CARRY_FIELDS` (zero on an SP start, dropped by DP), the writer widens it
- [x] S4 GPU on dolpung (rules below): an A/B timing in one allocation (`scripts/perf/ab_time_gh200.sbatch`, variants
      DP and `--precision single`, `configs/core2_gpu_recom.yaml`, `--restart-in` the v3 1990 restart); then the
      1-year twin DP vs SP with `configs/core2_gpu_recom_v3_1990.yaml` (the dt ramp; plain dt 2700 from that restart
      runs on 2006 forcing), both on the SAME commit, monthly output; `scripts/recom/core2_year_summary.py` on each
      and `scripts/recom/kerr_compare.py <base> sp:<sp run>:<dp run>` (point it at month 12; it compares the last
      month and the end states). The GPU run-to-run floor is in `docs/recom/VALIDATION.md` ("the error of
      bgc_every = 4": k = 1 twice). Report: SYPD, the twin's inventory drift (DIC, Alk, DIN, DFe, O2), CO2 flux,
      NPP, export, pointwise end-state differences, each against the floor and against the physics' own SP - DP
      ➕ from the export `snap_recom_sp_73e2171` (73e2171): A/B job 28439576 (the first, 28438360, ran but lost its
      timing: `gpu_profile.py` needs `<checkout>/.git/HEAD`); DP 28439582, SP 28439583 and a DP repeat 28439750 (the
      year's floor), chained on one node; table in `docs/recom/VALIDATION.md` and the STOP 4 report below
- [x] S5 docs: `docs/recom/VALIDATION.md` (an SP section), `docs/PERFORMANCE_CORE2_GPU.md` (the rows), the parent
      plan's Task 16 items ticked with numbers, a lesson in `docs/PORTING_LESSONS.md`; the deviations list
      ➕ VALIDATION SP section + CORE2 twin table, 4 ledger rows, parent Task 16 ticked, lesson 2026-10-26 (lane SP)
- [x] ⛔ S6 STOP 4: write the report for the user (numbers, deviations, the recommendation with its reasons) into this
      plan and the hand-back; do NOT change any default

## S1 design (2026-10-26, written before coding)

**What upstream has.** Nothing to copy: FESOM2 refuses `USE_SINGLE_PRECISION` with `RECOM_COUPLED`
(`src/CMakeLists.txt:124`), and the two codes would not even agree on the kinds: REcoM's module sets its own
`WP = 8` (`recom_modules.F90:24`, so `Benthos`, `Alk_surf`, `GlodecayBenthos`, `GloCO2flux_seaicemask`, the carbonate
cache and REcoM's `dtr_bf`/`vert_sink`/`str_bf` dummies are real(8), `recom_sinking.F90:461`), while FESOM2 declares the
actual `dtr_bf`, `vert_sink`, `str_bf` as its own WP (`oce_modules.F90:304-305`). The SP + REcoM mode is therefore
defined here, as the #1207 scheme for the tracer loop plus REcoM in its own float64; every choice below that has no
upstream text is listed as a deviation (R0-R6).

**Upstream order of one BGC tracer in the loop** (`oce_ale_tracer.F90:729-930`, SP branch `:861-869`): advection +
Redi fill the WP_full `del_ttf`; REcoM's terms are added to `values` (`values += dtr_bf`, `+= vert_sink`, `+= str_bf`);
the carried reconstruction `acc = values·hnode + del_ttf + carry`, `T* = WP(acc/hnode_new)`,
`carry = WP(acc − T*·hnode_new)`; the implicit diffusion in a WP_full column with its carry update (`:1615-1622`).

**JAX SP step for full REcoM, `bgc_every = 1`** (all new code under `PR.sp()` branches; the DP trace is unchanged):

| piece | precision | how |
|---|---|---|
| `State.bgc`, `bgc_old` | WP (f32) | as T/S |
| `State.bgc_carry` (new, `(nod2D, nl, 31)`) | WP (f32) | the #1206 carry per BGC tracer; `None` in DP and in the passive mode; in `precision.CARRY_FIELDS` (zero on an SP start from a DP restart, dropped by a DP run, written to the restart like `carry_T`: D6) |
| `bgc_benthos`, `bgc_alk_surf`, `bgc_carb`, `bgc_diag`, `bgc_budget` | f64 | REcoM's own arrays (REcoM `WP = 8`); `precision.cast_state` keeps them f64 (`STATE_F64_RECOM`) |
| stage 1 (`recom_sources_sinks`, mocsy), sinking geometry, `recom_tracer_loop_terms`, the 2-D diagnostics, the budget sums | f64 | REcoM's own `WP = 8`: every WP input (T, S, `a_ice`, the forcing fields, the WP geometry, `areasvol`, the REcoM node constants of `ForcingStatic`) is widened exactly with `PR.full` at the boundary, and REcoM's constants are explicit float64 (a dtype-less constructor would be float32 in the x64-off trace; `test_recom_sp.test_sp_step_has_no_implicit_promotion` finds any f32 operand meeting an f64 one). A nested `jax.enable_x64(True)` around REcoM was tried first and dropped: it traces, but the program is lowered once in the x64-off context and int64 index types from the island break the lowering (`jnp.select`'s argmax, job 28437901) |
| the stage-1 output `X1` | f64 | not rounded: the advection reads `V1 = WP(X1)` (FESOM2's WP `values` after `recom()`), and the residual `(X1 − V1)·hnode` reaches the carry through the content below (R1) |
| advection + explicit Redi of `V1` | WP values, f64 `del_ttf` | `step._tracers_advect_sp`'s per-tracer part (`advect_one_fct(content=True)` + `PR.full(gm_redi.*(content=True))` with the WP thicknesses), factored out as `step.tracer_content_sp` and shared by T/S and the BGC stack (vmapped / looped over the tracer axis like the DP stage); `bgc_old ← V1` |
| REcoM's loop terms `dtr_bf`, `vert_sink`, `str_bf` | f64 | computed from `X1` in f64 (REcoM's code), they enter as f64 content: `vmod = X1 + Δ` (R2) |
| T* | f64 acc, WP result | `tracer_adv.ale_reconstruct_carry` on `vmod`: `acc = vmod·hnode + del_ttf + carry`, one rounding, residual into the carry. Every REcoM increment (stage 1's change, the loop terms) and the transport share ONE carried rounding, so the SP budget closes like DP's |
| implicit vertical diffusion with REcoM's `bc_surface` | f64 column, carry | `tracer_diff.impl_vert_diff_one(..., carry=...)` (the T/S kernel) per tracer, f64 `Z3d` (`Z3d_diff`, as T/S); the surface fluxes are REcoM's real(8) values and enter the f64 rhs unrounded (R3) |
| the tracers the later sinking groups read (ballasting density from the finished detritus) | WP | `finish` returns the WP diffused values (FESOM2's WP `values`), widened for REcoM's code |
| halo | as T/S | `V1` halo-complete from the step start; the T*/diffused values at halo nodes are refreshed by the final `exch` of the stack (the carry is node-local, not exchanged, as `carry_T`) |

**Budget / conservation**: the conserved SP quantity is `Σ (X·hnode + carry)·areasvol` (upstream counts the carry as
content, `fesom_conservation.F90:197`); `wiring.inventory(..., carry=)` adds it. Budget terms stay as DP
(`content(X1 − X)`, `content(vmod − X1)`, surface, clip), all f64.

**Restart**: `bgc_carry` written (f64 on disk) when present, read back as WP; a DP restart into an SP run starts it
at zero; a DP run drops it (D6, as the T/S carry). REcoM's restarted f64 fields are unchanged.

**Refusals kept / added**: SP + full REcoM only with `bgc_every = 1` (k > 1 refused: the interval sums and the k-step
stage have no SP scheme); SP physics guards unchanged (zstar without w_split, TKE, mEVP, forced); in
`recom.wiring.check_step_inputs` and `RunConfig.validate`. **Passive mode** keeps its `sp_estimate` (cost estimate,
no carry; not the scheme): the real scheme does not fall out for free (it would need the carry in the passive State
and the every-k sums), so it stays as is, documented.

**Deviations (no upstream text; for the user at STOP 4)**:
- R0. SP + REcoM itself (upstream refuses the build).
- R1. Stage 1's result is not rounded into a WP `values` before the loop: its f32 rounding residual enters the
  carry (an SP build of FESOM2 would round `values` when `recom()` writes back, losing sub-ulp sources and sinks).
- R2. REcoM's loop terms enter as f64 content (in FESOM2 they are added to WP `values` before the carried
  reconstruction, i.e. rounded once more).
- R3. REcoM's surface fluxes enter the WP_full diffusion rhs unrounded (FESOM2's `bc_surface` is a WP function).
- R4. REcoM's node constants on `ForcingStatic` (seafloor area, SAAR columns) pass through the WP cast of the
  forcing constants and are widened in REcoM (one f32 rounding; REcoM computes them in real(8)); the D8 rule.
- R5. REcoM's code reads the WP model state and geometry (T, S, `a_ice`, `zbar_3d_n`, `areasvol`, latitude, the
  forcing) widened, as an SP FESOM2 would hand its WP arrays to a real(8) REcoM (`hnode` is f64 anyway).
- R6. `bgc_carry` is in the restart (as D6 for T/S).
Why R1-R3: a deep-ocean remineralisation increment of DIC or O2 per step (~1e-7 mmol m-3) is ~1e-3 of the f32 ulp of
DIC (~1.2e-4 at 2000 mmol m-3); rounded into WP `values` without a carry it would be lost every step, systematically.

**Tests (S2)**, in `fesom_jax/tests/test_recom_sp.py` (pi, CPU, compute node): (a) DP lock + the REcoM DP tests
unchanged; (b) one SP step from the twin's step 0 vs the DP step, every BGC tracer `|SP − DP|/max|DP|` next to T/S's
of the same pair; dtypes per policy; no implicit f32/f64 mixing outside the REcoM f64 island; (c) the SP budget over 4
and 30 steps with the carry in the inventory, and the negative control (carry zeroed every step); (d) pi `dist_4`
sharded SP vs dense SP; (e) restart round trip through `zarr_output` (carry kept, f64 on disk, back to WP); (f) the
refusals.

## ⛔ STOP 4 report (2026-10-26): single precision with full REcoM — for the user's decision

**What it is.** `precision: single` together with full REcoM (`recom.bgc_every: 1`, the CORE2 production physics) on
branch `recom/sp` (73e2171 code, 7454287 tests, docs after). FESOM2 has no such build (it refuses
`USE_SINGLE_PRECISION` with `RECOM_COUPLED`), so the mode is ours and opt-in; no default changes. The BGC tracers get
the #1207 scheme of T and S (WP values, float64 tendencies, their own rounding carry `State.bgc_carry`); REcoM's own
code stays float64 (REcoM's `WP = 8`) and every REcoM increment enters the one carried rounding as float64 content.

**Speed** (4 GH200, CORE2, dt 2700, the v3 1990 restart):

| | DP | SP | SP/DP |
|---|---|---|---|
| A/B on one node (job 28439576, 2 reps each) | 120.1 ms/step = 61.6 SYPD | 76.6 ms/step = **96.6 SYPD** | 0.64 |
| one year in the production loop (monthly output, CMDBUF on) | 1679 s = 51.5 SYPD | 1196 s = **72.3 SYPD** | 0.71 |
| peak memory per GPU | 14.8 GB | 9.3 GB | 0.63 |

**Fidelity.** pi, CPU: the scheme alone (stage 3 on the same transport fields) agrees with DP to <= 2.5e-7 of each
tracer's max; the budget closes to a median 8e-14 (worst 1.7e-11) with the carry, 1e-7 to 7e-7 without it; sharded =
dense to the precision ratio of the DP pair; restarts carry the BGC carry. CORE2, one year SP against DP (same commit):
global inventories DIC 1.1e-8, Alk -1.3e-9, DIN 4.5e-8, DSi 7.4e-8, DFe -4.4e-7, O2 -4.0e-7; December CO2 flux
+8.5e-5, NPP +4.3e-5, POC export -2.9e-4; annual means equal to three digits (CO2 uptake 2.032 Pg C/yr in both, NPP
47.363 / 47.365, export 5.63 / 5.63). The DP run repeated is bitwise in the physics and within <= 3e-11 in the BGC
inventories, so these differences are not GPU noise: they are the single-precision physics' own divergence from DP
(T rms 1.6e-3 in December, 3.2e-3 at the end; S 4e-5 / 8e-5), carried into the biology. Pointwise, the dissolved
tracers stay below T's rms (<= 1.5e-3), plankton and detritus amplify it (1e-2 to 6e-2). For scale: `bgc_every = 4`
(the other speed lever, 157 SYPD) misses NPP by -8 % and export by -14 % after one month.

**Deviations** (all stated in code and in the S1 design above; R0-R6 are new, D1-D8 are the physics SP's, accepted
2026-10-20/21):
- R0 SP + REcoM itself (upstream refuses the build);
- R1 stage 1's result is not rounded into WP `values` before the tracer loop: its rounding residual goes into the
  carry (an SP FESOM2 would round when `recom()` writes back and lose sub-ulp sources and sinks: a deep DIC/O2
  remineralisation step, ~1e-7 mmol m-3, is ~1e-3 of the float32 ulp of DIC);
- R2 REcoM's loop terms (benthic return flux, sinking, deposit) enter as float64 content (FESOM2: added to WP
  `values`, rounded once more);
- R3 REcoM's surface fluxes enter the float64 diffusion column unrounded (FESOM2's `bc_surface` is a WP function);
- R4 REcoM's node constants on `ForcingStatic` (seafloor area, SAAR columns) pass through the WP cast and are widened
  (one float32 rounding);
- R5 REcoM reads the WP model state and geometry widened (as an SP FESOM2 would hand its WP arrays to a real(8)
  REcoM);
- R6 `bgc_carry` is in the restart (as D6 for T/S);
- not a deviation but a limit: `bgc_every > 1` refuses SP; the passive mode keeps its cost estimate (no carry).

**Open items.** (1) The SP budget closes to <= 1.7e-11, not to DP's 7e-15, in the five tracers that grow by orders of
magnitude from REcoM's tiny initial value (DetCalc, DICremin, PhyCalc, DOC, DON): a small systematic loss, ~1e-11 of
the step's content change, not growing from 4 to 30 steps; stage 3 alone conserves to <= 3e-13, so it enters where
stage 1 meets the tracer step; not isolated (ruled out: Redi, the thickness commit, the wet mask). (2) DOC amplifies
a sharded summation order ~9x more in SP than in DP (1.2e-4 after 2 pi steps). (3) No Fortran anchor exists for this
mode; the only references are our DP run and the physics SP's Fortran zstar pair. (4) One year only: the physics SP
was accepted after a 10-year pair (salt +2e-8 psu a year); the BGC has not had that test.

**Recommendation.** Accept SP + full REcoM as an opt-in for runs where speed matters more than bit-closeness to DP
(spin-ups, ensembles, sensitivity runs): +57 % SYPD in the A/B (+40 % in the production loop), no sign of a leak in a
year (the conserved inventories move like the physics' salt), the biology's annual means unchanged to three digits.
Keep DP the production default (your decision of 2026-10-26 stands; nothing changed). If you want SP + REcoM for
production, the next steps would be a 10-year SP/DP pair (~10 GPU node-h on 4 GH200) with the yearly inventory drift
judged as the physics' T3 criterion, and the budget item (1) closed first. Reasons against: it is a mode upstream
refuses, with no Fortran reference, and the biology's pointwise state differs from DP by 1-6 % in plankton after a
year (the physics' chaos, not a defect, but it means SP and DP runs are not interchangeable at the gridpoint level).

**Cost of the lane:** CPU 1.6 node-h (34 jobs on `compute`), GPU 1.82 node-h on dolpung (= 7.3 GH200-h: two A/B jobs
0.47, three CORE2 years 1.35), one dolpung node at a time.

## Rules

- Nothing heavy on the login node (the user's rule after a crash): pytest, XLA compiles, model runs on compute nodes
  (`-A ab0995 -p compute --hint=nomultithread`, `--time=30:00` where it fits, `--mem=0` for CORE2) or dolpung; the
  login node only for git, grep, reading, sacct. `scripts/recom/run_wiring_tests.sbatch [DEV=4] <pytest args>` is a
  ready test wrapper. 4 in-process CPU devices crash XLA:CPU's all_gather on CORE2 (lesson 2026-10-26): CORE2 on CPU
  runs on one device; the sharded tests use pi.
- **dolpung: at most 2 nodes for the whole user, all sessions** (other sessions' jobs, e.g. `omip-*`, `fjx_*`, count):
  `squeue -p dolpung -u a270088` before every submit, chain with `--dependency`, so this lane holds at most one node
  while another session holds one. Account `mh1571`, `--hint=nomultithread`.
- Wait on Slurm with a background shell `until` loop on `sacct` (exits on any terminal state); never poll from the
  model, never with agents.
- The editable install points at `~/port_jax`: run this worktree's code with `PYTHONPATH=<worktree>` (the sbatch
  wrappers do). No persistent XLA cache (`JAX_COMPILATION_CACHE_DIR` must stay unset).
- Never delete anything (no rm, no scancel, no git clean), never push, never touch other worktrees or branches;
  English commit messages without attribution lines; stage files one by one (never `git add <dir>`).
- Every deviation from FESOM2 is stated in code, plan and lesson, never decided silently.

## Deliverable

Commits on `recom/sp`; the S1 design; the S2 numbers; the S4 SYPD and twin table; the deviations; the STOP 4 report;
CPU and GPU node-hours used.
