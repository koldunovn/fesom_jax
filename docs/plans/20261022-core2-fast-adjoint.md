# The CORE2 adjoint on the fast configuration (CORE2 GPU speed plan, Tier C)

## Overview
- **Why.** The forward runs ~19x faster per model time than in the adjoint's configuration (1 GH200, DP, CG, ice every
  step, dt 1800: ~93 ms/step; the fast stack: 7.39 ms/step at dt 2700). The adjoint gained only the hardware (A100 ->
  GH200, 1.45 -> 0.64 s/step on the year window, job 28336373); the single-device GPU options gave nothing. The
  user (2026-10-22): put the adjoint on the fast configuration, "постепенно и осторожно".
- **What.** The year-window adjoint (`scripts/adjoint/ice_cbar.py tkegrad`, the Y4raw7 setting, J = window-mean MLD
  over natl_subpolar, five controls c_k, c_eps, cd, alpha, k_gm) moved step by step onto the fast stack: ice every
  second step, split-explicit SSH, dt 2700, single precision, then 4 GPUs and the GPU options. Each step is opt-in
  and is judged by the adjoint-vs-FD check of the year-window work (central FD, h 1e-2, error <= 10 %), with FD runs
  of the new configuration (cheap now: a 10-day FD run is under a minute).
- **Order:** the cheap decisive measurement first (where the 640 ms of an adjoint step go), then one physics change
  at a time, from the least to the most risky for the gradient.

## Context
- Branch `perf/core2-adj-fast` (worktree `wt_adjfast`) = adjoint-yr-y3 33ae98d + perf/core2-gpu 0378994 (merge
  dbd5bb0, ten files by hand). The adjoint driver: `scripts/adjoint/core2_adj.py` (`build_model`, `assert_physics`
  admits only the reference physics: DP, CG, dt 1800, ice every step), `scripts/adjoint/ice_cbar.py tkegrad` (the
  year-window runs, `--setting`, `--ice-*`), FD runs `ice_cbar.py fwd/fdbatch`, scoring `runs/comb/tools/fdwin.py`
  (48 steps a day hard-coded).
- Fast-stack pieces and their state for AD: ice every N steps (`lax.cond`, differentiable; `gpu.ice_static` is
  forward-only); split-explicit (`ssh_se`, gradient tests exist, sharded = dense); SP (`precision.py`, the sharded
  runner refuses `return_grad_fn` in SP); the sharded adjoint (`run_steps_sharded_forced(return_grad_fn=True)`,
  validated against the dense gradient in Paper §3; ragged transport refused, padded/coloured AD-correct).
- A100 reference: the Y4 year window (2013-01-01) 7.15 h, 1.45 s/step; GH200 0.64 s/step (job 28336373, cancelled
  at 60 %, the options variant at the same pace).

## Development Approach
- Gate-first; every relaxation of `assert_physics` is a named option, off by default; the reference adjoint stays
  bitwise (the dp_hlo_hash adj_core2 program, the 2-step gradient check `runs/adjfast/grad_cmp.sh`).
- A step is accepted only with: (1) its adjoint vs its own FD at 10 days (all five controls <= 10 %); (2) a speed
  number (s/step, same node, A/B against the previous step); (3) the user's yes at the STOPs below.
- CPU tests on compute nodes; GPU work on dolpung, at most one node of this session (two with fesom3 counted).

## Implementation Steps

### Task 0: the merged branch is both of its parents
- [x] merge perf/core2-gpu into the adjoint line (dbd5bb0)
- [x] the dense 2-step adjoint gradient bitwise adjoint-yr-y3's (CPU, job 28341427: 5 leaves bitwise)
- [x] the fast forward on pi bitwise perf/core2-gpu's (CPU, job 28341428: 60 restart arrays bitwise)
- [x] both sides' unit tests: SP / ice-2 / run config / mesh / state / restart 68 passed (28341435), SE 48 passed
  (28341436); the TKE-cut tests' pinned jaxpr hashes moved with perf's column-kernel dispatch (the scans' moveaxis
  reordered; the merge traces perf/core2-gpu's program exactly) -> re-pinned (0815708)
- [x] ➕ `core2_adj.load_state` refused the merged State (the SP carry fields): fixed (1168889)
- [x] the Y4raw7 adjoint on GPU within the repeat spread of adjoint-yr-y3: 12-day window (the year base has
  boundaries every 12 days; the 1-day job 28341444 had none), job 28341619: J bitwise, the five gradients within
  5e-5 relative of y3's (y3's own two repeats differ by up to 3e-5), 1.0 s/step both

### Task 1: where an adjoint step goes
- [ ] nsys of two reverse chunks of the Y4raw7 window on one GH200 (forward replay vs reverse, by area:
  mEVP frozen sweep, TKE, CG and its implicit adjoint, tracers, remat recompute); ~1 GPU-h
- [ ] ledger section; decide the order of Tasks 2-5 by the numbers

### Task 2: ice every second step in the adjoint
**Files:** Modify `scripts/adjoint/core2_adj.py` (`FAST_PHYSICS`, `with_fast_physics`, `assert_physics(cfg, fast)`,
`build_model(fast=...)`), `tke_cbar.py` / `ice_cbar.py` (`--fast`, the base-run check compares the pieces); Create
`scripts/tests/test_core2_adj_fast.py`; runs in `runs/adjfast/t2/`.
- [x] the option (`--fast ice2`, the value from core2_gpu_fast.yaml); the start State gets the ice-cycle fields;
  CORE2 2-step ice2+se adjoint vs FD 3.5e-10 (job 28341899, 5 tests passed)
- [x] ⚠️ a 10-day J_MLD check with one h is not a bar: the REFERENCE fails it too (c_k +14 %, alpha +21 %, c_eps FD
  ~0; job 28341980) -> the protocol the reference passed: 30 days from 2014-01-01, h 1e-2 and 3e-2
- [x] 30 days (job 28342331): reference 0.3-0.8 %, ice2 0.3-2.7 % (k_gm 2.7 %); adjoint 0.761 -> 0.690 s/step (-9 %),
  forward 0.124 -> 0.099 s/step (-20 %), one GH200 each, four processes on the node
- [ ] ⛔ STOP: table to Nikolay (given in the session 2026-10-22)

### Task 3: split-explicit SSH in the adjoint
- [x] `--fast se` (SE tables, flat rings from a CG restart, kept from an SE restart; a CG window from an SE
  restart refused)
- [x] 30 days from 2014-01-01 (job 28342332): se 0.01-2.3 %, ice2+se 0.4-2.2 % against FD (the reference 0.3-3.5 %);
  but on ONE GPU the SE adjoint is SLOWER: 0.846 s/step (ref 0.761), ice2+se 0.799 (ice2 0.690), while the forward
  is faster (0.109 / 0.089 against 0.124 / 0.099): the reverse of 60 checkpointed substeps costs more than the CG's
  implicit adjoint solve. On 4 GPUs the CG's global reductions may turn this around -> both on 4 GPUs (jobs
  28344098 ice2+se, 28344555 ice2)
- [ ] ⛔ STOP: table to Nikolay

### Task 4: dt 2700 in the adjoint
- [x] `--fast dt2700`; the tools turn days into steps at the window's dt (`core2_adj.fast_dt`); a window refuses
  a restart written at another dt (its AB2 history, `dt_stage`)
- [ ] the start State: the production driver with `dt_ramp` from the 2013-01-01 hindcast restart, one day with
  ice2+se in DP -> `runs/adjfast/spin/fesom.2013.002.00000` (dt_stage 2700; job 28342168)
- [x] the start States `runs/adjfast/spin/fesom.2013.002.00000` and `fesom.2014.002.00000` (jobs 28342276, 28342439)
- [x] ice2+se+dt2700 from 2014-01-02, two repeats (jobs 28342821 + 28345209): ⚠️ fd_batch.py turned days into steps at
  dt 1800, so the FD covered 1440 steps = 45 days (fixed); the adjoint over those 45 days: all five within 1.15 %
  (k_gm 0.01-0.11 %), the repeats agree to 2e-5 (c_k)...5e-4 (k_gm). Speed without compile: 0.650 s/step = 20.8 s
  per model day (ice2+se at dt 1800: 30.9; reference 29.1)
- [ ] ⛔ STOP

### Task 5: 4 GPUs (sharded adjoint chunk, double precision)
Swapped with the SP task by Nikolay 2026-10-22 ("меняем таски местами, да"): SP lives on the sharded path only.
- ⚠️ the sharded runner's gradient path (`return_grad_fn`, NN-GM) differentiates the whole window and passes no
  `adjoint_cfg`: the year-window cuts (Y4raw7) and the ice adjoint settings would not apply. Needed: a sharded
  chunk function `chunk_fn(carry, xs, u)` for `adjoint.chunked_grad` (folded State + accumulators, costs as global
  sums over owned lanes, boundaries gathered to the host)
- [x] `fesom_jax/adjoint_sharded.py` (the chunk under shard_map, folded carry, run constants through xs and
  `adjoint.keep`: a closed-over array is embedded in the HLO as a constant), `build_model(devices=N)`, `--devices`
- [x] CPU, 2 steps, Y4raw7 (test_core2_adj_sharded): with ice2+se the forward equals the dense one to 1e-12 and the
  production sharded forward to 4e-14; the gradient equals the dense one to 2.8e-9-4.9e-7 (job 28343746). With the
  CG no two programs agree to round-off (same trip counts, iterates differ at the 1e-5 stop: 2e-5 after two steps)
- [x] ➕ the sharded SE gradient was NaN: a pad lane's velocity is unbounded (pad Coriolis 1.0 s^-1), a dead
  viscosity slot read lane 0, its coefficient overflowed, 0 x inf in the reverse pass -> `ssh_se` masks the dead
  slots' velocity difference before the coefficient (21aa839; forward unchanged, SE tests 94 passed)
- [x] GPU, 30 days from 2014-01-01, 4 GH200: ice2+se 0.313 s/step without compile (1 GPU 0.643, x2.05), gradients
  within 2e-5...2e-3 of the 1-GPU ones (job 28344098); ice2 (CG) 0.338 (1 GPU 0.547, x1.62; job 28344555) -> SE
  stays (on 4 GPUs it is the faster). ⚠️ the 30-day SE boundary had NaN on every pad lane of the U ring (the pad
  Coriolis): the state-cotangent norm was NaN, the control gradients not -> pad lanes keep U (36c0151)
- [x] ice2+se+dt2700 on 4 GH200 with its own FD, 30 days from 2014-01-02 (job 28346791): 0.10-1.71 % against FD;
  0.321 s/step without compile = 10.3 s per model day = 1.04 h per year (+396 s compile, 73 programs); the trace
  norms finite (the pad fix). Reference on one GH200: 29.1 s per model day (2.96 h/year), on one A100 ~7 h/year
- [ ] ⛔ STOP

### Task 6: single precision on the sharded adjoint chunk, then the GPU options
- SP exists only on the sharded forward path (`run_steps_sharded_forced`: host casts of the State, forcing,
  Params and mesh; the jitted call inside `precision.scope`); the dense step was never run in SP, and
  `jax.device_put` inside `precision.scope` narrows float64 host arrays to float32 (probe: hnode and the cost
  accumulators would be lost): placement outside the scope, the jitted fwd/bwd calls inside it.
- [x] the chunk in `precision.scope("single")`, compiled with `precision.compiler_options` (the library's jit helpers
  read `chunk_fn.precision`), constants / carry / forcing cast on the host as the production runner, Params cast to
  WP inside (the controls float64), the sums float64, point-slope forcing; dense SP refused (c1bff2e)
- [x] CPU (job 28346856): the forward equals the production SP sharded forward to 1e-7; the gradient finite, against
  the DP one 9e-5...6e-2 (cd, the smallest: two models, not the SP adjoint's error)
- [x] ice2+se+dt2700+sp on 4 GH200 with its own FD, 30 days from 2014-01-02 (job 28347180): 0.61-1.11 % against FD
  (the SP FD is noisier at h 1e-2: alpha 4.09 vs 3.96 at 3e-2, k_gm -0.684 vs -0.617; against h 3e-2 alone
  0.7-1.1 %); 0.244 s/step = 7.8 s per model day = 0.79 h per year (+401 s compile, 94 programs); trace norms finite
- [ ] ⛔ STOP: the table to Nikolay (2026-10-22 evening)
- [ ] the GPU options in the reverse program, at dt 1800 (ice2+se+sp, 4 GH200): the profile of one chunk
  (`scripts/perf/adj_profile.py`, job 28350488) and an A/B of the env options on one chunk with the control gradient
  checked (`adj_ab_gh200.sbatch`, job 28351354: SE unroll 10, column unroll, Pallas column kernels, mEVP unroll 12,
  ordered sums, all). The wide halos (SE, mEVP) need their host tables in the sharded chunk: only if the profile
  shows the reverse pass's exchanges to matter; ⛔ STOP
- [x] the profile (job 28350488, no options): kernels 352 ms/step under nsys, NCCL 54 % (950 launches/step: the SE
  substeps' and mEVP subcycles' exchanges, forward, recompute and transpose), ops helpers 10 %, SE 10 %, halo
  packing 6 %, ice thermo 5 %, the rest < 3 % each
- [x] ⚠️ the first A/B (28351354) set four options job-wide (sbatch --export splits commas) -> '+' separator and a
  guard (d13b0da). Clean A/B (28355056), one chunk of 48 steps, fwd + bwd ms/step: none 216.5; all five 132.8
  (-38.7 %); leave one out: column kernel ~10 %, SE unroll ~5 %, ordered sums ~2 %, column unroll ~0.7 %, mEVP
  unroll 0; the chunk gradient moves 4e-4 (SP, another operation order)
- [x] the SE wide halo in the sharded chunk (FESOM_SE_WIDE, cb3ff94): CPU forward 1e-12, gradient 2.8e-9...4.9e-7 of
  the dense (job 28355082); GPU A/B against all (28355248); the ice adjoint's cost by ice_mode frozen (28355270)
  decides whether the mEVP wide halo is worth a wide frozen sweep
- [x] SE wide halo on the GPU (28355248): forward 10.7 -> 9.5 ms/step, reverse 123.2 -> 124.4: nothing on the sum;
  without the ice derivative (28355270) 104.9 ms/step: the ice adjoint ~21 %
- [x] the profile with the five options (28357325, fwd 10.6 + bwd 121.6 ms/step): kernels 229 ms/step, NCCL 51 % (950
  launches), SE 16 % of which the divergence's gather transpose 10.6 % (24 ms, 1067 launches), halo packing 9 %,
  ice ~12 %, the rest < 3 % each
- [ ] ⛔ STOP (2026-10-23 morning): candidates, each opt-in and gated by the sharded-vs-dense gradient: (a) the SE
  divergence with an explicit transpose (an inverse elem -> (node, slot) table, a gather instead of the atomic
  scatter; a new Mesh field), est. -10 %; (b) the mEVP wide halo inside the frozen sweep (K transposed sweeps
  between exchanges), est. part of the ice's 21 %; (c) the GPU options as the adjoint's default in the job scripts

### Task 7: the year window on the fast adjoint
Approved by Nikolay 2026-10-22 ("один потом два": Task 7, then the GPU options), ~21 GPU-h.
- [x] the July start State: one day from states_2013/fesom.2013.182.00000 with dt_ramp -> spin/fesom.2013.183.00000
- [x] Y4raw7, ice2+se+dt2700+sp, 4 GH200, 365 days from 2013-01-02 (job 28348743), chunk 32 (one day), stride 24:
  ⚠️ the reverse sweep BLOWS UP: equal to the reference (dt 1800 DP, the cancelled year run) to ~1 % from day 365
  back to day ~255 (c_k 3.04 vs 2.99, k_gm -0.1875 vs -0.1867), then the state cotangent grows ~x10 per 15 days
  backward (fastest Jun-Sep and Apr-May, x1.3 in January) to 1.5e19 at day 0; carried by S, T, tke, w_e, uv — not
  the ice. The 30/45-day checks were January windows and could not see it
- [x] ➕ DECISION Nikolay 2026-10-22: dt 2700 dropped for the adjoint ("забудь про 2700 ... переходим на 1800": we would
  chase its instabilities forever); the fast adjoint = ice2 + se + sp on 4 GPUs at dt 1800.
- [x] ➕ ⚠️ Nikolay: the SE substeps follow dt — the se piece had kept the dt-2700 M 60 at dt 1800 (dtbt 30 s, 1.5x the
  SE work; the 30-day dt-1800 checks of Tasks 3 and 5 ran with it). Now M = ceil(60 dt / 2700): 40 at dt 1800 = the
  CFL guard's M_min, the production 45 s substep (eb29000)
- [x] year adjoints Jan 2013 (job 28350486) and Jul 2013 (28350487) at dt 1800, M 40, chunk 48: NO blow-up (norm
  grows smoothly to ~1e3 at day 0, as the reference); against the REFERENCE's year FD (dt 1800 DP CG, another
  model): Jan c_k +2.6 %, c_eps +2.8 %, cd 0.0 %, alpha +0.4 %, k_gm -6.9 %; Jul +7.8 / +9.5 / +10.3 / +2.0 / +5.1 %.
  0.213 s/step without compile = 10.2 s per model day = 1.04 h per year (+~410 s compile). The fast config's own
  year FD (h 3e-2): Jan (job 28354435) PASSES: c_k +2.7 %, c_eps -2.9 %, cd +0.2 %, alpha -0.2 %, k_gm +7.0 %;
  Jul (job 28354436) PASSES: c_k +7.6 %, c_eps -8.0 %, cd +9.0 %, alpha +2.1 %, k_gm -4.4 % (the earlier year
  settings left 12-15 % on July starts)
- [x] the profile (28350488) at dt 1800. Held, obsolete: 28350097 (dt-2700 dump), 28350189 (dt 2400), 28348881 (dt-2700 profile),
  28350188 / 28350353 / 28350354 (dt 1800 with M 60)
- [ ] ➕ (superseded) Nikolay 2026-10-22: "на 45 минутах ... разваливается" -> 40 or 30 min, stabler, important for the adjoint:
  the timestep piece takes any dt (`dt2400`, 7daf4df); year adjoints (no FD) Jan 2013 at dt 1800 (ice2+se+sp, job
  28350188) and dt 2400 (job 28350189, spin/dt2400/); the cotangent maps of the dt-2700 run at days 260-220 (job
  28350097); the Jul dt-2700 year (28348848) held
- [x] lessons (567d371)
- [ ] ledger (docs/PERFORMANCE_CORE2_GPU.md: the adjoint section); ⛔ STOP: the table to Nikolay

## Costs (estimates)
- Task 0: done but the checks (~2 CPU node-h, 0.5 GPU-h). Task 1: ~1 GPU-h. Tasks 2-5: per task ~0.5-1 day of work,
  ~2-4 GPU-h (FD 10 runs x 10 days + adjoint 10 days + A/B). Task 6: ~1-2 days, ~5 GPU-h. Task 7: ~10-20 GPU-h.

## Post-Completion
- The year-window decisions of the adjoint handoff (docs/plans/20260927-adjoint-handoff-20261020-halfyear.md) are
  re-taken on the fast configuration; the dense reference adjoint stays available for comparison.
