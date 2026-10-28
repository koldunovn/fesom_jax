# CORE2 GPU speed: from 113 SYPD toward fesom3-jax's ~1000

## Status 2026-10-21 (morning) — numbers in `docs/PERFORMANCE_CORE2_GPU.md`

- 4 GH200, CORE2 production loop at dt 2700 with monthly output (chunk 32): **double precision 21.3 ms/step = 347
  SYPD** (core2_gpu.yaml + the placed-constants cache); **single precision + ordered sums + the cache 11.7 ms/step
  = ~630 SYPD** steady (~590 for a 10-year job with its two compiles). Steady dt 1800: DP 20.4, SP 12.9, SP + ordered
  sums 10.7, + command buffers 10.36 ms/step.
- Merged into perf/core2-gpu: ordered sums (opt-in `gpu.ordered_sums`; machine-dependent: DP -13.7 % on A100, +14 %
  on GH200, SP -16 % on GH200), single precision (Task 13, opt-in `precision: single`, DP lock green), the
  placed-constants cache (on with executable reuse, `FESOM_PLACE_CACHE=0` off, bitwise on CPU and GPU), the
  host-thread-independent Chebyshev bounds, `CMDBUF=on` in the run script.
- Waiting for the user: STOP 4 (single precision: the one-year gate passes; the budget criterion, D6/D8),
  STOP 3 for ice every second step (branch perf/core2-ice2, not merged: -7.6 % on A100, and FESOM2 itself moves the
  SH ice volume +38 % by December at `ice_ave_steps = 2`, the port +36 %), which options go into which config.

## Overview

The fesom3-jax port reached ~1020 SYPD for CORE2 on one dolpung node (4 GH200). FESOM2-JAX runs the same mesh at
43.7 ms/step on 4 GH200 at dt 1800 (113 SYPD, production physics, speed options off; JUPITER kernel bench,
2026-07-23: padded halo, constant forcing — so the production loop is somewhat slower). At the
point where the two ports are comparable (double precision, CG, dt 1800, 4 GH200) fesom3-jax took 20.2 ms/step
(16.9 with the SSH solve gathered onto every GPU), so we are ~2.6x slower before any of its algorithm changes.

This plan brings fesom3-jax's measured levers into FESOM2-JAX in three tiers, chosen with the user on 2026-10-20:

- **Tier A (priority): production speed with results at the Fortran level.** Plumbing, kernels, launch counts,
  exchange layout. The CPU replay gates against the C/Fortran references stay bitwise; GPU output stays within the
  GPU's own run-to-run noise.
- **Tier B: physics-level changes, judged statistically** — dt 2700, the sea ice every second step, single
  precision, and (conditionally) split-explicit SSH. Each is checked against Fortran FESOM2 with the same option
  wherever FESOM2 has it, and goes into production only after its twin passes and the user signs off.
- **Tier C: the adjoint.** Must keep working through A and B (no accuracy loss, no memory blow-up); real adjoint
  speed-ups are a later, separate plan. The user, 2026-10-20: "we need adjoint to work" — the adjoint gate runs
  after every tier, and a change that breaks it does not land.

**Isolation (the user, 2026-10-20): this work must not affect the running adjoint experiments.** All of it happens
in the worktree `/work/ab0995/a270088/fesom_jax_dev/wt_perf` (branch `perf/core2-gpu`). Never edit, switch
branches in, or run jobs from the main checkout `/home/a/a270088/port_jax` or any adjoint/ice worktree (`wt_y3`,
`wt_icefarc_*`, `wt_comb`, ...), and never touch their branches.

Hardware: start on Levante A100 (1 node, 4 GPUs, the CORE2 production layout), move to dolpung GH200 as it
becomes available — "whatever gives us compute time fastest". Every change must work on both.

Rough targets (estimates from fesom3-jax's factors, NOT measurements): Tier A ~17-20 ms/step on 4 GH200 at
dt 1800 (250-290 SYPD), ~30 ms on 4 A100 (~165 SYPD); after Tier B ~620-780 SYPD on 4 GH200. 1000 SYPD needs, in
addition, the Pallas margin fesom3-jax found for its split-explicit and EVP steps. 1000 SYPD = 4.93 ms/step at
dt 1800, 7.40 ms/step at dt 2700.

## Context (from discovery)

**fesom3-jax's route** (`~/fesom3_jax/docs/PERFORMANCE.md`, `docs/plans/20261016-gpu-speed.md`; CORE2, 4 GH200,
steady ms/step):

| step | ms/step | SYPD |
|---|---|---|
| dp, CG, dt 1800 | 20.2 (16.9 gathered SSH) | 244-292 |
| + dt 2700, split-explicit + wide halo, EVP/SE inner subcycles unrolled | 12.7 | 583 |
| + single precision | 8.5 | 867 |
| + ice every second step, Pallas SE/EVP kernels, T/S batched, chunk 192 | 7.26-7.4 | ~1020 |

Caveats there: the single-precision fast path leaks salt (10 yr +2.2e-4 psu vs Fortran SP +2.3e-8; suspected XLA
algebraic simplification with excess precision removing f64->f32->f64 convert pairs; branch `sp-leak`), and the
10-year job with output ran at 742 SYPD (compiles + writes).

**fesom3-jax lessons we reuse:**
- profile with `nsys --cuda-graph-trace=node`; the default trace hides kernels inside XLA command buffers;
- the robust comm number is (unprofiled steady step) - (sum of compute kernels);
- Pallas column kernels for tridiagonal solves: the per-level loops cost ~59 ms/step on CORE2 on one A100,
  kernel 210.8 -> 173.7 ms/step; an XLA transpose next to a Pallas call made XLA lay the whole step out
  level-major (83 -> 142 ms) — transposes go inside a Pallas kernel; derivatives via `custom_jvp` whose tangent
  is the reference loops (`~/fesom3_jax/fesom3_jax/ops/column.py`);
- unrolling the EVP inner subcycles: -11 % on 4 GH200 (graph launches 912 -> 200 per step);
- SSH solve gathered onto every device (no collective inside the CG loop): -15 % on 4 A100, an option for small
  meshes only;
- EVP wide halo (one exchange every K subcycles): -6 % at K 8;
- no gain there: XLA command buffers on 4 GPUs, NCCL protocol/channel tuning, `ragged_all_to_all`, merging
  collectives once compute-bound, an EVP fusion kernel (bandwidth-bound: compare kernel time to the byte floor
  first); T/S batching +0.6 % (Kokkos ceiling ~2 %);
- chunk 192 vs 96 ~+1.5 %; host forcing read-ahead in processes (HDF5 is not thread-safe).

**FESOM2-JAX today:**
- CORE2 production physics = zstar + cvmix-TKE + mEVP + GM, dt 1800, `configs/core2_full.yaml`, `partition:
  dist_4`. JUPITER: 1 GH200 93.2 ms/step, 4 GH200 43.7 (`docs/JUPITER_SCALING_FINDINGS.md` §5-6; kernel bench,
  best transport, constant forcing); 4 A100 production physics ≈ 88.0 ms (same doc, line 264).
- ⚠️ `configs/core2_full.yaml` is also the adjoint's reference configuration: `scripts/adjoint/core2_adj.py:84`
  reads it and refuses to run (`:370`) when the Chebyshev preconditioner is on. So the speed options go into a new
  `configs/core2_gpu.yaml` (Tier A production) and `configs/core2_fast.yaml` (Tier B), never into
  `core2_full.yaml`.
- The editable install resolves `fesom_jax` to the main checkout (`/home/a/a270088/port_jax`), and
  `scripts/run_from_config.sbatch:22` `cd`s there: a job started from a worktree runs the main checkout's code
  unless the wrapper sets `PYTHONPATH` to the worktree (`core2_adj.py` has `_check_import_root` for this).
- Adjoint on `main`: the only preset is `AdjointConfig.long_window()` (`adjoint_stab.py:315`); ice adjoint modes
  are exact / rheology / prescribed / frozen (`ice.py:119-167`). The half-year preset `year_window_combined` and
  `frozen_sweep` live on unmerged branches (`adjoint-yr-y3`, ice branches).
- Existing opt-in speed options, combined 1-year climate cert passed with a pooled floor (July 2026): Chebyshev
  CG preconditioner `ssh: {cheb_degree: 3, cheb_kappa: 30}` (`fesom_jax/run_config.py:104-108`, CLI
  `--cheb-degree`), on-device forcing `forcing: {on_device: true}` (commented out at `configs/core2_full.yaml:31`,
  CLI `--forcing-on-device`); comm fusions already default. Production-loop gains on CORE2 -8/-14/-21/-27 %.
- No Pallas anywhere. Tridiagonal solves: `ops.tdma` (`fesom_jax/ops.py:121`, two `lax.scan`s over K + `moveaxis`)
  called from `tracer_diff.py:127`, `momentum.py:378-379` (u, v: same matrix), `gm.py:403-404` (x, y: same
  matrix); TKE has its own `_solve_tridiag` (`cvmix_tke.py:113`) and two mixing-length `lax.scan`s
  (`cvmix_tke.py:87`, `:109`). Vertical sums use `jnp.cumsum`/`lax.cumsum` (`eos.py:159`, `pgf.py:111/153-154`,
  `ale.py:224/253/360`) — profile decides whether they matter.
- mEVP subcycles: `lax.scan` with no unroll (`ice_mevp.py:268-270`; standard EVP the same at `ice_evp.py:279-281`).
  Adjoint modes have their own subcycle handling (`ice_step.py:128`).
- `IceConfig.ice_ave_steps` exists (`ice.py:68`) but only sets `ice_dt` (`ice_step.py:88`); the step itself runs
  the ice every step.
- Halo transport is chosen by CLI flag (`scripts/run_from_config.py:72-86`: `--padded`, `--coloured`,
  `--ragged`; default all_gather). `scripts/run_from_config.sbatch:27` turns XLA command buffers off.
- Benchmark instruments: kernel bench `scripts/bench/bench_forward_scaling.py` (compile-once; `--warmup` is
  accepted but unused), production-loop instrument `scripts/run_from_config.py --progress` (read from chunk 3: two
  executables compile, chunk 2 is inflated), `<mesh>_full_bench.yaml` twins with output off, cert scripts
  `scripts/bench/core2_allon_1yr_cert.sbatch` + `core2_allon_cert_floor_ext.sbatch`.
- dolpung: wrapper `/work/ab0995/a270088/fesom_jax_dev/runs/dolpung/job_gh200.sbatch` (adjoint), read-only aarch64
  venv + our overlay, account mh1571.

**Fortran FESOM2** (`~/fesom27/fesom2/src`): split-explicit SSH exists (`oce_ale_ssh_splitexpl_subcycl.F90`,
`dynamics%use_ssh_se_subcycl`, `MOD_DYN.F90:136`); `ice_ave_steps` exists (`fesom_module.F90:671-686`: forcing
and `oce_fluxes` every step, `ice_timestep` only on update steps, `ice_dt = ice_ave_steps*dt`; the counter
`ice_steps_since_upd` is in the restart, `MOD_ICE.F90:447`). Single precision: FESOM2 PR #1207
(`USE_SINGLE_PRECISION`, WIP in `~/fesom2_sp`). The R2/FORCA20 `fesom.x` binary is built from uncommitted code
(experimental ice FCT) and is not rebuildable; namelist-only changes (`step_per_day`, `ice_ave_steps`) can still
run with it.

**Tools to copy (not import)** from `~/fesom3_jax/scripts/`: `gpu_profile.py` (time / prof / noise / stats, 372
lines), `hlo_kernel_map.py` (nsys or ncu kernels -> source lines and areas via the optimised HLO's stack frames,
202 lines), `nccl_waits.py` (collective cost vs waiting, 72 lines). Kernel references:
`~/fesom3_jax/fesom3_jax/ops/column.py` (tridiagonal + column scan + transpose kernels), `ops/evp_kernel.py`,
`ops/se_kernel.py`.

## Development Approach

- **testing approach: gate-first.** Before each kernel or feature, write the equivalence test: kernel vs the XLA
  reference (Pallas interpret mode on CPU, and on a real GPU in the GPU test job), derivatives vs the reference's,
  the unchanged path bitwise. Then implement.
- complete each task fully before moving to the next; small, focused commits on `perf/core2-gpu`.
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task (success and edge cases).
- **CRITICAL: all tests must pass before starting the next task** — no exceptions.
- **CRITICAL: update this plan file when scope changes during implementation.**
- every Tier A change keeps the CPU replay gates bitwise (Pallas and other GPU-only paths behind one dispatch
  switch; the CPU path keeps today's XLA code) and keeps defaults byte-identical unless the task flips a default
  on purpose.
- GPU jobs run only from a committed worktree HEAD, never from a tree being edited. Jobs use the existing guarded
  wrappers `scripts/adjoint/job_cpu.sbatch` / `job_gpu.sbatch` (`--export=ALL,WT=<wt_perf>`: they `cd $WT`, export
  `PYTHONPATH=$WT`, and exit unless `fesom_jax` imports from `$WT`); any new wrapper (4-GPU, nsys, GH200) copies
  that guard (otherwise the editable install silently runs the main checkout).
- tests run on compute nodes, never on the login node (CORE2 / pytest rule): CPU via `job_cpu.sbatch` (full suite:
  `--cpus-per-task=64 --time=06:00:00 --mem=0`), GPU via `job_gpu.sbatch -m pytest -m gpu ...`.
- every new `test_*.py` is listed in `scripts/runs/suite_groups.txt` (`fesom_jax/tests/test_suite_groups.py`
  fails otherwise; `scripts/tests/` files go in the `scripts` group).
- after each task, append a lesson to `docs/PORTING_LESSONS.md` (standing rule) and a ledger row to
  `docs/PERFORMANCE_CORE2_GPU.md`.
- every GPU job is announced with its cost before submission; Slurm waits are background shell watchers
  (`sacct` every ~10 min, exit on any terminal state), never model-side polling.
- house rules: nothing deleted without an explicit yes; no permission changes; no Workflow tool; commit messages in
  English only, no Claude attribution or session links; never `git add <dir>` (stage files explicitly, check
  `git diff --cached --name-status`).

## Testing Strategy

- **unit tests** (CPU, every task): kernel-vs-reference in Pallas interpret mode; jvp/vjp of the dispatching
  function vs the reference; the default path unchanged; config parsing for every new key.
- **GPU tests** (`pytest -m gpu`, skipped without a GPU; run by `scripts/adjoint/job_gpu.sbatch`): kernels on
  the real device vs the XLA reference to a stated relative tolerance (Triton may contract a*b-c into an FMA, as
  XLA:GPU's own kernels do).
- **existing gates**: `fesom_jax/tests/test_identity_gates.py`, `test_core2_step.py`, `test_mevp.py`,
  `test_ice_adjoint_mode.py`, `test_gradient_sharded.py`, `test_halo.py`, `test_ops.py` must stay green.
- **GPU fidelity**: a change passes when its GPU output differs from the pre-change GPU output by no more than
  the pooled GPU rerun floor (>=4 reruns, `gpu_profile.py noise`), field by field, relative per field.
- **Tier B fidelity**: JAX twins vs the reference configuration, judged against interannual variability and the
  global salt and heat budgets; Fortran runs with the same option for comparison where available.
- **adjoint gate** (after each tier): a 1-day CORE2 gradient (`scripts/adjoint/core2_adj.py`,
  `AdjointConfig.long_window()`, ice `adjoint_mode="rheology"`, run from the worktree) matches the pre-change
  gradient within the GPU repeat spread measured by >=2 repeats BEFORE any change (Task 2); reverse s/step and
  peak memory logged. The adjoint runs with Pallas forced off (Task 4), so this gate checks that nothing broke,
  not speed.
- no UI, no e2e tests.

## Progress Tracking

- mark completed items with `[x]` immediately when done
- add newly discovered tasks with ➕ prefix
- document issues/blockers with ⚠️ prefix
- ⛔ marks a user stop: do not continue past it without the user's answer
- keep the ledger and this plan in sync with the work actually done

## What Goes Where

- **Implementation Steps** (`[ ]` checkboxes): code, tests, measurement jobs, docs in this repo.
- **Post-Completion** (no checkboxes): decisions and work outside this repo (paper, fesom3-jax, Fortran builds).

## Implementation Steps

### Task 0: Branch, worktree and ledger

**Files:**
- Create: `docs/PERFORMANCE_CORE2_GPU.md`

- [x] create branch `perf/core2-gpu` from 1566967 (= `drift-fix-jax` with this plan; `main` is 7bd3915, one commit
  behind) in the worktree `/work/ab0995/a270088/fesom_jax_dev/wt_perf`; `data` symlink as in `wt_y3`
  (2026-10-20; the running adjoint jobs import from `wt_y3`, untouched)
- [x] create the ledger `docs/PERFORMANCE_CORE2_GPU.md`: protocol (steady and production-loop timing, a100_80 pin,
  >=2 reps in one allocation, finite runs only, SYPD at the dt actually run), one table with commit, job id,
  machine, GPUs, dt, ms/step steady, ms/step production loop, SYPD, GPU-h, note
- [x] enter the baseline rows already measured, labelled by instrument: JUPITER kernel bench 1/4 GH200 (93.2 /
  43.7; padded, constant forcing), Levante 4 A100 production physics ≈ 88.0 (`JUPITER_SCALING_FINDINGS.md:264`),
  fesom3-jax's comparison rows (source cited); dolpung gets its own baseline in Task 2
- [x] smoke the guard on the login node: with `PYTHONPATH=$WT`, `fesom_jax.__file__` resolves inside `wt_perf`;
  without it, it resolves to the main checkout (proves the guard is needed); commit

### Task 1: Profiling and noise tools

**Files:**
- Create: `scripts/perf/gpu_profile.py`
- Create: `scripts/perf/hlo_kernel_map.py`
- Create: `scripts/perf/nccl_waits.py`
- Create: `scripts/perf/gpu_profile.sbatch`, `scripts/perf/gpu_profile_gh200.sbatch`
- Create: `scripts/tests/test_perf_tools.py`
- Modify: `pyproject.toml` (register the `gpu` marker under `[tool.pytest.ini_options] markers`)
- Modify: `fesom_jax/tests/conftest.py` (skip `gpu`-marked tests without a GPU)

- [x] write `scripts/tests/test_perf_tools.py` first: `hlo_kernel_map` parsing on a tiny synthetic optimised-HLO
  text + nsys CSV (kernel -> fusion -> source line -> area), the area table for `fesom_jax` module names, and
  `gpu_profile` steady-ms arithmetic (warm/timed split)
- [x] copy `gpu_profile.py` and adapt `time` / `prof` / `noise` / `stats` to our production path (the sharded
  integrator `run_from_config` uses, forcing overlapped as in production, finite check at the end, peak memory)
- [x] copy `hlo_kernel_map.py` with area buckets for our modules: ssh (CG), tracer_adv, tracer_diff (+tdma),
  momentum, pgf/eos (vertical sums), ale (zstar), cvmix_tke, gm/gm_redi, ice_mevp, ice_thermo/ice_adv, halo,
  forcing/jra55, other; launches per step as multiples of CG iterations / mEVP subcycles
- [x] copy `nccl_waits.py` (collective cost vs waiting)
- [x] sbatch wrappers for A100 (`-A ab0995_gpu -p gpu --constraint=a100_80`) and GH200 (from the dolpung adjoint
  wrapper; `--gpus=4`, one node), both with `--hint=nomultithread`, the import guard of `job_gpu.sbatch`, nsys
  `--cuda-graph-trace=node`, HLO dump `--xla_dump_hlo_as_text`
- [x] GPU tests run through `scripts/adjoint/job_gpu.sbatch -m pytest -m gpu fesom_jax/tests` (no new wrapper)
- [x] run the new tests on a compute node - must pass before Task 2 (job 28281705: 38 passed with `test_suite_groups.py` + `test_ops.py`)
- [ ] ➕ the tools' GPU path (the instrumented production run, nsys, HLO dump) is first exercised by Task 2's first job; fix there if it trips

### Task 2: Stage 0 — where a CORE2 step goes

**Files:**
- Modify: `docs/PERFORMANCE_CORE2_GPU.md`

- [x] GPU noise floor: `gpu_profile.py noise` on 4 A100 (jobs 28282325 off / 28282453 on; GH200 pending), 4 runs from the cold start, pooled per-field floor saved
  as JSON next to the ledger (the yardstick for every Tier A change); repeat on 4 GH200 when dolpung has room
- [ ] adjoint baseline: the adjoint gate's 1-day gradient (Testing Strategy) twice on one GH200 at the base commit
  — sets the gate's threshold from the measured repeat spread
- [x] steady + nsys profile, CORE2 prod physics dt 1800, speed options OFF and ON, on 4 A100 (1 A100 pending) (~1 GPU-h)
- [ ] the same on 1 and 4 GH200 (one dolpung node, ~0.5 GPU-h)
- [ ] side measurement: XLA command buffers on vs off (`scripts/run_from_config.sbatch:27`), 1 and 4 GPUs
- [ ] ledger section "Where a CORE2 step goes": per area kernel ms and launches/step; whole step: steady ms,
  kernels/step, compute sum, comm+wait+idle (steady - compute), host-late share; next to fesom3-jax's GH200 table
- [ ] apply the decision rules (Technical Details) and write the resulting Tier A order into this plan
- [ ] ⛔ STOP 1: report the profile and the proposed order to the user; wait for the yes
  - 2026-10-20 reported: half the 4-A100 kernel time is NCCL (all_gather halo + CG), host forcing not overlapped (13-29 ms/step), tridiagonal solves 0.3 ms. Proposed order: A5 transport → A4 gathered SSH → A6 forcing thread → re-profile → A3 / TKE scans / scatter helpers; A2 (Pallas tdma) demoted.
  - ✅ 2026-10-20 the user: "yes, go ahead in that order. 2 should be just an option" — the gathered SSH solve (A4) is opt-in, OFF by default everywhere (not in `core2_gpu.yaml` unless the user says so).

### Task 3: A1 — existing speed options on for CORE2 production (approved 2026-10-20)

**Files:**
- Create: `configs/core2_gpu.yaml` (= `core2_full.yaml` + the speed options; the GPU production config)
- Modify: `fesom_jax/tests/test_run_config.py`
- Modify: `fesom_jax/ice_step.py` or `fesom_jax/run_config.py` (the `ice_ave_steps` guard)

- [x] write the tests first (`test_run_config.py`, where the YAML-loading tests live): `core2_gpu.yaml` loads
  with `ssh.cheb_degree == 3`, `cheb_kappa == 30`, `forcing.on_device is True`, and is otherwise identical to
  `core2_full.yaml`; `core2_full.yaml` still has both off (the adjoint reference, `core2_adj.py:370`); a config
  with `ice: {ice_ave_steps: 2}` raises a clear error
- [x] create `core2_gpu.yaml` with a header comment: production GPU config, the July combined cert, this plan;
  `core2_full.yaml` stays unchanged
- [x] guard: refuse `ice_ave_steps != 1` until Task 12 implements it (today the YAML accepts it and silently
  doubles `ice_dt` while the ice still runs every step, `ice_step.py:88`)
- [ ] production-loop A/B (`core2_full` vs `core2_gpu`, 4 A100, 2 reps, one allocation), ledger row
- [x] run tests (job 28282396: `test_run_config.py` + `test_ice_step.py`, 18 passed) - must pass before Task 4

### Task 4: A2 — Pallas column kernels for the tridiagonal solves

**Files:**
- Create: `fesom_jax/ops_column.py`
- Create: `fesom_jax/tests/test_ops_column.py`
- Modify: `fesom_jax/ops.py` (`tdma` dispatches; multi-rhs entry; static `use_pallas` argument)
- Modify: `fesom_jax/momentum.py:378-379`, `fesom_jax/gm.py:403-404`, `fesom_jax/tracer_diff.py:127-141`
- Modify: `fesom_jax/cvmix_tke.py:113` (and the mixing-length scans at `:87`, `:109` if Task 2 shows they matter)
- Modify: `scripts/adjoint/core2_adj.py` (forces `use_pallas=False`)

- [ ] write `test_ops_column.py` first (fesom3-jax does the same in `tests/test_ops_numerics.py:189-215`):
  - kernel (Pallas interpret mode, CPU) == `ops.tdma` bitwise for random well-conditioned systems with
    identity rows for inactive levels (our callers' contract, `tracer_diff.py:115-118`, `gm.py:392-394`,
    `cvmix_tke.py:266-269`), 1 and 2 right-hand sides, float64;
  - the reciprocal variant == TKE's `_solve_tridiag` bitwise (it computes `fxa = 1/m; cp = c*fxa`,
    `cvmix_tke.py:113-126`, where `ops.tdma` divides);
  - `jax.grad` through the dispatch with `interpret=True` == the reference gradient bitwise;
  - dispatch is OFF on CPU and when `use_pallas=False` (CPU gates untouched);
  - `@pytest.mark.gpu`: kernel vs reference on the device to a stated relative tolerance
- [ ] `ops_column.py`: Thomas kernel in two variants (divide as `ops.tdma`, reciprocal as `_solve_tridiag`), one
  thread per column, `KERNEL_COLS` columns per program, no kmin/kmax (identity rows carry the masking — port
  fesom3-jax's kernel without its active-range logic); Pallas transpose kernel so no XLA transpose sits next to the
  call; `custom_jvp` with the XLA loops as tangent; switch `FESOM_PALLAS=0/1` (default on for CUDA GPUs only)
  plus the static `use_pallas` argument
- [ ] ⚠️ under `jax.grad` the `custom_jvp` tangent re-runs the XLA loops' primal (fesom3-jax `column.py:219-226`),
  so a GPU adjoint with Pallas on would run both; `core2_adj.py` forces `use_pallas=False` (the adjoint is
  unchanged until a Tier C plan)
- [ ] multi-rhs: u/v, GM x/y, and T/S (`impl_vert_diff` builds the same matrix twice, `tracer_diff.py:131-141`;
  the test asserts the two matrices are bitwise equal before sharing) each use one elimination
- [ ] route `momentum`, `gm`, `tracer_diff` and TKE's `_solve_tridiag` through it; keep each call site's
  inputs/outputs unchanged; `tracer_adv.py:543` (w_split only) is covered by the dispatch inside `ops.tdma`
- [ ] CPU gates (`test_identity_gates.py`, `test_core2_step.py`, `test_ops.py`) + GPU tests (`pytest -m gpu`) +
  `test_gradient_sharded.py` green
- [ ] GPU fidelity vs the Task 2 floor; steady + production-loop timing on 1 and 4 A100 (and GH200 when free);
  compile time logged; ledger row
- [ ] run tests - must pass before Task 5

### Task 5: A3 — mEVP subcycles unrolled on the forward path

**Files:**
- Modify: `fesom_jax/ice_mevp.py:268-270` (and `ice_evp.py:279-281` for symmetry)
- Modify: `fesom_jax/ice.py` (`IceConfig` field for the unroll factor, default byte-identical)
- Modify: `fesom_jax/tests/test_mevp.py`, `fesom_jax/tests/test_ice_adjoint_mode.py`

- [ ] write tests first (CPU, bitwise): forward AND `jax.grad` with unroll k (8, full) == unroll 1 for 2 ice
  steps, in ice adjoint modes `exact` and `rheology`, with and without `adjoint_subcycles` set
- [ ] add `unroll=k` to BOTH scans (`ice_mevp.py:265-271`: the body is already `jax.checkpoint`-wrapped and splits
  into two scans when `adjoint_subcycles` is set) behind an `IceConfig` field; default 1 (byte-identical). The
  adjoint driver builds its config from the same YAML and so inherits the setting — acceptable because it is
  bitwise; reverse memory is checked in the adjoint gate
- [ ] measure on 4 A100: factor 8 / 24 / full — steady ms/step AND compile time (cf. the dars zstar compile
  cliff), each with XLA command buffers on and off (the gain depends on `run_from_config.sbatch:27`); pick the
  GPU setting for `core2_gpu.yaml`; ledger rows
- [ ] GPU fidelity vs the floor (should be exact: same arithmetic)
- [ ] run tests - must pass before Task 6

### Task 6: A4 — SSH solve gathered onto every device (opt-in, small meshes)

**Files:**
- Modify: `fesom_jax/ssh.py`
- Modify: `fesom_jax/run_config.py` (`ssh.gathered: bool`, default false)
- Create: `fesom_jax/tests/test_ssh_gathered.py`

- [x] write tests first (fake CPU devices, 4): gathered sharded solve == the dense single-device solve bitwise on
  identical global inputs (zstar on, with and without `cheb_degree 3`); gradient of a scalar of eta matches the
  single-device gradient. NOT bitwise vs today's sharded path: its `psum` reduction order differs, so the
  iteration count may differ — the GPU comparison against today's path uses the noise floor / solver tolerance
- [x] implement: per solve, gather `rhs`, `x0` (`st.d_eta`, `step.py:414`) and `hbar` (the zstar increment is
  rebuilt matrix-free from `hbar` each step, `ssh.py:593-641`); pass the global `SSHOperator` and the mesh geometry
  it uses (`elem_nodes`, `gradient_sca`, `edge_tri`, `edge_cross_dxdy`) into `shard_map` as replicated inputs
  (today only per-device slices exist, `ssh.py:174`); run the single-device CG redundantly; keep the owned part;
  keep `_ssh_implicit`'s `custom_jvp` (`ssh.py:680`); no collective inside the loop
- [x] config key + CLI override, default OFF (the user, 2026-10-20: "just an option")
- [ ] measure 4 A100 / 4 GH200 with and without; ledger rows; it stays OFF in `core2_gpu.yaml` (4 A100 done: job 28283307, -3.9 %; GH200 pending)
- [x] run tests - must pass before Task 7 (test_ssh_gathered 40 passed, job 28283207; test_step_sharded 22 passed + 1 skip in 3:09 h, job 28286185; the rest 71 passed, job 28286186)

### Task 7: A5 — halo transport at dist_4

**Files:**
- Modify: `fesom_jax/run_config.py` (a `halo:` key, CLI flags stay as overrides) — only if the winner is not the
  default (all_gather)
- Modify: `configs/core2_gpu.yaml`, `fesom_jax/tests/test_run_config.py`

- [x] measure all_gather / padded / coloured at 4 A100 (jobs 28283046/47, before Tasks 4-6 per the STOP 1 order): padded ties, coloured -11 % ⇒ keep all_gather (GH200 later)
- [x] the winner IS the default (all_gather) ⇒ no config key needed
- [x] ledger rows
- ➕ [x] compact all_gather transport + unrolled level sweeps (the layout flip, jobs 28286309-11, 28286769, 28287270): `gpu:` block in core2_gpu.yaml, 73.9 -> 56.9 ms on 4 A100
- ➕ [ ] next per the fesom3-jax side-by-side (user 2026-10-20): fewer collectives (mEVP wide halo, batched exchanges, gathered SSH with compact), then assemblies without atomic scatter-adds
- [ ] run tests - must pass before Task 8

### Task 8: A6 — production loop

**Files:**
- Modify: `scripts/run_from_config.py`
- Modify: `fesom_jax/zarr_output.py` / `fesom_jax/ushow_output.py` (writes on a thread)
- Modify: `fesom_jax/tests/` (output and restart tests)

- [ ] write tests first: output files byte-identical with threaded vs synchronous writes; restart bit-identity
  unaffected by chunk length
- [ ] chunk length sweep for CORE2 (96 / 192 / 240 / 480) with forcing-table memory logged; set the default
- [x] forcing overlapped with the device: next chunk's forcing built on one worker thread (1c82624, bitwise the serial build on CPU; job 28284314: host gap 13.2 -> 0.08 ms/step, -12.4 %)
- [ ] fewer distinct executables per
  run (equal chunks per month) — compile is in every job's wall time (no persistent XLA cache, standing rule)
- [ ] a 1-year production-like run with monthly output on 4 A100: job SYPD including compile and writes; ledger
- [ ] run tests - must pass before Task 9

### Task 9: Conditional Tier A items (only if the Task 2 profile shows them)

- [ ] ➕ mEVP wide halo (one exchange every K subcycles, redundant compute on a K-ring) — own sub-plan section
  first
- [ ] ➕ fused CG iteration (2-3 kernels instead of ~11) — check the matvec against the byte floor first
- [ ] ➕ optimisation barriers where the profile shows XLA building giant fusions
- [ ] each: gate-first tests, measurement, ledger row; drop anything under 2 % of the step

### Task 10: Tier A acceptance

**Files:**
- Modify: `docs/PERFORMANCE_CORE2_GPU.md`, `configs/core2_gpu.yaml`

- [ ] 1-year CORE2 run with `core2_gpu.yaml` vs `core2_full.yaml`, judged with the pooled floor (reuse
  `scripts/bench/core2_allon_1yr_cert.sbatch` logic)
- [ ] adjoint gate (Testing Strategy) vs the Task 2 baseline; reverse s/step and peak memory logged
- [ ] full CPU suite (`scripts/adjoint/job_cpu.sbatch`, 64 cores, 6 h) + GPU tests green
- [ ] final Tier A table: 1/4 A100, 1/4 GH200, steady and production-loop, SYPD at dt 1800
- [ ] ⛔ STOP 2: numbers + which options go into `core2_gpu.yaml`; wait for the user

### Task 11: B2 — dt 2700 (Fortran first; can start during Tier A)

**Files:**
- Create: `scripts/perf/compare_twins.py`
- Create: `scripts/tests/test_compare_twins.py`
- Create: `configs/core2_fast.yaml`

- [ ] Fortran FESOM2 CORE2 at `step_per_day = 32` with the R2 binary (namelist change only), 1 year on CPU; if
  stable, 10 years (CPU node-h stated before submitting)
- [ ] write `test_compare_twins.py` first (synthetic monthly series: budget drift, interannual-std yardstick,
  map RMS), then `compare_twins.py` (global salt/heat content, SST/SSS map RMS, ice area/volume NH/SH per month,
  max |diff|/interannual std)
- [ ] JAX 10-year twin at dt 2700 vs dt 1800, both at the Tier A HEAD; the Fortran pair as the comparison
- [ ] `core2_fast.yaml` gets `dt: 2700` only if the twin passes; ledger
- [ ] ⛔ STOP 3 (dt): results to the user before `core2_fast.yaml` is used for anything

### Task 12: B1 — sea ice every second step (`ice_ave_steps = 2`)

**Files:**
- Modify: `fesom_jax/state.py` (ice counter + running means), `fesom_jax/step.py`, `fesom_jax/ice_step.py`
- Modify: `fesom_jax/integrate.py`, `fesom_jax/integrate_sharded.py` (the peeled first step, `:807-822`)
- Modify: `scripts/adjoint/core2_adj.py` (`chunk_fn`), the restart reader/writer
- Modify: `fesom_jax/run_config.py`, `configs/core2_fast.yaml`
- Create: `fesom_jax/tests/test_ice_ave_steps.py`

- [x] read FESOM2's handling: `ocean2ice` runs every step and keeps running means of T, S, ssh, u_w, v_w since
  the last ice update (`ice_oce_coupling.F90:191-239`); the update flag is decided from the counter
  (`fesom_module.F90:671-686`), and the counter is in the restart (`MOD_ICE.F90:447`); ⚠️ user decision:
  fesom3-jax found FESOM2 zeroes `u_w`/`v_w` before averaging — copy the Fortran or fix it
  (2026-10-21: verified in e96a2b13 `ice_oce_coupling.F90:169-174, 194-195` and 4ccdd35a `:213, :238`; MIRRORED,
  not fixed — the fix stays the user's decision; also mirrored: the bulk formulae read the srfoce means every step)
- [x] write tests first: `ice_ave_steps = 1` bitwise unchanged (all gates); with 2: ice update on alternate
  steps, `ice_dt = 2*dt`, the 5 running means match a hand-rolled loop, fluxes from the held ice state on the
  in-between step; a restart written mid-cycle (odd step) resumes bitwise; chunk lengths 239/240 (the driver peels
  step 1, so scan lengths are odd) give the same trajectory (`test_ice_ave_steps.py` + the teacher-forced dist_4
  test in `test_step_sharded.py`, job 28295509; N = 1: the production chunk's StableHLO sha256 is unchanged with
  and without the gpu set, job 28295377; restart mid-cycle at 2e-8, not bitwise — the restart-seam class)
- [x] implement: carry the counter and the 5 running means in State and the restart; split `ice_surface_step`
  into the every-step part (bulk formulae, ocean fluxes, running means) and the update-only part (dynamics,
  advection, thermodynamics). How to skip the update-only part is decided by measurement: `lax.cond` on the counter
  vs a static schedule; XLA:GPU conditionals likely read their predicate on the host (one sync per step; verify),
  so time both on 4 A100 before choosing (b5a38fa, b7203db: `lax.cond`; -7.6 % ≈ half the ~5.9 ms ice update, so
  no static schedule was built; the held thermo outputs are State fields too)
- [x] remove the Task 3 guard for `ice_ave_steps = 2`
- [x] 1-year pairs: Fortran ice-2 vs ice-1 (R2 binary, namelist only) and JAX ice-2 vs ice-1; `compare_twins.py`
  (JAX: ice-2 one year 28295449 vs the ice-1 dt 1800 twin, and vs the same-machine ice-1 half year 28295530 —
  NOT inside the noise; Fortran 28296443/44 from the R2 1989 restart: FESOM2's ice-2 - ice-1 equals the JAX one to a
  few % in every month, December SH ice volume +38 % vs +36 % — the change is the scheme's, not the port's; ledger
  Tier B bullets, `runs/ice2/fortran_pair/pair_table.json`)
- [ ] timing on 4 A100 / 4 GH200; ledger (4 A100 done, job 28295418: 38.56 -> 35.62 ms/step; GH200 not run)
- [ ] ⛔ STOP 3 (ice): results to the user before it enters `core2_fast.yaml`

➕ **Design notes (2026-10-20 night, from the upstream source e96a2b13; not implemented yet):**
- Order per step in `fesom_module.F90`: `ocean2ice` (:1100) BEFORE the update decision (:1129-1135), so it sees the
  PREVIOUS step's `ice_update`/`ice_steps_since_upd`; then `ice_timestep` only when `ice_update` (:1144-1148);
  `oce_fluxes_mom` + `oce_fluxes` EVERY step (:1165-1166) from the last ice-thermo outputs. Init: counter =
  `ice_ave_steps - 1`, `ice_update = .true.` (:600-601); `ice_dt = ice_ave_steps*dt` (`ice_setup_step.F90:56`).
- `ocean2ice` (`ice_oce_coupling.F90:146-196`): on an update step T/S/ssh are the current values, otherwise the
  running mean `(old*k + new)/(k+1)`. ⚠️ **Upstream quirk:** `u_w`/`v_w` are zeroed for every node first (:170-174)
  and then "averaged" `(u_w*k + uw)/(k+1)` (:194-195), so with `ice_ave_steps = 2` the ice step sees `uw/2`, half
  the ocean surface velocity, in the ice-ocean drag. `oce_fluxes_mom` reads the same `srfoce_u/v` EVERY step
  (`ice_oce_coupling.F90:81-83`), so on the skipped steps the ice-ocean stress the OCEAN receives uses
  `uw/(k+1)` too. A shadow port would repeat it; a fix is a FESOM2 change — the user's call.
- Deliberately not implemented on the night of 2026-10-20: ~13 optional State fields (srfoce means, the stored
  thermo fluxes `flx_h`, `flx_fw`, `evaporation`, `ice_sublimation`, `thdgr`, `thdgrsn`, `real_salt_flux`, the
  thermo-entry `a_ice`) with their restart/partition plumbing, while the SP work changes State in parallel; the
  counter can be derived from the absolute step (`step % ice_ave_steps == 0`, a deviation: upstream restarts carry
  it).
- JAX needs new optional State fields (None when `ice_ave_steps == 1`, so DP/adjoint programs stay identical):
  the counter, the srfoce running means (temp, salt, ssh, u, v) and the ice-thermo flux outputs `oce_fluxes` reads
  on skipped steps; `ice_surface_step` split into the ice update (EVP + FCT + cut-off + thermo, under `lax.cond`)
  and the every-step fluxes; the restart carries the new fields (`MOD_ICE.F90:468` writes the counter).
- Expected gain: the ice update is ~2.5 ms of the 21.3 ms GH200 step ⇒ ~-6 % at `ice_ave_steps = 2`.

### Task 13: B3 — single precision

**Files:**
- Create: `docs/plans/20261020-core2-single-precision.md` (sub-plan, written before any code)

- [ ] read FESOM2 PR #1207 (`~/fesom2_sp`) and fesom3-jax's `single-precision` and `sp-leak` branches (read-only;
  cherry-pick ideas onto our own branch only)
- [ ] write the sub-plan: precision switch, double-precision islands (all reductions, CG and its dots, density
  anomaly, hydrostatic pressure, forcing time interpolation, tracer accumulation), how XLA is kept from removing
  the f64<->f32 conversions, conservation gate from day one (1 and 10 years vs DP; vs Fortran SP if it builds)
- [ ] ⛔ user reviews the sub-plan before implementation
- [ ] implement per the sub-plan (its own tasks and tests)
- [x] ⛔ STOP 4: conservation result before any long run uses single precision — user 2026-10-21: SP accepted
  (10-year pair 28304114/5)

### Task 14: B4 — split-explicit SSH (conditional)

- [x] only if, after Tier A, SSH solve + its collectives is >~15 % of the 4-GPU step: write a sub-plan (FESOM2's
  `oce_ale_ssh_splitexpl_subcycl.F90` as the reference; CORE2 only; wide-halo variant), user yes, then implement —
  19.5 % at 817 SYPD; sub-plan `20261021-core2-split-explicit.md` (Sergey's SM AB3-AM4, not FESOM2's FB), user yes
  2026-10-21
- [ ] otherwise record "not needed" with the numbers in the ledger

### Task 15: Fast configuration end to end

**Files:**
- Modify: `configs/core2_fast.yaml`, `docs/PERFORMANCE_CORE2_GPU.md`

- [ ] `core2_fast.yaml` = `core2_gpu.yaml` + the Tier B options the user signed off
- [ ] 10-year production-like run with monthly output on 4 GH200 and on 4 A100: job SYPD incl. compile and
  writes, chunk SYPD; `compare_twins.py` vs the dt 1800 DP reference
- [ ] adjoint gate again on the Tier B HEAD (the adjoint still runs the reference physics; check nothing broke)

### Task 16: Verify acceptance criteria

- [ ] every Tier A item measured, kept or dropped with numbers in the ledger
- [ ] CPU replay gates bitwise; GPU fidelity within the floor; adjoint gate passed
- [ ] full CPU suite + GPU tests green
- [ ] each Tier B option has its twin result and the user's decision recorded

### Task 17: [Final] Documentation

- [ ] `docs/USER_GUIDE.md`: new config keys (`ssh.gathered`, `halo`, ice unroll, `ice_ave_steps`, precision),
  which configuration to use for what
- [ ] `docs/PARALLELISM.md`: CORE2 GPU section updated (the old "JAX ≈ Kokkos" / per-step numbers)
- [ ] `docs/PORTING_LESSONS.md`: one entry per task (as they happen)
- [ ] move this plan to `docs/plans/completed/`

## Technical Details

**Decision rules for the Tier A order (Task 2):**
- tridiagonal solves + level scans + their transposes > ~20 % of the step -> Pallas column kernels first;
- mEVP launches per subcycle dominate the ice share -> unroll first;
- comm + wait + idle > ~25 % of the 4-GPU step -> gathered SSH solve and transport choice first;
- any area < 2 % of the step -> dropped.

**Pallas kernel contract:**
- inputs/outputs in our (entity, level) layout; any level-major copy made by a Pallas transpose inside the call;
- same operation order as the XLA reference; FMA contraction by Triton accepted on GPU (GPU-only path);
- no active-range arguments: inactive levels are identity rows set by the callers (as for `ops.tdma` today);
  two Thomas variants, dividing (`ops.tdma`) and reciprocal (TKE's `_solve_tridiag`);
- `custom_jvp` with the XLA reference as the tangent: forward- and reverse-mode derivatives equal the
  reference's; under `jax.grad` the tangent re-runs the XLA loops, so the adjoint keeps Pallas off;
- switches: `FESOM_PALLAS=0/1` (default on for CUDA, off elsewhere) and a static `use_pallas` argument that
  the adjoint driver sets to False; CPU never traces a Pallas call.

**Timing protocol:**
- steady ms/step: compile excluded, executable reused, long chunk, wall clock with one wait at the end;
- production loop: `run_from_config --progress`, chunks 3+;
- A100 pinned `a100_80`; >=2 reps in one allocation; only finite end states counted; SYPD at the dt run
  (17,520 steps/yr at 1800 s, 11,680 at 2700 s).

**Footprint and budget:**
- dolpung: at most one node at a time from this session (account limit of two, shared with fesom3-jax);
  check `squeue -p dolpung -u a270088` before submitting;
- estimates: Stage 0 ~2 GPU-h; Tier A ~40 GPU-h; Tier B ~20 GPU-h + a few CPU node-h; every job announced with its
  cost; ask before anything approaching ~100 GPU-h.

**Execution:** Tasks 0-3 in the main session. Tasks 4, 5, 6 can run as up to three high-effort agents in their
own worktrees (each with this plan's task as its brief, a deliverable and a time box), only after STOP 1. Tier B
item by item, B2's Fortran run in parallel with Tier A.

## Post-Completion

*Items requiring manual intervention or external systems — informational only*

**Decisions for the user:**
- STOP 1-4 above; the FESOM2 `u_w`/`v_w` averaging question (Task 12); whether `ssh.gathered` and a non-default
  halo go into `core2_gpu.yaml`.
- whether the adjoint gate should use the half-year preset (`year_window_combined`), which needs
  `adjoint-yr-y3` merged into `main` first; the plan uses `AdjointConfig.long_window()` from `main`.

**External:**
- the model paper's performance numbers are NOT changed by this plan; whether a faster configuration enters the
  GMD paper is a separate decision (paper protocol: `scripts/bench/fig10prod/`, levers-off figure).
- results worth telling the fesom3-jax session (e.g. if our profile finds something they missed) go through the
  user; we never touch their branches.
- a later Tier C plan: Pallas backward for the tridiagonal solve (the transposed system), single precision in the
  adjoint's forward recompute, reverse-sweep memory.
