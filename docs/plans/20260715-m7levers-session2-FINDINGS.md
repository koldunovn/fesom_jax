# M7-levers session 2 (2026-07-15 afternoon) — findings, written as the work happens

*Branch `perf/kokkos-m7-levers`. Continues `docs/HANDOFF-20260715-m7-levers-speed.md`.*

## 0. Standing user directives recorded this session

- **Kokkos jobs are the Kokkos session's business — NEVER submit them from here.** 13 jobs I
  submitted per handoff §5 were cancelled on user order ("it's not your business, kokkos session
  will do it when the time comes"); their queue/ledger discipline must not be disturbed. Memory:
  `kokkos-session-owns-kokkos-jobs`. What the paper still needs from their side is listed in §5 of
  the handoff — surface it to the user, read their ledger read-only.
  (Leftover from the cancelled jobs: 5 stub dirs `jaxp_*` + `abcpu./abenv.262746*` out/err files
  under `/work/ab0995/a270088/port2/m7/` — a few KB of inert logs; their session can sweep them.)
- Their session-10 dividend survey (row0 vs h14, jobs 26274324–45) covers post-M7 GPU numbers at
  core2@1-2N, farc@2-8N, dars@2-8N, NG5@4-16N — first harvest: core2@1N h14 0.0754 s/step
  (−30.6 % vs row0). The figure's missing pieces remain: core2@4/8N GPU, Kokkos CPU anchors at
  core2/farc, all 32N points, and the config-matching caption statement.

## 1. Fusion attribution A/B (handoff §4.1) — SUBMITTED, pending queue

`bench_core2_fusion_ab_d8.sbatch` (26274790, 2 nodes) + `bench_dars_fusion_ab_d32.sbatch`
(26274791, 8 nodes): same-allocation main-vs-branch, interleaved ×2, mEVP ON, padded both legs.
Leg isolation via PYTHONPATH (fesom_jax is an EDITABLE install → a worktree leg silently imports
the branch tree otherwise; verified PYTHONPATH wins). Main worktree: `~/port_jax_main_ab`
@ ceea184 (+ `data/` symlink — worktrees don't carry untracked data, see §4).
`bench_forward_scaling` times the warm jitted chunk only ⇒ the clamp fix can't contaminate the Δ.

## 2. On-device forcing interpolation (ladder #1) — IMPLEMENTED, gates in flight

Per the plan (`docs/plans/20260715-fesom-jax-ondevice-forcing.md`), all four order-of-work items:

1. `JRA55Reader.bracket_schedule(dates, nb)` — per-chunk coefficient tables + per-step schedule;
   `_getcoeffld` fires per bracket ROLL. The step() refresh predicate factored into
   `_needs_refresh` (pure refactor). Gate: `test_bracket_schedule_matches_step` — bit vs a step()
   walk AND exactly the step-walk's `_getcoeffld` call count. **GREEN** (job 26275851 phase 1).
2. `surface_forcing.combine_step_forcing` + `ForcingTables/ForcingSched/ForcingDeviceConst`;
   `SurfaceForcing.stack_tables`. Gate: `test_combine_device_matches_host_bit` — bit for all
   fields, platform-probed ≤1 ULP allowance ONLY for prec_rain/prec_snow on non-IEEE-divide
   backends (see §3). **GREEN** (26275851 phase 1, 10/10 with the probe).
3. Driver wiring: `run_steps_sharded_forced(..., forcing_tables/sched/const)` — tables are a
   node-sharded per-chunk constant, the scan xs is the tiny replicated schedule, StepForcing is
   built in-scan (fuses into the step). `run.py` behind `forcing: {on_device: true}` (default OFF
   ⇒ byte-identical); `--forcing-on-device` on `run_from_config.py`. Fixed in passing: the reuse
   cache key was missing `use_coloured` (latent, pre-existing).
4. Gates: `test_fold_forcing_tables_places_nodes` (host wiring, green locally);
   `test_forcing_tables_matches_seq_serial` (tables == device-built seq BYTE + ≤1e-9 vs the
   production host-seq) + `..._npes2` (dist_2, full folded byte) — **definitive run: job
   26276016** (earlier runs raced my own edits). GPU A/B + integration-level identity gate ready:
   `bench_core2_ondevice_forcing_ab.sbatch` + `configs/core2_full_bench.yaml` (240 steps,
   48/chunk, interleaved ×2; the two legs' end-of-run restarts must be bit-identical on GPU).

Scope: applies to the global-host-build path = CORE2/dars/FORCA20 production chains (they don't
use `--local-forcing`). The NG5 local-forcing path raises for now (next increment).

## 3. FINDING: XLA CPU's f64 divide is not correctly rounded

`x / 1000.0` on XLA CPU is 1 ULP off vs numpy on ~13 % of operands — a RUNTIME divisor operand
and `--xla_cpu_enable_fast_math=false` don't help (it's the backend's vectorized divide itself);
CUDA `div.rn.f64` is IEEE. Mul-add at cancellation magnitudes, subtract, and the whole g2r
rotation ARE bit-exact. Consequence for any jnp-vs-numpy bit contract with a division: bit on
GPU, ≤1 ULP on CPU. Divisors are kept as runtime operands anyway (`ForcingDeviceConst.prec_div`)
so no backend constant-folds them into reciprocal multiplies. Lesson:
`ONDEVICE_FORCING_XLA_CPU_DIVIDE` in PORTING_LESSONS.md.

## 4. FINDING: `test_forced_multistep_owned_matches[2]` fails PRE-EXISTING, identically everywhere

uv owned max|Δ| = 8.969e-03 > the 1e-7 clean budget — BYTE-IDENTICAL failure value on
main@ceea184 (26275895), clean b296ee1 (26275933) and the dirty branch (26275876). So: not the
fusions, not the forcing WIP (and the identical number re-confirms the fusions change nothing on
this path). The test was never in the branch's green set (26267782 = jra55/transports/ssh/EVP
only = the "171"). Read: stale calibration — 3 forced-KPP steps put FCT tracer flips ~1e-4 into
the state and uv inherits it one step later, but uv sits in the test's CLEAN (1e-7) field set.
Deselected in `ondevice_forcing_gates.sbatch` with a pointer here; recalibrate separately (move
uv/uvnode/Kv/Av to the soft set for this test, or compare fewer steps). Do NOT gate the merge on
it. Also learned: **git worktrees don't carry untracked `data/`** — the have_forcing skip made a
worktree leg look green (rc=0, "1 skipped"); `data/` is now symlinked into both worktrees
(`~/port_jax_main_ab`, `~/port_jax_b296`).

## 5. RESULTS (landed 16:00–16:45)

- **Sharded-step tail GREEN: 26273635, 13 passed rc=0 (2 h 11 m)** ⇒ the branch's merge gate
  passed; the gated state (`293293a`) merged to main per the handoff.
- **Fusion attribution BANKED (CORE2-8, 26274790): main 79.32/79.26 → branch 72.55/72.44
  ms/step = −8.6 %**, same allocation, interleaved ×2, mEVP+padded, reps agree to 0.1 %.
  dars-32 leg (26274791) still queued.
- **FINDING (extends §3): XLA FMA-contracts `rdate·coef_a+coef_b` INSIDE jit** — ~1e-9 rel on
  the forcing fields (100 % of elements; eager does NOT contract, so eager bit-gates pass while
  in-scan values differ). `--xla_allow_excess_precision=false`, `optimization_barrier`, bitcast
  round-trips all fail to prevent it. Seed growth measured: T rel ~1.4e-7 over 7 KPP steps;
  mEVP sigma rel ~3e-2 over 12 all-on steps (ocean fields ≤4e-5, NO field at O(1) — the
  field-wise profile rules out a bracket/wiring bug). ⇒ **`forcing.on_device` ON is
  VALUE-equivalent, not bit-identical** (same softening precedent as multi-GPU roundoff
  nondeterminism); OFF (default) stays byte-identical. Gates restructured: eager bit-gates for
  values (GREEN, 10/10), `_TABLES_ATOL=1e-6` step gates + slow-field 1e-4-rel wrong-bracket
  tripwires for trajectories (jobs 26278426/27), GPU A/B two-tier identity gate (26278428 —
  PASS-BIT would mean XLA:GPU doesn't contract; measures it either way).
- Driver smoke (26276087): both legs ran the full run_from_config path cleanly (the wiring
  works; host= 0.1–0.5 s/chunk both legs — CPU-side the host build is not the bottleneck);
  restart profile = the FMA-seed picture above.

## 6. GPU A/B RESULT (26278428) — the lever pays −12 % on the production loop

CORE2 all-on (zstar+TKE+mEVP+GM, dt=1800) at dist_4/1 node, 240 steps, 48-step chunks,
interleaved ×2, warm chunks (3–5; chunks 1–2 carry the two compiles):

| leg | host/chunk | device/chunk | per step |
|---|--:|--:|--:|
| off (host stack) | 1.2 s | 4.2–4.3 s | **112.5 ms** |
| on (tables) | 0.6 s | 4.1–4.2 s | **99 ms** |

**⇒ −12 % total step time** (host build halved, device down slightly from the smaller H2D),
reproduced across both reps. All gates green; CPU gates: `test_jra55` 10/10,
`test_forcing_sharded` 7/7 (per-field relative slow-field budgets — an ABSOLUTE budget across
State fields is meaningless, ssh_rhs is O(1e6); job 26278992), driver smoke PASS (26278427).

**Identity on GPU:** the naive off-vs-on compare at 240 steps FAILS 1e-4 budgets — but the
off₁↔off₂ CONTROL (plain rerun, same flag) decorrelates just as much (T rel 5.5e-3, uv 8.6e-3
vs off↔on T 1.2e-2, uv 8.5e-3): the known multi-GPU forced-path run-to-run nondeterminism
([[fesom-jax-perchunk-recompile-and-nondeterminism]]) masks any flag effect at this window.
The A/B's identity gate is now FLOOR-CONTROLLED (flag effect ≤10× the rerun floor per slow
field) — **PASS** on the 26278428 restarts. Whether XLA:GPU FMA-contracts the combine is
unobservable through this noise; short-window correctness lives in the CPU gates.

**Open next:** flip-default decision for `forcing.on_device` (user call — value-equivalent,
not bit-identical, to the host path); NG5 local-forcing increment; CGPOLY/EVPWIDE (ladder #2);
the re-measure envelope (§4 of the handoff) once dars-32 fusion A/B (26274791) lands.

## 7. 1-YEAR CLIMATE CERTIFICATION: PASS (job 26280985)

Three legs (off_1 / off_2 = rerun-noise control / on), CORE2 all-on, full year 1958
(17,520 steps @ dt=1800), annual means of monthly output, per-field max & rms rel:

| field | floor (off₂−off₁) max / rms | on−off₁ max / rms |
|---|---|---|
| temp  | 6.55e-3 / 3.68e-5 | 6.41e-3 / 3.49e-5 |
| salt  | 1.09e-3 / 6.55e-6 | 9.74e-4 / 6.26e-6 |
| u     | 3.82e-2 / 2.21e-4 | 2.89e-2 / 2.05e-4 |
| v     | 3.65e-2 / 2.11e-4 | 2.59e-2 / 1.93e-4 |
| ssh   | 8.95e-4 / 1.55e-5 | 1.08e-3 / 1.56e-5 |
| a_ice | 4.56e-2 / 7.42e-4 | 3.48e-2 / 7.60e-4 |
| m_ice | 1.70e-2 / 2.22e-4 | 1.65e-2 / 2.32e-4 |

**Every field's ON−OFF deviation sits at or below the OFF−OFF rerun floor** (ssh at 1.2×,
well inside the 3× gate): over a full year, switching the flag is statistically
indistinguishable from rerunning the same year — climate-equivalent, certified. The lever is
now: −12 % step time, all short-window gates green, 1-yr certified. Remaining before
flipping any default: the user's call (bit-comparison workflows should keep it OFF).
