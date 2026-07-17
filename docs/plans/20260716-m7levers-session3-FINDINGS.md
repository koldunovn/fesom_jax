# M7-levers session 3 — findings (2026-07-16, live document)

*Branch `perf/kokkos-m7-levers`. Continues `docs/plans/20260715-m7levers-session2-FINDINGS.md`
per `docs/HANDOFF-20260716-m7levers-session3.md`. Plan: finish the speed ladder, then the paper
re-measure with `forcing.on_device` OFF.*

## 1. dt=120 dars fusion A/B harvested (first action) — WASH, question CLOSED

Job 26299153, bench-finite CLEAN all 4 legs (max_uv=5.752 both legs): main 332.78/333.05 vs
branch 332.93/333.07 ms/step = **+0.05 %**. Confirms the dt=60 wash at the near-production dt.
Banked into the session-2 findings table. The fusions stay quoted as a small-shard/latency-regime
lever (−8.6 % CORE2-8, ~0 at dars-32).

## 2. Reducer selection policy UNIFIED (paper re-measure prerequisite (a))

`paper_jax` commit `57a71d4`: `make_numbers._row` now takes the FASTEST (min `sstep_s`) matching
row — the same policy as `fig_scaling._best_per_ngpu` — so re-runs entering the bench glob can
never split the figure from the text macros (0713 gotcha 1 / 0714 prerequisite (a), done
consciously). Exposed + fixed a latent under-filter: the "MPI-decomposed Kokkos twin" per-node
macros (`kokkosOneNode*/TwoNode*/FourNode*`) matched BOTH backends and leaned on JSON row order —
now pinned `label='Kokkos (CPU)'`. `_numbers.tex` byte-identical on the current `scaling.csv`
(the fix is pure hardening today).

**⚠️ Discovered for the re-measure:** the reducer globs `scripts/logs/bench_*.out` (77 files,
including the OLD contaminated-protocol rows). Under best-per-point selection an old
faster-but-dishonest row would beat an honest new one — **ARCHIVE (move, never delete) the old
`.out` logs out of the glob before `make data` consumes the re-measure** (e.g.
`scripts/logs/pre_m7_protocol/`). Also: the figure/macros pin `halo=ragged` — the transport-
envelope decision (padded ≤16 GPUs, coloured at NG5-64) must drop/replace that pin in the same
paper pass.

## 3. Ladder #1 — NG5 local-forcing on-device increment: DONE (commit `5f462c7`)

`forcing.on_device` now composes with `--local-forcing` instead of raising:
`LocalForcing.stack_tables_partitioned` + `forcing_const_partitioned` build the bracket
tables/trig on the LOCAL sub-mesh (bit-identical to the global tables' local shards —
`bracket_schedule`'s `_gather` is per-node) and scatter `[P, …, Lmax]` via one generic
`_scatter_last`; `run.py` routes the combined path in the chunk loop.

- **Pytest gate GREEN** (26300806, 8 passed + 1 pre-existing deselected):
  `test_local_forcing_tables_equal_global_partition` — strict byte equality of tables + sched +
  const vs the global partition at synth npes=4, plus all existing forcing gates.
- **Driver smoke GREEN (26301615, dist_1): local-tables ≡ global-tables restarts
  BIT-IDENTICAL (leg B == leg C, every field), host-vs-tables within the FMA-seed slow-field
  budget (worst uv 3.8e-5 vs 1e-4).** First attempt (26301028, dist_4 + 4 fake CPU devices)
  SEGFAULTED in ALL THREE legs incl. two that run no new code — an XLA:CPU fake-device ×
  full-all-on-step harness limit, NOT a wiring issue (lesson in PORTING_LESSONS).
  **Increment COMPLETE** — NG5 now gets both host-forcing levers combined when opted in.

## 4. Ladder #2 — CGPOLY (Chebyshev-preconditioned CG): IMPLEMENTED, gates GREEN, A/Bs landing

Commit `00f6e3c`. Degree-k Chebyshev polynomial over the DIAG-scaled operator replaces the
MITgcm M⁻¹ when enabled (opt-in: `ssh: {cheb_degree: k}` config / `FESOM_CG_CHEB=k` bench env;
default OFF ⇒ byte-identical). `(degree, lam_min, lam_max)` ride as STATIC meta on `SSHOperator`
through partition/fold/shard_map; bounds from a fixed-seed host power iteration
(λmax×1.05, λmin=λmax/κ, κ=30 default). Apply = k SpMV+halo, NO dot products.

- **Synthetic sanity (login node):** M⁻¹ symmetric to 3e-16, SPD, bounds cover the spectrum,
  diag/cheb iters ratio 3.6× at k=3.
- **Gates GREEN (26301449, 12/12 incl. all pre-existing S.6):** on the captured REAL CORE2
  step-1 rhs at equal unpreconditioned-residual tolerance:
  **MITgcm 127 iters → cheb k=2: 55 (2.31×) → k=3: 42 (3.02×)** — far past the E.3 ≥1.8×
  verify. Sharded k=3: iteration count device-identical, owned d_eta ≤1e-9 budget.
  Early-stopped iterates agree to 2.3e-5 rel (both are valid soltol-1e-5 solutions —
  CGPOLY ON is solver-tolerance-equivalent, NOT byte-identical ⇒ same class as the C's
  early stop; default-ON would need its own climate cert, per the kokkos E.5 precedent).
- **CORE2-8 A/B (26301616, bench-finite CLEAN ×4, max_uv identical): OFF 72.26/71.78 →
  ON 57.06/57.20 ms/step = −20.7 %.** The "flat at small scale is EXPECTED" spec warning was
  WRONG in the good direction here: with iters÷3 at k=3, total halo exchanges per solve ALSO
  drop (~254 → ~168/step at 127→42 iters) on top of psums 254→84 — and CORE2-8 is the
  latency-bound point (cf. the fusions' −8.6 % from collective count alone).
  Stacked ladder at CORE2-8: 79.3 (pre-branch) → 72.5 (fusions) → **57.1 ms/step (CGPOLY)**.
- **dars-32 dt120 A/B (26301703, bench-finite CLEAN ×4, max_uv identical): OFF 333.56/333.78 →
  ON 321.66/321.52 ms/step = −3.6 %.** The compute-dominated anchor where the fusions WASHED —
  CGPOLY still pays there because it cuts real WORK, not just collective count (net SpMV
  applications drop ~1.5× at k=3 with iters÷3), on top of psums÷3. Regime map so far:
  −20.7 % latency-bound (CORE2-8), −3.6 % compute-bound (dars-32).
- **NG5-64 JUDGE A/B (26301617, bench-finite CLEAN ×4, max_uv identical 0.698): OFF
  515.01/500.25 → ON 460.23/458.00 ms/step = −9.6 %** (−8.2 % against the faster OFF rep —
  the OFF legs spread 2.9 % within-job, the ON legs 0.5 %). **CGPOLY regime map COMPLETE:
  −20.7 % latency-bound (CORE2-8) / −9.6 % at scale (NG5-64) / −3.6 % compute-bound
  (dars-32) — the lever pays EVERYWHERE measured.** Bonus observation: the OFF leg (507.6
  mean, fusions in) vs the pre-branch coloured NG5-64 number (543.3, job 26232225) suggests
  the fusions are worth ~−6.5 % at NG5-64 — the point session 2 left unmeasured — but that is
  a CROSS-allocation comparison (quote cautiously; within-job OFF spread alone is 2.9 %).
- Protocol note: the bench's `--warmup` flag is DEAD (parsed, never used — timing is the 2nd
  call of one compiled N-step run from cold); all A/Bs are the standard cold 25-step protocol,
  comparable with the fusion A/Bs.
- **Ladder Gate-0 leftovers (see docs/plans/20260716-levers-on-scaling-figure.md):** CG1R
  default NO (CGPOLY already cut the psum pool 3×); EVPWIDE decide on an NG5-64 phase profile
  (1 extra 16-node job), not speculatively; optional cheb degree/kappa tuning (1 cheap
  CORE2-8 job).

## 5. Ladder #4 — TKE decomp profile: DONE (26301643) — the Kokkos spill does NOT transfer

CORE2 all-on dist_4/1-node, FESOM_REUSE_EXE=1 (the post-recompile-fix protocol): all-on
90.55 ms/step; component deltas — **TKE 2.84 ms (~3 %)**, zstar 5.44, GM 9.36, mEVP ice
14.34, bare ocean 58.03 (matches the 58 ms bench base). The Kokkos `tke_column_loop`
register-spill catastrophe is an ENGINE property, not a physics one: the scan-based XLA
TKE is ~3 % of the step ⇒ no TKE kernel lever needed on the JAX side. Measured, not
assumed — ladder #4 CLOSED.

## 6. RE-MEASURE envelope — rows as they land (protocol: scripts/bench/remeasure/README.md)

**core2 (26301823, all 8 rows bench-finite CLEAN, max_uv=1.120 every row, reps ≤0.5 %):**

| ngpu | transport | per_step rep1/rep2 (ms) |
|---|---|---|
| 1 | (allgather tag; zero halo) | 192.68 / 192.52 |
| 2 | padded | 126.51 / 126.62 |
| 4 | padded | 78.30 / 77.99 |
| 8 | padded | **68.20 / 68.54** |

**farc (26301826, all 6 rows bench-finite CLEAN, max_uv=2.793 every row, reps ≤1 %):**

| ngpu | transport | per_step rep1/rep2 (ms) |
|---|---|---|
| 4 | padded | 310.86 / 311.51 |
| 8 | padded | 205.18 / 205.73 |
| 16 | padded | 163.71 / 165.41 |

Monotone strong scaling (4→8 doubling efficiency 76 %, 8→16 62 %).

**forca20 (26301827, all 4 rows bench-finite CLEAN, max_uv=2.973 every row, reps ≤0.5 %):**

| ngpu | transport | per_step rep1/rep2 (ms) |
|---|---|---|
| 16 | ragged | 438.25 / 440.38 |
| 32 | ragged | 295.34 / 296.53 |

16→32 doubling efficiency 74 %. The production-chain dist_32 COMPILE-HANG did NOT reproduce in
the bench harness (compile ~72 s) — that failure mode is a run_from_config-chain property, not
a forca20-32 property.

**dars 8/16/32 (26301824, all 6 rows bench-finite CLEAN at dt=120 — the 150-step horizon
HOLDS incl. dist_32; max_uv 3.71 every row, reps ≤0.9 %):**

| ngpu | transport | per_step rep1/rep2 (ms) |
|---|---|---|
| 8 | padded | 942.21 / 942.13 |
| 16 | ragged | 497.76 / 495.58 |
| 32 | ragged | 290.04 / 292.55 |

Doubling efficiency 95 % (8→16) and 85 % (16→32).

**dars 64 (26301825, both rows bench-finite CLEAN, max_uv=3.707):** ragged, per_step
242.27 / 243.76 ms — 16→32→64 doubling efficiency 85 % → 60 % (the curve flattens as the
shards shrink to 48k nodes/GPU).

**ng5-32 (26301829, both rows bench-finite CLEAN, max_uv=1.989, reps 0.03 %):**
coloured, per_step 876.93 / 877.18 ms (peak_gpu 41.3 GiB).

**ng5-64 (26301830, both rows bench-finite CLEAN, max_uv=1.989):** coloured, per_step
485.85 / 488.48 ms — 32→64 doubling efficiency 90 %.

**ENVELOPE COMPLETE (2026-07-16): 15/15 points, every row bench-finite clean, reps ≤1 %.**
Best-per-point summary (ms/step): core2 192.6/126.6/78.1/68.4 @1/2/4/8 · farc 311/205/164
@4/8/16 · dars 942/497/291/243 @8/16/32/64 (dt=120) · forca20 439/296 @16/32 ·
ng5 877/487 @32/64.

**⇒ CORE2 now SCALES 4→8 GPU (78.1→68.4) instead of anti-scaling** — the padded+fusions
stack inverted the paper's "small mesh anti-scales past one node" §5 story (the 0714 handoff's
predicted upgrade: "saturates rather than degrades"). Paper-pass consequences: (a)
`fig_scaling`'s acceptance check EXPECTS anti-scaling 4→8 — it will now report "scales", fine
(it prints, doesn't assert); (b) §5 text re-scope alongside the transport-envelope edit.
NOTE cross-protocol: 150-step rows are NOT comparable with the 25-step A/B numbers (the
150-window amortizes the expensive early steps — CORE2-8 comes out 68.4 vs 72 in the 25-step
window); all ratios stay within-protocol.

## 7. Queue/keep-out notes

- `m7abenv` jobs 26299413/26299414 (+ 16-node pending) are the KOKKOS session's — untouched.
- CGPOLY-ON changes d_eta within soltol ⇒ the paper re-measure (protocol-consistent with the
  63-yr production figures) runs with cheb OFF **and** on_device OFF; CGPOLY is quoted as an
  opt-in lever with its own A/B numbers (user decision pending on any default flip).

## 8. PAPER DATA PASS (2026-07-16 pm, paper_jax f764162)

- 54 old-protocol GPU logs ARCHIVED (moved) to `scripts/logs/pre_m7_protocol/`; CPU-campaign
  + prod logs retained; the 7 new `bench_rm_*.out` + refreshed decomp/prod logs form the glob.
- Transport-envelope selection landed: fig_scaling `_jax_full_best` + make_numbers pins
  dropped (kept: the allgather pins on the decomposition rows, `ngpu=1` on `\coreStepCPU` —
  its sentence describes the single-process run; best-row would have swapped in the 8-proc
  1.51 s row).
- Macro shifts (old→new): coreStep .087→.078, coreSYPD 57→63, coreStepTwoNode .123→.068
  (anti-scaling INVERTED), darsEff 91/81/56→95/85/60, ngEffSixtyFour 72→90, ngSYPD
  1.12→1.35, commShareFour 44→38; 128-GPU macros now "--" (no new-protocol data — §5
  turnover claims removed).
- §5 edits: strong-scaling paragraph rewritten (saturates-not-degrades, farc ×1.9, fusion
  buy-back ~9 % sentence + source comments), dars validity note simplified (all rows dt=120
  verified-finite), caption transport note, 8.1 GiB peak. **In-text TODO: the Kokkos dashed
  curves are m524-VINTAGE** — refresh + re-scope the comparison sentence when the post-M7
  Kokkos ledger lands (their session; do not submit).
- Two macros re-measuring under the new protocol: coreMs* decomposition (26311383, script
  moved into the glob + steps=150) and coreStepProd (26311413, + padded leg). Old prod log
  to be archived when the new one lands (avoid cross-protocol best-row mixing).
- CPU campaign macros left on the old logs DELIBERATELY (self-contained accessibility story,
  pre-fusion protocol noted; re-measuring the CPU campaign is out of scope).

## 9. FINAL macro re-measures + SESSION CLOSE (evening)

- **Decomp (26311383, 150-step, all finite):** oce 54.96 → `coreMsOce`=55; ice +14.8 → 15;
  kpp+gm +15.0 → 15 (was 58/18/15 under the old protocol).
- **Prod physics (26311413, 150-step, all finite; padded 87.9 / ragged 89.8 / allgather
  85.5 ms):** `coreStepProd` 0.094→0.085, `coreSYPDProd` 52→58 — allgather is the best
  transport at CORE2-4 prod physics. Old prod log archived; every quoted macro is now
  single-protocol.
- paper_jax commits this session: 57a71d4 (policy), f764162 (re-measure pass), 69d8d10
  (decomp/prod macros). PDF builds clean; only the retired 128-GPU macros are "--".
- **Task #4 (paper re-measure) COMPLETE** except the Kokkos-comparison refresh, which is
  BLOCKED on the Kokkos session's post-M7 ledger (in-text TODO at the §5 comparison
  sentence; their curves are m524-vintage).

### Open user decisions at session close
1. Push main (local 293293a, 9 ahead) + merge this branch (forcing levers certified,
   CGPOLY gated; both opt-in, OFF = byte-identical).
2. Kokkos comparison data for fig10 (core2@4/8N GPU, CPU anchors, 32N; post-M7 NG5) —
   theirs to produce.
3. Gate-0 leftovers for the levers-ON figure plan: EVPWIDE profile? cheb tuning? (defaults:
   skip / optional) — then the combined 1-yr cert (docs/plans/20260716-levers-on-scaling-figure.md).

## 10. Levers-ON track started (evening; user: next priority — Kokkos refresh DEFERRED while they optimize)

- **CGPOLY tuning A/B (26312901, all legs finite):** off 72.52 · k2κ30 58.18 · k3κ30 57.38 ·
  k4κ30 57.27 · k3κ100 62.75 · k4κ100 56.61 ms/step. Top three within ~1.3 % (≈ noise, single
  reps); k3κ100 clearly worse. **Verdict: keep k=3/κ=30** — the config with the full evidence
  chain (gates + 3-scale A/Bs); a ≤1 % nominal k4κ100 gain doesn't buy re-validation.
- **Combined all-levers 1-yr cert SUBMITTED (26312994, CHEB_DEGREE=3):** 3 legs (off/off-rerun/
  on = --forcing-on-device --cheb-degree 3), the session-2 floor-controlled design; expect
  ~1¾ h (26280985 took 1:43).
- **Far points re-added to the baseline figure grid (user):** dars-128 (26312903), ng5-128
  (26312905), forca20-64 (26312980), forca20-128 (26312981) — the old figure had forca20
  64/128 (319.6/478.2 ms, old protocol, curve turned over); my envelope had silently
  shortened that curve to 32. All four under the frozen 150-step protocol.
- New plumbing: `--cheb-degree` on run_from_config; `FESOM_CG_CHEB_KAPPA` env in the bench.

## 11. ALL-ON envelope — first results (2026-07-17 morning) + an NG5 regression caught

Production-loop A/B (ms/step, chunks 2–5 mean, off/on ×2 interleaved; jobs 26315972/75/76/77):

| mesh | ngpu | OFF | ON | Δ |
|---|---|---|---|---|
| core2 | 1 / 2 / 4 / 8 | 387 / 308 / 239 / 224 | 371 / 289 / 216 / 200 | −4.3 / −6.2 / −9.7 / −10.8 % |
| farc | 4 / 8 / 16 | 500 / 411 / 364 | 405 / 316 / 265 | −19.1 / −23.1 / −27.1 % |
| forca20 | 16 / 32 | 848 / 690 | 534 / 399 | **−37.0 / −42.2 %** |
| ng5 | 32 | 1486 | 1985 | **+33.6 % ⚠️ REGRESSION** |

The ng5-32 split is diagnostic: host share fell 13.2→1.0 s/chunk (the forcing lever WORKS)
but device time rose 50→80 s/chunk. NG5's ON leg is unique three ways: the only user of the
NEW local-tables path, coloured transport, KPP; and CGPOLY there runs at REAL iteration
counts (the kernel A/Bs ran 25 cold steps where CG has not spun up — a κ=30 mis-fit to the
NG5 operator would hide in the kernel A/B and explode here). Decomposition job 26322951
(off / tables-only / cheb-only at ng5-32) attributes it before any retune.

**forca20-64 baseline row (26312980, CLEAN):** 257.98 / 267.98 ms — the curve still
descends at 64 (56 % doubling eff from 32).

## 12. NG5 regression ATTRIBUTED (26322951) — tables, not CGPOLY; per-mesh lever set adopted

Decomposition at ng5-32 (production loop, --local-forcing in all legs): off 1494.8 ·
**cheb-only 1469.3 (−1.7 %, CGPOLY fine)** · **tables-only 2026.6 (+35.6 % — the on-device
TABLES path is the whole regression, device-side)**. Every winning mesh (core2/farc/forca20)
runs TKE + the GLOBAL tables path; NG5 is KPP + the local-tables path — **OPEN INVESTIGATION
`TABLES_NG5_DEVICE_REGRESSION`**: disambiguate KPP-vs-local-tables later (e.g. a CORE2 kernel
A/B kpp+tables vs tke+tables); not needed for the figure.

**Figure decision: the optimized configuration is PER-MESH** (levers are per-config opt-ins):
core2/farc/dars/forca20 = tables + cheb; NG5 = local-forcing + cheb, tables OFF (documented
with the regression number). ng5_b (tables+cheb, 26315978) CANCELLED while still pending;
both NG5 jobs resubmitted with ON = cheb-only (26323963 ng5-32 ×2 reps, 26323964 ng5-64).

## 13. ⚠️ ALL-ON INSTRUMENT UNDER INVESTIGATION (2026-07-17 midday)

Two anomalies in the ao (production-loop) harness:
1. **ao dars-8 ran at ~100 s/step** (26315973 TIMEOUT after 2 chunks; chunk2 == chunk1 ⇒ NOT
   compile) — vs 0.94 s/step for the same mesh/count in the kernel bench.
2. **ao core2-4 OFF measured 239 ms/step where the session-2 ondev A/B measured 112.5** on
   the SAME config + instrument class (26278428). The off/on RATIOS in §11 are internally
   fair (interleaved, same env) but the ABSOLUTE ao numbers are 2× suspect — the optimized
   figure must not use them until resolved.

Env differences vs the certified session-2 template: (a) my ao harness passes a per-point
TRANSPORT flag (session-2 ran the dist_4 default = allgather); (b) allocator settings
(PREALLOCATE=false) are shared with session-2 but untested at dars scale. Instrument-check
job 26325214 (core2 dist_4, OFF only, 4 legs: {noflag, --padded} × {prealloc off, on})
splits it. ao dars_b (64) CANCELLED pending the verdict; dars_a partial (TIMEOUT) not banked.
The ng5-TKE decision job (26324321, still queued) shares the harness — its RATIOS still
decide the NG5 lever set; absolutes inherit the caveat.
