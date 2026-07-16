# Plan: the ALL-OPTIMIZATIONS-ON scaling figure (user-requested 2026-07-16)

*User decision (2026-07-16): the paper's main scaling figure stays protocol-consistent
(levers OFF, matching the 63-yr fidelity runs); the levers are quoted separately as A/Bs.
THIS plan adds a second, clearly-labeled "optimized configuration" scaling measurement once
the optimization campaign is finished — showing what the model delivers with everything ON.*

## 0. What "all optimizations ON" means (as of this writing)

| lever | state | fidelity class | evidence |
|---|---|---|---|
| exchange-pair + CG psum fusions | already ON everywhere (merged) | byte-identical | −8.6 % CORE2-8; wash dars-32 |
| getcoeffld clamp fix + trig cache | already ON (merged) | byte-identical | host-side |
| `forcing: {on_device: true}` | opt-in | VALUE-equivalent (FMA/CPU-div) | −12 % CORE2 production loop (26278428); **1-yr cert PASS** (26280985) |
| `--local-forcing` (+ on_device now composes) | opt-in, NG5-scale | bit-identical to global build | 26300806/26301615 |
| CGPOLY `ssh: {cheb_degree: 3}` | opt-in | solver-tolerance-equivalent | 3.02× fewer iters; −20.7 % CORE2-8, −3.6 % dars-32; NG5-64 pending |
| CG1R / EVPWIDE | NOT implemented | solver class | conditional — go/no-go in Gate 0 |

## 1. Gate 0 — declare the optimization campaign DONE

1. Harvest the NG5-64 CGPOLY A/B (26301617) — completes the lever's regime map.
2. CG1R go/no-go: only if the NG5-64 result shows the CG psums STILL dominant after CGPOLY
   (unlikely — iters÷3 already cut the psum pool 3×). Default: NO.
3. EVPWIDE go/no-go: only if an NG5-64 phase profile shows mEVP exchanges dominant after the
   pair fusion. Requires a decomp-style profile at NG5-64 (1 extra 16-node job). Default:
   decide on the profile, do not implement speculatively.
4. Optional cheap tuning pass (one 2-node CORE2-8 job): cheb degree k∈{2,3,4} × kappa∈{30,100}
   grid via `FESOM_CG_CHEB` — we shipped k=3/κ=30 on first try; a −2-3 % further win is
   plausible but not required.

## 2. Gate 1 — certify the COMBINED configuration (the kokkos E.5 precedent)

The on_device lever is certified alone; CGPOLY is not, and no one has certified them TOGETHER.
One combined cert covers everything the figure will claim:

- **1-yr CORE2 all-on chain, ALL levers ON vs the production OFF baseline**, using the
  session-2 3-leg floor-controlled design (ON, OFF, OFF-rerun; annual-mean per-field deviation
  of ON−OFF must sit at/below the OFF−OFF nondeterminism floor). Reuse the 26280985 cert
  harness; add `ssh: {cheb_degree: 3}` to the ON config.
- If the combined cert FAILS: cert CGPOLY alone to isolate (the only uncertified lever);
  fall back to quoting the failing lever as A/B-only and drop it from the ON figure.
- User sign-off on the certified lever set before any measurement is labeled "optimized".

## 3. The measurement — PRODUCTION-LOOP protocol, not the kernel bench

**Key instrument fact:** `bench_forward_scaling` CANNOT see the forcing lever — it times a
pre-built on-device forcing stack, so the host build + H2D the lever removes is outside the
timed window (the −12 % lives in the chunk loop's `host=` share). CGPOLY it can see; forcing
it cannot. ⇒ The all-ON figure measures **`run_from_config` production throughput**, the thing
a user actually gets:

- Template: `bench_core2_ondevice_forcing_ab.sbatch` (240 steps, 48-step chunks, chunk 1
  carries compile, reduce ms/step over chunks 2–5, `FESOM_REUSE_EXE=1`, a100_80).
- Both legs per point in ONE job: `off` = production baseline, `on` = all certified levers
  (`--forcing-on-device` + `ssh.cheb_degree=3` [+ `--local-forcing` at NG5]) — the figure can
  then show baseline and optimized curves from the SAME allocation, and the off leg
  cross-checks the kernel-bench envelope.
- Grid: the same points as the baseline envelope — core2 1/2/4/8, farc 4/8/16,
  dars 8/16/32/64 (dt=120), forca20 16/32, ng5 32/64 — per-point banked-best transport,
  ×2 reps, bench-finite-style health line per leg (chunk diagnostics or end-state check).
- Output/restarts OFF (pure step throughput; I/O is quoted separately in the paper already).
- Cost: ~2× the baseline envelope (two legs) ≈ 130 node-hours GPU + the 1-yr cert chain.

## 4. Paper integration

- New macros (`\coreSYPDOpt`, `\ngStepOpt`, …) and either dashed "optimized" curves on the
  existing fig10 panels or a small companion panel — decide with the user at figure time.
- Caption language: "optimized configuration (on-device forcing + Chebyshev-preconditioned
  CG), validated against the production configuration over a 1-yr integration (§X)" — the
  value-equivalence caveat stated once, referencing the cert.
- The baseline figure, table, and 63-yr fidelity sections stay untouched.

## 5. Sequencing / dependencies

1. NOW: baseline envelope + NG5-64 CGPOLY A/B finish (in queue).
2. Gate 0 decisions (CG1R/EVPWIDE/tuning) — evidence + user sign-off.
3. Gate 1 combined 1-yr cert (~1 chain job + comparison) — user sign-off on the lever set.
4. The all-ON production-loop envelope (~130 node-hours).
5. Paper pass: macros + figure + caption (together with the already-queued §5 re-scope).
