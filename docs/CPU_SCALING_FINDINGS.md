# FESOM2-JAX CPU strong-scaling campaign — FINDINGS

*Levante `compute` (128-core Milan, 256 GB), jaxlib 0.10.1, full model (ice+KPP+GM, JRA-1958,
PHC IC), compile excluded from every number. Campaign run 2026-07-27. Logs:
`scripts/logs/{gate_cpu_*,cpu_topo_sweep,cpu_camp_*}.*.out`.*

**This file supersedes the "JAX-CPU is effectively single-node" conclusion of 2026-07-13**
recorded in `docs/PARALLELISM.md` and asserted in the model paper, §5. That conclusion was
correct for the transport it was measured with and wrong as a statement about the backend.

---

## 1. The headline: multi-node CPU works, and the halo TRANSPORT was the whole story

CORE2 (127k nodes), 8 ranks/node × 16 cores, gloo collectives:

| config | transport | ms/step |
|---|---|---|
| 1 node × 8 ranks | padded | 1258.8 |
| 2 nodes × 16 ranks | padded | 2889.0 |
| **2 nodes × 16 ranks** | **coloured** | **1004.5 / 1015.3 / 1011.8** (3 runs) |

Padded loses 2.3× crossing the node boundary; coloured **gains** over the one-node baseline.
At 2 nodes coloured is 2.9× faster than padded in the identical allocation. All legs
physically identical: `nonfinite_T=0`, `max_uv=0.705 m/s`.

Why: padded sends one dense `all_to_all` whose cost grows with rank count — brutal on a TCP
fabric. Coloured sends K=Δ neighbour-only `ppermute` rounds, a cost set by mesh connectivity,
not by P. The 2026-07-13 campaign predates the coloured transport (Phase 8d, 2026-07-14), so
it could only ever have measured the pathological case.

## 2. The MPI backend is a dead end — but for a recordable reason

`docs/PARALLELISM.md` listed `jax_cpu_collectives_implementation='mpi'` as the untested escape
hatch. It is testable: jaxlib 0.10.1's `libjax_common.so` ships `xla::cpu::MpiCollectives` plus
a complete MPItrampoline loader, and an MPIwrapper shim built against Levante's OpenMPI
(`/work/ab0995/a270088/mpiwrapper-ompi412`, `MPITRAMPOLINE_LIB`) makes a 2-node `psum` pass.

**But the real model always dies:**

```
jax.errors.JaxRuntimeError: UNKNOWN: MPI: Communicator requested from a thread that is not the
one MPI was initialized from. Multiple threads/devices per process are not yet supported.
```

XLA's `MpiCollectives` requires every collective on the MPI-init thread; JAX's runtime issues
them from its own threads. The toy gate passes only because it stays on the main thread. So
the MPI path cannot run this model at all, at any scale — and it does not need to, since gloo
+ coloured scales. Good outcome for reproducibility: a stock `pip install jax` suffices.

Two launcher facts worth keeping: Levante's OpenMPI 4.1.2 is **not** built with SLURM PMI, so
`srun --mpi=pmix` cannot start it (`MPI_Init_thread` fails) — use `mpirun`. And `mpirun`
inherits its slot count from the batch allocation, so `--ntasks-per-node` must match the
intended `--map-by ppr:N:node`.

## 3. The coloured/gloo defect (first seen at CORE2 npes=8; §3b has the full blast radius)

`Gloo collective permute failed: ... Connection reset by peer`, signal 6, reproduced **four
times** at CORE2 npes=8: 1 node × 8 ranks (×2), 2 nodes × 4 ranks, and 1 node × 8 ranks × 16
cores. The sweep brackets it on both sides — CORE2 coloured runs fine at npes = 2, 4, 16, 32
and 64:

| npes | 2 | 4 | **8** | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| ms/step | 2605.7 | 1718.4 | **ABORT** | 1011.8 | 992.3 | 1205.5 |

**It is the partition, not the rank count**: farc at npes=8 runs coloured without complaint
(6143 ms). So `dist_8` on CORE2 produces an exchange structure that gloo's collective-permute
rejects, while the same P on a different mesh is fine. Not root-caused; workaround is padded at
the affected rungs, the transports being forward bit-exact.

⚠️ This section was written when CORE2 npes=8 was the only known case. It is not — see §3b:
the same defect also takes out CORE2 npes=128, farc npes=32 and dars npes=64, which is what
caps three of the four ladders.

## 3b. The "coordination service" failures are the SAME gloo defect

Two rungs — core2 npes=128 and farc npes=32 — appeared to die in
`xla::CoordinationServiceAgent::SetError`, which looked like a second, independent problem, and
neither OOM nor job-start contention explained it (they reproduce in isolation; core2 8-node
peaked at 82 GB, farc 4-node at 157 GB; larger rungs of the same ladders pass).

With the full log captured, the first rank to die reports:

```
jax.errors.JaxRuntimeError: UNKNOWN: Gloo collective permute failed:
    [gloo/transport/tcp/pair.cc:547] Connection closed by peer
```

**It is §3's defect.** The coordination service abort is only the cascade — one rank dies, the
service tears down the rest. A control run with `heartbeat_timeout_seconds=900` died the same
way and its message confirms it was *reacting* to another task's death, killing the heartbeat
hypothesis outright.

So there is **one** defect, not three, and it now has a bigger blast radius than §3 implied:
it takes out core2 npes=8, core2 npes=128 and farc npes=32, while core2 16/32/64 and farc
8/16/64/128 pass. Structural, per (mesh, npes) — not flaky, not load-dependent.

**Hypothesis tested and REFUTED** (`scripts/bench/analyse_colouring.py`, job 26489188). The
idea was that a colour class is a *partial* permutation, leaving ranks idle on one or both
sides, and that gloo's collective-permute cannot tolerate the asymmetry. The round structure
says otherwise — in both directions:

| config | outcome | rounds partial | (rank, round) idle-both slots |
|---|---|---|---|
| core2 npes=8 | **FAILS** | 1 / 3 | 4 |
| core2 npes=16 | passes | 5 / 6 | 42 |
| farc npes=32 | **FAILS** | 8 / 8 | 114 |
| farc npes=64 | passes | 8 / 8 | 207 |

The passing configurations are *more* partial and have an order of magnitude more idle ranks
than the failing ones. Partial permutations are not the discriminator, and neither is the
presence of full rounds (core2 npes=8 has two full rounds and fails; core2 npes=16 has one and
passes).

⚠️ **REFINED — see §11.6. "Keyed to (mesh, npes)" is too narrow**: farc npes=128 passes at
CG-preconditioner degree 0 (5 runs) and fails at degree 6 (3 runs), same mesh, same ranks, same
layout. The trigger includes the *exchange pattern*, not just the partition.

**What IS established: the defect is independent of process layout, and deterministic for a
fixed configuration.** dars npes=64 fails identically at 2 ranks/node × 32 nodes and
at 1 rank/node × 64 nodes; CORE2 npes=8 fails at 8×16 on one node, at 4×32 across two, and at
8 ranks/node × 16 cores. Same npes, different placements, same abort — while CORE2 *passes* at
npes=64. So it is a property of the exchange structure the partition produces, not of node
count, ranks per node, or the fabric. That also means **it cannot be dodged by re-splitting**:
a failing rung fails at every layout, which is why three ladder points fall back to padded.

Affected: core2 npes=8, core2 npes=128, farc npes=32, dars npes=64.
**Root cause remains open** — a genuine upstream candidate alongside
[the ragged transpose bug](JAX_RAGGED_A2A_BUG.md), with a clean reproducer: CORE2 `dist_8`,
`--halo coloured`, `JAX_PLATFORMS=cpu`, gloo, 8 ranks — fails in the first collective, every
time, on one node or two.

Campaign handling: the three affected rungs are measured with **padded** instead (jobs
26489196/26489197). The transports are forward bit-exact, so the curve keeps its meaning, but
those points carry padded's multi-node penalty and are labelled as such — they are not
comparable to the coloured points beside them.

*Process note:* two diagnostic rounds were wasted because the leg script piped mpirun through
`tail -14`, which kept the stack trace and discarded the error message above it. Legs now write
a full log to `scripts/logs/full_<name>.<jobid>.out`.

## 4. Per-node topology: how to split 128 cores

Fixed hardware (2 nodes, 256 cores), CORE2, coloured — the split matters more than the count:

| ranks/node × cores | npes | ms/step |
|---|---|---|
| 4 × 32 | 8 | FAILED (the CORE2 npes=8 defect above) |
| 8 × 16 | 16 | 1011.8 |
| **16 × 8** | 32 | **992.3** |
| 32 × 4 | 64 | 1205.5 |

Padded at the same 16×8 split: **11682 ms** — 11.8× slower than coloured, the clearest single
statement of what the transport is worth on CPU.

For scale, one process using all 128 cores runs CORE2 at **5.76 s/step** (0.9 SYPD, the paper's
§5 number). Eight ranks × 16 cores on the same 128 cores: 1.26 s/step — 4.6× faster from the
grouping alone. XLA:CPU's thread pool saturates near 8–16 cores per device on these
gather-bound kernels; domain decomposition is the parallelism that keeps paying.

## 5. Host memory is the binding constraint on big meshes — and it tracks the NODE

Every rank calls `load_mesh` on the whole global export and builds the global `State` + PHC IC
on the host before sharding. But the measured totals do **not** behave like a per-rank cost:

| mesh | ranks/node × cores | per-node peak | outcome |
|---|---|---|---|
| core2 | 16 × 8 | 92 GB | OK |
| farc | 16 × 8 | 238 GB | OOM |
| farc | 8 × 16 | — | OK |
| dars | 8 × 16 | 232 GB | OOM |
| dars | 4 × 32 | 227 GB | OOM |
| ng5 | 4 × 32 | — | OOM |
| ng5 | 2 × 64 | — | OOM |

Halving the ranks per node barely moved the dars total (232 → 227 GB), which looked like a
per-*core* cost. **It isn't** — that reading was an artifact: both runs were OOM-killed, so both
report a peak pinned at the node ceiling, and equal ceilings say nothing about the underlying
demand. The single-rank probe (job 26489102) settles it:

| mesh | cores | RSS |
|---|---|---|
| core2 | 16 | 24.25 GB |
| core2 | 128 | 24.37 GB |

**Core count is irrelevant to memory** (0.5% apart across an 8× change). The driver is the
per-rank host-side **setup transient**: every rank materialises the global `State` and PHC IC
and `build_sharded_mesh`'s `[P, Lmax, …]` gather (every partition's padded arrays, in every
process) before taking its own shard.

The one clean, *completed* big-mesh measurement gives the magnitude — and it is far worse than
the OOM-killed runs suggested. dars at 2 ranks/node **succeeded** at **237.7 GB/node = 119 GB
per rank**, against 149.8 GB for the same mesh unsharded at npes=1. **Sharding across 16 ranks
cut per-rank memory only 1.26×.** Per-rank cost is essentially the global footprint, nearly
independent of npes, so:

> **ranks per node ≈ node RAM / global model footprint** — and nothing about adding nodes helps.

That single rule predicts everything observed: dars (119 GB) fits 2 ranks/node and only just;
ng5 at ~2.3× dars ≈ 270 GB exceeds a whole 251 GB node, so it OOMs at **1 rank/node**, exactly
as measured. Levante has no fatter CPU node (all `compute` = 251 GB). Big meshes are therefore
forced onto few fat ranks — the least efficient configuration, since the probe also shows one
rank gains only 1.4× going from 16 to 128 cores (8686 → 6124 ms).

The real fix is a shard-on-load path (mmap the export, build only the local shard) — not
attempted here, and the single highest-value follow-up for CPU execution at scale.

## 6. Strong-scaling ladders

**CORE2** (127k nodes, dt=1800, 16 ranks/node × 8 cores, coloured):

| nodes | ranks | ms/step | SYPD |
|---|---|---|---|
| 1 | 16 | 1103.2 | 4.47 |
| **2** | 32 | **990.2** | **4.98** |
| 4 | 64 | 1292.8 | 3.81 |
| 8 | 128 | *(retry, coordination abort)* | |

Padded anchor, 1 node: 1200.0 ms. CORE2 peaks at 2 nodes and then degrades — the same
saturation the GPU campaign found for this mesh, and unsurprising at 127k nodes / 4k per rank.

**Padded's compile time explodes with rank count**, which is a second cost beyond its step
time: at npes=128 the padded fallback spent **1337.7 s (22 min) compiling** — the first attempt
died on a 30-min walltime with the timed run still ahead of it — where coloured compiles the
same case in ~33 s. A dense `all_to_all` over 128 ranks is evidently expensive to lower as well
as to run.

**farc** (638k nodes, dt=900, 8 ranks/node × 16 cores, coloured):

| nodes | ranks | ms/step | SYPD |
|---|---|---|---|
| 1 | 8 | 6143.1 | 0.40 |
| 2 | 16 | 4122.6 | 0.60 |
| 4 | 32 | *(retry, coordination abort)* | |
| **8** | 64 | **3581.9** | **0.69** |
| 16 | 128 | 6205.2 | 0.40 |

Padded anchor, 1 node: 6043.7 ms — at npes=8 the transports tie, as expected when there is
barely any halo to exchange. farc improves out to 8 nodes (1.7× for 8× the hardware) and then
falls over at 16.

**dars** (3.16M nodes, dt=180, 2 ranks/node × 64 cores, coloured) — **the first dars numbers on
CPU**:

| nodes | ranks | ms/step | Mnodlev/s | SYPD | efficiency |
|---|---|---|---|---|---|
| 8 | 16 | 16445.3 | 11.0 | 0.030 | 1.00 (base) |
| **16** | **32** | **8614.3** | **20.9** | **0.057** | **0.95** |
| 32 | 64 | gloo defect (§3) | | | |

**This is the campaign's best scaling result: 1.91× for 2× the nodes, ~95% parallel
efficiency**, and 20.9 Mnodlev/s is the highest CPU throughput measured — above farc (8.6) and
CORE2 (6.1). Bigger meshes amortise the framework overhead better, matching the GPU campaign
and the 2026-07-13 study. Two caveats keep it honest: in absolute terms this is 0.057 SYPD, not
a production configuration; and per *node* it is only 1.3 Mnodlev/s against CORE2's 4.0,
because the memory ceiling forces dars onto 2 fat ranks/node (64 cores each), the split the
sweep showed to be the least efficient.

**ng5 (7.4M nodes) does not run on Levante CPU at all** — four configurations, all OOM-killed:
4 ranks/node, 2 ranks/node, and 1 rank/node at both npes=16 and npes=8 (the last peaking at
242 GB). By the §5 rule this is arithmetic, not bad luck: one rank needs ~270 GB against a
251 GB node, so no rank count or node count can help. The blocker is the setup transient —
global `State` plus the `[P, Lmax, …]` gather — not the physics, the transport, or the mesh
size per se. A shard-on-load path would very likely make NG5 feasible; without it, NG5 on CPU
is unreachable on this machine.

## 7. What this changes in the paper

`sections/05_performance.tex:326` currently states that across CPU nodes "every launch variant
we tried crashes, hangs, or anti-scales, so JAX-CPU execution is effectively single-node."
That sentence needs rewriting, not footnoting.

## 8. Proposed replacement for the paper's §5 CPU paragraph

`sections/05_performance.tex` currently says JAX-CPU "is effectively single-node". That is now
false, so the sentence needs replacing rather than footnoting. Draft below — **the conclusion
of the paragraph does not change**: many-CPU production still belongs to the Fortran original
and the Kokkos shadow. What changes is the reason. It is no longer "the runtime cannot do it"
but "it scales, at a throughput that remains far from the Fortran original, and the largest
mesh does not fit in memory".

> Nor is this mode where many-CPU production belongs, though not for the reason we previously
> reported. With the coloured halo transport --- the same `ppermute`-round exchange that is
> fastest on multi-GPU at scale --- the model does scale across CPU nodes: on dars (3.16M
> nodes) it runs at 16.4\,s per step on eight nodes and 8.6\,s on sixteen, a speed-up of
> 1.91$\times$ for twice the hardware, and reaches 20.9\,M node-levels per second, the highest
> CPU throughput we measure. The transport is what decides this. The dense padded exchange we
> used previously grows with rank count and anti-scales across nodes --- on CORE2 at two nodes
> it costs 2.9$\times$ the coloured transport, and 11.8$\times$ at sixteen ranks per node ---
> whereas the coloured rounds scale with mesh connectivity rather than with the number of
> ranks. Two limits nevertheless keep many-CPU production with the other two codes. In absolute
> terms the throughput remains far below the Fortran original, which reaches 87.8 SYPD on 512
> ranks over four nodes. And host memory binds before parallelism does: every rank materialises
> the global model state before taking its shard, so per-rank memory is essentially the global
> footprint --- 119\,GB per rank for dars against 149.8\,GB unsharded --- which restricts dars
> to two ranks per node and puts NG5 (7.4M nodes) out of reach on a 251\,GB node at any rank
> count. Both are implementation limits rather than properties of the framework, and a
> shard-on-load path would lift the second.

Numbers to add to `_numbers.tex`, and the stale ones to retire:

| macro | value | note |
|---|---|---|
| `\darsCPUStepEight` | 16.4 s | dars, 8 nodes × 2 ranks, coloured |
| `\darsCPUStepSixteen` | 8.6 s | dars, 16 nodes × 2 ranks |
| `\darsCPUThroughput` | 20.9 | M node-levels / s |
| `\coreStepCPUmp` | 0.99 s | CORE2 best, 2 nodes × 16 ranks (replaces the 1.51 s / 3.3 SYPD single-node figure) |

⚠️ The existing paragraph's 1.51\,s / 3.3 SYPD sentence describes 8 processes × 16 cores with
the **padded** transport on one node. It is still true, but it is no longer the best CPU
configuration, and quoting it alone now understates the CPU story by 1.5×.

## 9. The GPU's one-process-many-devices topology does NOT transfer to CPU (tested, negative)

Tempting idea, since on GPU one process drives all 4 devices and therefore pays for the global
host state **once per node**, whereas our CPU launch pays once *per rank*
(`integrate_sharded.py:54`: "each process builds the full global host array and places only its
local shards"). Job 26492778 tested it with the coloured transport — the same retest that
rescued the multi-node verdict. It fails on both counts:

| config | CORE2 ms/step |
|---|---|
| 1 node, 16 processes × 8 cores | 1103.2 |
| 1 node, 1 process × 16 in-process devices | **2915.0** (2.6× slower) |
| 2 nodes, 32 processes | 990.2 |
| 2 nodes, 1 process/node × 16 devices | **2067.7** (2.1× slower) |
| dars npes=32, 2 nodes, 1 process/node × 16 devices | **OOM-killed** |

- **Speed:** the 2026-07-13 "in-process is ~1.7× slower" result holds, and is worse here. Unlike
  the single-node verdict, this one did not expire with the transport change.
- **Memory: it does not even solve the problem it was proposed for.** A single process holding
  16 devices keeps the global host arrays *and* all sixteen local shards alive simultaneously,
  so folding 16 global copies into one buys much less than the process count suggests. dars
  still OOMs.
- **One real change:** multi-node in-process *runs* now (it hung in 0.10.1 testing in July). Not
  useful, but the old failure mode is gone.

⇒ The many-process launch stays. The memory wall has exactly one fix: **shard-on-load** — build
each rank's own slice from the export rather than building the global and slicing it. That is a
loader change, not a launch change, and it is also the only route to NG5 on CPU (one global copy
is ~270 GB against a 251 GB node, so even a perfect topology cannot help).

## 10. Why CORE2 tops out near 5 SYPD — the cost decomposed (job 26492578)

The gap to Fortran (87.8 SYPD, 512 ranks / 4 nodes) has been factored before as ~2.7× kernel
efficiency × ~2.5× parallel efficiency. This campaign localises the parallel half and adds the
piece that was previously assumed rather than measured.

**Intra-rank thread scaling is the disaster.** One rank, CORE2, npes=1:

| cores | ms/step |
|---|---|
| 16 | 8685.8 |
| 128 | 6123.9 |

**1.4× for 8× the cores (~17% efficiency).** XLA:CPU's thread pool cannot feed a fat core count
on gather-bound unstructured kernels. Consequently the only parallelism that pays is domain
decomposition: on identical hardware, 1 rank × 128 cores = 6123.9 ms vs 16 ranks × 8 cores =
1103.2 ms — **5.5× purely from regrouping the same cores**. That is why the topology sweep
matters more than the core count, and why big meshes forced onto few fat ranks (§5) lose so much.

**Collective latency is a third of the step.** Measured at the best config (2 nodes × 16 ranks):

| quantity | value |
|---|---|
| CG iterations per step | **127** (exactly; `hit_cap=0`, so converged, not capped) |
| scalar `psum`, 32 ranks | **0.877 ms** median (0.835 min) |
| 1 MiB/rank `psum`, 32 ranks | 25.2 ms |
| ⇒ 127 iters × 3 collectives × 0.877 ms | **334 ms** of a 983.9 ms step (**34%**) |

A millisecond to sum one number across 32 ranks is ~50× what NCCL costs on GPU. This is the
mechanism behind the 2-node saturation: adding ranks makes every one of those ~380 collectives
dearer while shrinking each rank's share of work.

**Component split** (same config): full 983.9 ms, ocean-only 724.2 ms ⇒ ice+KPP+GM = 259.8 ms
(26%). Nothing is pathological; cost is spread.

**Actionable today — CGPOLY.** The Chebyshev-preconditioned CG (already implemented, opt-in via
`FESOM_CG_CHEB=1`) cuts iterations ~3×:

| leg | ms/step | SYPD |
|---|---|---|
| baseline | 983.9 | 5.01 |
| **CGPOLY** | **830.1** | **5.94** |

−15.6%, the best single lever available without new kernels. It is *less* than the ~223 ms the
latency arithmetic predicts, because each Chebyshev iteration does more work — so waiting is a
large part of the cost but not all of it. On GPU the same switch gave −21% on CORE2-8.

⇒ Remaining headroom is bounded: even zero collective latency leaves ~650 ms/step (≈7.6 SYPD),
still ~12× off Fortran. Closing that needs the kernel-efficiency half — FFI custom kernels — as
the 2026-07-13 study already concluded.

## 11. Root cause of the CPU scaling ceiling: JAX's all-reduce is O(P) — and a −29% fix

Branch `perf/cpu-cg-collectives`. Jobs 26493332 / 26493370 / 26493520 / 26493572 / 26493910.

### 11.1 The measurement that explains everything

In-graph collective cost (chain of dependent collectives inside ONE jit; the difference between
chain lengths cancels dispatch, so this is what the CG's `while_loop` actually pays):

| ranks | `psum` scalar | `psum` 32 KB | `ppermute` 32 KB |
|---|---:|---:|---:|
| 2 | 0.041 ms | 0.118 | 0.039 |
| 4 | 0.107 | 0.322 | 0.049 |
| 8 | 0.183 | 0.736 | 0.049 |
| 16 | 0.363 | 1.772 | 0.060 |
| 32 | **0.812** | 6.919 | 0.615 |

**The all-reduce doubles when the rank count doubles — O(P), a ring.** MPI does it in O(log P):
over 2→32 ranks that would be ~5×, we measure 19.7×. The neighbour exchange is *flat* in rank
count (0.039→0.060 out to 16; the jump at 32 is the node boundary, not P).

This is the whole answer to "why does Kokkos not suffer from its exchanges". The algorithm is
identical — 2 halo exchanges + 2 all-reduces per CG iteration, ~127 iterations — but one of our
four collectives has a cost that grows with rank count, so every added rank makes ~254
collectives/step dearer while shrinking each rank's work. Extrapolated to Fortran's 512 ranks,
gloo would need ~13 ms per all-reduce against MPI's tens of microseconds.

*(The earlier per-call figure of 0.877 ms was NOT inflated by dispatch — in-graph is 0.812 ms.
The §10 estimate stands.)*

### 11.2 The fix: trade the O(P) collective for the flat one

A degree-k Chebyshev preconditioner does k SpMV+halo per apply and NO dot products, cutting the
ITERATION count — i.e. exactly this trade. CORE2, 32 ranks:

| degree | CG iters | ms/step | SYPD |
|---|---:|---:|---:|
| 0 | 127 | 994.9 | 4.95 |
| 1 | 81 | 806.5 | 6.11 |
| 2 | 55 | 747.9 | 6.59 |
| 3 | — | 724.0 | 6.81 |
| 4 | 35 | 726.6 | 6.78 |
| **6** | — | **705.3 / 711.3** | **6.99** |
| 8 | — | 714.0 | 6.92 |
| 12 | — | 732.3 | 6.75 |

**Degree 6 is the CPU optimum: −29%.** On GPU the optimum was degree 1 (−21%) — the platforms
have opposite collective economics, so the knob must be tuned per backend, not inherited.

### 11.3 It also removes the multi-node degradation

| nodes | ranks | degree 0 | degree 6 |
|---|---:|---:|---:|
| 1 | 16 | 1103.2 | 951.4 |
| 2 | 32 | 990.2 | **711.3** |
| 4 | 64 | 1292.8 (**worse**) | **716.2** (flat) |

The 4-node anti-scaling in §6 was the O(P) all-reduce, not a property of the backend. It is
gone. CORE2 still does not scale *past* 2 nodes, but at 64 ranks it holds ~2000 points/rank —
the mesh has run out of work, which is an honest saturation rather than a defect.

**New best CORE2 on CPU: 705 ms = 7.0 SYPD, vs the campaign's 4.98 (+41%).**

⚠️ CGPOLY changes the iteration path, so results differ at roundoff from the C reference: it is
opt-in (`FESOM_CG_CHEB=6`) and making it default on CPU needs the usual 1-year certification.

### 11.4 Two hypotheses tested and refuted

**(a) "The CG's 2-D halo wants padded, not coloured."** Reasoning was that a dense `all_to_all`
is one round against coloured's six, so a latency-bound 2-D field should prefer it. Measured at
32 ranks, main halo coloured, varying ONLY the solver's halo (new `FESOM_SSH_HALO` knob):

| CG halo | ms/step |
|---|---:|
| coloured | **1008.4** |
| padded | 1537.2 (+52%) |
| allgather | 4877.2 (+384%) |

Wrong, and backwards: a dense all-to-all is **P−1 messages per rank** (31 here), coloured is
**K=Δ** (6). Coloured wins on message COUNT, not on bandwidth — which also re-explains why it
wins for the 3-D fields. Bit-exactness of the knob verified (job 26493520: `max|diff| = 0` for
both overrides). The experiment still paid: swapping only the CG halo moves the step 1.0 s →
4.9 s, so the CG halo *is* the dominant communication — coloured simply already makes it cheap.

**(b) "The greedy edge-colouring wastes rounds."** Only sometimes, and not enough to matter:
core2 npes=32 uses K=7 where Δ=6 (one spare round), while core2 npes=16 and dars npes=32 are
already optimal (K=Δ). Not worth an optimal-colouring implementation.

### 11.5 The lever's size depends on how collective-bound the point is

Re-running the campaign's best points with `FESOM_CG_CHEB=6`:

| mesh | nodes × ranks/node | points/rank | degree 0 | degree 6 | change | SYPD |
|---|---|---:|---:|---:|---|---:|
| CORE2 | 2 × 16 | 4.0 k | 990.2 | **711.3** | **−28%** | 4.98 → **6.94** |
| CORE2 | 4 × 16 | 2.0 k | 1292.8 | **716.2** | **−45%** | 3.81 → **6.89** |
| farc | 8 × 8 | 10.0 k | 3581.9 | **1991.3** | **−44%** | 0.69 → **1.24** |
| dars | 8 × 2 | 197 k | 16445.3 | 15918.2 | −3.2% | 0.030 → 0.031 |
| dars | 16 × 2 | 99 k | 8614.2 | 8508.9 | −1.2% | 0.057 → 0.058 |

**The gain tracks the collective share, not the mesh.** CORE2 and farc at their best points are
collective-bound (few thousand to ten thousand points per rank), and CGPOLY removes ~3/4 of the
all-reduces: −28 to −45%. dars is compute-bound — the §5 memory ceiling forces it onto 2 fat
ranks/node, so it holds 99–197 k points per rank and the collectives are already a small share:
−1 to −3%, i.e. nothing.

So the two CPU limits interact: the memory wall pushes big meshes into a regime where the
collective fix does not help, and the collective ceiling caps small meshes where memory is fine.
Fixing shard-on-load (§5) would let dars run at 16 ranks/node — a better split *and* a regime
where CGPOLY pays.

**Updated CPU bests: CORE2 705 ms / 7.0 SYPD, farc 1991 ms / 1.24 SYPD, dars 8509 ms / 0.058
SYPD.** Against the campaign's headline numbers that is +41%, +80%, +1%.

### 11.6 CORRECTION to §3b: the gloo defect is keyed to the exchange PATTERN, not just (mesh, npes)

§3b concluded the defect is "structural, per (mesh, npes) — not flaky". The reproductions behind
that (core2 npes=8 4/4, farc npes=32 3/3, dars npes=64 2/2) are real, but they only support
"these configurations always fail" — not that the partition is the whole key. farc npes=128
breaks the claim:

| farc npes=128, 16 nodes × 8 ranks, coloured | runs | outcome |
|---|---|---|
| CG preconditioner degree 0 | 5 (jobs 26489019, 26494664 ×2, campaign) | **all PASS** (6122.7 / 6172.0 / 6205.2 ms) |
| CG preconditioner degree 6 | 3 (jobs 26494414, 26494664 ×3) | **all FAIL** — `Gloo collective permute failed` |

Identical mesh, rank count and layout; only the preconditioner differs. So the defect is
**deterministic for a fixed configuration** (5/5 and 3/3, not intermittent) but the configuration
includes the *pattern of collective permutes*, which the preconditioner changes.

Note the direction: degree 6 issues **fewer** exchanges per step than degree 0 (27 iterations ×
7 halos = 189, vs 127 × 2 = 254), yet it is the one that fails. So the trigger is not the total
count but plausibly the **rate/burstiness** — a degree-6 apply fires 6 collective permutes
back-to-back with only a small SpMV between them, where degree 0 spaces them out. "Connection
reset by peer" under a tight burst is consistent with resource exhaustion or a race in gloo's
connection handling. Still not root-caused, but this is a sharper reproducer for upstream than
"partition dist_8".

**Practical consequence: the two levers conflict at that rung.** CGPOLY degree 6 cannot be used
at farc npes=128, so farc's best stays 8 nodes / 1991 ms (degree 6). It does not affect CORE2
(2–4 nodes) or dars, where degree 6 runs clean.

### 11.7 How un-bit-identical is CGPOLY? (job 26495313)

"Not bit-identical" is a category, not a magnitude. CORE2, degree 6 vs degree 0 from the same
initial state, 4 devices:

| field | 3 steps max\|diff\| | 20 steps max\|diff\| | 20 steps rms |
|---|---|---|---|
| T | 8.137e-03 °C | 7.897e-03 | 4.562e-06 |
| S | 6.647e-04 psu | 6.332e-04 | 4.343e-07 |
| uv | 2.088e-03 m/s | 1.335e-02 | 6.733e-06 |
| eta_n | 4.518e-05 m | 1.855e-05 | 1.050e-06 |
| d_eta | 1.624e-04 | 1.867e-05 | 1.023e-06 |

`max_uv` is identical at both lengths (0.464 / 1.135 m/s). The differences are at
solver-tolerance level — worst case ~8 mK at one point, rms ~5 µK, against the 0.004 °C
JAX-vs-Fortran SST difference already reported in the paper — and they do **not** grow
systematically from 3 to 20 steps (T, S, eta flat or smaller; only uv grows).

⚠️ 20 steps is 10 hours of model time and says nothing about chaotic growth over a 62-year
hindcast. The 1-year certification remains the gate for default-ON; this only establishes that
the immediate perturbation is small, not that the climate is unchanged.
