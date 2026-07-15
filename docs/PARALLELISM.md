# Parallelisation guide — which halo, which launch, when

fesom-jax has **three interchangeable halo-exchange transports** and **three launch
topologies**. This page says which to pick for a given (goal, mesh, hardware) — with the
measurements behind the advice (Levante benchmark campaign, 2026-07-13, jax/jaxlib 0.10.1) —
and records what does *not* work and why. The short version lives in the README's
["which parallelisation option to use when"](../README.md#which-parallelisation-option-to-use-when)
table; this is the long version.

## The three halo transports

Every sharded run exchanges halo (partition-boundary) values ~30× per step. Same owner
values in all three cases — only the transport differs:

| transport | flag | backends | gradient (reverse-mode AD) | per-device volume | when |
|---|---|---|---|---|---|
| **all_gather** | (default) | CPU + GPU | ✅ correct (the AD **oracle**) | ~the whole global field (47 MB/3-D field at CORE2, 2.65 GB at NG5) | small npes; correctness gates; the reference every other mode is tested against |
| **ragged** (`lax.ragged_all_to_all`) | `use_ragged=True` / `--ragged 1` | **GPU only** | ❌ **broken** — JAX's transpose is wrong, O(1) error ([JAX_RAGGED_A2A_BUG.md](JAX_RAGGED_A2A_BUG.md)) | minimal (only the actual halo: 0.1–0.3 MB at CORE2) | forward-only, and now only a **middle band (~32 GPUs)**: padded beats it ≤16, coloured beats it at 64 (−9 %, NG5). Its cost is a P-way all-to-all, so it grows with the device count |
| **padded** (slot-padded dense `lax.all_to_all`) | `use_padded=True` / `--halo padded` | **CPU + GPU** | ✅ **correct by construction** — the transpose of `all_to_all` is another `all_to_all` | ragged × pad factor: 1.8×@P4, 3.4×@P8, 5.4×@P16, 15.3×@P32, 22×@P64, 41×@P128 — grows ~linearly in P (`probe_pad_factor.py`), still far under all_gather | **any CPU sharded run**; **GPU forward at ≤16 GPUs** — fastest measured transport there; any sharded gradient ≤32 GPUs |
| **coloured** (K bipartite-coloured `lax.ppermute` rounds) | `use_coloured=True` / `--coloured` / `--halo coloured` | **CPU + GPU** | ✅ **correct by construction** — `ppermute` transposes to the inverse `ppermute` | 1.0–1.4× the true halo at ANY P — the only transport whose volume does **not** grow with P; cost is K = Δ ≈ 3–14 rounds | **the largest configurations: FASTEST measured at NG5-64** (543 ms vs ragged 592, padded 742). Loses on small meshes (the CG's ~254 tiny 2-D exchanges/step make K collectives the dominant cost). **The AD-correct AND fastest choice for big sharded gradients** |

The padded transport (Phase 8c, merged from `experiments/padded_halo_a2a/`) pads every
per-neighbour chunk to the max pair-chunk (`pad_slot`) and does ONE tiled `all_to_all` of
`P × pad_slot` lanes; index maps are host-built in `shard_mesh._ragged_exchange_map`
(`pad_src` / `pad_valid` / `pad_slotpos`). Gates: forward AND `jax.grad` vs the all_gather
oracle on the real CORE2 partitions — **bit-exact at dist_2/4/8, ≤2 ulp at dist_16/32**
(summation-order roundoff, categorically different from the ragged transpose's O(1) error);
`fesom_jax/tests/test_halo.py::test_padded_*`.

## The three launch topologies

1. **Single process, N devices** — the GPU standard (1 process sees 4 A100s) and the CPU
   *testing* mode (`XLA_FLAGS=--xla_force_host_platform_device_count=N` fake devices).
   ⚠️ On CPU this topology is for gates/tests, not throughput: it saturates at ~16 devices,
   and the **all_gather halo crashes outright at ≥32 in-process CPU devices** (deterministic
   XLA `rendezvous.h` check-failure in AllGatherThunk; padded runs fine to 128).
2. **Multi-process, 1 device each, gloo — the CPU production topology** (`JDIST_CPU=1`):
   `srun -n <npes>` with `--xla_force_host_platform_device_count=1` per process; the bench
   driver then sets `jax_cpu_collectives_implementation=gloo` + `jax.distributed.initialize()`
   (SLURM auto-detect). MPI-rank-like; **~1.7× faster than the best in-process config** at
   the same npes. Optimum on a 128-core Levante node: **8 processes × 16 cores**.
3. **Multi-process across nodes, GPUs** (`JDIST=1`): 1 process/node × 4 GPUs, `srun`;
   unchanged, the dars/NG5 multi-node path.

**Do not attempt multi-node CPU on jaxlib 0.10.1** — all four variants tested fail or
anti-scale: 128 fake devices/node crashes (coordination-service heartbeat death) or hangs to
the wall limit; 16 procs/node × 2 nodes *runs* but at 12.9 s/step (6× slower than the same
npes on one node — gloo-over-TCP latency); 4 nodes aborts. JAX-CPU is effectively
**single-node**. The untested escape hatch is `jax_cpu_collectives_implementation='mpi'`
(accepted by this jaxlib; needs an MPIwrapper shim built against the system MPI).

## Measured: GPU three-transport comparison (2026-07-13, same day / same nodes)

Full model (ice+KPP+GM, JRA-1958, PHC IC), compile excluded, A100s. Jobs 26227814 (CORE2 4 GPU),
26227815 (CORE2 8 GPU = 2 nodes), 26227983 (dars dist_16 = 4 nodes). Logs:
`scripts/logs/bench_{core2_halo3,dars_halo16}.*.log`.

| mesh | GPUs | all_gather | ragged | padded |
|---|---|---|---|---|
| CORE2 (127k) | 4 (1 node) | 90.1 ms | 86.1 ms | **80.1 ms** |
| CORE2 (127k) | 8 (2 nodes) | 185.5 ms | 116.1 ms | **78.1 ms** |
| dars (3.16M) | 16 (4 nodes) | — (OOMs/hangs at this scale) | **512.9 ms** | 525.0 ms (+2.4 %) |
| dars (3.16M) | 32 (8 nodes, job 26228413) | — | 319.6 ms | 349.1 ms (+9.2 %) — **but see below** |

**On GPU the padded transport is the fastest CORE2 exchange — it crosses the node boundary flat
(80→78 ms) where ragged loses a third (86→116) — and ties ragged at dars-16** (equal peak memory,
22.0 vs 22.1 GiB). So the CPU campaign's conclusion holds on GPU at both ends of the mesh range:
the AD-correct transport costs nothing over the forward-only optimum (here it often *wins*: at
latency-bound halo sizes one dense tiled `all_to_all` beats per-neighbour ragged bookkeeping).
Same-day ragged/all_gather controls agree with the older two-transport rows (86.7/93 @4, 122.7/187.5
@8, 513 @dars-16) within a few percent.

⚠️ **The dars-32 "ragged +9 %" row DID NOT REPRODUCE — treat it as one noisy sample, not a result.**
A second same-day/same-node run (job 26232224, 2026-07-14) measured **ragged 346.9 ms vs padded
348.7 ms — a 0.5 % tie.** Padded reproduced to 0.1 % across the two runs (349.1 → 348.7); **ragged
swung 8.6 %** (319.6 → 346.9). So at dars-32 the two transports are level within run-to-run variance,
and *ragged is the noisy one* — a useful thing to know before quoting any ragged margin. (This row is
also quoted in the paper's §5; it needs the same softening.) The honest statement is: **padded ≈ ragged
at dars-32**, and the crossover claim should not rest on a single allocation.

### Why — and how to predict the crossover without burning a job

The crossover is not a mystery to be measured mesh-by-mesh: it follows from the padded transport's
wire volume, which is fixed by the **partition files alone**.

`lax.all_to_all` demands a *uniform* message size — that is precisely what buys the static shape and
the correct transpose. So padded sizes every message at `pad_slot = max` over **all** (d,e) pairs of
the d→e chunk, and sends one such message to **every rank, neighbour or not**: the message to a
non-neighbour is a slot of pure zeros, and the message to a real neighbour is its chunk zero-padded
up to `pad_slot`. Each device therefore puts `P × pad_slot` lanes on the wire, where ragged puts only
the real chunks (`send_max` lanes). So

    pad factor = P · pad_slot / send_max

and it grows ~linearly in `P` for the dumbest possible reason: **a spatial partition gives every
device ~6–7 real neighbours no matter how large `P` gets, but padded allocates a slot for all `P`.**
The other ~`P−7` slots are zeros. `scripts/bench/probe_pad_factor.py` computes this in seconds on a
login node (no mesh, no GPU):

| config | P | pad_slot | pad lanes | ragged lanes | pad factor | real nbrs | occupancy | measured padded vs ragged |
|---|---|---|---|---|---|---|---|---|
| CORE2 | 4 | 167 | 668 | 381 | 1.8× | 3.0 | 45.5 % | **−7 %** (padded wins) |
| CORE2 | 8 | 117 | 936 | 274 | 3.4× | 2.5 | 18.5 % | **−33 %** (padded wins) |
| dars | 16 | 691 | 11,056 | 2,029 | 5.4× | 5.5 | 11.2 % | +2.4 % (tie) |
| dars | 32 | 767 | 24,544 | 1,602 | 15.3× | 5.9 | 4.2 % | +9.2 % (ragged wins) |
| NG5 | 64 | 662 | 42,368 | 1,927 | 22.0× | 6.4 | 3.2 % | (job 26231361) |
| NG5 | 128 | 494 | 63,232 | 1,552 | 40.7× | 6.8 | 1.6 % | — not worth running |

(node lanes; `elem` tracks it within 0.1×. Occupancy = fraction of the padded buffer that is real
data.) The measured winner flips exactly where the pad factor does: **padded wins while the pad
factor is ≲5, ties at ~5, and loses beyond ~15.** Two consequences worth stating plainly:

- **Padded's wire volume grows with P while ragged's does not** — and the *local* field it is
  refreshing shrinks as ~N/P. Those scissors close. Per device, per exchange, one `[Lmax, nl]` f64
  node field at NG5 (nl=70):

  | | ragged | **padded** | local field (`Lmax·nl·8`) | all_gather (global field) |
  |---|---|---|---|---|
  | 64 GPUs | 1.08 MB | **23.7 MB** | 66.1 MB | 4,146 MB |
  | 128 GPUs | 0.87 MB | **35.4 MB** | **33.4 MB** | 4,146 MB |

  At 128 GPUs padded puts **more bytes on the wire (35.4 MB) than the local array it is refreshing
  (33.4 MB)**, to do a job ragged does in 0.87 MB. (Per *device* — both transports remain ~100×
  under all_gather, which ships the whole 7.4 M-node global field, 4.1 GB, to everyone. Padded is
  not "worse than gathering everything"; it is just paying an unbounded padding tax.) No timing run
  is needed to know padded loses there — which is why the NG5-128 A/B (≈64 GPU-h) was not run.
- Padded's cost is *pure padding*, not algorithmic: at P=128 it is 98.4 % zeros on the wire. See the
  proposed fix below.

### Implemented: the coloured-`ppermute` transport (Phase 8d, `use_padded`'s sibling)

**Status: MERGED, fully gated, and it WINS AT THE LARGEST SCALE.** `use_coloured=True` / `--coloured`
/ `--halo coloured`. Correctness is settled — forward bit-exact and gradient-exact vs the `all_gather`
oracle (all 3 kinds, dist_2/4/8), bit-identical to `padded` through a full sharded step, end-to-end
gated through the CG (`test_halo.py::test_coloured_*`,
`test_step_sharded.py::test_coloured_step_matches_allgather`).

Full model, same-day/same-node GPU A/B (jobs 26232647/8, 26232224, 26232225). **K = Δ = the max
number of real neighbours = the number of `ppermute` rounds:**

| mesh | GPUs | K | all_gather | ragged | padded | **coloured** |
|---|---|---|---|---|---|---|
| CORE2 | 4 | 3 | 89.9 ms | 86.5 ms | **80.4 ms** | 92.5 ms |
| CORE2 | 8 | 3 | 185.7 ms | 116.1 ms | **78.5 ms** | 84.5 ms (−27 % vs ragged) |
| dars | 32 | 12 | — | **346.9 ms** | 348.7 ms (+0.5 %) | 385.4 ms (+11 %) |
| **NG5** | **64** | **10** | — | 592.1 ms | 741.9 ms (+37 %) | **543.3 ms (−9 % vs ragged)** |

**At NG5-64 — the biggest mesh at the largest scale — the coloured transport is the FASTEST, beating
ragged by 9 % and padded by 37 %.** That is the whole point of it: the regime where the AD-correct
transport used to be most expensive (padded's pad factor is 22× there) is now the regime where it is
*cheapest*. A sharded gradient at NG5-64 costs **less** than the forward-only ragged halo, not more.

**Why it flips with P.** `ragged_all_to_all` is semantically a **P-way** all-to-all — its cost grows
with the device count — whereas coloured's cost is bounded by `K = Δ ≈ 3–14` rounds *regardless of P*
(a spatial partition's neighbour count does not grow). At P=32 ragged's 32-peer collective still beats
12 sequential permutes; by P=64 it does not. Meanwhile padded's `P·pad_slot` volume is growing
quadratically-ish in the same direction. So the three curves cross at different places, and **coloured
is the only one whose cost does not grow with P.**

⚠️ **A single-scale micro-benchmark led me to exactly the wrong conclusion here — do not repeat it.**
`scripts/bench/bench_halo_micro.py` at dars/dist_16 (job 26232853, per exchange) says:

| | lanes | 3-D MB | 2-D µs/exch | 3-D µs/exch |
|---|---|---|---|---|
| padded | 11,056 | 5.04 | **93.6** | 518.3 |
| coloured | 2,443 | 1.11 | 273.4 | 509.3 |
| ragged | 2,029 | 0.93 | 169.6 | **269.9** |

From this it is tempting to declare "`ragged_all_to_all` IS the coloured transport done natively in
one collective, so coloured can never beat it." **That is false, and NG5-64 disproves it** — the
micro-benchmark is one point (P=16), and the ranking is P-dependent. What the micro-benchmark *does*
establish, and what remains true, is the cost structure: coloured LOSES the tiny 2-D exchange
(latency: K collectives vs 1) and WINS the big 3-D one (bytes). The CG fires ~2 × ~127 = **~254 2-D
exchanges per step** against a few dozen 3-D ones, which is why coloured loses on small meshes where
the 2-D population dominates — and why it wins once the 3-D fields get big enough to pay for K.

Follow-ups, in priority order:

1. **Hybrid:** `padded` for the CG's 2-D ssh exchange (93.6 µs, the best 2-D number, ~254 of ~280
   exchanges/step) + `coloured` for the 3-D fields. It takes each AD-correct transport where it
   already wins, and should extend coloured's NG5-64 lead *and* rescue it at dars-32/CORE2. Both map
   sets are already built and `SSHHalo`/`HaloCtx` route independently ⇒ a wiring change, not a new
   transport. **This is now the highest-value item in the halo layer.**
2. **NG5-128** (P=128, K=14): the P-scaling argument predicts coloured's lead *grows*. Previously
   dismissed because padded could not win there — that reasoning does not apply to coloured.
3. Fewer rounds: K = Δ is set by the *max*-degree device; merging small colour classes trades a little
   padding back for fewer collectives.

#### The design (why it is AD-correct)

`all_to_all` sends one message per rank **by definition**, so "cap padded at the real neighbours" is
a contradiction — the only lever it offers is `pad_slot`, not the message count. The fix has to
change the collective. `lax.ppermute` does point-to-point sends along a *static* permutation, so:

1. **Edge-colour the neighbour graph** (left = senders, right = receivers ⇒ *bipartite*, so König
   gives exactly Δ colours, not Vizing's Δ+1). Each colour class is a partial permutation: every
   device sends to ≤1 and receives from ≤1 peer — i.e. one legal `ppermute`.
2. **One `ppermute` per colour**, each with its **own** slot width (the max chunk *in that class*,
   not the global max). K = Δ ≈ 3–14 rounds.

Volume (`probe_pad_factor.py`, last columns) — it lands at **ragged's optimum, 1.0–1.4×, at every
scale**, because per-round slots kill the chunk-variance tax as well as the zero-slot tax:

| | ragged | padded | **coloured-ppermute** | K |
|---|---|---|---|---|
| CORE2-4 | 0.15 MB | 0.3 MB | **0.15 MB (1.0×)** | 3 |
| dars-32 | 0.62 MB | 9.4 MB | **0.83 MB (1.3×)** | 12 |
| NG5-64 | 1.08 MB | 23.7 MB | **1.51 MB (1.4×)** | 10 |
| NG5-128 | 0.87 MB | 35.4 MB | **1.22 MB (1.4×)** | 14 |

**29× less wire volume than padded at NG5-128 — with a correct transpose.** All four primitive
assumptions are verified on XLA:CPU: `ppermute` runs on CPU; its reverse-mode transpose is exact;
the composite *gather → ppermute* (the real shape of a halo exchange) is exact in forward **and**
gradient against a dense oracle; and partial permutations are legal, with omitted destinations
receiving **exactly zeros** and correct (zero) cotangents for non-senders.

**The one thing the volume model could not predict is latency — and that is what decides it.** The
transport trades 1 fused dense collective for K, and FESOM's exchanges are dominated by the CG's
~254 tiny 2-D ones per step, where collective count beats bytes. Hence the measured CORE2 loss above,
despite coloured shipping the fewest bytes of any transport. Keep this in mind before proposing any
"obviously cheaper" collective: **in this model, count the exchanges before you count the bytes.**

### What this means for the two production chains

| chain | partition | halo today | pad factor | switch to padded? |
|---|---|---|---|---|
| CORE2 hindcast | `dist_4` (1 node) | all_gather (the default — no flag passed) | 1.8× | **yes — measured 90.1 → 80.1 ms, ≈11 % free** |
| FORCA20 hindcast | `dist_16` (live; the config's `dist_32` compile-hangs) | ragged (`--ragged`) | **6.1×** | **no — predicted ~tie-to-slight-loss** |

FORCA20/dist_16 sits at pad factor 6.1×, essentially dars-16's 5.4× — and dars-16 measured padded
**2.4 % _slower_** than ragged. So the "switch the production chains to padded for a free 5–12 %"
idea is right for CORE2 (which is still on `all_gather`, the slowest of the three) and **wrong for
FORCA20**, which is already on the transport the pad factor says it should be on. Confirming the
FORCA20 prediction is a cheap 4-node A/B if it ever matters; the forward chain needs no gradient, so
there is no correctness reason to move it either.

## Measured: CORE2 full model, one Levante node (128-core Milan, dt=1800, 25 steps)

Full model = KPP + GM/Redi + prognostic ice + real JRA55-1958 + PHC IC; compile excluded;
SYPD = 4.93 / per-step-seconds. Jobs 26218456–26219539.

| topology | halo | npes | ms/step | SYPD |
|---|---|---|---|---|
| 1 fake device (dense) | — | 1 | 5952 | 0.83 |
| in-process | all_gather | 8 | 4330 | 1.14 |
| in-process | all_gather | 32 / 128 | **crash** (rendezvous bug, deterministic) | — |
| in-process | padded | 8 / 16 / 32 / 64 / 128 | 2837 / 2523 / 2635 / 3209 / 5131 | 1.74 … 0.96 |
| **multi-process gloo** | **padded** | **8×16 cores** | **1508** | **3.27** |
| multi-process gloo | padded | 16 / 32 / 64 procs | 1630 / 2101 / 3406 | 3.02 / 2.35 / 1.45 |

XLA flag probes at the optimum (fast-math incl. no-honor-nans/infs; concurrency-optimized
scheduler): **no effect** (±0.5%) — the kernels are gather/memory-bound, not FLOP-bound.
Cost decomposition (in-process 16): base ocean 55%, KPP+GM 28%, ice EVP 20% — uniform
slowness, no pathological component.

## Measured: mesh-size scaling (per-node throughput, M node-levels/s — dt-independent)

| mesh (nodes) | in-process best | multi-process best | notes |
|---|---|---|---|
| CORE2 (127k) | 2.4 | **4.0** | |
| farc (638k) | 2.0 | **5.3** | multiproc 3× faster than in-process here; needs the 512 GB node (per-rank setup) |
| dars (3.16M) | **3.1** | — (per-rank setup OOM) | in-process only; needs the **1 TB node** (OOM at 240/480 GB); 58.4 s/step, 633 s compile |

Per-node efficiency *improves* with mesh size under multiproc — more work per shard
amortises the fixed per-step overhead.

## Reference points (same CORE2 configuration)

| | SYPD | footprint | throughput/node (Mnodlev/s) |
|---|---|---|---|
| Fortran FESOM2 | 87.8 | 512 ranks / 4 CPU nodes | 27.1 |
| Kokkos-CPU | 80.4 | 512 ranks / 4 CPU nodes | 24.8 |
| **JAX GPU** | **53–57** | 1 node / 4×A100 | ~67 |
| JAX CPU (best, padded + multiproc) | 3.27 | 1 node (its ceiling) | 4.0 |

So JAX-CPU is ~7× per node behind Fortran-CPU, factored as **~2.7× kernel efficiency**
(the same "array-framework tax" Veros reports — Häfner et al. 2021 measure their JAX
backend within 1.0–1.4× of *pyOM2* at ≤256 cores, but on a structured C-grid, against a
research Fortran code, and parallelised with real MPI via mpi4jax) **× ~2.5× parallel
efficiency** (XLA:CPU cannot express Fortran's 512 cache-resident ranks; multi-node is
broken). Conclusion: **CPU is for development, tests, gates, laptops, and sharded-gradient
verification — production throughput is GPU (JAX ≈ Kokkos there) or Fortran.**

## Known upstream bugs found here (jax/jaxlib 0.10.1)

1. `lax.ragged_all_to_all` reverse-mode transpose wrong (O(1), everywhere) — the original
   blocker; full record + repro: [JAX_RAGGED_A2A_BUG.md](JAX_RAGGED_A2A_BUG.md).
2. XLA:CPU all_gather rendezvous check-failure at ≥32 in-process devices (deterministic).
3. `jax.distributed` CPU multi-node: coordination-service heartbeat death / hangs with
   ≥128 fake devices per process; hybrid (procs × in-process devices) hangs too.

## Quick recipes

```bash
# CPU, one node, the production topology (padded + 8 procs x 16 cores):
export JAX_PLATFORMS=cpu JDIST_CPU=1 XLA_FLAGS="--xla_force_host_platform_device_count=1"
srun -n 8 -c 16 python scripts/bench/bench_forward_scaling.py --npes 8 --halo padded --full 1 ...

# CPU gates/tests (in-process fake devices — fine at npes<=16, padded only past that):
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    python -m pytest fesom_jax/tests/test_halo.py

# GPU forward, by scale (all four transports are forward BIT-EXACT — pick purely on speed):
#   <=16 GPUs (either mesh class): --halo padded    — fastest measured (80/78 ms CORE2 4/8)
#   ~32 GPUs:                      --ragged 1       — but padded is level there (0.5%), so either
#   >=64 GPUs, big mesh:           --halo coloured  — FASTEST at NG5-64 (543 vs ragged 592, padded 742)
#   JDIST=1 for multi-node in every case.
# GPU sharded GRADIENTS: use_coloured=True (big meshes — AD-correct AND fastest) or use_padded=True
#   (<=32 GPUs). NEVER use_ragged: its transpose is broken and run_steps_sharded refuses it.

# Which transport will win at a new (mesh, npes)? Ask the partition files, not a GPU job —
# seconds on a login node; pad factor <~5 => padded, >~15 => ragged:
python scripts/bench/probe_pad_factor.py --only dars-32
```
