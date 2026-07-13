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
| **ragged** (`lax.ragged_all_to_all`) | `use_ragged=True` / `--ragged 1` | **GPU only** | ❌ **broken** — JAX's transpose is wrong, O(1) error ([JAX_RAGGED_A2A_BUG.md](JAX_RAGGED_A2A_BUG.md)) | minimal (only the actual halo: 0.1–0.3 MB at CORE2) | forward-only production at ≥32-GPU scale (NG5-class, where all_gather OOMs and padded is untested); at ≤16 GPUs padded ties or beats it (table below) |
| **padded** (slot-padded dense `lax.all_to_all`) | `use_padded=True` / `--halo padded` | **CPU + GPU** | ✅ **correct by construction** — the transpose of `all_to_all` is another `all_to_all` | ragged × padding factor: 1.8×@P4, 3.4×@P8, 6.3×@P16, 11.6×@P32 (measured, CORE2) — still 50–140× under all_gather | **any CPU sharded run**; **any sharded gradient** past all_gather's memory/speed limit; **GPU forward at ≤16 GPUs** — fastest measured transport there (table below) |

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
| dars (3.16M) | 32 (8 nodes, job 26228413) | — | **319.6 ms** | 349.1 ms (+9.2 %) |

**On GPU the padded transport is the fastest CORE2 exchange — it crosses the node boundary flat
(80→78 ms) where ragged loses a third (86→116) — and ties ragged at dars-16** (equal peak memory,
22.0 vs 22.1 GiB). So the CPU campaign's conclusion holds on GPU at both ends of the mesh range:
the AD-correct transport costs nothing over the forward-only optimum (here it often *wins*: at
latency-bound halo sizes one dense tiled `all_to_all` beats per-neighbour ragged bookkeeping).
Same-day ragged/all_gather controls agree with the older two-transport rows (86.7/93 @4, 122.7/187.5
@8, 513 @dars-16) within a few percent. **The crossover is at 16–32 GPUs on big meshes**: at
dars-32 ragged pulls ahead by 9 % (the pad factor grows with P — 11.6×@P32 measured at CORE2), so
ragged remains the forward default at ≥32 GPUs; below that, prefer padded everywhere. NG5-64
padded-vs-ragged pending (job 26228433) — expected to widen ragged's lead, guidance unchanged
either way; a gradient at that scale still uses padded (the only correct choice there).

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

# GPU forward, <=16 GPUs (either mesh class): --halo padded — fastest measured (see the
#   three-transport table above). >=32 GPUs on huge meshes: --ragged 1 (padded untested there);
#   JDIST=1 for multi-node either way.
# GPU sharded GRADIENTS: use_padded=True in run_steps_sharded(..., return_grad_fn=True).
```
