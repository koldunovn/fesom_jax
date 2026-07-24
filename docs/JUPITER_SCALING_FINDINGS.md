# FESOM2-JAX on JUPITER (GH200) — campaign FINDINGS

*Executed 2026-07-23 per `docs/JUPITER_SCALING.md`. Machine: JUPITER booster, quad-GH200
nodes (4 GPUs/node, 72-core Grace each, NDR InfiniBand), account `e-sta-destine`.
Numbers: `scripts/bench/jupiter/reduce_jupiter.py` over `scripts/logs/jupiter/*.out`.*

This file records what was **measured on the machine**. It supersedes every `⚠️VERIFY`
marker in `JUPITER_SCALING.md`.

---

## 1. The environment — the doc's "crux" was a non-issue (§2)

`JUPITER_SCALING.md` predicted jaxlib-on-aarch64 would be the whole risk and recommended an
NGC container. **No container is needed.** Working GPU JAX in ~3.5 minutes, pure wheels:

```bash
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5
python -m venv $HOME/jaxenv && source $HOME/jaxenv/bin/activate
pip install -U pip wheel
pip install "jax[cuda12]==0.10.1"       # aarch64 cp313 wheels EXIST
pip install -e ".[dev,viz]"
```

- **Pin 0.10.1 — the exact Levante campaign version.** Zero API drift, and it makes the
  GH200-vs-A100 comparison a hardware comparison rather than a version comparison.
- **The JSC `jax/0.8.1` module is CPU-ONLY** ("An NVIDIA GPU may be present on this machine,
  but a CUDA-enabled jaxlib is not installed"). It cannot be used. This is the single most
  useful negative result for anyone repeating this.
- The venv is built on the module Python, so **the module must be loaded before activating
  it** or you get `libpython3.13.so.1.0: cannot open shared object file`.
- Login nodes have a GH200, so gate J0 runs interactively — but they are **shared**; a
  contended login GPU measured core2/1 at 290 ms vs 98.8 ms on a compute node. Never time
  anything on the login node.

Verified env: `scripts/bench/jupiter/env_jupiter.sh` (all `CHANGEME`s resolved).

## 2. Data — no 24 GB transfer from Levante was needed (§1)

The doc budgeted a `.npy` mesh-export transfer (ng5 ~24 GB) and said `data/mesh_core2` ships
in the repo. **Both wrong.** The repo ships only the tiny `pi` mesh (`fesom_jax/data/mesh_pi`);
on Levante `data/` was a symlink to `/work`. But `scripts/prepare_mesh.py` is standalone and
C-verified, so every export was **generated on JUPITER** from the raw FESOM meshes the Kokkos
twin already staged:

| mesh | nod2D | nl | export | prepare | PHC IC cache |
|---|---:|---:|---:|---:|---:|
| core2 | 126,858 | 48 | 298 MB | 6.5 s | 12.8 s |
| farc | 638,387 | 48 | 1.5 GB | 32 s | 58.8 s |
| dars | 3,160,340 | 57 | 5.7 GB | ~3 min | 364.7 s |
| ng5 | 7,402,886 | 70 | 16 GB | ~6 min | 850.7 s |

`scripts/bench/jupiter/prepare_mesh_jupiter.sbatch` does both steps per mesh (idempotent —
it skips whatever already exists). Run it on a compute node: peak host RAM is several times
the export size.

Verified data paths (all pre-staged, read in place):

| what | path |
|---|---|
| JRA55-do forcing | `/e/scratch/hclimrep/koldunov1/meshes/JRA55-do-v1.4.0` |
| PHC3.0 winter IC | `…/meshes/phc3.0/phc3.0_winter.nc` |
| SSS | `…/JRA55-do-v1.4.0/PHC2_salx.nc` |
| runoff | `…/JRA55-do-v1.4.0/runoff.nc` — **not** `CORE2_runoff.nc` as on Levante |
| chlorophyll | `…/meshes/Sweeney/Sweeney_2005.nc` |
| raw meshes + `dist_<N>` | ng5 → `/e/scratch/hclimrep/…/ng5`; core2/dars/farc → `/e/scratch/e-sta-destine/…/meshes` |
| JAX `.npy` exports (generated) | `/e/scratch/e-sta-destine/koldunov1/fesom_jax_meshes` |

Partitions exist for every rung used (ng5 to 8192, dars to 4096). No partitioner was run.

## 3. Gates (§3)

| gate | result |
|---|---|
| **J0** env | PASS — `CudaDevice`, jax 0.10.1, float64 OK |
| **J1** 1 node / 4 GPU | PASS — padded 45.15 ms, coloured 52.31 ms, bench-finite clean |
| **J2** 2 nodes / 8 GPU over IB | **PASS** — see below |
| **J3** bench-finite every row | PASS — 0 non-finite rows across the whole campaign |

**J2, the load-bearing gate.** NCCL is on the InfiniBand fabric with GPUDirect RDMA, using
the **JSC defaults** — no `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` tuning was needed (the Kokkos
twin found the same: JSC defaults beat hand-tuning):

```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [RO]
NET/IB/0..3/GDRDMA/Shared          # all four 200G HCAs, one per superchip
```
Zero `NET/Socket` for the inter-node ring. `NCCL_DEBUG=INFO` cost <0.2% (46.23 vs 46.15 ms).

**Cross-check on physics agreement:** `max_uv = 1.308 m/s` was *identical* across padded and
coloured and across 4 and 8 GPUs — the transports and decompositions agree physically, not
merely numerically.

### Benign noise to expect
`cuda_vmm_allocator.cc: VMM cuMemCreate with FABRIC+POSIX_FD handle types failed:
CUDA_ERROR_NOT_PERMITTED ... will retry with simpler handle types` — hundreds of lines per
run. **Harmless and bounded**: it is emitted at allocation time only (never per step), the
retry succeeds, and timings are unaffected. JUPITER's quad-GH200 nodes are not an MNNVL
fabric domain, so the fabric-shareable handle is simply unavailable. Filter it from logs.

## 4. THE transport result — Levante's rule does not transfer

`JUPITER_SCALING.md` §4 inherited the Levante rule *"padded ≤16 GPUs, coloured beyond"*.
**On JUPITER that rule is wrong**, and following it would have recorded every 8- and 16-GPU
point ~4× too slow — a fake collapse exactly where the paper claims scaling.

Measured during the campaign (four of our own jobs, 116 nodes, running concurrently):

| point | padded | coloured | ratio |
|---|---:|---:|---:|
| core2 8 GPU (2N) | 190.4 / 181.5 / 204.0 / 205.5 / 220.0 / 224.2 ms | **49.1 / 48.9** | 4.2× |
| core2 16 GPU (4N) | 339.9 ms | **60.8 / 62.3** | 5.5× |
| farc 8 GPU (2N) | 432.0 / 446.1 ms | **156.3** | 2.8× |
| farc 16 GPU (4N) | 413.2 ms | *(pending)* | — |
| dars 16 GPU (4N) | 204.9 ms | **183.0** | 1.12× |
| dars 32 GPU (8N) | 234.5 / 243.4 ms | **133.4 / 139.6** | 1.76× |

`scripts/bench/jupiter/diag_2node.sbatch` isolated the cause on one allocation. It is **not**
step count (50 → 224 ms vs 150 → 204 ms), **not** the `timeout` wrapper (204.0 vs 205.5 ms),
and **not** drift (repeat of leg A: 224 → 220 ms). It is the transport.

### What the mechanism is NOT — a hypothesis that was tested and REJECTED
The obvious explanation was self-contention: the campaign ran four of our own jobs (116
nodes) concurrently, and padded — pads every halo buffer to the global max, exchanges P-way,
so bandwidth-bound — should suffer where coloured (1.0–1.4× the true halo in K rounds at any
P, tiny volume) would not. **This was written up as the finding, then disproved.**
`transport_map_jupiter.sbatch` ran with *nothing else of ours in the queue* and measured
core2/8 padded at **189.2 ms** — indistinguishable from the "contended" 181–224 ms.
**Self-contention is not the cause.** Recorded here because the wrong version was believed
for about an hour and the reasoning looked sound.

### Second hypothesis — "the fabric degraded" — ALSO REJECTED, by direct measurement
`scripts/bench/jupiter/fabric_probe.py` measures NCCL all-reduce bandwidth through the same
jax.distributed stack the model uses (`fabric_probe.sbatch`, 2026-07-24 02:0x, the same window
in which padded was measuring 4× slow):

| | all-reduce @256 MB | alg GB/s | **bus GB/s** |
|---|---:|---:|---:|
| 1 node, 4 GPU (NVLink only, no fabric) | 5.45 ms | 46.9 | **70.4** |
| 2 nodes, 8 GPU (first inter-node hop) | 7.66 ms | 33.4 | **58.5** |
| 4 nodes, 16 GPU | 10.18 ms | 25.2 | **47.2** |
| 8 nodes, 32 GPU | 11.44 ms | 22.4 | **43.4** |

**The fabric is healthy.** Crossing from NVLink-only to inter-node costs just 17%
(70.4 → 58.5 GB/s), and 58.5 GB/s bus is ~58% of the 100 GB/s/node injection limit
(4 × NDR200) — a normal NCCL all-reduce efficiency. The decay with node count is the expected
all-reduce behaviour, not a defect. **So padded's ~4× disadvantage is a real property of the
transport on this machine, not a degraded network.**

> **Probe limitation, stated plainly.** The small-message rows (0.34–0.42 ms at 4 KB) are
> dominated by **JAX per-call dispatch**, not collective latency — the 1-node NVLink case
> shows the same 0.344 ms floor, and inside the model's single jitted 150-step function no
> such Python dispatch is paid. This probe therefore measures **bandwidth** honestly and says
> **nothing** about the latency regime — which is precisely the regime a halo exchange lives
> in (CORE2 issues ~254 tiny 2-D exchanges per step). A latency characterisation needs the
> collectives timed *inside* one jitted graph; that is not done here.

### What remains unexplained
| when | job | core2/8 padded | other jobs of ours |
|---|---|---:|---|
| 22:53 | gates | **46.2 / 46.2** | none (2 tiny prep jobs) |
| 23:0x | core2 campaign | 190.4 / 181.5 | 3 jobs, 52 nodes |
| 23:15 | diag legs A–D | 224.2 / 204.0 / 205.5 / 220.0 | 4 jobs, 116 nodes |
| 01:0x | transport map | 189.2 | **none** |

Padded was fast **once**, before ~23:00, and has been ~4× slower in every one of ~8 subsequent
attempts, under load and idle alike. Coloured is flat throughout (51.1 → 49.1 → 48.9 ms).
With both self-contention and fabric degradation ruled out, **the single 46.2/46.2 ms gates
measurement stands unexplained.** It is one observation against many; the campaign treats it
as suspect rather than as padded's true best case, but it has NOT been explained away and
should not be silently dropped.

### The measured instability (why the transport map was abandoned)
`transport_map_jupiter.sbatch` ran on a quiet queue with the three transports **interleaved
within each rep**, and was **cancelled after 2 of 8 points** because the baseline was drifting
faster than the effect being measured:

| point | transport | rep 1 | rep 2 | spread |
|---|---|---:|---:|---:|
| core2:8 | **coloured** | 49.10 | 48.98 | **0.2%** |
| core2:8 | padded | 189.16 | 162.29 | 16% |
| core2:8 | ragged | 322.73 | 612.53 | 90% |
| core2:16 | **coloured** | 66.25 | 69.55 | 5% |
| core2:16 | padded | 324.99 | 253.67 | 28% |
| core2:16 | ragged | 553.51 | 750.10 | 36% |
| farc:8 | coloured | 209.36 | 282.53 | 35% |
| farc:8 | padded | 230.27 | 405.72 | 76% |
| farc:8 | ragged | 154.21 | 734.76 | **376%** |

Two things follow. (1) **The bandwidth-bound transports are not merely slower, they are
irreproducible** — a single rep cannot rank them, and `ragged` winning farc:8 rep 1 (154 ms)
is not a result. (2) Coloured was tight at core2 (0.2–5%) and had degraded to 35% at farc an
hour later, i.e. the machine's inter-node behaviour was drifting *during* the map. Continuing
would have spent ~64 more node-hours producing a ranking that would not reproduce. The booster
was ~28% allocated to other users throughout (1611 alloc / 2766 idle), so external congestion
is the most likely driver, but this was not isolated.

**What is safe to conclude, and all this campaign claims:** coloured is the fastest AND the
only reproducible multi-node transport in every head-to-head measured here, so it is what all
multi-node points use. **A definitive JUPITER transport map — the thing that would replace
Levante's rule with a measured one — is NOT delivered** and needs a stable window plus more
reps per point.

### Practical rule for JUPITER: `--halo coloured` for every multi-node point

That is the operational conclusion, and it is well supported: coloured won every head-to-head
measured here and is the only transport that reproduces. What is **not** established is *why*
padded loses by 4×, or what padded's ceiling on this machine is.

A third mechanism was also checked and rejected: **pad factors are not the cause.** The padded
transport pads every device's halo to the global max, so an unbalanced partition would tax
everyone — but all partitions are near-perfectly balanced (`padfac` 1.00–1.01 for core2, farc
and dars at every rung; measured with `partit.read_partition` + `shard_mesh.local_sizes`).

**Three hypotheses proposed, three rejected by measurement** (self-contention, fabric
degradation, pad imbalance). The transport gap is real and reproducible; its mechanism is open.

## 5. GH200 vs A100 — the comparison, done like-for-like

Two different A100 baselines exist and they must not be mixed:

1. **`docs/PARALLELISM.md`: "JAX GPU 53–57 SYPD, 1 node / 4×A100"** — this is the **LEGACY**
   `ice+kpp+gm` model, *not* the paper's production physics. Dividing a production-physics
   JUPITER row by it would report a physics change as a hardware speedup.
2. **The fig10 v2 production campaign** (`scripts/bench/fig10prod/README.md`) — same code,
   same jax 0.10.1, 150 steps, ≥2 reps, bench-finite gate, per-point banked-best transport,
   and the same per-mesh physics/dt. **This is the comparable baseline**, and the JUPITER
   sbatch was verified flag-for-flag against it (core2 dt=1800 +GM; farc dt=900; dars dt=120;
   ng5 dt=180; all `--ice 1 --mevp 1 --tke 1 --kpp 0 --zstar 1`).

`scripts/bench/jupiter/calib_vs_levante.sbatch` measures the legacy model on GH200 so
baseline (1) is usable too:

| CORE2, 4 GPUs, identical LEGACY physics | ms/step |
|---|---:|
| 4×A100 (Levante, `PARALLELISM.md` 53–57 SYPD) | 86.5–93.0 |
| 4×GH200 (JUPITER, measured) | **41.46** (41.47 / 41.45) |
| **GH200 / A100** | **2.09–2.24×** |

Same allocation, same job, the production model costs **14% more** than legacy
(47.36 / 47.37 vs 41.47 / 41.45 ms) — which is exactly the error that would have been
smuggled into a hardware ratio.

### Overnight reproducibility — the campaign's compute numbers STAND
The campaign ran while four of our own jobs were live and while the transport behaviour was
unstable (§4), so the whole result set was re-checked the next day
(`scripts/bench/jupiter/repro_1node.sbatch`, job 1033980, 2026-07-24, fresh node, quiet queue).

**Single-node points were chosen deliberately: they touch ZERO inter-node fabric** (4 GPUs on
one node, NVLink only), so they separate "are the kernels stable?" from "is the network
stable?" — and both headline GH200-vs-A100 ratios come from single-node points.

| point | campaign (07-23) | repro (07-24) | delta |
|---|---:|---:|---:|
| core2 1 GPU padded | 98.6 / 98.8 | 99.06 | +0.3% |
| core2 2 GPU padded | 64.1 / 64.3 | 63.96 | −0.4% |
| core2 4 GPU padded | 43.71 / 44.51 | 43.97 | centred |
| farc 4 GPU padded | 146.50 | 146.48 | **0.01%** |
| core2 1 GPU coloured | 93.19 / 93.77 | 93.84 | +0.1% |
| core2 2 GPU coloured | 64.11 | 64.10 | 0.02% |

**6/6 reproduce within 0.5%, across both transports.** Consequences:
1. The kernels and per-GPU throughput are stable overnight ⇒ **both published A100 ratios are
   confirmed to 2 s.f.** (core2/4: 88.0/43.97 = 2.00× vs published 2.01×; farc/4:
   283.0/146.48 = 1.93×).
2. Whatever instability this machine exhibits is a **fabric-only** phenomenon touching the
   multi-node rungs; it does not reach compute.

The ng5 remeasure independently corroborates this on multi-node points: its P=64 and P=256
anchors both reproduced (235.9 and 247.7 vs 243.4 and 257.2).

> **NOT re-verified.** The dedicated multi-node reproducibility probe
> (`repro_multinode.sbatch`, job 1033981 — dars 16/32/64/128 and farc 8/16) was **cancelled
> before it ran** at the user's instruction once the single-node set came back clean. So the
> two multi-node claims — **dars turning over at 64** and **farc peaking at one node** — still
> rest on the campaign's original measurement set, taken under self-contention. They are the
> claims most exposed to fabric instability and should be re-run before publication.

### Error bars (measured, not assumed)
- **rep-to-rep inside one allocation: 0.05%** (41.47 vs 41.45; 47.36 vs 47.37)
- **allocation-to-allocation: ~5%** (core2/4 production: 45.15 in the gates job, 47.36 in the
  calib job, 43.71–44.51 in the campaign job)

Allocation variance is ~100× the rep noise. **That is the real error bar on any cross-machine
ratio**, and it is why every rung of a curve is run inside ONE allocation
(`bench_scaling_jupiter.sbatch` sub-allocates with `srun` rather than submitting per-rung
jobs). The Levante campaign lost a session to an "allocation fluke" at 128 GPUs; this is the
protective measure.

## 6. THE RESULTS — strong scaling, 1 → 256 GPUs

Production physics per mesh, 150 steps, ≥2 reps, best transport per point, **every row
bench-finite clean (85 rows, 0 dropped)**. SYPD at the production dt. `GH200/A100` is against
the Levante fig10 v2 production campaign at the same GPU count (§5).

| mesh | GPUs | ms/step | SYPD | par.eff | GH200/A100 |
|---|---:|---:|---:|---:|---:|
| **core2** 127k | 1 | 93.2 | 52.9 | 100% | 2.46× |
| | 2 | 64.1 | 76.9 | 73% | 2.28× |
| | **4** | **43.7** | **112.8** | 53% | 2.01× |
| | 8 | 46.1 | 106.8 | 25% | 1.60× |
| | 16 | 60.8 | 81.0 | 10% | — |
| **farc** 638k | **4** | **146.5** | **22.4** | 100% | 1.93× |
| | 8 | 156.3 | 21.0 | 47% | 1.27× |
| | 16 | 170.3 | 19.3 | 22% | 0.90× |
| | 32 | 170.2 | 19.3 | 11% | — |
| | 64 | 247.7 | 13.3 | 4% | — |
| **dars** 3.16M | 16 | 183.0 | 3.59 | 100% | 2.46× |
| | 32 | 133.4 | 4.93 | 69% | 2.04× |
| | **64** | **121.1** | **5.43** | 38% | 1.87× |
| | 128 | 137.5 | 4.78 | 17% | 1.38× |
| **ng5** 7.4M | 32 | 316.8 | 2.07 | 100% | 2.38× |
| | **64** | **235.9** | **2.79** | 67% | 1.86× |
| | 128 | 637.1 ‡ | 1.03 | 12% | — |
| | 256 | 247.7 | 2.65 | 16% | — |

‡ ng5 P=128 is a **reproducible cliff, not a bad measurement** — reconfirmed at 643.2 ms on a
fresh allocation with both bracketing anchors reproducing. See below. It is *not* a turnover:
P=256 recovers.

### The two headline conclusions

**(a) GH200 is a consistent 1.8–2.5× over A100 at matched GPU counts** on identical physics
and identical code — largest where the kernel dominates (core2/1 2.46×, dars/16 2.46×) and
decaying toward the strong-scaling limit (core2/8 1.60×, dars/128 1.38×). These kernels are
gather/memory-bound, so this tracks HBM3-vs-HBM2e bandwidth, not FLOPs.

**(b) The strong-scaling knee moves LEFT — every mesh turns over at roughly HALF the device
count it did on Levante.** dars peaked at 64 GPUs here and was still climbing at 128 on
Levante; farc peaks at a single node here and improved to 16 GPUs on A100. This is the
arithmetic consequence of (a): ~2× faster GPUs drain a shard's work in half the time while
the communication cost per step is unchanged, so the compute/communication crossover arrives
at half the shard count. **JUPITER buys throughput per GPU, not more GPUs' worth of scaling.**
The independent Kokkos twin turning over on dars at the same 64 GPUs (§7) confirms the knee
is a property of mesh × machine, not of this implementation.

### The ng5-128 cliff — REAL and REPRODUCIBLE (remeasured 2026-07-24), not a turnover
ng5 at 128 GPUs is **bracketed by faster points on both sides** — 243.4 ms at P=64 and
257.2 ms at P=256. A strong-scaling turnover cannot recover at 2× the device count, so this
was initially recorded as a suspected outlier with the caveat "do not publish until
reproduced in isolation".

**It has now been reproduced in isolation, and the caveat is withdrawn**
(`scripts/bench/jupiter/remeasure_ng5_128.sbatch`, job 1033966: a fresh 64-node allocation on
entirely different hardware — jpbo-051/053/054 vs the campaign's jpbo-036/037/038 — with an
empty queue):

| ng5 coloured | campaign (2026-07-23) | remeasure (2026-07-24) | agreement |
|---|---:|---:|---:|
| P=64 anchor | 243.4 / 261.1 | **235.9** | reproduces (3% faster) |
| **P=128** | **644.0 / 638.5** | **643.2** | **0.1%** |
| P=256 anchor | 257.2 / 260.4 | **247.7** | reproduces (3.7% faster) |

Both bracketing anchors reproduced *on the same allocation that produced the 643 ms*, so the
allocation is demonstrably healthy and the cliff is not machine state. **The measurement was
correct; the effect is genuine and deterministic.**

**Compile time reproduces too, and it is the load-bearing clue.** Compile is *excluded* from
`per_step` (the bench times the 2nd warm call on an already-compiled executable), so it never
contaminates the measurement — but it is host-side XLA work that no network condition can
inflate, and it is ~2× at exactly P=128 across both allocations:

| ng5 P | 32 | 64 | **128** | 256 |
|---|---:|---:|---:|---:|
| compile, campaign | 75–87 s | 76 s | **155–166 s** | 69–75 s |
| compile, remeasure | — | 65.9 s | **161.0 s** | 76.5 s |

Runtime *and* compile both ~2.6× at one single value of P, reproducibly, on two different node
sets ⇒ the **compiled program at P=128 differs**, and the cause is upstream of the fabric.

**Mechanisms tested and REJECTED** (do not re-test):
- *A bad node*: the campaign's two slow reps ran on **disjoint** 32-node subsets
  (jpbo-038-[17-48] vs jpbo-036-[33-48]+jpbo-037-[01-16]) and agreed to 0.9%, while the
  **union** of those same 64 nodes ran P=256 at 257 ms. The remeasure then reproduced it on
  entirely different hardware.
- *Partition balance*: ng5 `dist_128` is as balanced as every other rung —
  myDim max/mean **1.006**, padfac **1.013**.
- *Colouring rounds K*: K jumps 10 → 14 between P=64 and P=128, but is **15 at P=256** where
  the time fully recovers. K alone cannot explain a penalty that vanishes at higher K.
- *Wiring / graph size*: the static coloured exchange at P=128 is **on trend in every metric**,
  not pathological — its total packed halo buffer is *smaller* than P=64's:

  | ng5 P | K | total packed buffer/device | max/mean slot |
  |---|---:|---:|---:|
  | 32 | 10 | 13097 | 2.69 |
  | 64 | 10 | 10794 | 2.46 |
  | **128** | **14** | **8736** | 3.17 |
  | 256 | 15 | 7176 | 3.43 |

  And P=256 compiles **45** ppermutes (15 rounds × 3 kinds) in 69–76 s while P=128 compiles
  **42** in 155–166 s — nearly the same collective count, half the compile time. So neither
  buffer volume nor collective count explains the compile blow-up.

**The discriminator RESOLVED it: the cliff is P-SPECIFIC, not transport-specific.** P=128 was
run under the two transports the campaign never used there (`AB_MAX_NPES=64` had capped the
A/B). **All three transports collapse at P=128**, and coloured — far from being the culprit —
is by a wide margin the least damaged:

| ng5 P=128 | per_step | compile | vs its own P=64 |
|---|---:|---:|---:|
| **coloured** | **643.2 / 637.1 / 646.6** | 161–166 s | 2.65× (243 → 643) |
| padded | 3222.2 / 3220.8 | 554–561 s | 5.95× (541 → 3221) |
| ragged | 3035.6 | 512 s | — |

Both padded reps agree to **0.04%** — even the pathology is deterministic. Compile tracks
runtime across all three (161 s → 554 s → 512 s against anchors of 66–77 s), which is the same
signature seen in coloured alone: **whatever happens at P=128 inflates compile and runtime
together, for every transport.** A transport bug cannot do that; a bad network cannot inflate
host-side compile at all.

Anchors re-run at the END of the same job confirm no drift over its 1 h 40 m:
P=64 235.9 → **251.6**, P=256 247.7 → **248.9**.

**Status (updated 2026-07-24 evening): ROOT CAUSE FOUND — one XLA fusion decision.** At
exactly the ng5/dist_128 shape (`Lmax_nod=59637`) XLA's `multi_output_fusion` merges the
per-step surface-flux balance reduction (`sss_runoff._area_mean`) into the 1,425-op vmapped
ice-thermodynamics fusion, flipping it from `kind=kLoop` (pure elementwise, what P=64/256
get) to `kind=kInput` (reduction-emitter codegen) — and that kernel runs **383 ms once per
step** (nsys job 1036137), which is the whole cliff. The zstar×ice joint dependence is the
dataflow ice-thermo → water_flux/virtual_salt → balance mean → zstar SSH consumers that
exists only with both on. Fix: `FESOM_BALANCE_BARRIER=1` (`lax.optimization_barrier` on the
`_area_mean` input, byte-identical numerics), confirmation job 1036604. Full chain of
evidence in the handoff. Earlier same day, the ablation had localised it: At P=128, same allocation: baseline-prod 649.5 ms and tke-OFF
644.0 ms (both CLIFF, true compile ~58–64 s) vs zstar-OFF 222.2 ms, ice-OFF 193.4 ms,
ocean-only 171.3 ms (all healthy, ratios 0.83–0.91 vs their own P=64). The excess is a
JOINT term: removing either member removes the same ~430–456 ms, so it belongs to no
single component's ops — the signature of a global execution-mode switch (collective
overlap loss), matching the serialized-collective arithmetic (~9,400 ppermute rounds/step
× ~65 µs ≈ 610 ms). Compile inflation decouples: it follows zstar alone (ice-OFF keeps
zstar: healthy runtime, compile still 2×). Full record, verdict table and the running
HLO-diff (job 1036111) + nsys (1036137) experiments in
**[`HANDOFF-20260724-ng5-128-cliff.md`](HANDOFF-20260724-ng5-128-cliff.md)**. Eliminated
(now twelve): bad node · machine state · node balance · elem/edge balance · coloured buffer
volume · colouring rounds K · SSH `nnz_max` (max/mean 1.011) · device count (dars at P=128
is normal) · raw collective performance at 64/128/256 GPUs (`ppermute` **flat**: 64.5 /
65.6 / 70.1 µs) · CG non-convergence (iteration counts **identical** at all P: 86/99/97,
`hit_cap=0`) · partition exchange structure at elem/edge kinds (clean at all P,
`partition_structure.py`) · any SINGLE physics component (the ablation).

Two facts now constrain any explanation:

**(a) The model does the SAME WORK at every P** — same CG iterations, same residuals, same
collective counts, same shapes, same `max_uv`. P=128 executes *identical* work 2.65× slower.
This kills every "extra work" explanation. **It also means there is no correctness problem at
P=128** — the solve converges normally and returns the same answer.

**(b) Compile is ~2× inflated at P=128 for every transport.** ⚠️ Note the bench's `compile=`
field is *compile + the first 150-step run*, so `150 × per_step` must be subtracted; after
doing so, true compile is 60–69 s (coloured), 71–78 s (padded), 57 s (ragged) at P=128 against
25–40 s at every other P. Compile is host-side, so no network condition can explain it.

Together: *the same mathematics, expressed as a program that both compiles and runs ~2× slower,
at one value of P.* The decisive untried experiment is an **optimized-HLO diff at
P=64/128/256** (op counts, fusions, collective lowering) — see the handoff.

**How to report ng5 in the paper.** Best-of-all-reps after the remeasure:

| ng5 coloured | P=32 | P=64 | P=128 | P=256 |
|---|---:|---:|---:|---:|
| best ms/step | 316.8 | **235.9** | 637.1 ‡ | 247.7 |

Two separate statements, which must not be conflated:
1. **ng5 peaks at P=64 (235.9 ms) and gains nothing beyond it** — P=256 (247.7 ms) does not
   beat P=64, it merely returns to about the same level. So ng5 *does* saturate at 64 GPUs,
   consistent with the campaign's "the knee moves left" conclusion (§6b).
2. **P=128 is a reproducible cliff sitting on top of that saturation** — a 2.7× spike between
   two points that are themselves flat. It is *not* the turnover, and it must not be drawn as
   one; a monotone "turns over after 64" curve would misrepresent the P=256 recovery, while
   silently dropping P=128 would hide a real, five-times-confirmed effect.

The defensible presentation is to plot all four points and annotate P=128 as an unexplained,
reproducible anomaly under investigation.

## 7. Cross-check against the Kokkos code twin (same machine, same mesh)

`port_kokkos/docs/JUPITER_FLEET_RESULTS.md` ran on the same JUPITER booster. Their `gN` is
N **nodes** (4N GPUs), so at matched GPU counts on dars:

| dars | Kokkos (dp, s25+STAGE) | JAX (coloured, this campaign) | JAX/Kokkos |
|---|---:|---:|---:|
| 16 GPU (g4) | 117.4 ms | 183.0 ms | 1.56× |
| 32 GPU (g8) | 67.7 ms | 133.4 ms | 1.97× |
| **64 GPU (g16)** | **50.0 ms** | **121.1 ms** | 2.42× |
| 128 GPU (g32) | 50.0 ms (flat) | 137.5 ms | 2.75× |

Two things this establishes:

1. **The dars knee at 64 GPUs is real and code-independent.** Kokkos goes flat 64→128, JAX
   regresses. Two independent implementations turning over at the same device count says the
   limit belongs to the mesh's comm structure on this fabric, not to either code.
2. **The JAX gap is the known "array-framework tax"** (`docs/PARALLELISM.md` factors it as
   ~2.7× kernel efficiency), and it *widens* with device count (1.6× → 2.8×) — JAX's parallel
   efficiency degrades faster than the hand-written Kokkos kernels'.

Caveat: absolute ms are not strictly like-for-like — the Kokkos fleet runs *their* uniform
benchmark config, not this campaign's per-mesh production physics. The **knee location** is
the comparable quantity; the ratio is indicative, not a certified code-to-code benchmark.

## 7. Ops notes

- `--gres=gpu:4 --ntasks-per-node=1 --gpu-bind=none`, one process per node,
  `jax.distributed.initialize(local_device_ids=range(4))` — works exactly as the doc predicted.
  `jax.distributed` auto-detects SLURM; no launcher glue.
- Sub-node rungs (1, 2 GPUs) need no `CUDA_VISIBLE_DEVICES` juggling: the model takes
  `jax.devices()[:npes]`.
- **Per-run `timeout`** is in `bench_scaling_jupiter.sbatch`. Levante lost a job to a hung
  padded halo at 64 GPUs where a stall is indistinguishable from a slow compile; without it,
  one hang eats a 64-node allocation. (Measured to be perf-neutral: 204.0 vs 205.5 ms.)
- **Do not run campaign jobs concurrently** — see §4. Our own four jobs (116 nodes) were
  enough to quadruple the padded points. Serialize, or use coloured throughout.
- ng5 memory is comfortable: **48.5 GiB/GPU peak at 32 GPUs** (96 GB HBM3). The doc's
  memory-borderline worry did not materialise at these rungs.
- Compile is excluded from timings but not from wall clock: ~20 s core2, ~40 s farc,
  ~50 s dars/16, ~105 s ng5/32. No persistent XLA cache (project rule).
- The dars zstar **compile cliff** (>3 h at ~435k lanes/GPU on A100) did **not** fire:
  dars/16 is 197k lanes/GPU and compiled in 51.7 s.
- An `emergency_maintenance` reservation (1199 nodes, `MAINT,IGNORE_JOBS`) was active during
  this campaign. Our nodes were outside it, but check `scontrol show res` before trusting
  fabric-sensitive numbers.
