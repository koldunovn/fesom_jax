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
| | **64** | **243.4** | **2.70** | 65% | 1.80× |
| | 128 | 638.5 ⚠️ | 1.03 | 12% | — |
| | 256 | 257.2 | 2.55 | 15% | — |

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

### ⚠️ The ng5-128 anomaly — an outlier, NOT a turnover
ng5 at 128 GPUs (638.5 / 644.0 ms, reps agreeing to 0.9%) is **bracketed by faster points on
both sides** — 243.4 ms at 64 and 257.2 ms at 256. A turnover cannot recover at 2× the device
count, so this is specific to the 128-GPU/32-node configuration. Two candidate mechanisms
were tested and **both fail to explain it**:
- *Colouring rounds*: K does jump 10 → 14 between P=64 and P=128 — but it is **15 at P=256**,
  where the time recovers fully. Measured K per kind (nod/elem/edge, identical across kinds):

  | ng5 P | 32 | 64 | 128 | 256 | | dars P | 16 | 32 | 64 | 128 |
  |---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
  | K | 10 | 10 | 14 | 15 | | K | 9 | 12 | 12 | 13 |
  | halo/GPU | 1725 | 1347 | 1026 | 764 | | halo/GPU | 1237 | 1019 | 834 | 635 |

  Halo volume per GPU *falls* monotonically, so this is a latency/topology effect, not volume.
- *Contention*: rep 2 ran after every other job of ours had finished and matched rep 1.

Re-measured across all three transports on a quiet fabric by
`scripts/bench/jupiter/transport_map_jupiter.sbatch`. **Do not publish the ng5 curve until
this point is explained or reproduced in isolation.**

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
