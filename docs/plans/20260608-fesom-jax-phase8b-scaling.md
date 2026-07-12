# FESOM2 → JAX Port — Phase 8b: scaling the sharded model (farc → dars → NG5)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (roadmap — sharding = locked decision 5).
**Predecessor:** Phase 8 (`docs/plans/completed/20260607-fesom-jax-phase8-sharding.md`), tag
`v1.1-multi-gpu` (commit `7b16d27`) — the model is N-vs-1 forward- AND gradient-correct, validated on real
A100s. **That correctness is the ORACLE for everything here.** Phase 8b changes *how fast/how big*, never
*what's computed*: every step must stay N-vs-1 == single-device (the Phase-8 gates), only faster.
**Created:** 2026-06-08. **Status:** 🚧 IN PROGRESS — B.0 (the halo rewrite) first.

---

## 0. Scope (READ FIRST)

### Why this phase exists — the user's scaling concern, stated honestly

The user asked: *"do you expect that you will be comparable in terms of scaling with C and kokkos? the
worry is that you copy too much data all the time."* **That worry is correct about the Phase-8 code as
shipped, and B.0 is the fix.** Phase 8's halo exchange is `lax.all_gather` (`halo.py`): every device
gathers *every other device's entire padded local array* `[P, Lmax, …]`, then reads its halo lanes out of
it. That moves **O(P · N_local)** bytes per exchange and gets WORSE as rank count grows — the opposite of
scaling. It was the right choice for the 2–4-device *correctness* gate (simplest verifiable collective,
trivially transpose-correct for AD) but it is **non-scaling by construction**.

The C MPI port and the Kokkos port both move only the **halo** (the partition-boundary copies) to the
**neighbours that need it**, point-to-point — **O(boundary_size)**, independent of total rank count. To be
"comparable in terms of scaling with C and Kokkos," the JAX port must do the same. JAX 0.10.1 exposes
exactly that primitive: **`lax.ragged_all_to_all`** (per-neighbour ragged send/recv sizes), and crucially
it has a **registered transpose + jvp** (`jax._src.lax.parallel._ragged_all_to_all_{transpose,jvp}`), so
the differentiability the whole port is built for survives the swap.

So the honest answer to the user: *with `all_gather` — no, it won't scale; with `ragged_all_to_all` (B.0) —
the communication volume matches the C/Kokkos model (halo-only, point-to-point), and then the remaining gap
is XLA-vs-hand-tuned-MPI overlap, which we measure on the farc→dars→NG5 ladder against the Kokkos numbers.*

### What B.0 does NOT change

Forward physics, the partition source (`dist_<NP>`), the redundant-halo compute model, the reductions
(`psum`), the masks, the AD seams. B.0 is a **drop-in replacement of one primitive** behind the existing
`HaloCtx.exchange` interface. The Phase-8 N-vs-1 forward gate (`test_step_sharded.py`) and the AD gate
(`test_gradient_sharded.py`) are the bit-for-bit oracles — B.0 is done when both stay green with the new
primitive (forward byte-identical to the `all_gather` path; gradient transpose-correct).

### The mesh ladder (Kokkos prior art — validate against it, don't reinvent)

All meshes + their `dist_<NP>` partitions live on `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/`. The Kokkos
port already ran this exact ladder — `port_kokkos/docs/SCALING_{FARC,DARS,NG5}.md` are the reference
numbers + the known deep-mesh gotchas. Mirror their setup; compare wall-clock/throughput honestly.

| mesh  | nodes  | levels | dt (cold) | target hardware                 | Kokkos doc        |
|-------|--------|--------|-----------|---------------------------------|-------------------|
| CORE2 | 127 k  | 48     | 1800 s    | the correctness gate (Phase 8)  | —                 |
| farc  | 638 k  | 48     | 900 s     | ~1 A100                         | `SCALING_FARC.md` |
| dars  | 3.16 M | 57     | 180 s     | 4 GPU / 1 node                  | `SCALING_DARS.md` |
| NG5   | 7.4 M  | 70     | 180 s     | multi-node (`jax.distributed`)  | `SCALING_NG5.md`  |

**⚠️ Timestep (CFL) is mesh-specific — `dt=1800` is CORE2-only.** Finer meshes need a smaller `dt` or they
go CFL-unstable. The Kokkos cold-PHC-start values (`SCALING_DARS.md`, `SCALING_M524.md`): CORE2 1800, farc
900, dars/NG5 180 (`dt=240` is CFL-unstable from cold on dars/NG5; production runs use 240 only post-spinup).
The benchmark (`bench_forward_scaling.py --dt`) MUST pass the per-mesh `dt` — same `dt` for the ragged-vs-
all_gather pair so the comparison stays apples-to-apples. Mesh `.npy` bundles live on **`/work`**
(`/work/ab0995/a270088/fesom_jax_meshes/mesh_{farc,dars,ng5}`), exported by the C port
(`jobs/jax_mesh_export_{farc,dars,ng5}.sh` on the `jax-mesh-export` branch).

---

## 1. Task ladder

### B.0 — STEP 0 (MANDATORY): replace `all_gather` with `ragged_all_to_all`

The load-bearing rewrite. Everything downstream is meaningless until this lands (scaling numbers on an
`all_gather` halo measure the wrong thing).

**Files:** `fesom_jax/shard_mesh.py` (build the ragged maps), `fesom_jax/halo.py` (the new primitive),
`fesom_jax/tests/test_halo.py` / `test_step_sharded.py` / `test_gradient_sharded.py` (gates).

- [x] **B.0a — ragged exchange maps (`shard_mesh.RaggedExchange`).** ✅ DONE (RevLog #1). Built per device,
      per kind from the **same owner map** (`_owner_map`) `_exchange_map` uses — NOT the C `ComStruct` —
      because the owner map is uniform across `nod`/`elem`/`edge` (the `Partition` has no `com_edge2D`) and
      is provably consistent with the `all_gather` oracle. Per device `d`:
      - **send**: `send_idx` = local interior lanes concatenated **ordered by destination device 0..P-1**,
        plus dense `send_sizes[P]` + `send_offsets[P]` (the `input_offsets`). `operand = field[send_idx]`.
      - **recv**: `recv_idx` = local halo lanes concatenated **ordered by source device 0..P-1**, plus
        `recv_sizes[P]` + `recv_offsets[P]` (the `output_offsets`). Scatter the recv buffer into these halo
        lanes; interior + pad untouched (broadcast/overwrite). `send_sizes[d,e]==recv_sizes[e,d]`.
      Stored as `ShardedMesh.exchange_ragged: {kind: RaggedExchange}`. Gate
      `test_shard_mesh.py::test_ragged_exchange_reproduces_allgather` (host numpy, npes=2/4) PASSES — the
      ragged maps reproduce the `all_gather` exchange on every valid lane. ⚠️ One owned lane sent to several
      neighbours is gathered multiple times into `operand` (correct — the transpose scatter-ADDs the
      cotangents back, why AD stays correct). ⚠️ The builder uses Python per-halo-lane loops — fine for
      CORE2/farc/dars; vectorize for NG5 (and consider export-caching the maps) at B.3.
- [x] **B.0b — the new `halo_exchange`.** ✅ IMPLEMENTED (gate pending B.0c). `halo.halo_exchange_ragged`:
      `operand = field[send_idx]` (gather) → `recv = lax.ragged_all_to_all(operand, zeros(recv_max),
      send_offsets, send_sizes, out_offsets, recv_sizes, axis_name='p')` → `jnp.where(halo_mask,
      recv[recv_gather], field)` (gather-back + masked select — NOT `.at[].set`, so it stays cleanly linear
      for AD with no duplicate-index hazard). **Decisive `ragged_all_to_all` semantics** (verified vs the
      docstring's size-2 example): `output_offsets[d,e]` = where d's slice lands ON RECEIVER e (the receiver
      applies an internal `all_to_all` to recover its local write offsets) ⇒ `out_offsets = recv_offsets.T`
      and the constraint `send_sizes == all_to_all(recv_sizes)` holds by `send_sizes = recv_sizes.T`. Picked
      via `HaloCtx(use_ragged=, exch_ragged=, recv_max=)`; the all_gather path is the default + fallback.
      `halo_ctx=None` stays the dense identity no-op (single-device suite still byte-identical). Wired into
      `integrate_sharded.run_step_sharded(use_ragged=)` (the SSH/CG halo still uses all_gather — a B.0d
      follow-on). Standalone `halo.run_halo_exchange_ragged` for the isolated gate. ⚠️ `run_steps_sharded`
      / `run_gm_diag_sharded` not yet ragged-wired (not needed for the B.0c gates).
- [~] **B.0c — GATE (GPU-only).** ⚠️ **`lax.ragged_all_to_all` is UNIMPLEMENTED on XLA:CPU** (`UNIMPLEMENTED:
      HLO opcode 'ragged-all-to-all' is not supported by XLA:CPU ThunkEmitter`, confirmed job 25438390 — the
      existing all_gather S.3 tests stayed green, only the 6 new ragged tests hit the CPU gap). So B.0 can
      ONLY be gated on real GPUs (NCCL). The ragged tests now **SKIP on CPU** (platform guard) so the
      single-device/CPU suite stays clean. GPU gate (`scripts/debug/phase8b_b0_gpu.sbatch`, job 25438454):
      (1) **isolated primitive** `test_halo.py::test_ragged_primitive_matches_allgather` — ragged ==
      all_gather forward (byte-id on valid lanes) + grad (transpose), npes 2 & 4, all three kinds;
      (2) **step-level wiring** `test_step_sharded.py::test_ragged_step_matches_allgather` —
      `run_step_sharded(use_ragged=True)` == `(use_ragged=False)` byte-identical on owned. The host-side
      B.0a gate already proves the index math; the GPU gate proves the collective + its AD transpose.
      **RESULT (job 25438454):** FORWARD ✅ byte-identical to all_gather (all kinds, npes 2 & 4 — the
      collective + my offset args are correct). GRADIENT ❌ — JAX's `ragged_all_to_all` autodiff transpose
      is WRONG (grad max|Δ| ≈ 4.3@npes2 → 8.0@npes4, ~linear in P; the forward is exact + all-linear, so the
      bug is JAX's `_ragged_all_to_all_transpose`, summing over the axis instead of point-to-point). The
      step-level forward differs by 1.3e-7 (pad-lane handling: ragged leaves pad, all_gather sets pad=lane0
      — climate-close, my `<1e-12` was too strict → relaxed to field-appropriate). **So the FORWARD is
      validated; the AD is deferred to B.0d.** `use_ragged=True` is forward-only safe (default False keeps
      grad correct). Lesson appended to `PORTING_LESSONS.md`; the grad test is `xfail` (B.0d).

### B.0d — fully-correct ragged backward via `custom_vjp` (DEFERRED — not needed for forward scaling)
- [ ] Wrap `halo_exchange_ragged` in `jax.custom_vjp` so we control the transpose (JAX's is broken, B.0c).
      Backward option (A, simpler): reuse the proven `all_gather` exchange's VJP (correct on every meaningful
      lane; backward still O(P·N_local)). Option (B, fully scaling): hand-written reverse `ragged_all_to_all`
      (`input_offsets=recv_offsets`, `send_sizes=recv_sizes`, `output_offsets=send_offsets.T` [add to
      `RaggedExchange`], `recv_sizes=send_sizes`). Re-gate the AD on GPU (the `xfail` flips to pass).
      ⚠️ Needed for **training-at-scale** with the ragged halo; a forward-only scaling run does NOT need it.

### B.1 — farc (638 k × 48) forward scaling on ~1 A100

- [ ] Load the farc mesh + read its `dist_<NP>` (the `partit.py` reader should just work; confirm the
      ASCII format matches). Build `ShardedMesh` for farc.
- [ ] Forward run (a few hundred steps, the user's "~200 steps") on 1 (and 2/4) A100; record wall-clock /
      throughput. Compare against `port_kokkos/docs/SCALING_FARC.md`.
- [ ] N-vs-1 correctness spot-check on farc (the gate is mesh-agnostic — same `test_step_sharded` logic on
      the farc fixtures) to confirm the partition + exchange maps are right at this size.

### B.2 — dars (3.16 M × 57) on 4 GPU / 1 node

- [ ] farc → dars is mostly "bigger mesh, more partitions" — surface any O(N) host-side build cost or
      memory ceiling in `build_sharded_mesh` (the `[P, Lmax, …]` materialization). Scaling vs `SCALING_DARS.md`.

### B.3 — MULTI-NODE scaling (dars + NG5) vs Kokkos — the headline (user-requested, in progress)
dars (3.16 M) and NG5 (7.4 M) **do not fit one 4×A100 node** (OOM even ocean-only — Kokkos starts both at
2 nodes too), so they are multi-node. Staged:
- [x] **STEP 1 — `jax.distributed` bring-up.** ✅ `scripts/multinode_sanity.{py,sbatch}` on 2 GPU nodes
      (8 A100, 1 task/node × 4 GPU): `jax.distributed.initialize()` (SLURM auto-detect) → confirmed
      `jax.devices()`=8 global + cross-process `psum`/`all_gather` inside `shard_map` + `device_put` of a
      global host array to a multi-host `NamedSharding` placing addressable shards per process. [job 25441355]
- [x] **STEP 2 — multi-process data placement (the HOST-BUILD REWRITE).** ✅ DONE + VALIDATED (RevLog #7).
      The whole data pipeline now builds HOST numpy (`State.zeros/rest` `xp=np`; `integrate_sharded._fold`
      polymorphic — numpy for concrete, `jnp` for tracers so grad-through-fold survives; `folded_*`/
      `_halo_arrays`/`_fold_forcing` host) and `run_steps_sharded` **ALWAYS** `device_put`s the folded inputs
      to a `NamedSharding` (single- AND multi-process — dropped the `process_count>1` guard). So GPU 0 never
      holds the full global: dars-8 `peak_gpu_after_setup = 8.10 GiB` (was 40+ GiB → setup OOM). Each process
      builds the GLOBAL host arrays (fit one node's RAM: dars ~22 GB, NG5 ~80 GB ≪ 256 GB) → device_put places
      only its addressable shards. PHC IC per mesh cached to `/work` (dars + NG5 done). If the global host
      arrays ever exceed one node's RAM, switch to per-subdomain loading
      (`make_array_from_process_local_data`) — deferred until needed.
- [~] **STEP 3 — dars multi-node full model** (dist_8 = 2 nodes done; dist_16/32 next), real JRA+PHC+ice,
      dt=180, ragged vs all_gather → vs Kokkos `SCALING_M524.md`. ✅ **dist_8 (job 25446699):** RAGGED runs
      **0.934 s/step** (peak_gpu 35.90 GiB); **ALLGATHER OOMs** (needs a 43.34 GiB collective buffer) — the
      FIRST hard ragged win (all_gather's O(P·N_local) doesn't fit at 8 GPU). vs Kokkos M524 CUDA dars-2N
      **0.814** → ~15% slower (XLA-vs-MPI collective overlap; CORE2 JAX was faster). ⚠️ JAX mesh nl=57; confirm
      Kokkos M524 dars level count (old doc says 47). NEXT: dist_16 (4N) + dist_32 (8N) ragged for the JAX
      scaling curve vs M524 (dars 4N 0.475, 8N 0.344).
- [x] **STEP 4 — NG5 multi-node full model** ✅ (RevLog #8). NG5 (7.4M × nl70) FULL model RUNS on **dist_32
      (8 nodes, 32 A100): 0.840 s/step** (job 25452144) — vs Kokkos M524 CUDA NG5-8N **0.810** → ~3.7% slower
      (the gap closes with scale: dars-2N 15% → dars-4N 8% → NG5-8N 4%). nl=70 needed NO level-cap fix (JAX
      reads `mesh.nl` dynamically; the Kokkos "64-level scratch cap" was C-specific). dist_16 (4N) + dist_8 (2N)
      OOM the per-GPU model working set (140 GiB at dist_16); dist_32 is the min that fits (66.63 GiB). The TWO
      enabling fixes: (1) `_to_global_sharded` → `make_array_from_callback` (the `device_put` of the folded
      global staged ~125.81 GiB on one GPU — the NG5-only wall); (2) **gather-free sharded Zarr output**
      (`zarr_output.py`, `--out-zarr`): each GPU writes its `Lmax`-chunk in parallel → 19 GB across 8 nodes,
      `owned_nod == nod2D` exact — the "step-0 global-gather" OOM avoided. The ragged win is concrete from
      dars-8 on (all_gather OOMs multi-node where ragged fits).

---

## B.3 REWRITE PREP — host-build the data pipeline (the dars/NG5 setup-OOM blocker)
*Prepared 2026-06-09 for next-session planning + execution. This is the one real change blocking
multi-node numbers.* **✅ DONE + VALIDATED 2026-06-09 (RevLog #7)** — dars-8 multi-node FULL now runs
(ragged 0.934 s/step; setup peak 8.10 GiB). ⚠️ The "also unblocks single-node dars-4" claim below was WRONG:
host-build removed the SETUP OOM, but dars-4 FULL still OOMs on the MODEL working set (48.6 GiB/GPU at dist_4);
dars needs 2 nodes (the Kokkos limit too). See RevLog #7.

**Symptom.** dars full model OOMs the GPU during SETUP (before any timing). Identical error for BOTH
all_gather and ragged: `RESOURCE_EXHAUSTED: Out of memory ... 1.34 GiB ... executable jit__where`
(single-node dars-4 earlier: same, `jit_broadcast_in_dim`). **1.34 GiB = 3.16M × 57 × 8 B = exactly ONE
full global 3-D node field.** Same for both halos ⇒ it is NOT the halo/model — it is the data build.

**Root cause (traced to the functions).** The pipeline materializes the FULL GLOBAL arrays as
`jax.numpy` (GPU) arrays on the default device (GPU 0) BEFORE sharding:
- `fesom_jax/state.py` `State.rest`/`State.zeros` (+ the bench `phc_state`/`perturbed_state`):
  `jnp.full`/`jnp.zeros`/`jnp.where` ⇒ the global State `[N, nl]` lands on GPU 0 (dars: ~1.34 GiB ×
  ~15 fields ≈ 20 GiB on ONE GPU).
- `fesom_jax/integrate_sharded.py` `_fold` + `folded_state`/`folded_mesh`/`folded_operator` +
  `_halo_arrays`: `jnp.asarray(...)` ⇒ the folded `[P·Lmax, nl]` global on GPU 0.
- `shard_mesh.partition_state`/`partition_forcing_*` already return HOST numpy ✓ — but `_fold`
  immediately re-uploads them to GPU 0.
So GPU 0 must hold the ENTIRE global model before `device_put` shards it ⇒ OOM at dars/NG5 scale.
(CORE2/farc fit only because their globals are small enough for one GPU. The B.3 STEP-2 `device_put`
shards the *folded* arrays — but they were already built on GPU 0, so it doesn't help.)

**Fix (CONTAINED — host-build + device_put-sharded; NOT the per-subdomain rewrite).**
1. Build the WHOLE pipeline as HOST numpy (`np`), never `jnp`, until the final placement:
   `state.py` (a numpy `rest`/`zeros` path or a `_rest_numpy`), bench `phc_state`/`perturbed_state`,
   and `_fold`/`folded_*`/`_halo_arrays` → `np.asarray`.
2. ALWAYS `device_put` the folded inputs to a sharding before `jax.jit(shard_map)` — single-process
   to a LOCAL `NamedSharding`, multi-process to the GLOBAL one (generalize the existing
   `_to_global_sharded`). `device_put` of a host-numpy global to a sharding places ONLY the
   addressable shards on device ⇒ GPU 0 never holds the full global.
3. Net: host holds the full global numpy (dars ~22 GB, NG5 ~80 GB — fit a 512 GB GPU node; keep
   `--mem=0`); each GPU holds only its 1/P shard. (Per-subdomain loading is a FUTURE step, only if a
   mesh's global exceeds one node's RAM — NG5 80 GB ≪ 512 GB, so host-build suffices for now.)

**This also fixes single-node dars-4** (the OOM was the build, not the sharded model: dars/4 ≈ 790k
nod/GPU fits 40 GB). **Verify in order:** dars-4 single-node full → dars-8 multi-node → dist_16/32 → NG5.

**Change-list (files):**
- `fesom_jax/state.py` — numpy path for `rest`/`zeros` (or a host builder used by the data pipeline).
- `fesom_jax/integrate_sharded.py` — `_fold` + `folded_state`/`folded_mesh`/`folded_operator` +
  `_halo_arrays` → `np`; make the `device_put`-to-sharding ALWAYS run (single-process too, to a local
  mesh sharding) so the global is never on one GPU.
- `scripts/bench/bench_forward_scaling.py` — `phc_state`/`perturbed_state` → numpy.
- GATE: the single-device suite + `test_step_sharded.py`/`test_gradient_sharded.py` must stay green
  (the host-build + device_put must be numerically identical — it only changes WHERE arrays live).

**Verification.** Add a peak-GPU-mem probe in setup; run dars-4 single-node full (must fit post-fix);
confirm CORE2/farc per-step unchanged; then dars-8 multi-node.

**Risks / open questions for the planning session:**
- Confirm `jax.jit(shard_map)` accepts device_put-sharded host-numpy single-process (the 8-GPU sanity
  showed it works for the global multi-process case).
- `folded_mesh` reconstructs the local `Mesh` INSIDE `shard_map` from the sharded inputs — unaffected,
  but re-confirm after switching its build to numpy.
- NG5 IC: the `_extrap_nod3D` numpy GS loop over 7.4 M nodes may be slow/fail (dars 3.16 M = 670 s) —
  caching kicked off (job 25445934, 4 h); fallback = the C-port PHC dump.
- Full-model XLA compile ≈ 15 min/config ⇒ run **one config per job** (don't batch; the subtraction
  timing's 2nd compile already proved too costly — using compile-once + warm-call now).

**Already prepped (so next session starts ready):** jax.distributed multi-node bring-up VALIDATED
(8 GPU, cross-node collectives correct); the multi-process `device_put` path + launchers committed;
farc + dars PHC IC cached on `/work`; NG5 IC caching running (25445934).

---

## 2. Acceptance

Phase 8b is done when the model runs N-vs-1-correct at farc/dars/NG5 scale with **halo-only point-to-point
communication** (not `all_gather`), and the wall-clock/throughput is **reported honestly against the Kokkos
SCALING_*.md numbers** — including, candidly, wherever XLA collective overhead leaves a gap vs hand-tuned
MPI. The gradient stays correct (the AD gate green with the ragged primitive). The 2-yr multi-GPU *climate*
run remains a separate follow-up (chaotic reduction-order divergence — same as the C, Phase-8 §0).

---

## Revision Log

### #8 — NG5 RUNS (the headline): the global-fold placement wall + gather-free Zarr output; dars curve (2026-06-09)
After the host-build fix, **dars scaled cleanly** — dars-8 (2N) 0.934 s/step, **dars-16 (4N) 0.513** (vs Kokkos
M524 0.814 / 0.475 — gap 15%→8%, 2N→4N speedup 1.82× ≥ Kokkos's 1.71×). **NG5 then hit a placement wall:**
dist_16 OOM'd (working set 140 GiB + I/O "exceeds base limit") and dist_32 OOM'd in a `jit__identity_fn`
allocating **125.81 GiB ≈ the full folded global `[P·Lmax,nl]` on ONE GPU** — even though the model working set
FIT (66.63 GiB at dist_32). Root cause: `_to_global_sharded`'s `jax.device_put(folded_global, NamedSharding)`
stages a global-sized copy on a single device (≈56 GB dars = fit; ≈125.81 GiB NG5 = OOM); it does NOT shrink
with node count. **Fix: `make_array_from_callback`** (per-shard host slicing, no global on device; CPU-verified
bit-identical to `device_put`, both gates green). Also built **`fesom_jax/zarr_output.py`** — gather-free
sharded output: each GPU writes its folded `Lmax`-chunk to Zarr in parallel (rank-0 metadata → barrier →
disjoint `addressable_shards` writes), `gid`/`owned` maps for `reconstruct_global`; round-trip CPU-exact, wired
as `--out-zarr`. **RESULT (job 25452144): NG5 dist_32 (8 nodes/32 A100) FULL model RUNS 0.840 s/step (vs
Kokkos 0.810, ~4%) AND writes 19 GB sharded Zarr** (T = 32 chunk files = 1/GPU; `owned_nod == nod2D` exact).
The headline NG5 multi-node goal is reached; the JAX port is ~4–8% off hand-tuned Kokkos CUDA at dars/NG5 scale
with halo-only (ragged) comms + process-local I/O. zarr v2 installed in the env. ⚠️ login can't reconstruct an
NG5 global (~8 GB > login per-proc cap) — use a compute node.

### #7 — HOST-BUILD REWRITE done + validated; dars-8 multi-node FULL runs (ragged fits, all_gather OOMs) (2026-06-09)
Implemented the B.3 REWRITE PREP host-build fix: `State.zeros/rest` take `xp=jnp|np` (default byte-identical;
`np` builds the global IC on the HOST); `integrate_sharded._fold` is now **polymorphic** (numpy for concrete
host arrays so the full global never lands on GPU 0; `jnp` for tracers so the S.8 grad-through-fold survives —
the one real trap, the IC-field gate folds a `jax.grad` tracer); `folded_mesh`/`_fold_forcing`/`_halo_arrays`
host-numpy; `run_steps_sharded` **ALWAYS** `device_put`s the folded inputs to a `NamedSharding` (single- AND
multi-process). Bench `phc_state`/`perturbed_state`/`stress_p` numpy + a peak-GPU-mem probe.
**GATES (CPU fake-devices, all GREEN):** State/partition byte-identical (68 tests) + direct `xp=np`-vs-`jnp`
byte-diff = 0; ocean sharded forward (npes 1/2/4); ocean grad (param `d/d k_ver` rel 3.8e-8, `a_ver` 4.1e-4 /
IC-field+masked-NaN / FD / **multistep** = the `run_steps_sharded` device_put path); forced assembled forward
+ backward (KPP+GM+ice — `_fold_forcing` + boundary_node device_put). Placement-only ⇒ every value identical.
**dars-8 (dist_8, 2 nodes, 8×A100, FULL JRA1958+PHC+ice, dt=180, job 25446699):** host-build VALIDATED —
`peak_gpu_after_setup = 8.10 GiB` (was 40+ GiB → the setup OOM is GONE); **RAGGED runs 0.934 s/step**
(peak_gpu 35.90 GiB), **ALLGATHER OOMs** (needs a 43.34 GiB collective buffer) — the FIRST hard ragged win
(all_gather's O(P·N_local) doesn't fit multi-node). vs Kokkos M524 CUDA dars-2N **0.814** → ~15% slower
(comparable; XLA-vs-MPI overlap — CORE2 JAX was *faster*). ⚠️ **Correction to the prep:** "host-build fixes
single-node dars-4" was WRONG — host-build removed the SETUP OOM (dars-4 reaches step compile), but dars-4
FULL OOMs on the MODEL working set (`hlo_rematerialization` floor 48.6 GiB/GPU at dist_4) — the SAME limit
Kokkos hits (dars needs 2 nodes). Separate the setup OOM (data build, fixed) from the model OOM (step
working set, needs more devices). The host↔GPU copy is one-time (`device_put` of the IC); all N steps run in
one `jax.jit(shard_map(lax.scan))` with the carry GPU-resident (no per-step host traffic). NEXT: dars-16/32
ragged (the JAX scaling curve vs M524 4N 0.475 / 8N 0.344) → NG5 (7.4M, IC cached — the headline goal).

### #6 — multi-node bring-up DONE; dars/NG5 blocked on a setup-OOM (data built on GPU 0) — REWRITE PREP (2026-06-09)
B.3 STEP 1 (jax.distributed) + STEP 2 (multi-process device_put) committed + validated: 2-node sanity
(8 A100, 1 proc/node × 4 GPU) confirms global devices + cross-process psum/all_gather in shard_map are
correct (config lesson: 1 proc/node with `local_device_ids=range(4)`; 4-procs/node raced the coordinator).
farc + dars PHC IC cached (real T). BUT every dars full-model run (single-node dist_4 AND 2-node dist_8,
all_gather AND ragged) OOMs the GPU in SETUP with the identical `1.34 GiB jit__where` (= one full global
3-D field) — traced to the data pipeline building the FULL GLOBAL arrays as `jnp` on GPU 0 before sharding
(`State.rest`/`phc_state` jnp; `_fold`/`folded_*`/`_halo_arrays` `jnp.asarray`). `partition_state` is
already numpy but `_fold` re-uploads it. **Fix = host-build the whole pipeline (numpy) + always
device_put-sharded** (contained; NOT per-subdomain) — also unblocks single-node dars-4. Full diagnosis +
change-list + verification + risks in the **"B.3 REWRITE PREP"** section above (prepared for next-session
planning). Also learned: the full-model XLA compile is ~15 min/config (run 1 config/job; switched the
timing from the 2-compile subtraction to compile-once + warm-call). NG5 IC caching kicked off (25445934).

### #5 — FULL model (real JRA+PHC+ice): JAX is COMPARABLE to Kokkos-CUDA; the "10×" was a compile bug (2026-06-09)
Two corrections (both user-flagged): (a) the bench must use the **REAL Kokkos setup** — PHC winter IC
(`phc_ic.load_phc_ic`, mesh-agnostic) + JRA55 1958 (`build_core_forcing`, mesh-agnostic) + prognostic ice
(real a_ice cover ⇒ the EVP does real work) — NOT synthetic constant forcing; (b) the timed window must
**exclude XLA compile**. The original bench warmed up at n=2 but timed n=N (different scan length ⇒ different
executable) and `run_steps_sharded` re-jits a fresh shard_map each call ⇒ the timed wall-time was
COMPILE-dominated, and the full model's much bigger graph (120-subcycle EVP scan + KPP + GM) compiles much
longer ⇒ a spurious "10× slower than Kokkos". FIX: `run_steps_sharded(return_executable=True)` returns the
jitted fn + inputs so the bench compiles ONCE then times a warm 2nd call; per-step via the **subtraction
method** `(t_N − t_W)/(N − W)` (the JAX analog of Kokkos "omit the first 5 steps" — cancels compile, dispatch
overhead, and the AB2 first-step transient). Also wired the full forced+ice path into `run_steps_sharded`
(ice_cfg/step_forcing/forcing_static/boundary_node_p, constant forcing across the scan — fair for timing).
**CORRECTED CORE2 npes=4 (4×A100, dt1800, real forcing), ms/step:** full allgather **92.6**, full ragged
**86.7**, vs **Kokkos CORE2 full GPU 1N = 117** ⇒ **JAX is COMPARABLE / slightly faster than hand-tuned
Kokkos-CUDA** (the 10× retracted). Decomposition: ocean+forcing 58.3 (63%), +ice/EVP +18.0 (19%), +KPP+GM
+14.6 (16%) — proportions like Kokkos's own profile; the EVP is healthy, not a bottleneck. **Ragged is now
slightly FASTER than allgather on the full model** (86.7<92.6) — the first ragged win: the full model fires
~500 exchanges/step (EVP 120×2 + CG ~127×2 + substeps) so ragged's per-exchange volume savings outweigh its
per-call overhead even single-node. Compile ~33 s (one-time). NEXT: farc-full (single node, vs Kokkos 309
ms); dars/NG5 full = multi-node.

### #4 — farc forward benchmark: the ragged win is a MULTI-NODE phenomenon, not single-node (2026-06-08)
CORE2+farc at npes=4, full-ragged (substep + CG), correct dt (CORE2 1800 / farc 900), job 25440595, 4×A100
single node. ms/step: CORE2 allgather 110.6 / ragged 144.7 (**+31%**); farc allgather 287.5 / ragged 298.4
(**+3.8%**). **The ragged penalty shrinks sharply with mesh size** (+31% → +3.8%), but ragged is still
marginally SLOWER on farc-4 — even though per device the halo is ~1019 nod vs ~160k owned (all_gather moves
~600× more *data*). **Why: on a SINGLE node the 4 A100s talk over NVLink (~600 GB/s), so all_gather's larger
volume is ~free (a 2D-node all_gather at farc-4 ≈ 5 MB/device ≈ sub-ms); the cost is per-collective-call
LATENCY/overhead, and ragged has MORE of it (a less-mature collective + my extra gather/scatter/where
kernels).** So the ragged win is fundamentally a **bandwidth-bound** regime — **high device count AND/OR
MULTI-NODE** (inter-node links ~25–100 GB/s ≪ NVLink, and all_gather's O(P·N_local) volume grows with P),
which is exactly the NG5 multi-node target (B.3). Single-node benchmarks CANNOT show the ragged win and we
should stop expecting them to. **Encouraging side-result: farc per-step ≈ 0.29 s at 4×A100 is in the
ballpark of the Kokkos `SCALING_M524.md` farc(dt900) s/step — the JAX FORWARD is computationally
competitive.** Also validated: the full-ragged forward (substep + CG, EVP path) runs end-to-end + STABLY on
a 638k mesh at dt=900. NEXT: the real ragged-win test is **multi-node** (NG5 via `jax.distributed`, B.3);
dars-4 single-node would only confirm the penalty-shrink trend + that dars loads.

### #3 — CORE2 forward benchmark: the SSH/CG halo is the dominant comm (ragged-ized it) (2026-06-08)
First forward-scaling benchmark (`scripts/bench_forward_scaling.{py,sbatch}`, CORE2, 4×A100, job
25440383): the multistep ragged FORWARD **runs end-to-end on GPU through the `lax.scan`** ✅ (the
lowering validation). Timing (ms/step): npes2 allgather 144.6 / ragged 164.8; npes4 allgather 109.6 /
ragged 142.6. **Ragged is SLOWER at CORE2 scale (+14% → +30%, penalty grows with npes).** Two honest
reasons: (1) CORE2 (127k / 2–4 dev) is far BELOW the crossover — when little data moves, `all_gather`'s
single mature NCCL collective beats `ragged`'s collective + the extra gather/scatter/where kernels;
ragged's O(boundary) only beats all_gather's O(P·N_local) at large P / large mesh. (2) **DECISIVE: the
benchmark barely tests ragged** — the dominant per-step comm is the **SSH CG solve** (`ssh_matvec` +
`ssh_precond`, each an `all_gather` halo, × ~127 CG iters ⇒ ~250 all_gather exchanges/step), which was
STILL all_gather; the ragged swap only touched the ~10–20 per-substep field exchanges. So both runs were
~95% all_gather (the CG) + ragged overhead on the minority. **FIX (this commit): ragged-ized the SSH/CG
halo** — `SSHHalo` gains `ragged`/`recv_max`/`use_ragged`; `ssh_matvec`/`ssh_precond` dispatch;
`run_step_sharded`/`run_steps_sharded` build `SSHHalo(use_ragged=)`. So a `use_ragged=True` run now
ragged-izes the dominant path too. (Forward-only safe — the CG-halo ragged backward also rides the B.0d
custom_vjp gap; default `use_ragged=False` keeps grad correct.) NEXT: benchmark **farc** (638k, above the
crossover) with full-ragged (substep + CG), npes 2/4, vs all_gather + the Kokkos `SCALING_FARC.md` numbers
— the first run that CAN show a ragged win. farc mesh exporting to `/work` (job 25440396).

### #2 — B.0b/c: ragged FORWARD validated on A100; JAX autodiff transpose is broken → B.0d (2026-06-08)
GPU gate (job 25438454, 4×A100): the halo-only `ragged_all_to_all` exchange == `all_gather` **byte-identical
on the forward** (all kinds, npes 2 & 4) — the maps + `output_offsets=recv_offsets.T` arg semantics +
NCCL movement are correct. But **JAX 0.10.1's `ragged_all_to_all` reverse-mode autodiff is WRONG** (grad
max|Δ| ≈ 4.3@2 → 8.0@4, ~linear in P → its transpose sums over the axis instead of routing point-to-point).
`halo_exchange_ragged` is all-linear with an exact forward, so it's JAX's transpose, not our composition.
The AD gate did its job (caught silent grad corruption). **Forward is validated**; AD deferred to **B.0d**
(`custom_vjp`: backward via the proven all_gather VJP, or a hand-written reverse a2a). `use_ragged=True` is
forward-only safe (default False = all_gather = correct grad). ⚠️ `ragged_all_to_all` is also UNIMPLEMENTED
on XLA:CPU, so all of this (and the scaling work) is **GPU-only** — fake CPU devices have no interconnect to
measure anyway. NEXT: forward scaling (B.1 farc → B.2 dars → B.3 NG5) on real GPUs, ragged vs all_gather +
absolute throughput vs the Kokkos `SCALING_*.md` numbers.

### #1 — B.0a DONE: the ragged exchange maps (2026-06-08)
`shard_mesh.RaggedExchange` + `_ragged_exchange_map` build the per-device halo-only send/recv maps from the
same `_owner_map` the `all_gather` `_exchange_map` uses (uniform across nod/elem/edge; consistent with the
oracle by construction). Stored on `ShardedMesh.exchange_ragged`. Host-numpy gate
(`test_ragged_exchange_reproduces_allgather`, npes=2/4) confirms the ragged maps reproduce the `all_gather`
exchange on every valid lane, all three kinds. `export/load_sharded_mesh` left untouched (ragged maps
default to None on load — cheap to rebuild; export-caching is an NG5-scale item). NEXT: B.0b — wire
`lax.ragged_all_to_all` into `halo.halo_exchange` (gather→a2a→scatter) behind the `HaloCtx` interface,
then B.0c gate it against the forward + AD oracles on CPU fake-devices then 2/4×A100.

### #0 — Plan created (2026-06-08)
Phase 8 closed + tagged `v1.1-multi-gpu`. Phase 8b scopes the scaling work the whole port targets, opening
with the mandatory B.0 halo rewrite (`all_gather` → `ragged_all_to_all`) — the direct fix for the user's
"you copy too much data" concern. Confirmed `lax.ragged_all_to_all` exists in JAX 0.10.1 with a registered
transpose + jvp (so AD survives), and that the `Partition.ComStruct` (rPE/rlist + sPE/slist, already parsed
by `partit.py`) maps onto its `(operand, input_offsets, send_sizes, output_offsets, recv_sizes)` model. The
Phase-8 gates (`test_step_sharded.py` forward + `test_gradient_sharded.py` AD) are the byte-for-bit oracles
for the swap. Ladder B.0 → farc → dars → NG5, validated against `port_kokkos/docs/SCALING_{FARC,DARS,NG5}.md`.
