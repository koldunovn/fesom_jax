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

| mesh  | nodes  | levels | target hardware                     | Kokkos doc        |
|-------|--------|--------|-------------------------------------|-------------------|
| CORE2 | 127 k  | 47     | the correctness gate (Phase 8)      | —                 |
| farc  | 638 k  | 48     | ~1 A100                             | `SCALING_FARC.md` |
| dars  | 3.16 M | 47     | 4 GPU / 1 node                      | `SCALING_DARS.md` |
| NG5   | 7.4 M  | 70     | multi-node (`jax.distributed`)      | `SCALING_NG5.md`  |

---

## 1. Task ladder

### B.0 — STEP 0 (MANDATORY): replace `all_gather` with `ragged_all_to_all`

The load-bearing rewrite. Everything downstream is meaningless until this lands (scaling numbers on an
`all_gather` halo measure the wrong thing).

**Files:** `fesom_jax/shard_mesh.py` (build the ragged maps), `fesom_jax/halo.py` (the new primitive),
`fesom_jax/tests/test_halo.py` / `test_step_sharded.py` / `test_gradient_sharded.py` (gates).

- [ ] **B.0a — ragged exchange maps from `ComStruct`.** Per device, per kind (`nod`/`elem`/`edge`), build
      from the `Partition`'s per-rank `ComStruct` (`rPE`/`rptr`/`rlist` recv, `sPE`/`sptr`/`slist` send):
      - **send**: `send_idx` = the `slist` local interior lanes concatenated **ordered by destination
        device 0..P-1** (expand the sparse `sPE` neighbour list to a dense length-P layout), plus dense
        `send_sizes[P]` and `input_offsets[P]`. (`operand = field[send_idx]` — gather the lanes to ship.)
      - **recv**: `recv_idx` = the `rlist` local halo lanes concatenated **ordered by source device
        0..P-1**, plus dense `recv_sizes[P]` and `output_offsets[P]`. (Scatter the received contiguous
        buffer into these halo lanes; interior + pad lanes untouched — the broadcast/overwrite semantics.)
      Store on `ShardedMesh` next to the existing `exchange` map (e.g. `exchange_ragged: {kind: RaggedMap}`).
      ⚠️ One owned lane may be sent to several neighbours (it appears in multiple `sPE` blocks) → it is
      gathered multiple times into `operand`; that is correct (the transpose then scatter-ADDs the
      cotangents back, which is why AD stays correct).
- [ ] **B.0b — the new `halo_exchange`.** `operand = field[send_idx]` (gather) →
      `recv = lax.ragged_all_to_all(operand, output_buf, input_offsets, send_sizes, output_offsets,
      recv_sizes, axis_name='p')` → `field.at[recv_idx].set(recv_chunk)` (scatter into halo lanes). Keep
      the `all_gather` implementation available behind a flag (`HaloCtx` carries which primitive) as the
      reference oracle and a fallback. `halo_ctx=None` stays the dense identity no-op (single-device path
      byte-identical — the whole single-device suite is still the proof).
- [ ] **B.0c — GATE.** (1) **Forward byte-identical** to the `all_gather` path: the S.3 identity gate +
      `test_step_sharded.py` npes==1 byte-id + the npes==2 owned field-appropriate match, all green with the
      ragged primitive (CPU fake-devices first, then 2/4×A100). (2) **AD transpose-correct**:
      `test_gradient_sharded.py` green with the ragged primitive (the registered transpose must scatter-add
      the halo cotangent back to the owner — verify `d/d(T0)` reconstruction `Bᵀ(g_p)` == dense as before).
      ⚠️ `ragged_all_to_all` may need ≥2 real devices to exercise the cross-device path; confirm the
      CPU-fake-device lowering works (it should — it's a standard collective) and fall back to a 2×A100 gate
      if not. Append the B.0 lesson to `PORTING_LESSONS.md`.

### B.1 — farc (638 k × 48) forward scaling on ~1 A100

- [ ] Load the farc mesh + read its `dist_<NP>` (the `partit.py` reader should just work; confirm the
      ASCII format matches). Build `ShardedMesh` for farc.
- [ ] Forward run (a few hundred steps, the user's "~200 steps") on 1 (and 2/4) A100; record wall-clock /
      throughput. Compare against `port_kokkos/docs/SCALING_FARC.md`.
- [ ] N-vs-1 correctness spot-check on farc (the gate is mesh-agnostic — same `test_step_sharded` logic on
      the farc fixtures) to confirm the partition + exchange maps are right at this size.

### B.2 — dars (3.16 M × 47) on 4 GPU / 1 node

- [ ] farc → dars is mostly "bigger mesh, more partitions" — surface any O(N) host-side build cost or
      memory ceiling in `build_sharded_mesh` (the `[P, Lmax, …]` materialization). Scaling vs `SCALING_DARS.md`.

### B.3 — NG5 (7.4 M × 70) multi-node — the headline

- [ ] `jax.distributed.initialize` for multi-node; partition across nodes. The deep-mesh fixes the Kokkos
      `SCALING_NG5.md` flags: **nl=70 > a hardcoded level cap somewhere**, and the **step-0 global-gather
      OOM** (the initial all-mesh gather in setup must be chunked/avoided at 7.4 M nodes). Scaling vs
      `SCALING_NG5.md`. This is the user's actual goal (the 7 M mesh, ~200-step scaling test).

---

## 2. Acceptance

Phase 8b is done when the model runs N-vs-1-correct at farc/dars/NG5 scale with **halo-only point-to-point
communication** (not `all_gather`), and the wall-clock/throughput is **reported honestly against the Kokkos
SCALING_*.md numbers** — including, candidly, wherever XLA collective overhead leaves a gap vs hand-tuned
MPI. The gradient stays correct (the AD gate green with the ragged primitive). The 2-yr multi-GPU *climate*
run remains a separate follow-up (chaotic reduction-order divergence — same as the C, Phase-8 §0).

---

## Revision Log

### #0 — Plan created (2026-06-08)
Phase 8 closed + tagged `v1.1-multi-gpu`. Phase 8b scopes the scaling work the whole port targets, opening
with the mandatory B.0 halo rewrite (`all_gather` → `ragged_all_to_all`) — the direct fix for the user's
"you copy too much data" concern. Confirmed `lax.ragged_all_to_all` exists in JAX 0.10.1 with a registered
transpose + jvp (so AD survives), and that the `Partition.ComStruct` (rPE/rlist + sPE/slist, already parsed
by `partit.py`) maps onto its `(operand, input_offsets, send_sizes, output_offsets, recv_sizes)` model. The
Phase-8 gates (`test_step_sharded.py` forward + `test_gradient_sharded.py` AD) are the byte-for-bit oracles
for the swap. Ladder B.0 → farc → dars → NG5, validated against `port_kokkos/docs/SCALING_{FARC,DARS,NG5}.md`.
