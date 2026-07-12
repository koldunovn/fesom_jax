# JAX `lax.ragged_all_to_all` — status record (CPU gap + reverse-mode AD) + repro + workaround

**Purpose:** durable record of everything we know about `jax.lax.ragged_all_to_all` so it isn't lost —
the two limitations that shaped the Phase 8b halo work, the evidence, JAX's actual transpose source, what
is and isn't confirmed, the version/upstream-awareness status, a draft bug report + minimal repro, and the
workaround we use. **Created 2026-06-09** (Phase 8b B.0c findings + a fresh upstream check).

---

## TL;DR

1. **`ragged_all_to_all` is UNIMPLEMENTED on XLA:CPU** — `UNIMPLEMENTED: HLO opcode 'ragged-all-to-all' is
   not supported by XLA:CPU ThunkEmitter` (job 25438390). It runs only on **GPU (NCCL)** / TPU. So all ragged
   validation + scaling is GPU-only; the CPU correctness gates use `all_gather`. (Guards: `bench_forward_scaling.py`
   `SKIP ragged on CPU`; `tests/test_halo.py` platform skip.)
2. **Its reverse-mode autodiff (transpose) gives the WRONG gradient — CONFIRMED in the bare primitive.** The
   **bare** `lax.ragged_all_to_all` violates the adjoint identity `⟨f(x),y⟩==⟨x,fᵀ(y)⟩` by **O(1) relative
   error** (sign-flipped at npes=2), with valid offsets and a byte-exact forward (isolation job 25454884,
   A100). So it is **JAX's `ragged_all_to_all` transpose**, not our composition. The `all_gather` halo's
   gradient is correct (the AD oracle).
3. **Version:** JAX **0.10.1 is the latest** (changelog, 2026-05-20) — no newer version to upgrade to, and the
   changelog never mentions `ragged_all_to_all`. No version escape hatch.
4. **Upstream awareness:** a GitHub-issues + changelog search found **no report** specific to the
   `ragged_all_to_all` transpose → it appears **unreported**.
5. **Workaround (version-independent):** wrap the exchange in `jax.custom_vjp`, backward via the proven
   `all_gather` VJP (Phase 8b **B.0d**). Forward scaling (all dars/NG5 numbers) does NOT use the backward, and
   training at smaller scale can use the all_gather halo (correct AD) — so this blocks only **training-at-scale
   with the ragged halo**, not anything shipped.

---

## Evidence (measured, A100, Phase 8b B.0c — job 25438454)

`tests/test_halo.py`:
- `test_ragged_primitive_forward_matches_allgather` (npes 2 & 4, kinds nod/elem/edge) **PASSES** — ragged ==
  all_gather forward, byte-identical on every valid lane.
- `test_ragged_primitive_grad_known_broken` (npes 2, nod) is **`xfail`** — `grad(sum(w · ragged_exchange(f)))`
  ≠ `grad(sum(w · allgather_exchange(f)))`, `max|Δ|` ~ O(npes).

The composition under test is `halo.run_halo_exchange_ragged` = **gather** `field[send_idx]` → `lax.ragged_all_to_all`
→ **gather-back + masked `where`**. All three sub-ops are linear in `field`, and the forward is exact, so a
*constituent transpose* is wrong; the gather/scatter/where transposes are standard JAX (trusted), which is why
B.0c attributed it to `ragged_all_to_all`'s transpose. **But the test measures the COMPOSITION, not the bare
primitive's adjoint** (see below).

---

## JAX's actual transpose rule (`jax/_src/lax/parallel.py`, main == 0.10.1)

```python
def _ragged_all_to_all_transpose(
    t, operand, output, input_offsets, send_sizes, output_offsets, recv_sizes,
    *, axis_name, axis_index_groups):
  if type(t) is ad.Zero:
    operand_t = ad.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    output_t = ad.Zero(output.aval) if ad.is_undefined_primal(output) else None
  else:
    zero = ad.zeros_like_aval(operand.aval)
    output_offsets_ = all_to_all(output_offsets, axis_name, 0, 0, tiled=True)
    input_offsets_ = all_to_all(input_offsets, axis_name, 0, 0, tiled=True)
    operand_t = ragged_all_to_all_p.bind(
        t, zero, output_offsets_, recv_sizes, input_offsets_, send_sizes,
        axis_name=axis_name, axis_index_groups=axis_index_groups)
    mask = control_flow.cumsum(
        lax.full(t.shape[0], 0, dtype='int32').at[output_offsets_].set(1)
        .at[output_offsets_ + recv_sizes].add(-1))
    mask = lax.expand_dims(mask, (*range(1, t.ndim),))
    mask = lax.broadcast_in_dim(mask, shape=t.shape, broadcast_dimensions=tuple(range(t.ndim)))
    output_t = lax.select(mask, lax._zeros(t), t)
  return [operand_t, output_t] + [None] * 4
```

**What it does:** it is NOT a `psum`/sum-over-axis (an earlier guess of ours was wrong). It builds a **reverse
`ragged_all_to_all`** — the cotangent `t` is the new operand, and the send/recv roles are swapped
(`output_offsets`↔`input_offsets`, `recv_sizes`↔`send_sizes`), with the offsets first synced via
`all_to_all(..., tiled=True)` and a `cumsum` mask zeroing the non-received region of the `output` cotangent.
**Structurally this is the right shape of a transpose** — which is exactly why we must isolate before blaming it.

---

## CONFIRMED: it's JAX's bare-primitive transpose (isolation run 2026-06-09, job 25454884, A100)

The bare-primitive adjoint identity `⟨f(x), y⟩ == ⟨x, fᵀ(y)⟩` for `f = shard_map(lax.ragged_all_to_all)`
(NO gather/scatter/where around it, valid multi-neighbour offsets, forward byte-exact) **FAILS**:

```
npes=2:  <f(x),y> = -1.157977e+02   <x,fT(y)> = +6.826251e+01   rel|Δ| = 1.589   (sign-flipped!)
npes=4:  <f(x),y> = -2.930010e+02   <x,fT(y)> = -5.021581e+02   rel|Δ| = 0.417
```

For a linear op the identity must hold to float tolerance; instead it's **O(1) relative error**. So the bug is
**definitively in JAX's `ragged_all_to_all` transpose**, NOT in our composition or offset maps. (The full
composition's adjoint also fails — rel|Δ| 0.12 @ npes2, 0.26 @ npes4 — consistent: it inherits the bare bug.)
Repro: `scripts/ragged_a2a_adjoint_repro.{py,sbatch}` (GPU-only). **The upstream report is now justified.**

⚠️ Note the error does NOT scale cleanly with npes (1.589 @ 2 vs 0.417 @ 4) — it's just *wrong* (O(1)),
magnitude varying with the random data. (The earlier "grows with device count" was the absolute grad-diff in
the composition, which depends on the field/weights — not a clean O(npes) law.)

---

## Self-contained, filing-ready repro (VERIFIED on 2×A100, job 25456662)

`scripts/debug/ragged_a2a_transpose_bug.py` — **jax + numpy only**, built on JAX's OWN `ragged_all_to_all`
docstring example (axis size 2). Output:

```
(1) forward             : [[1,3,0,0],[2,2,4,0]]   == docstring        -> OK
(2) transpose f^T(ones) : [[2,2,2],[2,2,0]]
    expected (by hand)  : [[1,1,1],[1,1,0]]                            -> WRONG
(3) adjoint identity    : <f(x),y>=+1.6176  <x,f^T(y)>=+2.3548  rel|Δ|=0.313  -> VIOLATED
```

**Key observation:** `fᵀ(ones)` is **exactly `axis_size`× the correct value** (2× at 2 devices; the unsent
`operand[2]` stays 0). The correct `fᵀ(ones)[i]` = "#output positions operand[i] lands in" = 1 for every sent
element (each is routed exactly once). So the transpose **accumulates each cotangent ~`axis_size` times** — an
over-count across the device axis. (On random `y` the error is 31%, not a uniform 2×, so it's a mis-route that
adds spurious cross-terms, not a pure scalar factor — i.e. the fix is in the routing, not a `/axis_size`.)

## Draft upstream bug report (ready to file)

> **Title:** `lax.ragged_all_to_all`: reverse-mode autodiff transpose is incorrect (gradient over-counts by
> ~axis_size); forward is correct
>
> **Repro:** `scripts/debug/ragged_a2a_transpose_bug.py` (≈60 lines, jax+numpy only) — JAX's own `ragged_all_to_all`
> docstring example (axis size 2) wrapped in `shard_map`. Checks: (1) forward == documented result; (2) the
> transpose of an all-ones cotangent vs the hand-derived gradient; (3) the adjoint identity `⟨f(x),y⟩ ==
> ⟨x, fᵀ(y)⟩`. **GPU/TPU required** (`ragged_all_to_all` is unimplemented on XLA:CPU).
>
> **Expected:** forward = `[[1,3,0,0],[2,2,4,0]]` (✓); `fᵀ(ones) = [[1,1,1],[1,1,0]]`; adjoint identity holds.
>
> **Actual (JAX 0.10.1, 2×A100, x64):** forward correct, but `fᵀ(ones) = [[2,2,2],[2,2,0]]` (exactly 2× =
> axis_size) and the adjoint identity is violated (rel|Δ| = 0.31). Downstream (a halo exchange) this corrupts
> `jax.grad` silently.
>
> **Likely locus:** `_ragged_all_to_all_transpose` in `jax/_src/lax/parallel.py` builds a reverse
> `ragged_all_to_all` (good shape) but the `all_to_all(input_offsets/output_offsets, tiled=True)` sync and/or
> the `cumsum` mask appear to route each cotangent to ~axis_size destinations instead of one. Happy to help.

---

## Can we suggest a fix to JAX? (honest assessment, post-isolation)

- **To JAX's transpose rule: a DIRECTION, not yet a one-line fix.** Now that it's isolated to the bare
  primitive, the rule's structure (reverse `ragged_all_to_all` with role-swap) is right, so the defect is in
  the **offset handling for general offset patterns** — the prime suspects are `all_to_all(output_offsets,
  tiled=True)` / `all_to_all(input_offsets, tiled=True)` (do they correctly invert the forward's "offsets are
  in the receiver's frame" convention for multi-neighbour sends?) and/or the `cumsum` mask if `output_offsets`
  aren't contiguous/sorted. Pinning the exact line needs minimizing our multi-neighbour offsets to the
  smallest failing case (≈ a day's work; we know the offset semantics from B.0a/B.0b). So: **suggesting the
  precise patch is "a bit more work, not too much"** — but the credible bug report (adjoint-identity failure +
  this suspect list) we can file now.
- **On our side: yes, a clear fix — `jax.custom_vjp` (B.0d).** Wrap `halo_exchange_ragged`; for the backward,
  reuse the **proven `all_gather` exchange's VJP** (correct on every meaningful lane; backward stays
  O(P·N_local), but the forward — the scaling-critical path — stays ragged). Optional fully-scaling variant:
  hand-write the reverse `ragged_all_to_all` ourselves (we control the offsets). This is version-independent
  and unblocks ragged training without waiting on upstream.

---

## Scope (so this stays in perspective)

The broken backward blocks ONLY **training-at-scale with the ragged halo**. Everything shipped is unaffected:
the forward scaling (CORE2/dars/NG5 — `v1.2-multinode-scaling`) never runs the backward, and training at
smaller scale can use the `all_gather` halo (correct AD). Default `use_ragged=False` = all_gather = correct grad.

## Sources
- JAX transpose source: `jax/_src/lax/parallel.py` `_ragged_all_to_all_transpose` (raw.githubusercontent.com/jax-ml/jax/main).
- JAX CHANGELOG (latest 0.10.1, 2026-05-20; no `ragged_all_to_all` entry).
- GitHub issues search — no `ragged_all_to_all` transpose report found.
- Our evidence: `tests/test_halo.py` (B.0c), plan `docs/plans/20260608-fesom-jax-phase8b-scaling.md` RevLog #2.
