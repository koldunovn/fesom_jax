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
2. **Its reverse-mode autodiff (transpose) gives the WRONG gradient** — measured on A100 (B.0c): the halo
   exchange built on it has a **byte-exact forward** (== `all_gather`) but a gradient that **mis-matches and
   scales with device count** (`max|Δ| ≈ 4.3 @ npes2 → 8.0 @ npes4`). The `all_gather` halo's gradient is
   correct (the AD oracle). **⚠️ Not yet isolated to the bare primitive — see "What is / isn't confirmed".**
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

## What IS and ISN'T confirmed (the isolation gap)

- **Confirmed:** our ragged *halo-exchange composition* has a wrong gradient on GPU (forward exact). Real, reproducible.
- **NOT confirmed:** that the bug is in JAX's `_ragged_all_to_all_transpose` itself vs. in *our* composition /
  offset maps (`send_offsets`, `out_offsets = recv_offsets.T`, `recv_sizes`, the gather/scatter index arrays)
  interacting with an assumption the transpose makes (e.g. that `output_offsets` are sorted/non-overlapping for
  the `cumsum` mask, or the `all_to_all(offsets, tiled=True)` sync).
- **The decisive test:** the **bare-primitive adjoint identity** — for the linear map
  `f = shard_map(lax.ragged_all_to_all)`, check `⟨f(x), y⟩ == ⟨x, fᵀ(y)⟩` (`fᵀ = jax.linear_transpose(f, x)`),
  with NO gather/scatter/where around it. If it fails → **JAX bug, file it**. If it holds → the bug is in our
  composition/offsets, fix on our side. Repro: `scripts/ragged_a2a_adjoint_repro.{py,sbatch}` (GPU-only).

**Run this before filing an upstream bug.** It's a small (2–4 GPU) job and decides whether the report is real.

---

## Draft upstream bug report (file AFTER the isolation repro confirms it's the bare primitive)

> **Title:** `lax.ragged_all_to_all` reverse-mode autodiff transpose produces an incorrect gradient (error
> grows with device/axis size)
>
> **Repro:** (the self-contained version of `scripts/ragged_a2a_adjoint_repro.py` — adapt the `ragged_all_to_all`
> docstring's size-2 example, wrap in `shard_map`, check the adjoint identity `⟨f(x),y⟩ == ⟨x, fᵀ(y)⟩`). GPU/TPU
> required (`ragged_all_to_all` is unimplemented on XLA:CPU).
>
> **Expected:** the adjoint identity holds to floating-point tolerance (the op is linear).
>
> **Actual:** it fails; the discrepancy grows with the number of devices (we saw `max|Δ| ≈ 4.3 @ 2 dev → 8.0
> @ 4 dev` in a downstream use). Forward is correct.
>
> **Env:** JAX 0.10.1, A100 + NCCL, `jax_enable_x64=True`.
>
> **Note:** `_ragged_all_to_all_transpose` (quoted above) builds a reverse `ragged_all_to_all`; the failure is
> likely in the `all_to_all(offsets, tiled=True)` sync and/or the `cumsum` mask for general (non-trivial,
> possibly multi-neighbour or unsorted) offset patterns — happy to help narrow it down.

---

## Can we suggest a fix to JAX? (honest assessment)

- **To JAX's transpose rule: not from inspection alone — that would be guessing.** The rule already does the
  role-swap a correct transpose needs; the bug (if it's here) is in the *details* (offset `all_to_all` / mask)
  for general offset patterns. Pinpointing it needs the isolation repro + reducing our offsets to a minimal
  failing case. We ARE well-placed to do that (we understand the offset semantics deeply — see B.0a/B.0b), so
  it's "a bit more work", not "too much": run the repro → minimize → then a precise fix or a precise report.
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
