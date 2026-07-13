# Padded dense-`all_to_all` halo — a CPU-capable, AD-correct ragged substitute

**Overnight prototype, 2026-07-13.** No repo file is modified: everything lives in this
folder; the model is adapted at runtime.

> **PRODUCTIZED same day** as the `use_padded` mode (`fesom_jax/halo.py::halo_exchange_padded`,
> host-built maps in `shard_mesh.py`, plumbed through `integrate_sharded`/`run_from_config`;
> gates in `fesom_jax/tests/test_halo.py`). Levante verification extended the gates to
> dist_16/32 and added the full-model CPU benchmark campaign — results + transport-choice
> guidance: **[`docs/PARALLELISM.md`](../../docs/PARALLELISM.md)**. This folder stays as the
> prototype record; new code should use `use_padded`, not `install()`.

## Why

Two documented blockers (see `docs/JAX_RAGGED_A2A_BUG.md`):

1. `lax.ragged_all_to_all` is **UNIMPLEMENTED on XLA:CPU** → on CPU the only halo is the
   `all_gather` broadcast, which receives ~the whole global field per device per exchange
   (~47 MB per 3-D field at CORE2; ~2.65 GB at NG5) — ×30 exchanges per step.
2. Its reverse-mode **transpose is wrong** (adjoint identity violated O(1), confirmed at the
   bare-primitive level) → even on GPU, gradients must fall back to the all_gather halo,
   blocking training-at-scale.

A dense `lax.all_to_all` exists and works on every backend, and its transpose is another
`all_to_all` (one of JAX's oldest, most-trusted rules). Padding each per-neighbour chunk to
the max pair-chunk size (`slot`) turns the ragged exchange into ONE dense `all_to_all` over
`P × slot` lanes — at CORE2 dist_4 that is **0.24 MB** per device per 3-D exchange instead
of 47 MB (true ragged: 0.14 MB; the padding overhead is 1.8×@P4 … 3.4×@P8 … 40×@P128).

## Files

- `padded_halo.py` — the exchange (`padded_exchange`) + a runtime adapter (`install(sm)`)
  that rebinds `halo_exchange_ragged` in `fesom_jax.halo` **and** `fesom_jax.ssh` (ssh.py
  binds the name at import). After `install`, running the model with `use_ragged=True`
  transparently uses the padded exchange. Slot widths are resolved per entity kind via the
  trace-static key `(recv_max, Lmax)`. Receiver-frame offsets are reconstructed in-trace as
  the exclusive cumsum of `recv_sizes` — exactly how `shard_mesh._ragged_exchange_map`
  builds `recv_offsets`, with the same canonical per-(e,d) chunk ordering on both sides.
- `gate_forward_grad.py` — oracle gates on the real CORE2 partitions: forward AND
  `jax.grad` vs the proven `all_gather` exchange, kinds nod/elem/edge, trailing shapes
  `()`, `(5,)`, `(3,2)`.
- `bench_padded.py` — copy of `scripts/bench/bench_forward_scaling.py` (3 marked deltas:
  path bootstrap, CPU-ragged skip removed, `--ragged 1` ⇒ padded adapter).
- `results/` — full logs of the A/B below.

## Gate results (this machine: M3 Max, CPU backend, jax/jaxlib 0.10.1)

`XLA_FLAGS=--xla_force_host_platform_device_count=N python gate_forward_grad.py N`

| npes | forward vs all_gather | grad vs all_gather |
|---|---|---|
| 2 | **bit-exact** (max diff 0.0), 9/9 | **bit-exact**, 9/9 |
| 4 | **bit-exact**, 9/9 | **bit-exact**, 9/9 |
| 8 | **bit-exact**, 9/9 | **bit-exact**, 9/9 |

## CORE2 full-model A/B (KPP+GM+ice+JRA55-1958, dt=1800, 20 steps, this machine)

| npes | halo | per step | vs all_gather | max_uv | peak RSS | CPU work (user s) |
|---|---|---|---|---|---|---|
| 4 | all_gather | 2376.9 ms | — | 1.135 | 25.2 GB | 1612 |
| 4 | **padded a2a** | **2370.5 ms** | −0.3 % (wash) | 1.135 | 23.2 GB | 1613 |
| 8 | all_gather | 2470.3 ms | — | 1.135 | 28.2 GB | 1746 |
| 8 | **padded a2a** | **2180.4 ms** | **−11.7 %** | 1.135 | 24.3 GB | 1603 |

**dist_8 + padded = 2180 ms/step (~2.26 SYPD) is the new best config on this machine**
(previous best: dist_4 all_gather, 2366–2392 ms). The padded halo restores the scaling
direction — 4→8 now improves instead of degrading — exactly as the volume analysis
predicted (all_gather cost is ~constant-per-device in npes; padded shrinks with shards).
`max_uv` is identical across all four runs (same protocol as the repo bench: 20 steps,
warmup 2, subtraction method; full logs in `results/`).

## How to run on Levante (GPU)

Same commands; the adapter is backend-agnostic. On GPU it competes with TRUE ragged for the
forward (expect padded ≈ ragged × small padding factor), but unlike ragged its **gradient
is correct**, so `use_ragged`-style training-at-scale becomes possible without the
`custom_vjp` workaround. To A/B on Levante:

```bash
python experiments/padded_halo_a2a/gate_forward_grad.py 4          # gates (any backend)
python experiments/padded_halo_a2a/bench_padded.py --ragged 1 ...  # padded
python scripts/bench/bench_forward_scaling.py --ragged 1 ...       # true ragged (GPU)
```

## Productization notes

- Move `padded_exchange` into `fesom_jax/halo.py` as a third mode (`use_padded`), with the
  slot maps built ON HOST in `shard_mesh.py` (a `PaddedExchange` sibling of
  `RaggedExchange`) instead of the in-trace reconstruction the prototype uses (the
  in-trace index math is [P]- and [Lmax]-sized integer ops per exchange — cheap, but free
  is better).
- Lift the `return_grad_fn=True with use_ragged=True` guard in `integrate_sharded.py` for
  the padded mode — its transpose is exact (gate above), the guard exists only because the
  RAGGED transpose is broken.
- The bench copy's deltas are marked `EXPERIMENT DELTA` for easy diffing against the
  original.

## Caveats

- jax/jaxlib version matrix on CPU (in-process collectives): 0.10.1 works at npes=4/8
  (npes=2 segfaults at MODEL scale — the small gates above pass); 0.10.2 deadlocks at
  npes≥4. Pin 0.10.1 (this env is pinned).
- Padding waste grows with npes (chunk-size skew): fine ≤ dist_32, ~40× vs true ragged at
  dist_128 — still 20× less volume than all_gather.
