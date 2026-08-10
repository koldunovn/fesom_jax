# The public adjoint API — design

Task 3 of `docs/plans/20260810-split-nn-eddy-and-promote-adjoint.md`. What `fesom_jax.adjoint`
must offer so the research code can stop importing `fesom_jax.integrate._run_steps`, and why the
surface is shaped the way it is. No code in this task; Task 4 implements it and carries the tests.

The mechanism itself is not in question — it is measured and written up in `ADJOINT_HORIZON.md`.
What is in question is where the seam goes, because right now there is none: chunked reverse
accumulation exists only as six closures inside `main()` of a 949-line script, closing over the
mesh, the SSH operator, the forcing builder, the freeze knobs, the stabilizer and the multiplier.

## The surface

Three functions and one protocol. Everything else stays where it is.

```
fesom_jax.adjoint
    scan_steps(step_fn, state, xs, *, rematerialize=True) -> state
    chunked_grad(chunk_fn, params, *, n_chunks, xs_fn, loss_fn, init_state,
                 chunk_steps, boundary_stride=1, checkpoint=None) -> ChunkedGrad
    ChunkedGrad(loss, grad, trace, host_gb, forward_seconds, reverse_seconds, ...)
    ReverseCheckpoint            # a Protocol, not an implementation
```

### `scan_steps` — the in-chunk runner

`step_fn(state, x) -> state`, over `xs` stacked on a leading axis. Returns the final state.

Two things distinguish it from `_run_steps`:

**The body contract is `(state, x) -> state`, not `(carry, x) -> (carry, None)`.** The `None` is
an artefact of `lax.scan`'s signature and has no business in an API. Ten call sites currently
write the adapter by hand; the library writes it once.

**There is no `segments` parameter.** `rematerialize=True` selects two-level √N rematerialization
with the segment count derived internally from `len(xs)` — the same `max(2, round(√N))` that all
five checkpointing call sites compute for themselves today, one of them by copying the formula
into a comment that calls it "the knob's correct scope". `rematerialize=False` is a plain scan,
which is what the five forward-only and forward-mode call sites pass as `checkpoint=False`.

### `chunked_grad` — the mechanism worth publishing

The caller supplies:

| what | why it is the caller's |
|---|---|
| `chunk_fn(state, xs, params) -> state` | everything model-specific — mesh, SSH operator, config, freeze knobs, stabilizer, how `params` becomes `Params` — lives in this closure and nothing else needs to know about it |
| `xs_fn(c) -> xs` | per-chunk forcing, built ON DEMAND. Not a list. See "why this is an argument and not a list" below |
| `loss_fn(final_state) -> scalar` | the library differentiates it to seed the sweep and never looks inside |
| `init_state`, `params`, `n_chunks`, `chunk_steps` | the window |
| `boundary_stride` | host-memory / recompute trade, default 1 |
| `checkpoint` | optional `ReverseCheckpoint`; `None` runs to completion in one process |

The library supplies:

- the forward over `n_chunks`, with chunk-boundary states parked on the **host** and the device
  peak equal to one chunk regardless of window length;
- second-level checkpointing: keep every `boundary_stride`-th boundary, recompute the spans
  between from the nearest kept one, for exactly one extra forward pass over the window;
- the reverse walk — a jitted `jax.vjp` over one chunk at a time, carrying the state cotangent and
  accumulating the parameter cotangent with `jax.tree.map`;
- the **buffer lifetimes**, which are a correctness hazard rather than an optimisation: the VJP's
  output must be blocked on before its inputs are freed, because `s_in`, `xs_c` and the previous
  cotangent are all live until it completes. The reference code carries a ⚠️ on that line. A
  second implementation would get it wrong, and freeing a live donated buffer does not fail
  loudly;
- the boundary-norm trace, ready for `adjoint_stab.amplification`.

`params` is a pytree, not a scalar. The reference implementation accumulates `g = g + g_c` on a
float64 scalar multiplier; the library must `jax.tree.map(add)` so a network's parameters work.
That is the whole point of the promotion — the NN-GM training loop differentiates a `gm_nn`
pytree, not a κ multiplier.

**Why `xs_fn` is a function and not a list.** Because building all the chunks' forcing up front
and holding it on device is what made the device peak grow with window length: 43.1 GB at 1 day
rising to 56.9 GB at 30, about 0.5 GB per simulated day of pure idle residency, extrapolating to
an OOM on an 80 GB card near 90 days. Building per chunk and freeing after `block_until_ready`
made the peak flat — 43.1 / 43.2 / 43.1 GB at 1 / 30 / 60 days — with bit-identical results,
because only when the arrays exist changed. An API that accepts a list invites the version that
does not scale, and it reads as the more natural one. It is not offered.

### `ChunkedGrad` — what comes back

`loss`, `grad` (pytree matching `params`), `trace`, `host_gb`, `forward_seconds`,
`reverse_seconds`, plus the window description (`n_chunks`, `chunk_steps`, `boundary_stride`) so a
result row can be reproduced from the row. The last point is not decoration: the 180-day ladder
record did not say it ran at boundary-stride 2 and the year rung did not say it ran at 19, and
resolving that confound cost two five-hour jobs.

**`trace` is returned, not screened.** It is the chunk-boundary cotangent norm, seeded with the
loss cotangent, so `len(trace) == n_chunks + 1`, and it is exactly what
`adjoint_stab.amplification(trace, chunk_steps)` consumes. The library produces it because the
reverse sweep produces it anyway; it does not decide what passes, because the bars (1.010/step
median, 0.020 log-spread) are the research project's, not the library's.

### `ReverseCheckpoint` — a protocol the library calls and does not implement

A year-long gradient is ~10 h of one A100 against a 12 h queue limit, and two years fit no slot at
all. What makes that tractable is that the reverse loop's entire carried state is one cotangent
`State` (1.93 GB), a scalar and an index — so the sweep can stop and resume. Without this in the
library, `adjoint_horizon.py` cannot move onto `chunked_grad` (Task 5) without losing the
capability that makes long windows reachable.

But the implementation knows about `/work` paths, a SLURM exit code, and a wall-clock budget.
None of that belongs in a library. So the library defines the protocol and calls it; the research
repository keeps `adjoint_ckpt.Checkpoint`, which already has exactly these methods:

```
have_boundaries -> bool          loss -> float
forward_resume(like)             -> (state | None, first_chunk)
put_boundary(c, state)           -> bool   # True => stop now, leg incomplete
load_boundaries(like)            -> dict
finish_forward(loss)             -> None
reverse_start(seed_cot, n_chunks, like) -> (cot, grad0, start_chunk, trace)
tick(next_chunk, cot, grad, trace)      -> bool   # True => stop now
```

Two signature changes are needed and both follow from `params` becoming a pytree:
`reverse_start` returns `grad0` and `tick` takes `grad`, and both are typed `float` today.

⚠️ **This is scope the plan's Task 4 does not list.** Recorded here rather than discovered at
Task 5.

## `_run_steps` stays private, and this is the reasoning

Not promoted as-is, for two independent reasons.

**Its contract would become public API.** `body: (carry, x) -> (carry, None)` over a `State`
pytree publishes both the scan artefact and the pytree's role as a scan carry.

**Its headline parameter cannot do what it looks like it does.** `segments` reads as the knob you
reach for when a window does not fit, and reaching for it makes things worse. The scheme stores
`S + N/S` carries, minimised at `S=√N`; a CORE2 `State` is 1.93 GB, so a 5-day window (N=240)
costs ≥ 2√240 × 1.93 ≈ 60 GB of carries plus ~18 GB of reverse working set against ~78 GB usable
on an 80 GB A100. Measured: `S=31` died at 69.24 GiB (job 26798871). A public function whose main
parameter is a trap is a hazard, not an API. `scan_steps` keeps the mechanism and drops the knob.

`_run_steps` therefore stays in `fesom_jax/integrate.py`, private, with a comment naming
`scan_steps` and `chunked_grad`. `integrate()` keeps using it.

⚠️ Its docstring currently quotes "43.1 GB at 1 day and 56.9 GB at 30 days" for chunked reverse
accumulation. That is the PRE-fix measurement, from before the per-chunk forcing lifetime was
fixed; the figure is now flat at 43.1 / 43.2 / 43.1. Fix it when the comment is added.

## The call sites, checked against this surface

Ten `_run_steps` calls move with the research code. Five of them do not need the adjoint at all.

| call site | today | becomes |
|---|---|---|
| `adjoint_horizon.py:592` (`run_chunk`) | `checkpoint=True, segments=√K` | `chunk_fn` body via `scan_steps` |
| `adjoint_horizon.py:615` (`one_day`, truth) | `checkpoint=True, segments=√48` | `scan_steps` |
| `horizon_scan.py:236` (`one_day`, truth) | `checkpoint=True, segments=√48` | `scan_steps` |
| `horizon_scan.py:305` (monolithic loss) | `checkpoint=True, segments=√n` | `scan_steps` — this is the superseded path, kept working, not fixed |
| `horizon_scan.py:317` (in-graph day loop) | `checkpoint=True, segments=√48` | `scan_steps` |
| `seam_gate.py:307` (`one_day`, stability arm) | `checkpoint=False` | `scan_steps(rematerialize=False)` |
| `seam_gate.py:390` (`run_chunk`) | `checkpoint=True, segments=√K` | `chunk_fn` body via `scan_steps` |
| `seam_gate.py:403` (`run_chunk_nockpt`, JVP) | `checkpoint=False` | `scan_steps(rematerialize=False)` |
| `student_year.py:146` (`chunk`) | `checkpoint=False` | `scan_steps(rematerialize=False)` |
| `student_year.py:156` (`first_chunk`) | `checkpoint=False` | `scan_steps(rematerialize=False)` |

Two observations that shaped the surface.

**`seam_gate.py:403` is not an adjoint use and neither is `:307`.** 403 is forward mode: a JVP has
no reverse pass and stores no tape, so `jax.checkpoint` there is pure overhead — semantically the
identity, measured as minutes of constant-folding inside
`jit(jvp_chunk)/jvp()/while/body/closed_call/checkpoint/checkpoint`. This is why `rematerialize`
is a real parameter with a real `False` case and not an internal default.

**Half the call sites want `rematerialize=False`.** If `scan_steps` had been designed only for the
adjoint, five of ten sites would have kept a hand-rolled `lax.scan` and the private import would
have survived in spirit.

Four further calls stay in `fesom_jax` and are a separate tidy-up: `determinism_probe.py:83`,
`sharded_grad_ab.py:158`, `scripts/paper/core2_lw_avgadj.py:261`,
`scripts/paper/core2_lw_signflip.py:160`.

## The numerical invariant

`dL/dm` from `chunked_grad` must equal `dL/dm` from the existing closures **to 3e-5** on the same
window, seed and flags. That number is not chosen for comfort: it is the agreement already
measured across boundary strides 1 / 2 / 3, and it is tighter than the model's own forward
reproducibility of 6e-4. If the extraction misses it, the extraction is wrong. **Do not adjust the
tolerance.**

Task 4 additionally asserts, on the pi mesh:

- the chunked gradient **equals** a monolithic `jax.grad` over a window short enough for both to
  fit — exact equality, since this is checkpointing and not approximation;
- the result is independent of chunk length (1, 2, 4 chunks over the same window);
- `boundary_stride > 1` changes nothing beyond 3e-5;
- `adjoint_stab.amplification(result.trace, chunk_steps)` accepts the trace and its length is
  `n_chunks + 1`.

## Names considered and rejected

`chunked_vjp` — it seeds with 1.0 on a scalar loss, so it is a gradient; the general VJP form is
available by passing your own `loss_fn`. `value_and_grad_chunked` — accurate and unusable at a
call site. `reverse_accumulate` — names the implementation rather than the result.
