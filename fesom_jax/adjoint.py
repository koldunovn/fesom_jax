"""Chunked reverse accumulation — the adjoint that fits, as a public interface.

``jax.grad`` over a long window does not fit one GPU, and no rematerialization setting makes it
fit. :func:`fesom_jax.integrate._run_steps`'s ``segments`` knob stores ``S + N/S`` carries,
minimised at ``S=√N``; a CORE2 :class:`~fesom_jax.state.State` is **1.93 GB**, so a 5-day window
(N=240) costs ≥ 2√240 × 1.93 ≈ 60 GB of carries plus ~18 GB of reverse working set against ~78 GB
usable on an 80 GB A100. Measured: ``S=31`` died at 69.24 GiB (job 26798871). Reaching for a
larger ``segments`` makes it worse, and so does a smaller one.

What fits is a different loop shape. Run the window forward in K-step chunks, park each
chunk-boundary ``State`` on the **HOST**, then walk the chunks backwards calling a jitted
:func:`jax.vjp` over one chunk at a time, carrying the state cotangent and summing the parameter
cotangent. The device then holds one chunk regardless of window length — measured **43.1 GB at
1 day, 43.2 at 30, 43.1 at 60**. This is exact: it is checkpointing, not an approximation.

The cost moves to host memory, at one ``State`` per kept boundary (1.93 GB per day at K=48). That
is what ``boundary_stride`` is for: keep every Nth boundary and recompute the spans between, for
exactly one extra forward pass over the window. A full year needs it — 366 × 1.93 = 706 GB against
a 515 GB node, against ~74 GB at stride 19. Validated at strides 1/2/3: ``dL/dm`` agrees to 3e-5,
better than the model's own 6e-4 forward reproducibility.

⚠️ **A long gradient can be finite, reproducible and still worthless, and the SEED STATE decides.**
With no network at all the same probe amplifies the reverse pass 1.0008/step from one seed and
1.0539/step from another — 1.2× against 3e5 over a 240-step window. :attr:`ChunkedGrad.trace` is
returned for exactly that reason; feed it to :func:`fesom_jax.adjoint_stab.amplification` before
using the gradient. This module produces the trace and does not decide what passes, because the
bars belong to the study, not to the library.

Design rationale, the call sites this replaces and the names considered: ``docs/ADJOINT_API.md``.
Measurements: ``docs/ADJOINT_HORIZON.md``.
"""

from __future__ import annotations

import functools
import math
import time
from typing import Any, Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp

from . import adjoint_stab
from .integrate import _run_steps

__all__ = ["scan_steps", "chunked_forward", "chunked_grad", "ChunkedGrad", "ReverseCheckpoint",
           "LegIncomplete"]


# ---------------------------------------------------------------------------
# the in-chunk runner


def _segments_for(length: int) -> int:
    """Segment count for two-level rematerialization: ``√N``, which is the floor.

    Not a parameter. The scheme stores ``S + N/S`` carries and every call site in this project
    computed exactly this expression for itself before the formula was moved here."""
    return max(2, int(round(math.sqrt(max(1, length)))))


def scan_steps(step_fn: Callable[[Any, Any], Any], state, xs=None, *,
               length: int | None = None, rematerialize: bool = True):
    """Run ``length`` (or ``len(xs)``) steps of ``step_fn`` and return the final state.

    ``step_fn(state, x) -> state``. ``xs`` is a pytree stacked on a leading axis, one slice per
    step, or ``None`` for a loop with no per-step input (then ``length`` is required).

    ``rematerialize=True`` (default) applies two-level √N rematerialization, so a reverse pass
    over the result stores ``≈2√N`` carries instead of ``N``. Use it for anything that will be
    differentiated in reverse mode.

    ``rematerialize=False`` is a plain scan. Use it for a **forward-only** run, and for
    **forward mode**: a JVP has no reverse pass and stores no tape, so :func:`jax.checkpoint`
    there is semantically the identity and pure overhead — measured as minutes of constant-folding
    scatter-adds inside ``jit(jvp_chunk)/jvp()/while/body/closed_call/checkpoint/checkpoint``.

    ⚠️ This deliberately has no ``segments`` parameter. See the module docstring: it reads like
    the knob you reach for when a window will not fit, and reaching for it makes things worse.
    Chunk the window with :func:`chunked_grad` instead.
    """
    if length is None:
        if xs is None:
            raise ValueError("scan_steps needs `length` when `xs` is None")
        length = int(jax.tree.leaves(xs)[0].shape[0])

    def body(carry, x):
        return step_fn(carry, x), None

    return _run_steps(body, state, xs=xs, length=length,
                      checkpoint=bool(rematerialize),
                      segments=_segments_for(length) if rematerialize else 0)


# ---------------------------------------------------------------------------
# chunked reverse accumulation


class ChunkedGrad(NamedTuple):
    """What :func:`chunked_grad` returns.

    The window description is part of the result on purpose. A row that does not say which
    ``boundary_stride`` produced it cannot be reproduced from the row: the 180-day ladder record
    did not, nor did the year rung, and resolving that confound cost two five-hour jobs."""

    loss: float
    grad: Any                       #: pytree matching ``params``
    trace: list                     #: chunk-boundary cotangent norms, ``n_chunks + 1`` of them
    host_gb: float                  #: boundaries parked on the host
    forward_seconds: float
    reverse_seconds: float
    n_chunks: int
    chunk_steps: int
    boundary_stride: int


class ReverseCheckpoint(Protocol):
    """Makes the sweep resumable across jobs. The library CALLS this; it does not implement it.

    A year-long gradient is ~10 h of one A100 against a 12 h queue limit and two years fit no slot
    at all, but the reverse loop's entire carried state is one cotangent ``State``, the running
    parameter cotangent and an index — so it can stop and continue. The implementation knows about
    filesystem layout, a process exit code and a wall-clock budget, none of which belong here."""

    have_boundaries: bool
    loss: float

    def forward_resume(self, like) -> tuple[Any, int]:
        """``(state, first_chunk)`` to resume the forward from, or ``(None, 0)`` to start over."""

    def put_boundary(self, c: int, state) -> bool:
        """Store boundary ``c``. Return True to stop the leg here."""

    def load_boundaries(self, like) -> dict: ...

    def finish_forward(self, loss: float) -> None: ...

    def reverse_start(self, seed_cot, n_chunks: int, like) -> tuple[Any, Any, int, list]:
        """``(cotangent, grad_so_far, start_chunk, trace)``."""

    def tick(self, next_chunk: int, cot, grad, trace: list) -> bool:
        """Record progress after a chunk. Return True to stop the leg here."""


class LegIncomplete(Exception):
    """A :class:`ReverseCheckpoint` asked to stop; the sweep is saved and not finished.

    Raised rather than returned so a partial result can never be mistaken for a gradient. The
    caller decides what to do — a batch driver typically exits with a code its resubmit script
    recognises."""

    def __init__(self, phase: str, chunk: int):
        super().__init__(f"leg stopped in {phase} at chunk {chunk}")
        self.phase, self.chunk = phase, chunk


# ⚠️ maxsize is SMALL on purpose. Each entry holds a compiled executable, and for CORE2 that is
# GB-scale on the device — this cache is not a place to be generous. It was 32 for one afternoon,
# during which a sweep over four adjoint configurations in ONE process died with
# `RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN`: eight executables (a forward and a backward
# per configuration) stayed resident because the cache still referenced every one of them, and the
# ninth had nowhere to load. That looked exactly like "the unfrozen adjoint does not fit", which
# was wrong and would have been a bad thing to believe.
#
# The cache exists for ONE call pattern: the same `chunk_fn` used repeatedly — a reproducibility
# repeat, or a finite-difference sweep's two forwards per step size. Four entries cover that with
# room to spare. **A sweep over CONFIGURATIONS still wants one process each**, which is what
# `scripts/nn_eddy/adjoint_horizon.sbatch` has always done and says in its header.
_JIT_CACHE = 4


@functools.lru_cache(maxsize=_JIT_CACHE)
def _jitted_forward(chunk_fn):
    return jax.jit(chunk_fn)


@functools.lru_cache(maxsize=_JIT_CACHE)
def _jitted_backward(chunk_fn):
    @jax.jit
    def bwd(state, xs, params, cot):
        _, vjp = jax.vjp(lambda s, p: chunk_fn(s, xs, p), state, params)
        return vjp(cot)
    return bwd


# Both caches are keyed on the `chunk_fn` OBJECT, and that is load-bearing rather than a
# micro-optimisation. `jax.jit` returns a fresh wrapper with a fresh compilation cache every time
# it is called, so jitting inside the function body would recompile the chunk — minutes on CORE2 —
# on every call. The reference implementation defined `fwd_chunk`/`bwd_chunk` ONCE outside its
# repeat loop for exactly this reason, and a caller that runs `chunked_grad` several times (a
# reproducibility repeat) or `chunked_forward` several times (a finite-difference sweep, two
# evaluations per step size) would otherwise pay for a full recompile each time and read it as the
# extraction having made the adjoint slower. `maxsize` bounds what the cache keeps alive.


def _free(tree) -> None:
    """Release device buffers now instead of at the next GC.

    ⚠️ Only ever after :func:`jax.block_until_ready` on everything that consumes them. A VJP's
    inputs stay live until it completes, and freeing a live buffer does not fail loudly."""
    jax.tree.map(lambda x: x.delete() if hasattr(x, "delete") else None, tree)


def chunked_grad(chunk_fn: Callable[[Any, Any, Any], Any], params, *,
                 init_state, n_chunks: int, xs_fn: Callable[[int], Any],
                 loss_fn: Callable[[Any], Any], chunk_steps: int,
                 boundary_stride: int = 1, free_xs: bool = True,
                 checkpoint: ReverseCheckpoint | None = None,
                 on_forward: Callable[[float, float, float], None] | None = None,
                 on_chunk: Callable[[int, float, Any], None] | None = None) -> ChunkedGrad:
    """``d(loss_fn(final_state)) / d(params)`` over ``n_chunks`` chunks, one chunk on the device.

    :param chunk_fn: ``(state, xs, params) -> state``, advancing the state by one chunk. Every
        model-specific thing — mesh, SSH operator, configuration, frozen terms, how ``params``
        becomes :class:`~fesom_jax.params.Params` — lives in this closure and nothing here needs
        to know about it. Jitted internally.
    :param params: the differentiated pytree. ``grad`` comes back with the same structure.
    :param xs_fn: ``c -> xs`` for chunk ``c``, built ON DEMAND. ⚠️ Not a list, and the difference
        is measured: holding every chunk's forcing on the device is ~0.5 GB per simulated day of
        idle residency and is the entire reason the device peak was NOT flat in window length
        (43.1 GB at 1 day rising to 56.9 at 30, extrapolating to an OOM near 90 days). Built per
        chunk and freed, the peak is flat and the results are bit-identical — only when the arrays
        exist changed. Returning a list from a closure over one defeats this.

        ⚠️ **Contract: the returned arrays are FREED after use, and ``xs_fn`` may be called more
        than once for the same ``c``** (the reverse walk re-asks, and ``boundary_stride > 1``
        re-asks again while rebuilding a span). Build fresh each call. Returning a cached array,
        or a view of one, hands the library a buffer it will delete out from under the next
        caller — and note that in JAX a full-width slice ``a[0:len(a)]`` is ``a`` itself, so a
        one-chunk window is the case that trips this first. The violation raises immediately with
        a message naming this contract; it does not corrupt anything.
    :param free_xs: release each chunk's inputs as soon as the chunk is done. Set False only if
        ``xs_fn`` genuinely serves cached arrays that must outlive the call — then the caller owns
        the residency this parameter exists to avoid.
    :param loss_fn: ``final_state -> scalar``. Differentiated once, to seed the sweep.
    :param chunk_steps: steps per chunk. Not used to run anything — ``chunk_fn`` decides that —
        but carried into the result and needed by
        :func:`fesom_jax.adjoint_stab.amplification` to turn the trace into a per-STEP rate.
    :param boundary_stride: keep every Nth chunk boundary on the host and recompute the spans
        between, for one extra forward pass. Host cost is ``n/stride + stride`` states, minimised
        near ``√n_chunks``. The gradient is unchanged: the recomputed states are the same states.
    :param checkpoint: optional :class:`ReverseCheckpoint`. Raises :class:`LegIncomplete` when it
        asks to stop.
    :param on_forward: ``(loss, host_gb, seconds)`` once the forward completes. A reverse sweep can
        run for hours; without this the first output of the whole call arrives at the end, and
        "the forward finished and parked N GB" is the line that says the run is worth waiting for.
    :param on_chunk: ``(chunk_index, cotangent_norm, grad_so_far)`` after each reverse chunk. The
        same reason. It is a reporting hook and must not mutate anything it is handed.

    :raises LegIncomplete: if ``checkpoint`` stopped the leg before the sweep finished.
    """
    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
    stride = max(1, int(boundary_stride))

    def get_xs(c):
        """``xs_fn(c)``, turning a contract violation into the sentence that explains it.

        Without this the caller sees JAX's ``Array has been deleted``, raised from inside their
        own slicing expression, with nothing connecting it to this function having freed the
        buffer one chunk earlier."""
        try:
            return xs_fn(c)
        except RuntimeError as exc:
            if "deleted" not in str(exc):
                raise
            raise RuntimeError(
                f"xs_fn({c}) returned a buffer chunked_grad had already freed. xs_fn must BUILD "
                f"each chunk's inputs fresh — it is called again for the same chunk on the "
                f"reverse walk, and again per span when boundary_stride > 1. Returning a cached "
                f"array, or a slice that aliases one (in JAX a[0:len(a)] IS a), hands this "
                f"function a buffer it deletes. Pass free_xs=False if the caching is deliberate."
            ) from exc

    free = _free if free_xs else (lambda _t: None)
    fwd_chunk = _jitted_forward(chunk_fn)
    bwd_chunk = _jitted_backward(chunk_fn)

    # --- forward, boundaries to the host ------------------------------------
    ta = time.time()
    if checkpoint is not None and checkpoint.have_boundaries:
        boundaries = checkpoint.load_boundaries(like=init_state)
        loss_v, fwd_t = float(checkpoint.loss), 0.0
        s_fin = None
    else:
        s_fin, boundaries = _forward(fwd_chunk, params, init_state, n_chunks, get_xs,
                                     stride, free, checkpoint)
        loss_v = float(loss_fn(s_fin))
        fwd_t = time.time() - ta
        if checkpoint is not None:
            checkpoint.finish_forward(loss_v)
    host_gb = sum(a.nbytes for a in jax.tree.leaves(boundaries)) / 1e9
    if on_forward is not None:
        on_forward(loss_v, host_gb, fwd_t)

    # --- seed the sweep -----------------------------------------------------
    # `_forward` always keeps the LAST boundary whatever the stride, so the final state is
    # available on a resume too and never has to be recomputed.
    if s_fin is None:
        s_fin = jax.device_put(boundaries[n_chunks])
    _, vjp_loss = jax.vjp(loss_fn, s_fin)
    cot = jax.block_until_ready(vjp_loss(1.0)[0])
    _free(s_fin)
    del s_fin

    # `grad` starts as None rather than a zeros pytree: the first chunk's cotangent DEFINES the
    # structure and dtypes, so nothing here has to guess them from `params`, whose leaves may be
    # Python scalars, and whose `None` fields are empty pytree nodes rather than leaves.
    if checkpoint is None:
        grad, c_start = None, n_chunks - 1
        trace = [float(adjoint_stab.cotangent_norm(cot))]
    else:
        resumed, grad, c_start, trace = checkpoint.reverse_start(cot, n_chunks, like=init_state)
        if resumed is not cot:
            _free(cot)
            cot = jax.device_put(resumed)
        trace = list(trace) if trace else [float(adjoint_stab.cotangent_norm(cot))]

    # --- reverse walk -------------------------------------------------------
    tb = time.time()
    for sup_lo in reversed(range(0, n_chunks, stride)):
        if sup_lo > c_start:                      # done in an earlier leg
            continue
        sup_hi = min(sup_lo + stride, n_chunks)
        # `boundaries` holds only every stride-th state; rebuild the span in between from the
        # nearest stored one. stride == 1 => the span IS `boundaries`.
        span = (boundaries if stride == 1
                else _rebuild_span(fwd_chunk, params, boundaries, sup_lo, sup_hi, get_xs, free))
        for c in reversed(range(sup_lo, min(sup_hi, c_start + 1))):
            s_in = jax.device_put(span[c])
            xs_c = get_xs(c)
            g_s, g_p = bwd_chunk(s_in, xs_c, params, cot)
            # ⚠️ ORDER MATTERS: `s_in`, `xs_c` and the OLD `cot` are live inputs to the VJP until
            # it completes. Block on its outputs first, then free.
            g_s, g_p = jax.block_until_ready((g_s, g_p))
            grad = g_p if grad is None else jax.tree.map(jnp.add, grad, g_p)
            _free(cot)
            free(xs_c)
            _free(s_in)
            del s_in
            cot = g_s
            trace.append(float(adjoint_stab.cotangent_norm(cot)))
            if on_chunk is not None:
                on_chunk(c, trace[-1], grad)
            # After chunk c the carried cotangent belongs to chunk c-1; -1 means the sweep is
            # complete. This is the only place a leg may end.
            if checkpoint is not None and checkpoint.tick(c - 1, cot, grad, trace):
                raise LegIncomplete("reverse", c - 1)
        span = None

    return ChunkedGrad(loss=loss_v, grad=grad, trace=trace, host_gb=host_gb,
                       forward_seconds=fwd_t, reverse_seconds=time.time() - tb,
                       n_chunks=n_chunks, chunk_steps=chunk_steps, boundary_stride=stride)


def chunked_forward(chunk_fn: Callable[[Any, Any, Any], Any], params, *,
                    init_state, n_chunks: int, xs_fn: Callable[[int], Any],
                    free_xs: bool = True):
    """Run the window forward in chunks and return the final state. No gradient, no boundaries.

    The same loop as :func:`chunked_grad`'s forward half, and the same ``xs_fn`` contract, but it
    parks nothing on the host. That is what a finite-difference check of ``chunked_grad`` wants —
    the perturbed evaluations must NOT store boundaries, must not write to a checkpoint and must
    not pay for either — and it is the shape of any long forward run whose per-chunk inputs are
    too large to hold all at once.
    """
    free = _free if free_xs else (lambda _t: None)
    s, _ = _forward(_jitted_forward(chunk_fn), params, init_state, n_chunks, xs_fn,
                    stride=1, free=free, checkpoint=None, keep_boundaries=False)
    return s


def _forward(fwd_chunk, params, init_state, n_chunks, xs_fn, stride, free, checkpoint,
             keep_boundaries=True):
    """Run the window, returning ``(final_state, {chunk_index: host_state} | None)``.

    The window start and the window end are always kept, whatever the stride."""
    s, c0 = init_state, 0
    kept = {0: jax.device_get(s)} if keep_boundaries else None
    if checkpoint is not None:
        st_r, c0 = checkpoint.forward_resume(like=init_state)
        if st_r is not None:
            kept = checkpoint.load_boundaries(like=init_state)
            s = jax.device_put(st_r)
        elif checkpoint.put_boundary(0, kept[0]):
            raise LegIncomplete("forward", 0)
    for c in range(c0, n_chunks):
        xs_c = xs_fn(c)
        nxt = jax.block_until_ready(fwd_chunk(s, xs_c, params))
        free(xs_c)                       # after block_until_ready: never free a live input
        if s is not init_state:
            _free(s)
        s = nxt
        if keep_boundaries and ((c + 1) % stride == 0 or c == n_chunks - 1):
            kept[c + 1] = jax.device_get(s)
            if checkpoint is not None and checkpoint.put_boundary(c + 1, kept[c + 1]):
                raise LegIncomplete("forward", c + 1)
    return s, kept


def _rebuild_span(fwd_chunk, params, stored, lo, hi, xs_fn, free):
    """Re-run chunks ``lo..hi-1`` from the stored boundary at ``lo``; host states by chunk index.

    The second level of checkpointing, and the reason a year is reachable. Level one trades device
    memory for recompute inside each chunk; this does the same to the host store. Keeping every
    ``stride``-th boundary and recomputing between costs ``n/stride + stride`` states — ~74 GB for
    a year at stride 19, against 706 GB at stride 1 on a 515 GB node.

    The price is exactly ONE extra forward pass over the window, because each chunk is recomputed
    at most once across the whole sweep. The gradient is unchanged; these are the same states."""
    s = jax.device_put(stored[lo])
    local = {lo: stored[lo]}
    for c in range(lo, hi - 1):
        xs_c = xs_fn(c)
        nxt = jax.block_until_ready(fwd_chunk(s, xs_c, params))
        free(xs_c)
        _free(s)
        s = nxt
        local[c + 1] = jax.device_get(s)
    _free(s)
    return local
