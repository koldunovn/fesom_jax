"""``fesom_jax.adjoint`` — the chunked reverse accumulation, on the pi mesh.

The claim being asserted is **exactness**, not adequacy. Chunking the reverse sweep and striding
the host boundaries are both checkpointing schemes: they change where states are stored and when
they are recomputed, and they must change no number. So the gates here are equality against a
monolithic ``jax.grad``, invariance to the chunk length, and invariance to the boundary stride —
each one a way the scheme could be silently wrong while still returning a finite, plausible,
reproducible gradient.

⚠️ The window is short and the pi mesh is small **on purpose**. This file guards the loop's
algebra, which is size-independent; it cannot guard the memory behaviour that motivates the module
(43.1 GB at 1 day, 43.2 at 30, 43.1 at 60), because that needs a CORE2 ``State`` and an A100. That
gate is the equivalence job in ``docs/ADJOINT_HORIZON.md``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import adjoint_stab, forcing, ic, ssh
from fesom_jax.adjoint import (ChunkedGrad, LegIncomplete, chunked_forward, chunked_grad,
                               scan_steps)
from fesom_jax.config import A_VER
# the private function scan_steps replaces — imported HERE, in the test that proves the two build
# the same graph, and nowhere else in the library or the scripts.
from fesom_jax.integrate import _run_steps
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.step import step

DT = 100.0          # the validated pi config
N_STEPS = 24        # short: both the chunked and the monolithic path must fit
K = 6               # steps per chunk => 4 chunks


@pytest.fixture(scope="module")
def fixture():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = load_mesh()
    return mesh, ssh.build_ssh_operator(mesh, dt=DT), forcing.surface_stress(mesh), \
        ic.initial_state(mesh)


def _params(kver):
    return Params(k_ver=jnp.asarray(kver, jnp.float64),
                  a_ver=jnp.asarray(A_VER, jnp.float64))


def _mean_sst(state, mesh):
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    return jnp.sum(jnp.where(wet0, state.T[:, 0], 0.0)) / jnp.sum(wet0)


def _advance(state, x, mesh, op, stress, params):
    """One step. ``x`` is a per-step stress scale, so the ``xs`` path is exercised rather than
    left as ``None`` — a chunked run whose chunks all take the same input would not notice a
    reverse sweep that fed chunk c's forcing to chunk c+1."""
    return step(state, mesh, op, stress * x, params, dt=DT, is_first_step=False)


# A fixed, non-uniform per-step stress scale on the HOST. Deterministic, and different for every
# step so a reverse walk that fed chunk c's input to chunk c+1 could not pass.
XS_HOST = np.asarray(1.0 + 0.05 * np.cos(np.arange(N_STEPS)), dtype=np.float64)


_CHUNK_FN: dict = {}


def _harness(fixture, chunk_steps):
    """``(chunk_fn, xs_fn, loss_fn, n_chunks, xs_all)``, with ONE ``chunk_fn`` per chunk length.

    ⚠️ **`chunk_fn` is memoised and the other four are NOT**, and both halves of that are
    load-bearing.

    Memoised, because `chunked_grad` caches its compiled forward and backward on the `chunk_fn`
    OBJECT: a helper that builds a fresh closure per call defeats that cache and leaves one
    executable per call resident. Run standalone this file passed; run inside the `unit` group it
    died with XLA's `INTERNAL: Failed to materialize symbols` — the CPU JIT out of dylib space,
    not a numerical failure (job 26851939, array task 5).

    NOT memoised for `xs_all`, because `chunked_grad` DELETES what `xs_fn` hands it, and
    `test_xs_fn_must_build_fresh_and_says_so` deliberately hands it an aliasing full-width slice
    of `xs_all` — twice, expecting an intact array the second time. Memoising the whole tuple
    (which this helper did for one afternoon) shares the freed array across the two calls, and the
    escape-hatch half of that test dies with `Array has been deleted` from inside its own slicing
    expression: the exact error the test exists to keep users from having to interpret, arriving
    from the test's own bookkeeping instead of from the contract (job 26853569)."""
    mesh, op, stress, st0 = fixture
    assert N_STEPS % chunk_steps == 0

    key = (id(fixture), chunk_steps)
    if key not in _CHUNK_FN:
        def chunk_fn(state, xs, params):
            return scan_steps(lambda s, x: _advance(s, x, mesh, op, stress, params),
                              state, xs, rematerialize=True)
        _CHUNK_FN[key] = chunk_fn

    def xs_fn(c):
        # ⚠️ Sliced on the HOST and put on the device fresh every call. Slicing a device array
        # instead would return the array ITSELF when the window is one chunk wide, and
        # `chunked_grad` frees what `xs_fn` hands it — see test_xs_fn_must_build_fresh.
        return jnp.asarray(XS_HOST[c * chunk_steps:(c + 1) * chunk_steps])

    return (_CHUNK_FN[key], xs_fn, (lambda s: _mean_sst(s, mesh)),
            N_STEPS // chunk_steps, jnp.asarray(XS_HOST))


def _monolithic(fixture, kver):
    """``(loss, dloss/dk_ver)`` in one ``jax.grad`` over the whole window."""
    mesh, op, stress, st0 = fixture
    _, _, loss_fn, _, xs_all = _harness(fixture, K)

    def loss(k):
        p = _params(k)
        fin = scan_steps(lambda s, x: _advance(s, x, mesh, op, stress, p),
                         st0, xs_all, rematerialize=True)
        return loss_fn(fin)

    v, g = jax.value_and_grad(loss)(jnp.asarray(kver, jnp.float64))
    return float(v), float(g)


def _chunked(fixture, kver, chunk_steps=K, stride=1, checkpoint=None, **kw):
    st0 = fixture[3]
    chunk_fn, xs_fn, loss_fn, n_chunks, _ = _harness(fixture, chunk_steps)
    return chunked_grad(chunk_fn, _params(kver), init_state=st0, n_chunks=n_chunks,
                        xs_fn=xs_fn, loss_fn=loss_fn, chunk_steps=chunk_steps,
                        boundary_stride=stride, checkpoint=checkpoint, **kw)


KVER = 1e-4


@pytest.fixture(scope="module")
def baseline(fixture):
    """One chunked run at K/stride=1, computed ONCE.

    Every invariance test compares against this. Recomputing it per parametrization cost four
    redundant traces of the whole window, which is most of a minute each and all of it compile."""
    return _chunked(fixture, KVER)


@pytest.fixture(scope="module")
def single_chunk(fixture):
    """The same window as ONE chunk — no boundary at all, so nothing is carried across one."""
    return _chunked(fixture, KVER, chunk_steps=N_STEPS)


# ---------------------------------------------------------------------------
# 1. the headline: chunked == monolithic
# ---------------------------------------------------------------------------
def test_chunked_equals_monolithic_grad(fixture, baseline, capsys):
    """Exact equality is the point. This is checkpointing, not an approximation.

    The tolerance is 1e-12 relative rather than 0.0 only because the two paths sum the parameter
    cotangent in a different ORDER — the monolithic scan accumulates inside one reverse pass, the
    chunked one adds four per-chunk contributions in Python — and float addition is not
    associative. Anything larger is a real disagreement."""
    l_ref, g_ref = _monolithic(fixture, KVER)
    res = baseline

    assert isinstance(res, ChunkedGrad)
    g = float(res.grad.k_ver)
    rel_l = abs(res.loss - l_ref) / max(abs(l_ref), 1e-300)
    rel_g = abs(g - g_ref) / max(abs(g_ref), 1e-300)
    print(f"\nmonolithic loss={l_ref:.17e}  dL/dk_ver={g_ref:+.17e}")
    print(f"chunked    loss={res.loss:.17e}  dL/dk_ver={g:+.17e}")
    print(f"relative: loss {rel_l:.3e}   grad {rel_g:.3e}")

    assert g_ref != 0.0, "the reference gradient is zero — the probe says nothing"
    assert rel_l == 0.0, f"the forward differs: {rel_l:.3e} (chunking must not touch it)"
    assert rel_g < 1e-12, f"chunked gradient differs from monolithic by {rel_g:.3e}"


# ---------------------------------------------------------------------------
# 2. invariance to the chunk length
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_steps", [N_STEPS // 2, K])
def test_grad_is_independent_of_chunk_length(fixture, single_chunk, chunk_steps, capsys):
    """2 and 4 chunks against 1 over the same window. The chunk length is a memory knob; if it
    moves the answer, the reverse walk is carrying the wrong state across a boundary."""
    ref = single_chunk
    res = _chunked(fixture, KVER, chunk_steps=chunk_steps)
    g_ref, g = float(ref.grad.k_ver), float(res.grad.k_ver)
    rel = abs(g - g_ref) / max(abs(g_ref), 1e-300)
    print(f"\nchunk_steps={chunk_steps} ({res.n_chunks} chunks): dL/dk_ver={g:+.17e}  rel={rel:.3e}")
    assert ref.n_chunks == 1, "the reference must be the un-chunked run"
    assert res.loss == ref.loss
    assert rel < 1e-12, f"{res.n_chunks} chunks disagrees with 1 chunk by {rel:.3e}"


# ---------------------------------------------------------------------------
# 3. invariance to the boundary stride
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stride", [2, 3, 4])
def test_boundary_stride_changes_nothing(fixture, baseline, stride, capsys):
    """Striding drops host boundaries and recomputes the spans between. Same states, same
    gradient. The project's measured agreement across strides 1/2/3 is 3e-5 on CORE2 over a year;
    on 24 pi steps there is no room for that much drift and the bar is round-off."""
    ref = baseline
    res = _chunked(fixture, KVER, stride=stride)
    g_ref, g = float(ref.grad.k_ver), float(res.grad.k_ver)
    rel = abs(g - g_ref) / max(abs(g_ref), 1e-300)
    print(f"\nstride={stride}: dL/dk_ver={g:+.17e}  rel={rel:.3e}  host {res.host_gb*1e3:.2f} MB "
          f"vs {ref.host_gb*1e3:.2f} MB")
    assert rel < 1e-12, f"boundary_stride={stride} disagrees with 1 by {rel:.3e}"
    assert res.boundary_stride == stride
    assert res.host_gb < ref.host_gb, \
        "a stride > 1 must park FEWER boundaries on the host; that is the only reason it exists"


# ---------------------------------------------------------------------------
# 4. the trace is what the amplification screen expects
# ---------------------------------------------------------------------------
def test_trace_feeds_the_amplification_screen(baseline, capsys):
    """The screen and the sweep must agree on what a trace is, or the screen silently reads a
    per-step rate off the wrong number of chunks. One boundary per chunk plus the seed."""
    res = baseline
    assert len(res.trace) == res.n_chunks + 1, \
        f"trace has {len(res.trace)} entries for {res.n_chunks} chunks; expected n_chunks + 1"
    assert all(np.isfinite(res.trace)), res.trace
    assert res.trace[0] > 0.0, "the seed cotangent has zero norm — the loss does not depend on T"

    amp = adjoint_stab.amplification(res.trace, res.chunk_steps)
    print(f"\ntrace={['%.4e' % t for t in res.trace]}")
    print(f"amplification: median={amp.median:.6f}/step  log_spread={amp.log_spread:.4f}  "
          f"worst_chunk={amp.worst_chunk}")
    assert np.isfinite(amp.median) and amp.median > 0.0
    assert len(amp.per_step) == res.n_chunks


# ---------------------------------------------------------------------------
# 5. resume: the protocol the library calls but does not implement
# ---------------------------------------------------------------------------
class _FakeCheckpoint:
    """In-memory :class:`~fesom_jax.adjoint.ReverseCheckpoint`, stopping after ``stop_after``
    reverse chunks. Enough to prove the resumed sweep is the same sweep — which is otherwise only
    testable by a ten-hour GPU job."""

    def __init__(self, stop_after=None):
        self.stop_after, self.done = stop_after, 0
        self.boundaries, self.have_boundaries, self.loss = None, False, 0.0
        self.saved = None            # (cot, grad, next_chunk, trace)

    def forward_resume(self, like):
        return None, 0

    def put_boundary(self, c, state):
        return False

    def load_boundaries(self, like):
        return self.boundaries

    def finish_forward(self, loss):
        self.loss = loss

    def reverse_start(self, seed_cot, n_chunks, like):
        if self.saved is None:
            return seed_cot, None, n_chunks - 1, []
        cot, grad, nxt, trace = self.saved
        return cot, grad, nxt, list(trace)

    def tick(self, next_chunk, cot, grad, trace):
        self.done += 1
        self.saved = (cot, grad, next_chunk, list(trace))
        return self.stop_after is not None and self.done >= self.stop_after


def test_resumed_sweep_equals_a_single_pass(fixture, baseline, capsys):
    """Stop the reverse walk mid-window, resume, and land on the same gradient.

    The forward is re-run on the resumed leg here (``have_boundaries`` stays False) because that
    is the weaker assumption: if the two legs' forwards disagreed, the gradient would too, and the
    test would catch it. A real checkpointer stores the boundaries and skips it."""
    ref = baseline

    ck = _FakeCheckpoint(stop_after=2)
    with pytest.raises(LegIncomplete) as exc:
        _chunked(fixture, KVER, checkpoint=ck)
    assert exc.value.phase == "reverse"
    stopped_at = exc.value.chunk

    ck.stop_after = None                       # second leg runs to the end
    res = _chunked(fixture, KVER, checkpoint=ck)
    g_ref, g = float(ref.grad.k_ver), float(res.grad.k_ver)
    rel = abs(g - g_ref) / max(abs(g_ref), 1e-300)
    print(f"\nleg 1 stopped with chunk {stopped_at} next; leg 2 finished")
    print(f"single pass dL/dk_ver={g_ref:+.17e}   resumed={g:+.17e}   rel={rel:.3e}")
    assert rel < 1e-12, f"the resumed sweep disagrees with a single pass by {rel:.3e}"
    assert len(res.trace) == ref.n_chunks + 1


# ---------------------------------------------------------------------------
# 6. scan_steps itself
# ---------------------------------------------------------------------------
def test_rematerialize_is_forward_transparent(fixture, capsys):
    """``jax.checkpoint`` trades compute for reverse-pass memory and is the identity forward.

    Asserted because ``rematerialize`` is a public parameter and half the call sites this module
    replaces pass ``False``: a run that changed its answer depending on it would be a trap of
    exactly the kind the module was written to remove."""
    mesh, op, stress, st0 = fixture
    _, _, _, _, xs_all = _harness(fixture, K)
    p = _params(1e-4)

    def run(remat):
        return scan_steps(lambda s, x: _advance(s, x, mesh, op, stress, p),
                          st0, xs_all, rematerialize=remat)

    a, b = run(True), run(False)
    for name in ("T", "S", "eta_n"):
        xa, xb = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert np.array_equal(xa, xb), f"{name} differs with rematerialize on/off"

    def loss(k, remat):
        pp = _params(k)
        fin = scan_steps(lambda s, x: _advance(s, x, mesh, op, stress, pp),
                         st0, xs_all, rematerialize=remat)
        return _mean_sst(fin, mesh)

    k0 = jnp.asarray(1e-4, jnp.float64)
    g_on = float(jax.grad(lambda k: loss(k, True))(k0))
    g_off = float(jax.grad(lambda k: loss(k, False))(k0))
    rel = abs(g_on - g_off) / max(abs(g_off), 1e-300)
    print(f"\ngrad remat=True {g_on:+.17e}  remat=False {g_off:+.17e}  rel={rel:.3e}")
    assert rel < 1e-12


def test_scan_steps_requires_a_length_without_xs():
    with pytest.raises(ValueError, match="length"):
        scan_steps(lambda s, x: s, 0.0, None)


@pytest.mark.parametrize("n", [6, 48, 50])
def test_scan_steps_builds_the_same_graph_as_the_call_it_replaces(fixture, n, capsys):
    """`scan_steps` must lower to what `_run_steps(checkpoint=True, segments=√N)` lowered to.

    This is the real equivalence gate for the extraction, and it is exact where a numerical
    comparison cannot be. Every migrated call site previously wrote

        _run_steps(body, state, xs=sfs, checkpoint=True, segments=max(2, round(√len(sfs))))

    with `body(carry, x) -> (carry, None)`; it now writes `scan_steps(one_step, state, sfs)` with
    `one_step(state, x) -> state`. If those produce the same jaxpr, no run can differ for any
    reason attributable to this change, at any window length, on any backend.

    ⚠️ Written because the CORE2 A100 comparison could NOT settle it. Two processes recompile,
    XLA picks kernels by timing them, and scatter-adds on an unstructured mesh are not
    deterministic: the same unchanged code measured 2.9e-5 and 1.0e-4 apart across processes
    (job 26844357), with the two versions' values INTERLEAVING. A difference that small cannot be
    attributed by running more samples of a noisy thing; it has to be removed by comparing the
    computation instead of its output.

    ⚠️ The lengths matter and are not decoration. `_run_steps` splits the scan into S segments of
    M = length // S and then runs `length - S*M` steps as a SEPARATE per-step-checkpointed tail —
    a branch that only exists when S does not divide the length. 6 (S=2, M=3, no tail) takes the
    first path; **48** is the real chunk length this project runs (S=7, M=6, tail of 6); 50 is
    S=7, M=7, tail of 1, the narrowest tail there is. A check at 6 alone would have proven the
    equivalence only for the branch that is never used in production."""
    mesh, op, stress, st0 = fixture
    p = _params(1e-4)
    xs = jnp.asarray(1.0 + 0.05 * np.cos(np.arange(n)), jnp.float64)

    def one_step(s, x):
        return _advance(s, x, mesh, op, stress, p)

    def old_body(carry, x):                       # the (carry, x) -> (carry, None) contract
        return one_step(carry, x), None

    segments = max(2, int(round(np.sqrt(n))))
    tail = n - segments * (n // segments)
    print(f"\nn={n}  segments={segments}  M={n // segments}  tail={tail}")
    old = jax.make_jaxpr(lambda s, x: _run_steps(old_body, s, xs=x, checkpoint=True,
                                                 segments=segments))(st0, xs)
    new = jax.make_jaxpr(lambda s, x: scan_steps(one_step, s, x, rematerialize=True))(st0, xs)

    so, sn = str(old), str(new)
    print(f"old jaxpr {len(so)} chars, new {len(sn)} chars")
    assert so == sn, (
        "scan_steps does not lower to what _run_steps(checkpoint=True, segments=√N) lowered to; "
        "the migrated call sites are NOT numerically equivalent to what they replaced")


def test_the_jitted_chunk_is_reused_across_calls():
    """Keyed on the chunk_fn OBJECT, so a repeat or an FD sweep does not recompile.

    ``jax.jit`` hands back a fresh wrapper with a fresh compilation cache each time it is called,
    so jitting inside ``chunked_grad`` would recompile the chunk — minutes on CORE2 — on every
    call. The reference implementation defined its ``fwd_chunk``/``bwd_chunk`` once outside its
    repeat loop; this asserts the library kept that property, because losing it would show up as
    the extraction having made the adjoint slower rather than as a bug."""
    from fesom_jax import adjoint as A

    def f(state, xs, params):
        return state

    def g(state, xs, params):
        return state

    assert A._jitted_forward(f) is A._jitted_forward(f)
    assert A._jitted_backward(f) is A._jitted_backward(f)
    assert A._jitted_forward(f) is not A._jitted_forward(g)


def test_progress_hooks_fire_once_per_chunk(fixture, capsys):
    """A reverse sweep runs for hours; the hooks are how anything is heard before it ends."""
    seen, fwd = [], []
    res = _chunked(fixture, KVER,
                   on_forward=lambda loss, gb, s: fwd.append((loss, gb, s)),
                   on_chunk=lambda c, n, g: seen.append((c, n, float(g.k_ver))))
    print(f"\non_forward {len(fwd)} call(s); on_chunk chunks {[c for c, _, _ in seen]}")
    assert len(fwd) == 1 and fwd[0][0] == res.loss
    assert [c for c, _, _ in seen] == list(reversed(range(res.n_chunks))), \
        "on_chunk must fire once per chunk, walking backwards"
    assert [n for _, n, _ in seen] == res.trace[1:], "the hook and the trace disagree"
    assert seen[-1][2] == float(res.grad.k_ver), \
        "the last on_chunk must carry the finished gradient, not a partial sum"


def test_chunked_forward_matches_the_gradient_run(fixture, baseline, capsys):
    """The forward-only path must be the SAME forward, or a finite-difference check built on it
    compares two different models and reports a disagreement that is not there.

    That is the use it exists for: ``adjoint_horizon.py``'s FD check evaluates the loss at
    ``m(1±h)`` and must not store boundaries or write to a checkpoint while doing it."""
    mesh, op, stress, st0 = fixture
    chunk_fn, xs_fn, loss_fn, n_chunks, _ = _harness(fixture, K)
    fin = chunked_forward(chunk_fn, _params(KVER), init_state=st0, n_chunks=n_chunks,
                          xs_fn=xs_fn)
    loss = float(loss_fn(fin))
    print(f"\nchunked_forward loss={loss:.17e}   chunked_grad loss={baseline.loss:.17e}")
    assert loss == baseline.loss, "the forward-only path is not the same forward"


# ---------------------------------------------------------------------------
# 7. the xs_fn contract, which this file broke on its first run
# ---------------------------------------------------------------------------
def test_xs_fn_must_build_fresh_and_says_so(fixture, capsys):
    """``chunked_grad`` frees what ``xs_fn`` returns, so ``xs_fn`` must build it.

    Found by writing this file's own ``xs_fn`` as ``xs_all[c*K:(c+1)*K]`` on a DEVICE array. For
    two or more chunks that is a genuine sub-slice and everything passed; for a ONE-chunk window
    ``a[0:len(a)]`` is ``a`` itself, so the library deleted the caller's cached array and the
    reverse walk failed on the next read. The trap is that it only appears at one chunk — the
    configuration a user reaches for first when trying the function out.

    JAX's own error (``Array has been deleted``) is raised from inside the caller's slicing
    expression and says nothing about who deleted it. This asserts the diagnosis, and that
    ``free_xs=False`` is a real escape hatch."""
    st0 = fixture[3]

    def aliasing_call(**extra):
        """A caller that slices a cached DEVICE array. Fresh `xs_all` each time, because the
        violating call frees it on its way to raising — by then the damage is already done, which
        is itself the reason the contract has to be stated rather than merely survivable."""
        chunk_fn, _, loss_fn, _, xs_all = _harness(fixture, N_STEPS)

        def xs_fn(c):
            return xs_all[c * N_STEPS:(c + 1) * N_STEPS]   # IS xs_all when there is one chunk

        assert xs_fn(0) is xs_all, \
            "this test's premise — that a full-width JAX slice aliases — no longer holds"
        return chunked_grad(chunk_fn, _params(KVER), init_state=st0, n_chunks=1, xs_fn=xs_fn,
                            loss_fn=loss_fn, chunk_steps=N_STEPS, **extra)

    with pytest.raises(RuntimeError, match="must BUILD each chunk's inputs fresh"):
        aliasing_call()
    print("\ncontract violation diagnosed rather than surfaced as 'Array has been deleted'")

    # the escape hatch: same caller, nothing freed, and the answer is the ordinary one
    res = aliasing_call(free_xs=False)
    assert np.isfinite(float(res.grad.k_ver))
    assert res.n_chunks == 1


# ---------------------------------------------------------------------------
# 8. the linearity guard
# ---------------------------------------------------------------------------
# A chunk VJP must be a linear map of the cotangent. The guard evaluates it on the first reverse
# chunk: "zero" asserts vjp(0) == 0 exactly, "full" also asserts vjp(2c) == 2·vjp(c) within a
# measured repeat spread. The planted defects are custom derivative rules that break linearity
# in the two ways the modes are built to see: adding a constant (caught by "zero") and squaring
# (homogeneous at zero, so only "full" sees it). They sit on a tiny synthetic model: the guard's
# algebra does not depend on the ocean, and the pi chunk runs separately to show a real model
# passes.

_W_SYN = np.random.default_rng(7).normal(size=(12, 12)) / 4.0


_NOISE_CALLS = [0]


def _call_counter():
    """Host side of the ``"noisy"`` plant: a new value on every call of the compiled VJP."""
    _NOISE_CALLS[0] += 1
    return np.float64(_NOISE_CALLS[0])


def _planted(kind):
    """Identity forward; the backward breaks linearity (``"add"``: g + 1, ``"square"``: g²),
    or (``"noisy"``) stays linear in ``g`` but changes from call to call, ``g·(1 + 1e-3·k)``
    with ``k`` a host call counter — a stand-in for a GPU chunk VJP whose rematerialised
    forward is not reproducible."""
    @jax.custom_vjp
    def f(x):
        return x

    def fwd(x):
        return x, None

    def bwd(_, g):
        if kind == "noisy":
            from jax.experimental import io_callback
            k = io_callback(_call_counter, jax.ShapeDtypeStruct((), jnp.float64))
            return (g * (1.0 + 1e-3 * k),)
        return ((g + 1.0) if kind == "add" else (g * g),)

    f.defvjp(fwd, bwd)
    return f


_SYN_FN: dict = {}


def _syn_chunk_fn(kind=None):
    """``(state, xs, params) -> state`` of a small nonlinear map, memoised per ``kind`` so the
    compiled chunk is reused (``chunked_grad`` caches on the object). ``state`` carries an
    integer counter so its cotangent has a float0 leaf; ``params`` may carry one too."""
    if kind not in _SYN_FN:
        plant = _planted(kind) if kind else (lambda x: x)
        W = jnp.asarray(_W_SYN)

        def chunk_fn(state, xs, params):
            def one(st, x):
                return {"x": jnp.tanh(params["a"] * (W @ st["x"]) + x), "n": st["n"] + 1}
            out = scan_steps(one, state, xs)
            return {"x": plant(out["x"]), "n": out["n"]}
        _SYN_FN[kind] = chunk_fn
    return _SYN_FN[kind]


def _syn(kind=None, check_linear=True, n_chunks=3, **kw):
    s0 = {"x": jnp.asarray(np.linspace(-1.0, 1.0, 12)), "n": jnp.asarray(0, jnp.int32)}
    params = {"a": jnp.asarray(0.9, jnp.float64), "k": jnp.asarray(2, jnp.int32)}
    return chunked_grad(_syn_chunk_fn(kind), params, init_state=s0, n_chunks=n_chunks,
                        xs_fn=lambda c: jnp.asarray(np.full(4, 0.1 * (c + 1))),
                        loss_fn=lambda s: jnp.sum(s["x"] ** 3), chunk_steps=4,
                        check_linear=check_linear, **kw)


@pytest.mark.parametrize("mode", [True, "zero", "full"])
def test_linearity_guard_passes_on_the_pi_chunk(fixture, baseline, mode, capsys):
    """A real model passes both modes, on CPU exactly (spread 0 ⇒ the 1e-9 floor), and the
    guard changes no gradient bit. ``True`` is ``"zero"``."""
    res = baseline if mode is True else _chunked(fixture, KVER, check_linear=mode)
    info = res.linear_check
    print(f"\ncheck_linear={mode!r}: {info}")
    assert info is not None and info["mode"] == ("zero" if mode is True else mode)
    assert info["chunk"] == res.n_chunks - 1, "the guard must run on the first reverse chunk"
    assert info["zero_max"] == 0.0
    if mode == "full":
        assert info["spread"] == {"state": 0.0, "params": 0.0}, "CPU repeats are bitwise"
        assert info["err"]["state"] <= 1e-9 and info["err"]["params"] <= 1e-9
        assert info["informative"] is True and info["informative_bar"] == 1e-6
    else:
        assert "informative" not in info
    off = _chunked(fixture, KVER, check_linear=False)
    assert off.linear_check is None
    assert float(off.grad.k_ver) == float(res.grad.k_ver)
    assert off.trace == res.trace


def test_linearity_guard_zero_catches_an_added_constant(capsys):
    with pytest.raises(RuntimeError, match="ZERO cotangent") as exc:
        _syn("add", check_linear="zero")
    print(f"\n{exc.value}")
    assert "miscompiled" in str(exc.value) and "non-determinism" in str(exc.value)
    _syn("add", check_linear=False)          # the guard, not the sweep, raised


def test_linearity_guard_full_catches_a_squared_cotangent(capsys):
    """Squaring maps 0 to 0, so ``"zero"`` passes; ``"full"`` sees vjp(2c) = 4·(…) ≠ 2·vjp(c)."""
    res = _syn("square", check_linear="zero")
    assert res.linear_check["zero_max"] == 0.0
    with pytest.raises(RuntimeError, match=r"vjp\(2c\)") as exc:
        _syn("square", check_linear="full")
    print(f"\n{exc.value}")


def test_linearity_guard_full_says_when_it_proves_nothing(capsys):
    """A chunk VJP that is linear but not reproducible (relative repeat spread ~2e-3, far above
    INFORMATIVE_SPREAD) passes "full" — the bound 10·s admits it — so the result must say the
    check was not informative, and a RuntimeWarning must say it at run time. The reproducible
    synthetic chunk is informative and warns about nothing (negative control)."""
    import warnings
    from fesom_jax.adjoint import INFORMATIVE_SPREAD
    with pytest.warns(RuntimeWarning, match="NOT informative") as rec:
        res = _syn("noisy", check_linear="full")
    info = res.linear_check
    print(f"\nnoisy: {info}\n{rec[0].message}")
    assert info["informative"] is False and info["informative_bar"] == INFORMATIVE_SPREAD
    assert info["spread"]["state"] > INFORMATIVE_SPREAD, "the plant must make repeats differ"
    assert info["err"]["state"] <= 10.0 * info["spread"]["state"], "and still pass the bound"
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # any warning from the clean chunk fails the test
        clean = _syn(None, check_linear="full").linear_check
    print(f"clean: {clean}")
    assert clean["informative"] is True
    assert clean["spread"] == {"state": 0.0, "params": 0.0}


def test_linearity_guard_float0_leaves_and_options(capsys):
    """Integer leaves in the state and in params (float0 cotangents) do not break the guard or
    the accumulation over several chunks; ``False`` skips; a bad value raises before any work."""
    for mode in ("zero", "full"):
        res = _syn(None, check_linear=mode)
        print(f"\n{mode}: grad a={float(res.grad['a']):+.6e}  k dtype={res.grad['k'].dtype}  "
              f"{res.linear_check}")
        assert res.grad["k"].dtype == jax.dtypes.float0
        assert res.linear_check["zero_max"] == 0.0 and res.linear_check["chunk"] == 2
    assert _syn(None, check_linear=False).linear_check is None
    with pytest.raises(ValueError, match="check_linear"):
        _syn(None, check_linear="yes")


def test_linearity_guard_runs_once_per_leg(fixture, baseline, monkeypatch, capsys):
    """A resumed sweep checks once in each leg, on that leg's first chunk."""
    from fesom_jax import adjoint as A
    calls = []
    real = A._check_linear

    def counting(*a, **k):
        calls.append(a[-1])                   # the chunk index
        return real(*a, **k)

    monkeypatch.setattr(A, "_check_linear", counting)
    ck = _FakeCheckpoint(stop_after=2)
    with pytest.raises(LegIncomplete):
        _chunked(fixture, KVER, checkpoint=ck)
    first_leg = list(calls)
    ck.stop_after = None
    res = _chunked(fixture, KVER, checkpoint=ck)
    print(f"\nguard chunks: leg 1 {first_leg}, leg 2 {calls[len(first_leg):]}")
    assert first_leg == [res.n_chunks - 1], first_leg
    assert calls[len(first_leg):] == [res.n_chunks - 3], calls
    assert res.linear_check["chunk"] == res.n_chunks - 3
    g, g_ref = float(res.grad.k_ver), float(baseline.grad.k_ver)
    assert abs(g - g_ref) <= 1e-12 * abs(g_ref)


# ---------------------------------------------------------------------------
# 9. per-field traces, the cotangent hook and the initial-state cotangent
# ---------------------------------------------------------------------------
def test_field_trace_decomposes_the_trace(baseline, capsys):
    """One ``{leaf: norm}`` per boundary, aligned with ``trace``; the squares add up to it."""
    res = baseline
    assert res.field_trace is not None and len(res.field_trace) == len(res.trace)
    for i, (tot, fields) in enumerate(zip(res.trace, res.field_trace)):
        s2 = sum(v * v for v in fields.values())
        assert abs(s2 - tot * tot) <= 1e-12 * tot * tot, (i, s2, tot * tot)
    names = list(res.field_trace[0])
    assert "T" in names and "uv" in names and all("." not in n for n in names)
    amp = adjoint_stab.amplification_by_field(res.field_trace, res.chunk_steps)
    assert set(amp) == set(names)
    top = sorted(res.field_trace[-1].items(), key=lambda kv: -kv[1])[:4]
    print(f"\nfields {len(names)}; largest at the initial state: "
          + ", ".join(f"{k}={v:.3e}" for k, v in top))
    print("median per-step rate: " + ", ".join(
        f"{k}={amp[k].median:.4f}" for k, _ in top))
    assert np.isfinite(amp["T"].median)


def test_on_cot_hook_and_state_cot(fixture, baseline, single_chunk, capsys):
    """The hook sees every boundary once, seed first, walking back to the initial state; it
    changes nothing; ``state_cot`` is d loss / d initial state of the monolithic window."""
    mesh, op, stress, st0 = fixture
    seen = []

    def hook(c, cot):
        seen.append((c, float(adjoint_stab.cotangent_norm(cot))))

    res = _chunked(fixture, KVER, on_cot=hook, field_trace=False)
    print(f"\non_cot boundaries {[c for c, _ in seen]}")
    assert [c for c, _ in seen] == list(range(res.n_chunks, -1, -1))
    assert [n for _, n in seen] == res.trace
    assert res.field_trace is None
    # the hook and the traces change no bit
    assert float(res.grad.k_ver) == float(baseline.grad.k_ver)
    assert res.trace == baseline.trace
    for a, b in zip(jax.tree.leaves(res.state_cot), jax.tree.leaves(baseline.state_cot)):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    # state_cot == d loss / d s0 of the monolithic window. Exact for ONE chunk (the same
    # segmentation, so the same program up to the host round trip). For four chunks the reverse
    # program re-runs each chunk's forward in a different fusion, and the early-stopped SSH
    # solve turns that round-off into a ~1e-8 difference in the T cotangent (job 27735668:
    # 8.4e-9 relative) — the same size as rematerialisation on/off moves the monolithic
    # gradient (8.7e-9). So four chunks are held to that measured spread, not to 1e-12.
    _, _, loss_fn, _, xs_all = _harness(fixture, K)
    p = _params(KVER)

    def loss(s, remat=True):
        return loss_fn(scan_steps(lambda c, x: _advance(c, x, mesh, op, stress, p),
                                  s, xs_all, rematerialize=remat))

    ref = jax.jit(jax.grad(loss))(st0)
    ref_nr = jax.jit(jax.grad(lambda s: loss(s, False)))(st0)

    def rel(a, b):
        la = [np.asarray(x) for x in jax.tree.leaves(a)]
        lb = [np.asarray(x) for x in jax.tree.leaves(b)]
        num = sum(float(np.sum((x - y) ** 2)) for x, y in zip(la, lb))
        den = sum(float(np.sum(y ** 2)) for y in lb)
        assert den > 0.0
        return (num / den) ** 0.5

    r1, r4, spread = rel(single_chunk.state_cot, ref), rel(res.state_cot, ref), rel(ref, ref_nr)
    print(f"state_cot vs monolithic d loss / d s0: 1 chunk {r1:.3e}, {res.n_chunks} chunks "
          f"{r4:.3e}; monolithic remat vs no remat {spread:.3e}")
    assert r1 <= 1e-12, r1
    assert r4 <= max(10.0 * spread, 1e-12), (r4, spread)


def test_field_trace_and_hook_across_a_resume(fixture, baseline, capsys):
    """A resumed leg pads the boundaries of the earlier leg with ``None`` so ``field_trace``
    stays aligned with ``trace``, and its hook reports only its own boundaries."""
    ck = _FakeCheckpoint(stop_after=2)
    seen1, seen2 = [], []
    with pytest.raises(LegIncomplete):
        _chunked(fixture, KVER, checkpoint=ck, on_cot=lambda c, _: seen1.append(c))
    ck.stop_after = None
    res = _chunked(fixture, KVER, checkpoint=ck, on_cot=lambda c, _: seen2.append(c))
    print(f"\nleg 1 hook {seen1}, leg 2 hook {seen2}")
    assert seen1 + seen2 == list(range(res.n_chunks, -1, -1))
    assert len(res.field_trace) == len(res.trace)
    n_prev = len(seen1)
    assert all(f is None for f in res.field_trace[:n_prev])
    for got, ref in zip(res.field_trace[n_prev:], baseline.field_trace[n_prev:]):
        for k in ref:
            assert abs(got[k] - ref[k]) <= 1e-12 * max(ref[k], 1e-300)
    amp = adjoint_stab.amplification_by_field(res.field_trace, res.chunk_steps)
    assert "T" in amp


# ---------------------------------------------------------------------------
# 10. several costs in one sweep
# ---------------------------------------------------------------------------
def _loss_fns(mesh):
    """Three costs of the final state: mean SST, mean surface salinity, mean surface height."""
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])

    def sss(s):
        return jnp.sum(jnp.where(wet0, s.S[:, 0], 0.0)) / jnp.sum(wet0)

    def ssh_(s):
        return jnp.sum(jnp.where(wet0, s.eta_n, 0.0)) / jnp.sum(wet0)
    return [lambda s: _mean_sst(s, mesh), sss, ssh_]


def _multi(fixture, stride=1, **kw):
    from fesom_jax.adjoint import chunked_grad_multi
    mesh, _, _, st0 = fixture
    chunk_fn, xs_fn, _, n_chunks, _ = _harness(fixture, K)
    return chunked_grad_multi(chunk_fn, _params(KVER), init_state=st0, n_chunks=n_chunks,
                              xs_fn=xs_fn, loss_fns=_loss_fns(mesh), chunk_steps=K,
                              boundary_stride=stride, **kw)


def _same_tree(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y))
                                      for x, y in zip(la, lb))


@pytest.fixture(scope="module")
def multi_run(fixture):
    from fesom_jax import adjoint as A
    calls, order = [], []
    real = A._check_linear

    def counting(*a, **k):
        calls.append(a[-1])
        return real(*a, **k)

    A._check_linear = counting
    try:
        res = _multi(fixture, on_cot=lambda k, c, cot: order.append((k, c)))
    finally:
        A._check_linear = real
    return res, calls, order


def test_multi_cost_equals_single_cost_sweeps(fixture, multi_run, capsys):
    """K = 3 costs in one sweep == three chunked_grad runs, bit for bit (CPU), through the host
    round trips of the parked cotangents and gradients."""
    mesh, _, _, st0 = fixture
    res, _, _ = multi_run
    chunk_fn, xs_fn, _, n_chunks, _ = _harness(fixture, K)
    for k, lf in enumerate(_loss_fns(mesh)):
        one = chunked_grad(chunk_fn, _params(KVER), init_state=st0, n_chunks=n_chunks,
                           xs_fn=xs_fn, loss_fn=lf, chunk_steps=K)
        print(f"\ncost {k}: loss {res.losses[k]:+.6e}  dL/dk_ver multi "
              f"{float(res.grads[k].k_ver):+.17e} single {float(one.grad.k_ver):+.17e}")
        assert res.losses[k] == one.loss
        assert _same_tree(res.grads[k], one.grad), f"cost {k}: params gradient differs"
        assert res.traces[k] == one.trace, f"cost {k}: trace differs"
        assert res.field_traces[k] == one.field_trace, f"cost {k}: field trace differs"
        assert _same_tree(res.state_cots[k], one.state_cot), f"cost {k}: state cotangent differs"
        assert float(one.grad.k_ver) != 0.0
    assert len({float(g.k_ver) for g in res.grads}) == 3, "the three costs must differ"


def test_multi_cost_stride_and_guard(fixture, multi_run, capsys):
    """Strides 1 and 2 agree; the linearity guard runs once (first chunk, cost 0)."""
    res, calls, _ = multi_run
    res2 = _multi(fixture, stride=2)
    for k in range(3):
        g1, g2 = float(res.grads[k].k_ver), float(res2.grads[k].k_ver)
        assert abs(g1 - g2) <= 1e-12 * abs(g1), (k, g1, g2)
    print(f"\nguard ran on chunks {calls}; linear_check {res.linear_check}")
    assert calls == [res.n_chunks - 1]
    assert res.linear_check["chunk"] == res.n_chunks - 1


def test_multi_cost_hook_order(multi_run, capsys):
    """The contract track D depends on: once per cost per boundary, boundaries from the seed
    (c = n_chunks) down to 0, costs ascending within a boundary."""
    res, _, order = multi_run
    expect = [(k, c) for c in range(res.n_chunks, -1, -1) for k in range(3)]
    print(f"\non_cot calls {len(order)}: {order[:6]} ...")
    assert len(order) == 3 * (res.n_chunks + 1)
    assert order == expect


def test_multi_cost_rejects_a_checkpoint_and_handles_float0(capsys):
    from fesom_jax.adjoint import chunked_grad_multi
    s0 = {"x": jnp.asarray(np.linspace(-1.0, 1.0, 12)), "n": jnp.asarray(0, jnp.int32)}
    params = {"a": jnp.asarray(0.9, jnp.float64), "k": jnp.asarray(2, jnp.int32)}
    kw = dict(init_state=s0, n_chunks=3, xs_fn=lambda c: jnp.asarray(np.full(4, 0.1 * (c + 1))),
              chunk_steps=4)
    with pytest.raises(ValueError, match="ReverseCheckpoint"):
        chunked_grad_multi(_syn_chunk_fn(None), params, loss_fns=[lambda s: s["x"][0]],
                           checkpoint=_FakeCheckpoint(), **kw)
    lfs = [lambda s: jnp.sum(s["x"] ** 3), lambda s: jnp.sum(jnp.sin(s["x"]))]
    res = chunked_grad_multi(_syn_chunk_fn(None), params, loss_fns=lfs,
                             check_linear="full", **kw)
    for k, lf in enumerate(lfs):
        one = chunked_grad(_syn_chunk_fn(None), params, loss_fn=lf, **kw)
        assert float(res.grads[k]["a"]) == float(one.grad["a"])
        assert res.grads[k]["k"].dtype == jax.dtypes.float0
        assert np.array_equal(np.asarray(res.state_cots[k]["x"]), np.asarray(one.state_cot["x"]))
    print(f"\nsynthetic K=2: grads a {[float(g['a']) for g in res.grads]}")
