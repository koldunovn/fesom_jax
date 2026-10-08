"""Checks of a derivative: dot tests, finite-difference sweeps, window maps and their singular values.

Everything a study needs to decide whether a tangent-linear or adjoint result can be trusted,
kept apart from the model so that it can be applied to any map — one step, a chunked window, a
toy matrix in a test.

- :func:`dot_test` — ``<y, M x>`` against ``<Mᵀ y, x>`` at several tangent amplitudes. A
  tangent-linear model that is not linear in its tangent (an early-stopped solve inside it, a
  clip) passes at amplitude 1 and fails at 1e-8, so one amplitude is not a test.
- :func:`fd_sweep` / :func:`plateau` — central differences of a scalar cost over a sweep of
  step sizes against an autodiff value, with the forward noise floor measured from repeats.
- :func:`forward_states`, :func:`window_jvp`, :func:`window_vjp` — the tangent-linear and
  adjoint maps of a CHUNKED window, both linearised about the same stored ("pinned") chunk-entry
  states. A tangent sweep that recomputes its primal trajectory on a GPU does not linearise
  about the trajectory the adjoint used (scatter-add order differs from run to run and the
  difference grows with the window), so a dot test between an unpinned tangent sweep and an
  adjoint sweep measures the two trajectories' divergence, not the transpose.
- :func:`power_iteration` — the leading singular value and vectors of a window map in a
  weighted norm; :func:`volume_weights` builds the usual weights; :func:`fdlin` checks the map
  against finite differences of the full model along a direction.

Norms and maps use the element velocity ``uv``, never ``uvnode``: ``uvnode`` is recomputed from
``uv`` inside every step (the mixing scheme's output), so below the surface the cotangent of the
incoming ``uvnode`` is zero. It is NOT zero at the surface: the forced step's bulk formulae and
the ice-ocean stress read the previous step's surface current ``state.uvnode[:, 0]`` before it
is recomputed, so ``uvnode[:, 0]`` carries a small cotangent (in a one-step VJP of a box-mean
temperature on CORE2: max 2.1e-7 against 2.6e-4 for ``uv``). It is a lagged copy of ``uv``, not
an independent degree of freedom, so a norm on ``uv`` alone stays the right one.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import adjoint
from .adjoint import _JIT_CACHE, _forward, _free, _jitted_forward

__all__ = ["tree_vdot", "dot_test", "DotRow", "fd_sweep", "FDRow", "plateau",
           "forward_states", "window_jvp", "WindowJVP", "NO_CONSTANT_FOLDING", "window_vjp", "WindowVJP",
           "power_iteration", "PowerResult", "volume_weights", "fdlin", "FDLinRow"]


# ---------------------------------------------------------------------------
# pytree arithmetic over floating leaves (float0 / integer leaves carry no tangent)


def _is_real(x) -> bool:
    dt = getattr(x, "dtype", None)
    if dt is None:
        return isinstance(x, float)
    return dt != jax.dtypes.float0 and jnp.issubdtype(dt, jnp.floating)


def tree_vdot(a, b) -> float:
    """``Σ_leaves Σ a·b`` over the floating leaves of two pytrees of the same structure."""
    tot = 0.0
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        if _is_real(x) and _is_real(y):
            tot += float(jnp.vdot(jnp.ravel(jnp.asarray(x)), jnp.ravel(jnp.asarray(y))))
    return tot


def _tree_norm(a) -> float:
    return math.sqrt(max(tree_vdot(a, a), 0.0))


def _scale(a, s):
    return jax.tree.map(lambda x: x * s if _is_real(x) else x, a)


def _tangent_like(primal, tangent=None):
    """A tangent for ``primal``: ``tangent`` (or zeros) on floating leaves, ``float0`` zeros on
    integer leaves (what :func:`jax.jvp` requires there). Runs inside a trace too."""
    def one(p, t):
        dt = jnp.result_type(p)
        if not jnp.issubdtype(dt, jnp.floating):
            return np.zeros(np.shape(p), jax.dtypes.float0)
        if t is None:
            return jnp.zeros(np.shape(p), dt)
        return jnp.asarray(t, dt)
    if tangent is None:
        return jax.tree.map(lambda p: one(p, None), primal)
    return jax.tree.map(one, primal, tangent)


# ---------------------------------------------------------------------------
# dot test


class DotRow(NamedTuple):
    amp: float
    lhs: float        #: ``<y, M(amp·x)>``   (tangent-linear)
    rhs: float        #: ``amp·<Mᵀy, x>``    (adjoint)
    rel: float        #: ``|lhs − rhs| / max(|lhs|, |rhs|)``
    rel_norm: float   #: ``|lhs − rhs| / (‖M(amp·x)‖·‖y‖)``


def dot_test(f, x, v, w, amps=(1.0, 1e-4, 1e-8), *, jvp=None, vjp=None, jit=True) -> list:
    """Adjoint dot test of the linearisation ``M`` of ``y = f(x)``: ``<w, M(a·v)>`` (tangent-
    linear) against ``a·<Mᵀw, v>`` (adjoint) for every amplitude ``a`` in ``amps``.

    :param f: the map (``None`` when ``jvp`` and ``vjp`` are given).
    :param x: the linearisation point (``None`` when ``jvp`` and ``vjp`` are given).
    :param v: tangent direction, a pytree like ``x``.
    :param w: cotangent, a pytree like ``f(x)``.
    :param jvp: optional ``tangent -> M tangent`` (e.g. :func:`window_jvp` on pinned states).
    :param vjp: optional ``cotangent -> Mᵀ cotangent`` (e.g. :func:`window_vjp`).
    :returns: one :class:`DotRow` per amplitude. The adjoint is evaluated ONCE and scaled.

    ``rel`` is the plain relative mismatch; it is large whenever ``<w, Mv>`` is small by
    cancellation although both sides are accurate. ``rel_norm`` divides by ``‖Mv‖·‖w‖`` (the
    Cauchy-Schwarz bound of the product) and is the number to hold to round-off.
    """
    if (jvp is None) != (vjp is None):
        raise ValueError("pass both jvp and vjp, or neither")
    if jvp is None:
        def _jvp_fn(x_, t_):
            return jax.jvp(f, (x_,), (_tangent_like(x_, t_),))[1]

        def _vjp_fn(x_, c_):
            return jax.vjp(f, x_)[1](c_)[0]
        if jit:
            _jvp_fn, _vjp_fn = jax.jit(_jvp_fn), jax.jit(_vjp_fn)
        jvp = functools.partial(_jvp_fn, x)
        vjp = functools.partial(_vjp_fn, x)
    rhs1 = tree_vdot(vjp(w), v)
    wn = _tree_norm(w)
    rows = []
    for a in amps:
        mv = jvp(_scale(v, a))
        lhs = tree_vdot(w, mv)
        rhs = a * rhs1
        d = abs(lhs - rhs)
        rows.append(DotRow(float(a), lhs, rhs, d / max(abs(lhs), abs(rhs), 1e-300),
                           d / max(_tree_norm(mv) * wn, 1e-300)))
    return rows


# ---------------------------------------------------------------------------
# finite differences of a scalar cost


class FDRow(NamedTuple):
    h: float
    fd: float           #: ``(J(x + h d) − J(x − h d)) / 2h`` (mean over repeats)
    ad: float           #: the autodiff value it is compared with, ``<dJ/dx, d>``
    rel_err: float      #: ``|fd − ad| / |ad|``
    noise_over_h: float  #: forward noise floor / h: the FD error the noise alone can cause


def fd_sweep(J, x, d, hs, ad, repeats: int = 1) -> list:
    """Central differences of the scalar ``J`` at ``x`` along ``d`` for every ``h`` in ``hs``.

    The forward noise floor is the spread (max − min) of ``repeats`` evaluations of ``J(x)``,
    ``J(x + h d)`` and ``J(x − h d)`` — 0 for a deterministic CPU forward, non-zero on a GPU
    (scatter-add order) or for any cost with evaluation noise. A row whose ``rel_err`` is not
    well above ``noise_over_h / |ad|`` is noise-limited, not evidence against ``ad``.
    """
    reps = max(1, int(repeats))
    j0 = [float(J(x)) for _ in range(reps)]
    base_spread = max(j0) - min(j0)
    rows = []
    for h in hs:
        h = float(h)
        xp = jax.tree.map(lambda a, b: a + h * b if _is_real(a) else a, x, d)
        xm = jax.tree.map(lambda a, b: a - h * b if _is_real(a) else a, x, d)
        jp = [float(J(xp)) for _ in range(reps)]
        jm = [float(J(xm)) for _ in range(reps)]
        fd = (float(np.mean(jp)) - float(np.mean(jm))) / (2.0 * h)
        noise = max(base_spread, max(jp) - min(jp), max(jm) - min(jm))
        rows.append(FDRow(h, fd, float(ad), abs(fd - float(ad)) / max(abs(float(ad)), 1e-300),
                          noise / h))
    return rows


def plateau(rows, rtol: float):
    """The FD plateau: the longest run of consecutive step sizes (sorted by ``h``) whose FD values
    agree with the run's median to ``rtol`` — decided from the FD values alone, so it tells which
    finite difference to believe without assuming the autodiff value is right.

    Returns ``None`` if no two consecutive rows agree, else a dict ``h_min``, ``h_max``, ``n``,
    ``fd`` (median over the run), ``ad``, ``rel_err`` (``|fd − ad| / |ad|`` of the median) and
    ``rows``."""
    rs = sorted(rows, key=lambda r: r.h)
    best = None
    for i in range(len(rs)):
        for j in range(len(rs), i + 1, -1):
            run = rs[i:j]
            med = float(np.median([r.fd for r in run]))
            if all(abs(r.fd - med) <= rtol * max(abs(med), 1e-300) for r in run):
                if best is None or len(run) > len(best):
                    best = run
                break
    if best is None:
        return None
    med = float(np.median([r.fd for r in best]))
    ad = best[0].ad
    return {"h_min": best[0].h, "h_max": best[-1].h, "n": len(best), "fd": med, "ad": ad,
            "rel_err": abs(med - ad) / max(abs(ad), 1e-300), "rows": best}


# ---------------------------------------------------------------------------
# chunked window maps on pinned states


def forward_states(chunk_fn, params, s0, n_chunks: int, xs_fn, *, free_xs: bool = True) -> list:
    """Run the window once and return the chunk-entry states on the HOST: ``n_chunks + 1``
    States, entry ``c`` entering chunk ``c`` and the last one the window end.

    Uses the same jitted chunk forward as :func:`fesom_jax.adjoint.chunked_grad` (cached on the
    ``chunk_fn`` object), so on CPU these are bit-for-bit the boundaries ``chunked_grad`` would
    store itself. Host cost: one State per chunk plus one."""
    free = _free if free_xs else (lambda _t: None)
    _, kept = _forward(_jitted_forward(chunk_fn), params, s0, n_chunks, xs_fn,
                       stride=1, free=free, checkpoint=None)
    return [kept[c] for c in range(n_chunks + 1)]


#: ``compiler_options`` for :func:`window_jvp` that skip XLA's constant-folding pass. On CORE2 the
#: tangent program holds many scatter-adds whose operands are all compile-time constants (zero
#: tangents of mesh-only terms, scattered with the mesh's constant index arrays); XLA folds each
#: on the host at tens of seconds apiece, and the tangent chunk took 761 s to compile on a CPU
#: node against 27 s without the pass (one production step).
NO_CONSTANT_FOLDING = {"xla_disable_hlo_passes": "constant_folding"}


@functools.lru_cache(maxsize=_JIT_CACHE)
def _jitted_tangent(chunk_fn_tl, options: tuple = ()):
    def tl(state, xs, params, t_state, t_params):
        return jax.jvp(lambda s, p: chunk_fn_tl(s, xs, p), (state, params),
                       (_tangent_like(state, t_state), _tangent_like(params, t_params)))
    return jax.jit(tl, compiler_options=dict(options)) if options else jax.jit(tl)


class WindowJVP(NamedTuple):
    tangent: Any        #: ``M·(t0, dparams)``: the state tangent at the window end (HOST)
    state: Any          #: the primal state at the window end (HOST); the pinned one if pinned
    trace: list         #: tangent norm at every chunk boundary, ``n_chunks + 1`` of them
    seconds: float


def window_jvp(chunk_fn_tl, params, s0, t0, dparams, n_chunks: int, xs_fn, *,
               pinned_states=None, free_xs: bool = True,
               compiler_options: dict | None = None) -> WindowJVP:
    """Tangent-linear model of a chunked window: the window-end state tangent caused by an
    initial-state tangent ``t0`` and a parameter tangent ``dparams`` (either may be ``None`` =
    zero; ``dparams`` has the structure of ``params``, and the same parameter tangent acts in
    every chunk).

    :param chunk_fn_tl: ``(state, xs, params) -> state`` for one chunk, built WITHOUT
        :func:`jax.checkpoint` (``scan_steps(..., rematerialize=False)``): forward mode stores no
        tape, and differentiating through remat costs minutes of constant folding on CORE2. It
        must be the same model as the ``chunk_fn`` of :func:`window_vjp`. Its jitted JVP is
        cached on the function OBJECT, like :func:`~fesom_jax.adjoint.chunked_grad`'s programs.
    :param pinned_states: the chunk-entry states from :func:`forward_states` (host list,
        ``≥ n_chunks`` entries). With them, chunk ``c ≥ 1`` linearises about
        ``pinned_states[c]`` instead of the primal the tangent sweep computed itself (chunk 0
        starts from ``s0``), so the tangent and the adjoint (:func:`window_vjp` with the same
        list) linearise the SAME trajectory. Without them the tangent sweep carries its own
        primal; on a GPU that trajectory drifts away from any other run of the window.
    :param xs_fn: ``c -> xs``, the :func:`~fesom_jax.adjoint.chunked_grad` contract (built fresh
        per call; freed after use unless ``free_xs=False``).
    :param compiler_options: XLA options for the tangent program (e.g.
        :data:`NO_CONSTANT_FOLDING`, which turns a CORE2 tangent-chunk compile of minutes into
        seconds). ``None`` (default) compiles exactly as before. A different option set is a
        different program: its primal may differ from the default one's in the last bits.
    """
    import time
    t_start = time.time()
    if pinned_states is not None and len(pinned_states) < n_chunks:
        raise ValueError(f"pinned_states has {len(pinned_states)} entries, need >= {n_chunks}")
    free = _free if free_xs else (lambda _t: None)
    tl = _jitted_tangent(chunk_fn_tl, tuple(sorted((compiler_options or {}).items())))
    s = s0
    t = _tangent_like(s0, t0)
    trace = [_tree_norm(t)]
    for c in range(n_chunks):
        if pinned_states is not None and c >= 1:
            s = jax.device_put(pinned_states[c])
        xs_c = xs_fn(c)
        s_next, t_next = jax.block_until_ready(tl(s, xs_c, params, t, dparams))
        free(xs_c)
        s, t = s_next, t_next
        trace.append(_tree_norm(t))
    return WindowJVP(tangent=jax.device_get(t), state=jax.device_get(s), trace=trace,
                     seconds=time.time() - t_start)


class _Pinned:
    """A :class:`~fesom_jax.adjoint.ReverseCheckpoint` that serves stored boundaries and never
    stops: it makes :func:`~fesom_jax.adjoint.chunked_grad` skip its own forward and walk back
    over the given states."""

    have_boundaries = True

    def __init__(self, states, loss):
        self._b = {c: st for c, st in enumerate(states)}
        self.loss = loss

    def forward_resume(self, like):
        return None, 0

    def put_boundary(self, c, state):
        return False

    def load_boundaries(self, like):
        return self._b

    def finish_forward(self, loss):
        pass

    def reverse_start(self, seed_cot, n_chunks, like):
        return seed_cot, None, n_chunks - 1, []

    def tick(self, next_chunk, cot, grad, trace):
        return False


class WindowVJP(NamedTuple):
    state_cot: Any      #: ``Mᵀ w`` w.r.t. the initial state (HOST)
    params_cot: Any     #: ``Mᵀ w`` w.r.t. ``params``
    result: Any         #: the :class:`~fesom_jax.adjoint.ChunkedGrad` (traces, guard, timing)


def window_vjp(chunk_fn, params, s0, w, n_chunks: int, xs_fn, *, pinned_states=None,
               chunk_steps: int = 1, **kw) -> WindowVJP:
    """Adjoint of a chunked window: ``Mᵀ w`` for a window-end state cotangent ``w``, w.r.t. the
    initial state and ``params``, through :func:`~fesom_jax.adjoint.chunked_grad` with the
    linear loss ``<w, s_end>``.

    With ``pinned_states`` (``n_chunks + 1`` host States from :func:`forward_states`) the sweep
    runs no forward of its own and walks back over exactly those states, so it transposes the
    map :func:`window_jvp` linearises with the same list. Without them ``chunked_grad`` runs its
    own forward (on CPU bit-for-bit the same states). ``chunk_steps`` only labels the result;
    further keywords (``check_linear``, ``on_cot``, ``field_trace``, ``boundary_stride``, …) go
    to ``chunked_grad`` (``checkpoint`` is not allowed with ``pinned_states``).
    """
    w_dev = jax.device_put(w)

    def loss_fn(s):
        return sum(jnp.vdot(jnp.ravel(a), jnp.ravel(b))
                   for a, b in zip(jax.tree.leaves(s), jax.tree.leaves(w_dev))
                   if _is_real(a) and _is_real(b))

    if pinned_states is not None:
        if "checkpoint" in kw:
            raise ValueError("window_vjp: pinned_states and checkpoint are exclusive")
        if len(pinned_states) < n_chunks + 1:
            raise ValueError(f"pinned_states has {len(pinned_states)} entries, need "
                             f"{n_chunks + 1} (every chunk entry and the window end)")
        kw["checkpoint"] = _Pinned(pinned_states[:n_chunks + 1],
                                   tree_vdot(pinned_states[n_chunks], w))
    res = adjoint.chunked_grad(chunk_fn, params, init_state=s0, n_chunks=n_chunks, xs_fn=xs_fn,
                               loss_fn=loss_fn, chunk_steps=chunk_steps, **kw)
    return WindowVJP(state_cot=res.state_cot, params_cot=res.grad, result=res)


# ---------------------------------------------------------------------------
# singular vectors in a weighted norm


def _named(tree):
    """``[(leaf path, leaf)]`` with the names of :func:`fesom_jax.adjoint_stab.field_norms`."""
    return [(jax.tree_util.keystr(p, simple=True, separator="."), x)
            for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _by_name(fn, tree, table, default):
    """Rebuild ``tree`` with ``fn(leaf, table.get(name, default))`` on the floating leaves."""
    leaves = []
    for name, x in _named(tree):
        leaves.append(fn(x, table.get(name, default)) if _is_real(x) else x)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tree), leaves)


def _wnorm2(tree, weights) -> float:
    tot = 0.0
    for name, x in _named(tree):
        wt = weights.get(name)
        if wt is not None and _is_real(x):
            tot += float(jnp.sum(jnp.asarray(wt) * jnp.square(x)))
    return tot


def _apply_w(tree, weights, inverse=False, mask=None):
    """``W·x`` (or ``W⁺·x``, zero where the weight is zero), restricted to ``mask``; leaves
    without a weight become 0."""
    def one(x, wt):
        if wt is None:
            return jnp.zeros_like(x)
        wt = jnp.asarray(wt, x.dtype)
        if inverse:
            wt = jnp.where(wt > 0, 1.0 / jnp.where(wt > 0, wt, 1.0), 0.0)
        return jnp.broadcast_to(wt, jnp.shape(x)) * x
    out = _by_name(one, tree, weights, None)
    if mask is not None:
        out = _by_name(lambda x, m: x if m is None else jnp.where(jnp.asarray(m), x, 0.0),
                       out, mask, None)
    return out


class PowerResult(NamedTuple):
    sigma: float      #: ``‖M v‖_out / ‖v‖_in`` at the last iterate
    v: Any            #: leading input vector (``‖v‖_in = 1``, supported on ``mask``)
    u: Any            #: ``M v / σ`` (``‖u‖_out = 1``)
    history: list     #: per iteration ``{"sigma", "rel_change"}``
    converged: bool


def power_iteration(apply_M, apply_MT, v0, *, weights: dict, mask: dict | None = None,
                    iters: int = 20, tol: float = 1e-3, out_weights: dict | None = None,
                    on_iter: Callable[[int, float], None] | None = None) -> PowerResult:
    """Leading singular value and vectors of a linear map ``M`` in weighted norms.

    ``‖x‖² = Σ_leaf Σ weights[leaf]·x²``, leaves named as in
    :func:`fesom_jax.adjoint_stab.field_norms` (``"T"``, ``"uv"``, ``"tke"``, …); a leaf with no
    entry has weight 0, so ``{"tke": 1.0}`` is a TKE-only norm and :func:`volume_weights` gives a
    volume-weighted prognostic one. ``out_weights`` (default ``weights``) measures the output.
    ``mask`` (``{leaf: bool array broadcastable to the leaf}``) restricts the INPUT, i.e. the
    seed region; leaves missing from it are unrestricted. Inputs live where the input weight is
    positive and the mask is set.

    :param apply_M: ``tangent -> M tangent`` (a State pytree each way, e.g. :func:`window_jvp`).
    :param apply_MT: ``cotangent -> Mᵀ cotangent`` in the EUCLIDEAN inner product (a VJP, e.g.
        :func:`window_vjp`); the weighted adjoint ``W_in⁺ Mᵀ W_out`` is formed here.
    :param v0: the seed (a State pytree), projected onto the admissible inputs and normalised.

    Iteration: ``v ← P W_in⁺ Mᵀ W_out M v / ‖·‖_in``, ``σ = ‖M v‖_out``; stops when ``σ`` changes
    by less than ``tol`` relative between two iterations (after at least two), or after
    ``iters``. Each iteration costs one ``apply_M`` and one ``apply_MT``, plus one ``apply_M``
    at the end for ``u``.
    """
    w_out = weights if out_weights is None else out_weights

    def support(x):
        """``x`` on the admissible inputs (input weight > 0 and mask set), 0 elsewhere."""
        def one(leaf, wt):
            if wt is None:
                return jnp.zeros_like(leaf)
            return jnp.where(jnp.broadcast_to(jnp.asarray(wt) > 0, jnp.shape(leaf)), leaf, 0.0)
        return _apply_w(_by_name(one, x, weights, None), {n: 1.0 for n, _ in _named(x)},
                        mask=mask)

    def normalise(x):
        n = math.sqrt(max(_wnorm2(x, weights), 0.0))
        if n == 0.0 or not math.isfinite(n):
            raise ValueError(f"power_iteration: input vector has norm {n} (seed outside the "
                             f"weighted, masked input region, or a non-finite map)")
        return _scale(x, 1.0 / n)

    v = normalise(support(v0))
    history, sigma, converged = [], None, False
    for it in range(int(iters)):
        mv = apply_M(v)
        s = math.sqrt(max(_wnorm2(mv, w_out), 0.0))
        rel = abs(s - sigma) / max(s, 1e-300) if sigma is not None else float("nan")
        history.append({"sigma": s, "rel_change": rel})
        if on_iter is not None:
            on_iter(it, s)
        if sigma is not None and it >= 2 and rel <= tol:
            sigma, converged = s, True
            break
        sigma = s
        g = apply_MT(_apply_w(mv, w_out))
        v = normalise(_apply_w(g, weights, inverse=True, mask=mask))
    mv = apply_M(v)
    sigma = math.sqrt(max(_wnorm2(mv, w_out), 0.0))
    u = _scale(mv, 1.0 / max(sigma, 1e-300))
    return PowerResult(sigma=sigma, v=v, u=u, history=history, converged=converged)


def volume_weights(state, mesh, *, fields=("T", "S", "uv", "eta_n"), scales=None) -> dict:
    """Norm weights for :func:`power_iteration`: volume-weighted prognostic fields, each divided
    by a scale² and by the wet volume (area for ``eta_n``), so ``‖x‖²`` is the sum of each
    field's squared volume-mean-rms in units of its scale.

    Volumes use the LIVE layer thicknesses of ``state``: nodes ``areasvol[:, k]·hnode[:, k]`` on
    ``node_layer_mask``; elements ``elem_area·helem[:, k]`` on ``elem_layer_mask`` (``uv``, both
    components); ``eta_n`` the surface area ``areasvol[:, 0]`` on wet surface nodes. Default
    scales: T 1 K, S 0.1 psu, uv 0.1 m/s, eta_n 0.1 m (override per field with ``scales``).
    ``uvnode`` is deliberately not offered (a diagnostic copy of ``uv``; its incoming cotangent is
    zero below the surface and small at it, where the forced step reads the previous step's
    surface current)."""
    sc = {"T": 1.0, "S": 0.1, "uv": 0.1, "eta_n": 0.1}
    sc.update(scales or {})
    node_vol = jnp.where(mesh.node_layer_mask, mesh.areasvol * state.hnode, 0.0)
    elem_vol = jnp.where(mesh.elem_layer_mask, mesh.elem_area[:, None] * state.helem, 0.0)
    surf = jnp.where(mesh.node_layer_mask[:, 0], mesh.areasvol[:, 0], 0.0)
    out = {}
    for f in fields:
        if f in ("T", "S"):
            out[f] = node_vol / (jnp.sum(node_vol) * sc[f] ** 2)
        elif f == "uv":
            out[f] = (elem_vol / (jnp.sum(elem_vol) * sc[f] ** 2))[..., None]
        elif f == "eta_n":
            out[f] = surf / (jnp.sum(surf) * sc[f] ** 2)
        else:
            raise ValueError(f"volume_weights: no volume rule for field {f!r} "
                             f"(T, S, uv, eta_n; build other weights by hand)")
    return out


# ---------------------------------------------------------------------------
# finite differences of a window map along a direction


class FDLinRow(NamedTuple):
    h: float
    rel_err: float          #: ``‖FD − TL‖ / ‖TL‖`` in the output norm
    per_field: dict         #: the same per leaf with a non-zero TL
    fd_norm: float
    noise_rel: float        #: ``noise_floor / (h·‖TL‖)``: the error the forward noise alone can cause
    regime: str             #: "linear", "noise-limited" or "beyond-predicted-range"


def fdlin(J_map, x, v, hs, tl, *, weights: dict | None = None, predicted_range=None,
          noise_floor: float | None = None) -> list:
    """Central differences of the full (nonlinear) map ``J_map`` at ``x`` along ``v`` against the
    tangent-linear response ``tl = M v``, for every ``h`` in ``hs``.

    :param weights: output norm as in :func:`power_iteration` (default: Euclidean over every
        floating leaf).
    :param predicted_range: the largest ``h`` for which the TL is expected to hold (e.g. the
        distance to a clip divided by the TL there), or a ``(h_min, h_max)`` pair; rows outside
        are marked "beyond-predicted-range".
    :param noise_floor: the forward noise of ``J_map`` in the output norm (e.g. the norm of the
        difference of two repeats); a row where that noise alone can explain half the measured
        error (``noise_rel ≥ rel_err / 2``) is "noise-limited". ``None`` ⇒ no row is.
    """
    def norm(t):
        if weights is None:
            return _tree_norm(t)
        return math.sqrt(max(_wnorm2(t, weights), 0.0))

    tl_n = norm(tl)
    if tl_n == 0.0:
        raise ValueError("fdlin: the tangent-linear response is zero")
    if predicted_range is None:
        lo, hi = 0.0, math.inf
    elif isinstance(predicted_range, (tuple, list)):
        lo, hi = float(predicted_range[0]), float(predicted_range[1])
    else:
        lo, hi = 0.0, float(predicted_range)
    rows = []
    for h in hs:
        h = float(h)
        fp = J_map(jax.tree.map(lambda a, b: a + h * b if _is_real(a) else a, x, v))
        fm = J_map(jax.tree.map(lambda a, b: a - h * b if _is_real(a) else a, x, v))
        fd = jax.tree.map(lambda a, b: (a - b) / (2.0 * h) if _is_real(a) else a, fp, fm)
        diff = jax.tree.map(lambda a, b: a - b if _is_real(a) else a, fd, tl)
        err = norm(diff) / tl_n
        per = {}
        for (name, d_leaf), (_, t_leaf) in zip(_named(diff), _named(tl)):
            if not _is_real(t_leaf):
                continue
            if weights is not None and name not in weights:
                continue
            tn = float(jnp.linalg.norm(jnp.ravel(t_leaf)))
            if tn > 0:
                per[name] = float(jnp.linalg.norm(jnp.ravel(d_leaf))) / tn
        nrel = (noise_floor / (h * tl_n)) if noise_floor is not None else 0.0
        if noise_floor is not None and nrel >= 0.5 * err:
            regime = "noise-limited"
        elif not (lo <= h <= hi):
            regime = "beyond-predicted-range"
        else:
            regime = "linear"
        rows.append(FDLinRow(h, err, per, norm(fd), nrel, regime))
    return rows
