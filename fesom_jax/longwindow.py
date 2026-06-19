"""Ensemble-averaged adjoint seam (Lea, Allen & Haine 2000) — the long-window method core.

A single long adjoint of a chaotic system blows up exponentially: its tangent-linear /
adjoint grows like ``e^{λt}`` (the leading Lyapunov exponent), so ``d(long-time-mean)/d(param)``
from one long backward pass is meaningless. The CLIMATE sensitivity is instead recovered by
AVERAGING many SHORT adjoint bursts seeded along a reference trajectory: the chaotic part of
each short gradient has ~random sign across the attractor and **cancels** in the ensemble mean,
while the systematic climate response **survives** (this is exactly why the per-burst window
must be long enough to develop the slow response yet short enough to stay below the blow-up —
the Task-C1 crux).

This module is the **generic mechanism** — no physics. The per-burst gradient kernel is
supplied by the caller (Lorenz-63 in the unit test; the frozen-ice FESOM adjoint of
``core2_paper_sensitivity`` in Tasks D1/E1). Three pieces:

* :func:`spread_indices` / :func:`seed_starts` — pick K start states spread along a saved
  trajectory (host-side index map; works for in-memory stacks AND on-disk snapshot files).
* :func:`average_grads` — streaming (Welford) mean + across-burst standard error, leafwise over
  a pytree gradient, with the project masked-NaN contract (a diverged burst's non-finite lanes
  are excluded, the mean stays finite).
* :func:`convergence` — the running-mean-vs-#bursts diagnostic (when does the averaged gradient
  stabilize?), with the burst count at which it settles.

All host-side numpy (the averaging is cheap; the cost is the per-burst adjoints the caller runs).
"""

from __future__ import annotations

import numpy as np
import jax


# ==========================================================================
# Seed selection — K starts spread along a reference trajectory
# ==========================================================================
def spread_indices(n_traj: int, K: int, rng=None) -> np.ndarray:
    """``K`` start indices spread across ``[0, n_traj)``: evenly spaced, optionally jittered
    within each sub-interval by ``rng`` (a ``numpy.random.Generator``). Host-side — usable
    whether the trajectory lives in memory or as index-addressed snapshot files on disk.

    Even spacing (``rng=None``) gives a deterministic, attractor-covering set (bin centres);
    a jittered set decorrelates the seeds (helpful when the snapshot cadence resonates with a
    slow cycle). Always returns ``min(K, n_traj)`` sorted unique-ish indices in range."""
    n_traj, K = int(n_traj), int(K)
    if K <= 0 or n_traj <= 0:
        return np.zeros(0, dtype=np.int64)
    if K >= n_traj:
        return np.arange(n_traj, dtype=np.int64)
    step = n_traj / K
    edges = np.arange(K, dtype=np.float64) * step          # left edge of each sub-interval
    off = (rng.random(K) if rng is not None else np.full(K, 0.5)) * step
    return np.clip(np.floor(edges + off).astype(np.int64), 0, n_traj - 1)


def seed_starts(traj_states, K: int, rng=None):
    """Gather ``K`` start states spread along ``traj_states`` (a python list/tuple of states,
    or a pytree with a leading time axis). Returns a python **list** of K states.

    For on-disk snapshots, call :func:`spread_indices` directly and load by index instead."""
    if isinstance(traj_states, (list, tuple)):
        idx = spread_indices(len(traj_states), K, rng)
        return [traj_states[int(i)] for i in idx]
    n = int(jax.tree.leaves(traj_states)[0].shape[0])
    idx = spread_indices(n, K, rng)
    return [jax.tree.map(lambda x: x[int(i)], traj_states) for i in idx]


# ==========================================================================
# Streaming ensemble average + across-burst standard error
# ==========================================================================
class _Welford:
    """Elementwise streaming mean / M2 over a sequence of same-shape arrays, **masking
    non-finite entries** (the project masked-NaN contract: a diverged burst's NaN/Inf lanes
    are excluded per-element, so the running mean stays finite). Each element keeps its own
    valid count."""

    __slots__ = ("count", "mean", "M2")

    def __init__(self):
        self.count = self.mean = self.M2 = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.count is None:
            self.count = np.zeros(x.shape)
            self.mean = np.zeros(x.shape)
            self.M2 = np.zeros(x.shape)
        finite = np.isfinite(x)
        xf = np.where(finite, x, 0.0)
        self.count = self.count + finite
        safe_n = np.where(self.count > 0, self.count, 1.0)
        delta = np.where(finite, xf - self.mean, 0.0)
        self.mean = self.mean + delta / safe_n
        delta2 = np.where(finite, xf - self.mean, 0.0)
        self.M2 = self.M2 + delta * delta2

    def result(self) -> dict:
        n = self.count
        var = np.where(n > 1, self.M2 / np.where(n > 1, n - 1.0, 1.0), 0.0)
        std = np.sqrt(np.maximum(var, 0.0))
        stderr = np.where(n > 0, std / np.sqrt(np.where(n > 0, n, 1.0)), 0.0)
        return dict(mean=self.mean, std=std, stderr=stderr, count=n)


def average_grads(grad_iter):
    """Stream-average an iterable of per-burst gradients → ``(stats, n_bursts)``.

    Each gradient is a scalar / array / pytree (reduced **leafwise**, elementwise). Non-finite
    burst entries are masked out (the result stays finite). ``stats`` is a dict of pytrees
    (matching the gradient structure) ``{'mean', 'std', 'stderr', 'count'}`` — ``mean`` is the
    ensemble-averaged gradient, ``stderr`` the across-burst standard error of that mean (the
    uncertainty report the plan requires). ``(None, 0)`` if the iterable is empty.

    Streaming (Welford) — holds one running accumulator per leaf, never the full ensemble."""
    welfords = None
    treedef = None
    n = 0
    for g in grad_iter:
        leaves, td = jax.tree.flatten(g)
        if welfords is None:
            welfords = [_Welford() for _ in leaves]
            treedef = td
        for w, leaf in zip(welfords, leaves):
            w.update(np.asarray(leaf))
        n += 1
    if welfords is None:
        return None, 0
    res = [w.result() for w in welfords]
    stats = {k: jax.tree.unflatten(treedef, [r[k] for r in res])
             for k in ("mean", "std", "stderr", "count")}
    return stats, n


# ==========================================================================
# Convergence diagnostic — running mean vs #bursts (a SCALAR observable)
# ==========================================================================
def convergence(grads, tol: float = 0.05) -> dict:
    """Running cumulative mean (and standard error) after each burst, for a **scalar**
    observable (e.g. ``d(time-mean MLD)/d(c_k)``) — the "how many bursts to stabilize?"
    diagnostic. ``grads`` is a 1-D sequence of per-burst scalars; non-finite bursts are masked.

    Returns ``{'running_mean'[k], 'running_stderr'[k], 'final', 'n_stable'}`` where ``n_stable``
    is the smallest burst count from which the running mean stays within ``tol`` (relative) of
    the final value — the convergence horizon."""
    g = np.asarray(grads, dtype=np.float64).ravel()
    if g.size == 0:
        return dict(running_mean=g, running_stderr=g, final=0.0, n_stable=0)
    finite = np.isfinite(g)
    gf = np.where(finite, g, 0.0)
    cum_n = np.cumsum(finite.astype(np.float64))
    safe_n = np.where(cum_n > 0, cum_n, 1.0)
    running_mean = np.where(cum_n > 0, np.cumsum(gf) / safe_n, 0.0)
    cum_sq = np.cumsum(np.where(finite, gf * gf, 0.0))
    var = np.where(cum_n > 1, (cum_sq - cum_n * running_mean ** 2) / np.where(cum_n > 1, cum_n - 1.0, 1.0), 0.0)
    running_stderr = np.where(cum_n > 0, np.sqrt(np.maximum(var, 0.0)) / np.sqrt(safe_n), 0.0)
    final = float(running_mean[-1])
    scale = max(abs(final), 1e-12)
    within = np.abs(running_mean - final) <= tol * scale
    n_stable = int(g.size)
    for k in range(g.size):
        if within[k:].all():
            n_stable = k + 1
            break
    return dict(running_mean=running_mean, running_stderr=running_stderr,
                final=final, n_stable=n_stable)
