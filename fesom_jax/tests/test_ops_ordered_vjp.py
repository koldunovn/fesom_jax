"""``ops.ordered_sum``'s explicit transpose (``FESOM_ORDERED_VJP=1``, given the forward index by ``ops.scatter_add``): the
gather ``vals_bar[i] = out_bar[seg[i]]`` equals the default transpose (a scatter of the cotangent) exactly, with entries
that have no target (``seg < 0``) getting 0; the forward is the same code either way. Light (CPU, no mesh)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import ops


def _case(seed, n=500, n_out=60, rest=(3,)):
    rng = np.random.default_rng(seed)
    seg = rng.integers(-1, n_out, size=n)                      # -1 = no target (the boundary sentinel)
    vals = jnp.asarray(rng.standard_normal((n,) + rest))
    bar = jnp.asarray(rng.standard_normal((n_out,) + rest))
    return seg, ops.inverse_map(seg, n_out), vals, bar


def _vjp(monkeypatch, explicit, seg, inv, vals, bar):
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    monkeypatch.setenv("FESOM_ORDERED_VJP", "1" if explicit else "0")
    y, f = jax.vjp(lambda v: ops.scatter_add(v, jnp.asarray(seg), inv.shape[0], inv=jnp.asarray(inv)), vals)
    return np.asarray(y), np.asarray(f(bar)[0])


def test_explicit_transpose_equals_the_default(monkeypatch):
    for seed, rest in ((0, ()), (1, (3,)), (2, (2, 4))):
        seg, inv, vals, bar = _case(seed, rest=rest)
        y0, g0 = _vjp(monkeypatch, False, seg, inv, vals, bar)
        y1, g1 = _vjp(monkeypatch, True, seg, inv, vals, bar)
        np.testing.assert_array_equal(y1, y0)                  # the forward: the same chain
        np.testing.assert_array_equal(g1, g0)                  # each entry gets exactly one cotangent value
        assert np.all(g1[seg < 0] == 0.0)


def test_without_the_index_the_default_is_taken(monkeypatch):
    seg, inv, vals, bar = _case(3)
    monkeypatch.setenv("FESOM_ORDERED_VJP", "1")
    _, f = jax.vjp(lambda v: ops.ordered_sum(v, jnp.asarray(inv)), vals)       # no seg_flat
    _, f1 = jax.vjp(lambda v: ops.ordered_sum(v, jnp.asarray(inv), jnp.asarray(seg)), vals)
    np.testing.assert_array_equal(np.asarray(f(bar)[0]), np.asarray(f1(bar)[0]))
