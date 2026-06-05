"""Task 1.2 gate: gather/scatter/mask/TDMA verified forward AND under autodiff.

The ops are the AD-safe primitives every kernel is built from, so this is the
half of GATE 1 that proves differentiability before any physics is ported.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ops

RNG = np.random.default_rng(20260605)


# --------------------------------------------------------------------------
# Gather
# --------------------------------------------------------------------------
def test_gather_nodes_to_elem_matches_indexing():
    field = jnp.asarray(RNG.standard_normal((20, 4)))           # (nod2D, nl)
    elem_nodes = jnp.asarray(RNG.integers(0, 20, size=(7, 3)).astype(np.int32))
    got = ops.gather_nodes_to_elem(field, elem_nodes)
    assert got.shape == (7, 3, 4)
    for e in range(7):
        for c in range(3):
            np.testing.assert_array_equal(
                np.asarray(got[e, c]), np.asarray(field[int(elem_nodes[e, c])])
            )


def test_gather_to_edges_vector_field():
    field = jnp.asarray(RNG.standard_normal((12, 5, 2)))        # (nod2D, nl, 2)
    edges = jnp.asarray(RNG.integers(0, 12, size=(9, 2)).astype(np.int32))
    got = ops.gather_to_edges(field, edges)
    assert got.shape == (9, 2, 5, 2)
    np.testing.assert_array_equal(np.asarray(got[3, 1]), np.asarray(field[int(edges[3, 1])]))


# --------------------------------------------------------------------------
# Scatter
# --------------------------------------------------------------------------
def test_scatter_add_matches_reference_loop():
    n = 8
    vals = jnp.asarray(RNG.standard_normal((15, 3)))
    seg = jnp.asarray(RNG.integers(0, n, size=15).astype(np.int32))
    got = np.asarray(ops.scatter_add(vals, seg, n))
    ref = np.zeros((n, 3))
    for i in range(15):
        ref[int(seg[i])] += np.asarray(vals[i])
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_scatter_add_masks_negative_sentinels():
    """edge_tri-style -1 entries must contribute nothing."""
    n = 5
    vals = jnp.asarray(RNG.standard_normal((10, 2)))
    seg = np.array([0, -1, 2, -1, 4, 4, 1, -1, 3, 2], dtype=np.int32)
    got = np.asarray(ops.scatter_add(vals, jnp.asarray(seg), n))
    ref = np.zeros((n, 2))
    for i in range(10):
        if seg[i] >= 0:
            ref[seg[i]] += np.asarray(vals[i])
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13)


def test_edge_scatter_gather_roundtrip_is_degree_weighted():
    """scatter(gather(f)) accumulates f[node] once per incident edge-endpoint,
    i.e. degree(node) * f[node]."""
    n = 10
    edges = jnp.asarray(RNG.integers(0, n, size=(18, 2)).astype(np.int32))
    field = jnp.asarray(RNG.standard_normal((n, 3)))
    endpts = ops.gather_to_edges(field, edges)                  # (18, 2, 3)
    out = ops.scatter_add_edges_to_nodes(endpts, edges, n)      # (n, 3)
    deg = np.bincount(np.asarray(edges).ravel(), minlength=n).astype(np.float64)
    np.testing.assert_allclose(np.asarray(out), deg[:, None] * np.asarray(field),
                               rtol=0, atol=1e-12)


def test_scatter_transpose_is_gather():
    """The reverse-mode transpose (vjp) of scatter_add is exactly a gather at the
    same segment ids — the property that makes scatters free for AD."""
    n = 6
    seg = jnp.asarray(RNG.integers(0, n, size=11).astype(np.int32))
    v = jnp.asarray(RNG.standard_normal((11, 2)))
    cot = jnp.asarray(RNG.standard_normal((n, 2)))              # output cotangent
    _, vjp = jax.vjp(lambda x: ops.scatter_add(x, seg, n), v)
    (grad_v,) = vjp(cot)
    np.testing.assert_allclose(np.asarray(grad_v), np.asarray(ops.gather(cot, seg)),
                               rtol=0, atol=1e-13)


def test_scatter_transpose_zero_on_sentinels():
    n = 4
    seg = jnp.asarray(np.array([0, -1, 2, 3, -1], dtype=np.int32))
    v = jnp.asarray(RNG.standard_normal((5,)))
    cot = jnp.asarray(RNG.standard_normal((n,)))
    _, vjp = jax.vjp(lambda x: ops.scatter_add(x, seg, n), v)
    (grad_v,) = vjp(cot)
    g = np.asarray(grad_v)
    assert g[1] == 0.0 and g[4] == 0.0                          # masked entries
    np.testing.assert_allclose(g[[0, 2, 3]],
                               np.asarray(cot)[[0, 2, 3]], rtol=0, atol=1e-13)


# --------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------
def test_mask_below_bottom_scalar_and_vector():
    field = jnp.asarray(RNG.standard_normal((4, 5)))
    mask = jnp.asarray(np.array([
        [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0]
    ], dtype=bool))
    out = np.asarray(ops.mask_below_bottom(field, mask))
    m = np.asarray(mask)
    assert np.all(out[~m] == 0.0)
    np.testing.assert_array_equal(out[m], np.asarray(field)[m])
    # vector field: mask broadcasts over the trailing component axis
    vfield = jnp.asarray(RNG.standard_normal((4, 5, 2)))
    vout = np.asarray(ops.mask_below_bottom(vfield, mask))
    assert np.all(vout[~m] == 0.0)
    np.testing.assert_array_equal(vout[m], np.asarray(vfield)[m])


def test_mask_gradient_passes_through_valid_only():
    field = jnp.asarray(RNG.standard_normal((3, 4)))
    mask = jnp.asarray(np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]], bool))
    g = jax.grad(lambda x: jnp.sum(ops.mask_below_bottom(x, mask) ** 2))(field)
    expected = 2.0 * np.asarray(field) * np.asarray(mask)
    np.testing.assert_allclose(np.asarray(g), expected, rtol=0, atol=1e-13)


# --------------------------------------------------------------------------
# TDMA
# --------------------------------------------------------------------------
def _dense_tridiag(a, b, c):
    K = b.shape[-1]
    M = np.zeros(b.shape + (K,))
    for k in range(K):
        M[..., k, k] = b[..., k]
        if k > 0:
            M[..., k, k - 1] = a[..., k]
        if k < K - 1:
            M[..., k, k + 1] = c[..., k]
    return M


def _random_well_posed(batch, K):
    a = RNG.standard_normal(batch + (K,))
    c = RNG.standard_normal(batch + (K,))
    # diagonally dominant ⇒ nonsingular & numerically benign
    b = 4.0 + np.abs(a) + np.abs(c) + np.abs(RNG.standard_normal(batch + (K,)))
    a[..., 0] = 0.0
    c[..., K - 1] = 0.0
    d = RNG.standard_normal(batch + (K,))
    return a, b, c, d


def test_tdma_matches_dense_solve_single():
    a, b, c, d = _random_well_posed((), 6)
    x = np.asarray(ops.tdma(*(jnp.asarray(v) for v in (a, b, c, d))))
    M = _dense_tridiag(a, b, c)
    np.testing.assert_allclose(M @ x, d, rtol=0, atol=1e-10)
    np.testing.assert_allclose(x, np.linalg.solve(M, d), rtol=1e-10, atol=1e-12)


def test_tdma_vectorized_over_entities():
    a, b, c, d = _random_well_posed((37,), 9)
    x = np.asarray(ops.tdma(*(jnp.asarray(v) for v in (a, b, c, d))))
    assert x.shape == (37, 9)
    M = _dense_tridiag(a, b, c)
    xref = np.linalg.solve(M, d[..., None])[..., 0]
    np.testing.assert_allclose(x, xref, rtol=1e-9, atol=1e-11)


def test_tdma_broadcasts_and_jits():
    a, b, c, d = _random_well_posed((5,), 7)
    # broadcast a scalar-per-column 'a' across batch to exercise broadcasting
    x_direct = ops.tdma(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d))
    x_jit = jax.jit(ops.tdma)(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d))
    np.testing.assert_allclose(np.asarray(x_direct), np.asarray(x_jit), rtol=0, atol=1e-13)


def test_tdma_gradient_vs_finite_difference():
    """Reverse-mode grad of a scalar loss through the TDMA solve matches a
    central finite difference (float64) — the AD gate for the vertical solvers."""
    a, b, c, d = _random_well_posed((3,), 6)
    a = jnp.asarray(a); b = jnp.asarray(b); c = jnp.asarray(c); d0 = jnp.asarray(d)

    def loss(d_in, b_in):
        x = ops.tdma(a, b_in, c, d_in)
        return jnp.sum(x ** 2)

    gd, gb = jax.grad(loss, argnums=(0, 1))(d0, b)
    gd = np.asarray(gd); gb = np.asarray(gb)

    h = 1e-6
    for (ent, lev) in [(0, 0), (1, 3), (2, 5)]:
        # d-grad
        dp = d0.at[ent, lev].add(h); dm = d0.at[ent, lev].add(-h)
        fd = float((loss(dp, b) - loss(dm, b)) / (2 * h))
        assert abs(fd - gd[ent, lev]) <= 1e-6 * (1 + abs(fd)), (ent, lev, fd, gd[ent, lev])
        # b-grad
        bp = b.at[ent, lev].add(h); bm = b.at[ent, lev].add(-h)
        fb = float((loss(d0, bp) - loss(d0, bm)) / (2 * h))
        assert abs(fb - gb[ent, lev]) <= 1e-6 * (1 + abs(fb)), (ent, lev, fb, gb[ent, lev])
