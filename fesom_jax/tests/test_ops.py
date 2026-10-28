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


@pytest.mark.parametrize("batch,K", [((37,), 9), ((4, 5), 48)])
def test_tdma_column_unroll_is_the_scan(batch, K, monkeypatch):
    """``FESOM_COLUMN_UNROLL=1`` (the level loop unrolled on the ``[..., K]`` arrays, no transpose)
    gives the ``lax.scan`` solve bit for bit, jitted, and the same gradient."""
    a, b, c, d = (jnp.asarray(v) for v in _random_well_posed(batch, K))
    loss = lambda d_in, b_in: jnp.sum(ops.tdma(a, b_in, c, d_in) ** 2)   # noqa: E731
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("FESOM_COLUMN_UNROLL", flag)
        out[flag] = (np.asarray(jax.jit(ops.tdma)(a, b, c, d)),
                     [np.asarray(g) for g in jax.jit(jax.grad(loss, argnums=(0, 1)))(d, b)])
    np.testing.assert_array_equal(out["1"][0], out["0"][0])
    for g1, g0 in zip(out["1"][1], out["0"][1]):
        np.testing.assert_allclose(g1, g0, rtol=1e-13, atol=0)


def test_unrolled_scan_matches_lax_scan_forward_and_reverse():
    xs = (jnp.asarray(RNG.standard_normal((6, 11))), jnp.asarray(RNG.standard_normal((6, 11))))
    body = lambda c, x: (c * 0.5 + x[0] - x[1], c + x[0])                  # noqa: E731
    init = jnp.zeros(6)
    for rev in (False, True):
        cs, ys = jax.lax.scan(body, init, tuple(v.T for v in xs), reverse=rev)
        cu, yu = ops.unrolled_scan(body, init, xs, reverse=rev)
        np.testing.assert_array_equal(np.asarray(cu), np.asarray(cs))
        np.testing.assert_array_equal(np.asarray(yu), np.asarray(ys).T)


def test_tke_level_scans_column_unroll_is_the_scan(monkeypatch):
    """The TKE min-scans and Thomas solve under ``FESOM_COLUMN_UNROLL=1`` equal their
    ``lax.scan`` forms bit for bit (values) and to round-off (gradients)."""
    from fesom_jax import cvmix_tke as ct
    N, K = 33, 48
    mxl = jnp.asarray(np.abs(RNG.standard_normal((N, K))) * 50.0)
    dzw = jnp.asarray(np.abs(RNG.standard_normal((N, K))) * 10.0 + 1.0)
    interior = jnp.asarray(RNG.random((N, K)) > 0.2)
    a, b, c, d = (jnp.asarray(v) for v in _random_well_posed((N,), K))

    def run():
        f = jax.jit(lambda m, z: (ct._fwd_min_scan(m, z, interior), ct._bwd_min_scan(m, z, interior)))
        s = jax.jit(ct._solve_tridiag)
        g = jax.jit(jax.grad(lambda m, dd: jnp.sum(ct._fwd_min_scan(m, dzw, interior) ** 2)
                             + jnp.sum(ct._solve_tridiag(a, b, c, dd) ** 2), argnums=(0, 1)))
        return [np.asarray(v) for v in (*f(mxl, dzw), s(a, b, c, d), *g(mxl, d))]

    monkeypatch.setenv("FESOM_COLUMN_UNROLL", "0")
    ref = run()
    monkeypatch.setenv("FESOM_COLUMN_UNROLL", "1")
    got = run()
    for i in range(3):
        np.testing.assert_array_equal(got[i], ref[i])
    for i in (3, 4):
        np.testing.assert_allclose(got[i], ref[i], rtol=1e-13, atol=0)


# --------------------------------------------------------------------------
# Ordered sums (gpu.ordered_sums): scatter_add through an inverse map
# --------------------------------------------------------------------------
def test_inverse_map_small_example():
    """Rows list their positions ascending; sentinels (-1, out of range) belong to no row; pads = n."""
    seg = np.array([[2, 0], [-1, 2], [0, 5], [2, 1]], np.int32)        # n = 8 entries, 3 rows
    inv = ops.inverse_map(seg, 3)
    np.testing.assert_array_equal(inv, [[1, 4, 8], [7, 8, 8], [0, 3, 6]])
    assert ops.inverse_map(seg, 3, width=5).shape == (3, 5)
    with pytest.raises(ValueError):
        ops.inverse_map(seg, 3, width=2)


@pytest.mark.parametrize("rest", [(), (4,), (3, 2)])
def test_ordered_sum_is_the_sequential_loop(rest, monkeypatch):
    """Exact against a numpy loop that adds each entry into its row in index order from 0.0 (the
    sequential scatter), including -1 sentinels and empty rows."""
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    N, E = 11, 40
    seg = RNG.integers(-1, N - 2, size=(E, 3)).astype(np.int32)          # rows N-2, N-1 empty
    vals = RNG.standard_normal((E, 3) + rest) * 10.0 ** RNG.integers(-8, 8, size=(E, 3) + (1,) * len(rest))
    ref = np.zeros((N,) + rest)
    for i, s in enumerate(seg.reshape(-1)):
        if s >= 0:
            ref[s] = ref[s] + vals.reshape((-1,) + rest)[i]
    inv = jnp.asarray(ops.inverse_map(seg, N))
    got = jax.jit(lambda v, s, iv: ops.scatter_add(v, s, N, inv=iv))(jnp.asarray(vals), jnp.asarray(seg), inv)
    np.testing.assert_array_equal(np.asarray(got), ref)


def test_ordered_sum_needs_the_switch(monkeypatch):
    """Without FESOM_ORDERED_SUMS the inv argument is ignored: the segment_sum program, bit for bit."""
    monkeypatch.delenv("FESOM_ORDERED_SUMS", raising=False)
    seg = jnp.asarray(RNG.integers(-1, 9, size=(30, 2)).astype(np.int32))
    vals = jnp.asarray(RNG.standard_normal((30, 2, 3)))
    inv = jnp.asarray(ops.inverse_map(np.asarray(seg), 9))
    f = lambda v, s, iv: ops.scatter_add(v, s, 9, inv=iv)
    g = lambda v, s, iv: ops.scatter_add(v, s, 9)
    assert jax.jit(f).lower(vals, seg, inv).as_text() == jax.jit(g).lower(vals, seg, inv).as_text()


def test_ordered_sum_gradient_is_the_gather(monkeypatch):
    """d/dvals of <w, scatter_add(vals)> = w[seg] on valid entries, 0 on sentinels — the same as the
    segment_sum's (exact here: one cotangent per entry)."""
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    seg = RNG.integers(-1, 8, size=(25, 3)).astype(np.int32)
    vals = jnp.asarray(RNG.standard_normal((25, 3, 2)))
    w = jnp.asarray(RNG.standard_normal((8, 2)))
    inv = jnp.asarray(ops.inverse_map(seg, 8))
    g_on = jax.grad(lambda v: jnp.sum(w * ops.scatter_add(v, seg, 8, inv=inv)))(vals)
    expect = np.where((seg >= 0)[..., None], np.asarray(w)[np.maximum(seg, 0)], 0.0)
    np.testing.assert_array_equal(np.asarray(g_on), expect)
    _, t = jax.jvp(lambda v: ops.scatter_add(v, seg, 8, inv=inv), (vals,), (vals,))
    np.testing.assert_allclose(np.asarray(t), np.asarray(ops.scatter_add(vals, seg, 8)), rtol=1e-14)
