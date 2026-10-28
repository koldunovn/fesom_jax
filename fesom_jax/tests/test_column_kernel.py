"""The Pallas column kernels (``fesom_jax.column_kernel``, opt-in ``gpu.column_kernel``) through Pallas's
interpreter on the CPU: the same values as the XLA sweeps they replace, and the reference sweeps' derivatives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import column_kernel as ck
from fesom_jax import ops

RNG = np.random.default_rng(11)


def _system(n, K):
    a = RNG.standard_normal((n, K))
    c = RNG.standard_normal((n, K))
    b = 4.0 + np.abs(a) + np.abs(c) + np.abs(RNG.standard_normal((n, K)))
    a[:, 0] = 0.0
    c[:, -1] = 0.0
    d = RNG.standard_normal((n, K))
    return tuple(jnp.asarray(v) for v in (a, b, c, d))


@pytest.mark.parametrize("n,K", [(37, 9), (300, 48)])     # 300: not a multiple of the 128-column block
def test_thomas_kernel_matches_tdma(n, K):
    a, b, c, d = _system(n, K)
    ref = np.asarray(ops.tdma(a, b, c, d))
    got = np.asarray(ck.thomas(ops._tdma_forward, ops._tdma_backward, a, b, c, d, interpret=True))
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0)


def test_thomas_kernel_gradient_is_the_reference_sweeps():
    a, b, c, d = _system(40, 12)
    w = jnp.asarray(RNG.standard_normal((40, 12)))
    f_k = lambda b_, d_: jnp.sum(w * ck.thomas(ops._tdma_forward, ops._tdma_backward, a, b_, c, d_,   # noqa
                                                interpret=True))
    f_r = lambda b_, d_: jnp.sum(w * ops.tdma(a, b_, c, d_))   # noqa: E731
    for gk, gr in zip(jax.grad(f_k, argnums=(0, 1))(b, d), jax.grad(f_r, argnums=(0, 1))(b, d)):
        np.testing.assert_allclose(np.asarray(gk), np.asarray(gr), rtol=1e-12, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_column_scan_min_scan_with_bool_mask(reverse):
    """The TKE min-scan shape: a float carry, a boolean mask among the inputs, the carry as the output."""
    n, K = 200, 48
    mxl = jnp.asarray(np.abs(RNG.standard_normal((n, K))) * 50.0)
    dzw = jnp.asarray(np.abs(RNG.standard_normal((n, K))) * 10.0 + 1.0)
    interior = jnp.asarray(RNG.random((n, K)) > 0.2)

    def body(prev, xs):
        m, z, i = xs
        new = jnp.where(i, jnp.minimum(m, prev + z), m)
        return new, new

    init = jnp.zeros((n,), mxl.dtype)
    ref_c, ref_y = ops.unrolled_scan(body, init, (mxl, dzw, interior), reverse=reverse)
    got_c, got_y = ck.column_scan(body, init, (mxl, dzw, interior), reverse=reverse, interpret=True)
    np.testing.assert_array_equal(np.asarray(got_y), np.asarray(ref_y))
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(ref_c))


def test_transpose_kernel():
    x = jnp.asarray(RNG.standard_normal((130, 47)))
    np.testing.assert_array_equal(np.asarray(ck.transpose(x, interpret=True)), np.asarray(x).T)


def test_enabled_needs_a_gpu(monkeypatch):
    monkeypatch.setenv("FESOM_COLUMN_KERNEL", "1")
    assert ck.enabled() == (jax.default_backend() == "gpu")
    monkeypatch.setenv("FESOM_COLUMN_KERNEL", "0")
    assert not ck.enabled()


@pytest.mark.gpu
def test_kernels_on_the_gpu():
    """The compiled Triton kernels against the XLA sweeps (round-off: Triton may contract into FMAs)."""
    a, b, c, d = _system(1000, 48)
    ref = np.asarray(jax.jit(ops.tdma)(a, b, c, d))
    got = np.asarray(jax.jit(lambda *v: ck.thomas(ops._tdma_forward, ops._tdma_backward, *v))(a, b, c, d))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    mxl = jnp.asarray(np.abs(RNG.standard_normal((1000, 48))) * 50.0)
    dzw = jnp.asarray(np.abs(RNG.standard_normal((1000, 48))) * 10.0 + 1.0)
    interior = jnp.asarray(RNG.random((1000, 48)) > 0.2)

    def body(prev, xs):
        m, z, i = xs
        new = jnp.where(i, jnp.minimum(m, prev + z), m)
        return new, new

    init = jnp.zeros((1000,), mxl.dtype)
    ref_y = ops.unrolled_scan(body, init, (mxl, dzw, interior))[1]
    got_y = jax.jit(lambda *v: ck.column_scan(body, init, v)[1])(mxl, dzw, interior)
    np.testing.assert_allclose(np.asarray(got_y), np.asarray(ref_y), rtol=1e-12, atol=1e-14)
