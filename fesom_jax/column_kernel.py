"""Column recursions as Pallas (Triton) GPU kernels: one thread per column through all levels (CORE2 GPU speed plan,
Tier A; opt-in ``gpu.column_kernel`` / ``FESOM_COLUMN_KERNEL=1``).

The level sweeps (``ops.tdma``, the TKE min-scans and Thomas solve) are sequential in the level and elementwise over
the columns. Under ``lax.scan`` they cost a kernel launch (or several) per level and a transpose to put the level in
front; unrolled in Python (``ops.unrolled_scan``, ``gpu.column_unroll``) XLA fuses a whole sweep into one kernel whose
concatenated output re-evaluates the recursion from the first level for every level — O(K²) work per column (the
backward substitution of ``impl_vert_visc`` on the elements: 1.35 ms per CORE2 step on 4 GH200, job 28291187). Here
each column is one thread that carries the recursion in registers: O(K) work, one launch per sweep. The kernel reads
and writes level-major copies (each level one coalesced access); they are made by a Pallas transpose kernel, which is
opaque to XLA, so XLA's layout assignment cannot re-lay the step level-major around it (the A5 layout flip).

The kernel body runs the same operations in the same order as ``ops.unrolled_scan`` (Triton may contract a
multiply-add into an FMA: round-off, as for the XLA paths). The tangent is the reference sweep's (``custom_jvp`` on
``ops.unrolled_scan``), so derivatives are those of the XLA code. The design follows fesom3-jax's
``ops/column.py`` (``tridiag_solve_kernel``, ``column_scan``, ``transpose_kernel``), which runs it on A100 and GH200.

``interpret=True`` runs the kernels through Pallas's interpreter on any backend (the CPU tests).
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax import lax

KERNEL_COLS = 128          # columns per program (a power of 2: Triton block sizes)
TILE = 64                  # the transpose kernel's square tile


def enabled() -> bool:
    """``FESOM_COLUMN_KERNEL=1`` and a GPU backend (read at trace time)."""
    return (os.environ.get("FESOM_COLUMN_KERNEL", "0").strip() not in ("", "0")
            and jax.default_backend() == "gpu")


def _kdt(dtype):
    return jnp.int8 if dtype == jnp.bool_ else dtype


def _transpose_body(x_ref, o_ref, *, rows, cols):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    i, j = pl.program_id(0).astype(jnp.int32), pl.program_id(1).astype(jnp.int32)
    ri = i * TILE + jnp.arange(TILE)
    cj = j * TILE + jnp.arange(TILE)
    m = (ri < rows)[:, None] & (cj < cols)[None, :]
    other = 0.0 if jnp.issubdtype(x_ref.dtype, jnp.floating) else 0
    t = plgpu.load(x_ref.at[pl.ds(i * TILE, TILE), pl.ds(j * TILE, TILE)], mask=m, other=other)
    plgpu.store(o_ref.at[pl.ds(j * TILE, TILE), pl.ds(i * TILE, TILE)], t.T, mask=m.T)


def transpose(x, interpret=False):
    """``x.T`` for a 2-D array, as a Pallas kernel (data movement only)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    rows, cols = x.shape
    return pl.pallas_call(
        functools.partial(_transpose_body, rows=rows, cols=cols),
        out_shape=jax.ShapeDtypeStruct((cols, rows), x.dtype),
        grid=(pl.cdiv(rows, TILE), pl.cdiv(cols, TILE)),
        compiler_params=None if interpret else plgpu.CompilerParams(num_warps=4),
        interpret=interpret)(x)


def _scan_body(*refs, body, reverse, nl, ncol, n, trees, dtypes):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    ni, nx, ny = n
    init_t, xs_t = trees
    di, dx, dy = dtypes
    ins, xr = refs[:ni], refs[ni:ni + nx]
    yr, cout = refs[ni + nx:ni + nx + ny], refs[ni + nx + ny:]
    i = pl.program_id(0).astype(jnp.int32)
    sl = pl.ds(i * KERNEL_COLS, KERNEL_COLS)
    ok = i * KERNEL_COLS + jnp.arange(KERNEL_COLS) < ncol

    def ld(ref, dt, *idx):
        other = 0.0 if jnp.issubdtype(ref.dtype, jnp.floating) else 0
        v = plgpu.load(ref.at[idx + (sl,)], mask=ok, other=other)
        return v != 0 if dt == jnp.bool_ else v

    def st(ref, v, *idx):
        plgpu.store(ref.at[idx + (sl,)], v.astype(ref.dtype), mask=ok)

    carry0 = tuple(ld(r, d) for r, d in zip(ins, di))

    def step(j, carry):
        k = jnp.int32(nl - 1) - j if reverse else j
        x = jax.tree.unflatten(xs_t, [ld(r, d, k) for r, d in zip(xr, dx)])
        c, y = body(jax.tree.unflatten(init_t, list(carry)), x)
        for r, v in zip(yr, jax.tree.leaves(y)):
            st(r, v, k)
        return tuple(jax.tree.leaves(c))

    carry = lax.fori_loop(jnp.int32(0), jnp.int32(nl), step, carry0)
    for r, v in zip(cout, carry):
        st(r, v)


def _scan_lm(body, reverse, interpret, init_t, xs_t, init_l, xs_lm):
    """The sweep on LEVEL-MAJOR inputs ``xs_lm`` (leaves ``[K, ncol]``); returns (carry leaves, level-major y
    leaves)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    nl, ncol = xs_lm[0].shape
    x0 = jax.tree.unflatten(xs_t, [jnp.zeros((ncol,), v.dtype) for v in xs_lm])
    c_ref, y_ref = jax.eval_shape(lambda c, x: body(c, x), jax.tree.unflatten(init_t, list(init_l)), x0)
    y_leaves = jax.tree.leaves(y_ref)
    dtypes = (tuple(v.dtype for v in init_l), tuple(v.dtype for v in xs_lm), tuple(v.dtype for v in y_leaves))
    out_shape = tuple(jax.ShapeDtypeStruct((nl, ncol), _kdt(y.dtype)) for y in y_leaves) + \
        tuple(jax.ShapeDtypeStruct(v.shape, _kdt(v.dtype)) for v in init_l)
    kern = functools.partial(_scan_body, body=body, reverse=reverse, nl=nl, ncol=ncol,
                             n=(len(init_l), len(xs_lm), len(y_leaves)), trees=(init_t, xs_t), dtypes=dtypes)
    out = pl.pallas_call(kern, out_shape=out_shape, grid=(pl.cdiv(ncol, KERNEL_COLS),),
                         compiler_params=None if interpret else plgpu.CompilerParams(num_warps=4),
                         interpret=interpret)(*(v.astype(_kdt(v.dtype)) for v in init_l),
                                              *(v.astype(_kdt(v.dtype)) for v in xs_lm))
    ys = tuple(v.astype(y.dtype) for v, y in zip(out[:len(y_leaves)], y_leaves))
    carry = tuple(v.astype(c.dtype) for v, c in zip(out[len(y_leaves):], init_l))
    return carry, ys, jax.tree.structure(y_ref)


def _float_mask(leaves):
    return tuple(jnp.issubdtype(v.dtype, jnp.inexact) for v in leaves)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4))
def _column_scan(body, reverse, interpret, init_t, xs_t, init_l, xs_l):
    xs_lm = tuple(transpose(v.astype(_kdt(v.dtype)), interpret).astype(v.dtype) for v in xs_l)
    carry, ys_lm, y_t = _scan_lm(body, reverse, interpret, init_t, xs_t, init_l, xs_lm)
    ys = tuple(transpose(v.astype(_kdt(v.dtype)), interpret).astype(v.dtype) for v in ys_lm)
    return carry, ys


def _reference(body, reverse, init_t, xs_t, init_l, xs_l):
    from .ops import unrolled_scan
    c, y = unrolled_scan(body, jax.tree.unflatten(init_t, list(init_l)), jax.tree.unflatten(xs_t, list(xs_l)),
                         reverse=reverse)
    return tuple(jax.tree.leaves(c)), tuple(jax.tree.leaves(y))


@_column_scan.defjvp
def _column_scan_jvp(body, reverse, interpret, init_t, xs_t, primals, tangents):
    init_l, xs_l = primals
    out = _column_scan(body, reverse, interpret, init_t, xs_t, init_l, xs_l)
    mi, mx = _float_mask(init_l), _float_mask(xs_l)

    def ref(*fl):
        it = iter(fl)
        il = tuple(next(it) if m else v for v, m in zip(init_l, mi))
        xl = tuple(next(it) if m else v for v, m in zip(xs_l, mx))
        return _reference(body, reverse, init_t, xs_t, il, xl)

    p = [v for v, m in zip(init_l, mi) if m] + [v for v, m in zip(xs_l, mx) if m]
    d = [v for v, m in zip(tangents[0], mi) if m] + [v for v, m in zip(tangents[1], mx) if m]
    _, dout = jax.jvp(ref, p, d)
    return out, dout


def column_scan(body, init, xs, reverse: bool = False, interpret: bool = False):
    """``ops.unrolled_scan(body, init, xs, reverse)`` as a Pallas kernel. ``xs`` leaves ``[ncol, K]``, ``init``
    leaves ``[ncol]``; ``body`` elementwise over the columns. Returns ``(carry, ys)`` like ``unrolled_scan``."""
    init_l, init_t = jax.tree.flatten(init)
    xs_l, xs_t = jax.tree.flatten(xs)
    carry, ys = _column_scan(body, reverse, interpret, init_t, xs_t, tuple(init_l), tuple(xs_l))
    x0 = jax.tree.unflatten(xs_t, [jnp.zeros(v.shape[:1], v.dtype) for v in xs_l])
    _, y_ref = jax.eval_shape(lambda c, x: body(c, x), init, x0)
    return jax.tree.unflatten(init_t, list(carry)), jax.tree.unflatten(jax.tree.structure(y_ref), list(ys))


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 6))
def _two_sweeps(fwd, bwd, a, b, c, d, interpret):
    """The Thomas solve: the forward elimination and the back-substitution, both on level-major copies, with one
    transpose in per input and one out."""
    lm = tuple(transpose(v, interpret) for v in (a, b, c, d))
    zero = jnp.zeros(a.shape[:1], a.dtype)
    _, (cp, dp), _ = _scan_lm(fwd, False, interpret, jax.tree.structure((0, 0)),
                              jax.tree.structure((0, 0, 0, 0)), (zero, zero), lm)
    _, (x,), _ = _scan_lm(bwd, True, interpret, jax.tree.structure(0), jax.tree.structure((0, 0)),
                          (zero,), (cp, dp))
    return transpose(x, interpret)


@_two_sweeps.defjvp
def _two_sweeps_jvp(fwd, bwd, interpret, primals, tangents):
    from .ops import unrolled_scan
    x = _two_sweeps(fwd, bwd, *primals, interpret)

    def ref(a, b, c, d):
        zero = jnp.zeros(a.shape[:1], a.dtype)
        _, (cp, dp) = unrolled_scan(fwd, (zero, zero), (a, b, c, d))
        return unrolled_scan(bwd, zero, (cp, dp), reverse=True)[1]

    _, dx = jax.jvp(ref, primals, tangents)
    return x, dx


def thomas(fwd, bwd, a, b, c, d, interpret: bool = False):
    """A tridiagonal solve given its two sweep bodies (``ops.tdma``'s or the TKE reciprocal form's), all inputs
    ``[ncol, K]`` of one float dtype. The tangent is ``ops.unrolled_scan``'s on the same bodies."""
    return _two_sweeps(fwd, bwd, a, b, c, d, interpret)
