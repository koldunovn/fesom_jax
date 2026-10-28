"""Working precision of a run: ``"double"`` (default) or ``"single"`` — FESOM2's ``USE_SINGLE_PRECISION`` build.

``"double"`` is the model as it always was: every helper below is the identity at trace time, so a double-precision
trace is the same jaxpr as before this module existed (checked by ``fesom_jax/tests/test_dp_lock.py``).

``"single"`` repeats upstream FESOM2's single-precision build (the stack ending in PR #1207; policy table and file:line
citations in ``docs/plans/20261020-core2-single-precision.md``): ``WP`` = float32 for the state, geometry, forcing
values, sea ice, the CG vectors and everything else, with float64 (``WP_full``) where upstream keeps real64 — layer
thickness ``hnode``/``hnode_new``, the tracer tendency accumulators and the rounding carry, the implicit vertical
diffusion column, the CG scalars / dot products / row sums, global integrals, the forcing time axis, the CVMix TKE
column, the ``visc_filt_bidiff`` arithmetic and the output means.

How (after fesom3-jax's ``precision.py``). A single-precision program is traced and run with ``jax_enable_x64`` off and
``jax_explicit_x64_dtypes = "allow"`` (:func:`scope`): every dtype-less array, Python float and float32 input is
float32 = ``WP``, an explicit ``.astype(jnp.float64)`` (:func:`full`) stays float64 = ``WP_full``, and mixed
float32/float64 arithmetic promotes to float64 as Fortran's mixed-kind expressions do; an assignment to a ``WP``
variable is an explicit :func:`wp`. :func:`sp` is the trace-time test (the x64 flag is part of JAX's trace context), so
SP-only algorithms branch on it as FESOM2 branches on ``use_tracer_carry`` / ``USE_SINGLE_PRECISION``.

Two more pieces the single-precision program needs:

* :func:`compiler_options` — ``xla_allow_excess_precision=False``. With XLA's default XLA:GPU folds every convert pair
  f64→f32→f64 into the float64 value, so the carry ``acc − f64(f32(acc/h))·h`` becomes 0 and the float32 roundings
  vanish (job 28288484 on A100; fesom3-jax job 28274339 on GH200). XLA:CPU keeps the pairs. Double-precision programs
  get no option (``None``), so they compile exactly as before.
* host casts (:func:`cast_tree`, :func:`cast_state`): float leaves to float32 — host numpy stays host (no full-size
  copy on device 0 of a shard stack), device arrays are cast in place (sharding kept) — except the ``WP_full`` State
  fields, which stay float64. Placement happens outside :func:`scope` (x64 on), so a float64 leaf is placed as
  float64; under x64 off a host float64 would be narrowed. JAX's x64 switch is per thread: any other thread that
  traces or places arrays for a single-precision run must enter :func:`scope` itself.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os

PRECISIONS = ("double", "single")
STATE_F64 = ("hnode", "hnode_new")          # WP_full storage upstream (MOD_MESH.F90:140-145)
# full REcoM's own carried arrays: REcoM's module sets its own WP = 8 (recom_modules.F90:24), so they stay float64 in a
# single-precision run (plan 20261026-recom-lane-sp, S1 design)
STATE_F64_RECOM = ("bgc_benthos", "bgc_alk_surf", "bgc_carb", "bgc_diag", "bgc_budget")
# the SP-only rounding carry (MOD_TRACER.F90:42), WP: T, S and (full REcoM) the BGC stack (nod2D, nl, ntr)
CARRY_FIELDS = ("carry_T", "carry_S", "bgc_carry")


def check(precision: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return precision


@contextlib.contextmanager
def scope(precision: str):
    """Trace and run a program inside this. ``"double"`` changes nothing; ``"single"`` makes float32 JAX's default
    (``WP``) while keeping explicit float64 (``WP_full``)."""
    check(precision)
    if precision == "double":
        yield
        return
    import jax
    from jax._src import config as jc
    with jax.enable_x64(False), jc.explicit_x64_dtypes("allow"):
        yield


def sp() -> bool:
    """True inside a single-precision trace (x64 off)."""
    from jax._src import config as jc
    return not jc.enable_x64.value


def wp_dtype():
    import jax.numpy as jnp
    return jnp.float32 if sp() else jnp.float64


def full(x):
    """``real(x, WP_full)``: float64 in single precision; the identity in double."""
    if not sp():
        return x
    import jax.numpy as jnp
    return jnp.asarray(x).astype(jnp.float64)


def wp(x):
    """``real(x, WP)`` / an assignment to a ``WP`` variable: float32 in single precision; the identity in double."""
    if not sp():
        return x
    import jax.numpy as jnp
    return jnp.asarray(x).astype(jnp.float32)


def lit(c):
    """A WP namelist constant inside a ``WP_full`` expression: float32-rounded in single precision (FESOM2 widens the
    WP value, e.g. ``real(tke_mxl_min, cvmix_r8)``), the Python value in double."""
    if not sp() or not isinstance(c, float):
        return c
    import numpy as np
    return float(np.float32(c))


def log10(x):
    """``jnp.log10``, also for a float64 ``x`` inside a single-precision trace: JAX's ``log10`` multiplies by a numpy
    float64 constant that ``lax.mul`` narrows to float32 when x64 is off and then refuses the dtype pair. In double
    precision ``jnp.log10`` itself (the double-precision trace is unchanged); in single precision the same product
    with the constant in ``x``'s dtype."""
    import jax.numpy as jnp
    if not sp():
        return jnp.log10(x)
    from jax import lax
    x = jnp.asarray(x)
    return lax.mul(lax.log(x), jnp.asarray(0.4342944819032518, x.dtype))


def full_tree(tree):
    """:func:`full` on every floating leaf of a pytree (identity in double)."""
    if not sp():
        return tree
    import jax
    import jax.numpy as jnp

    def leaf(a):
        if a is None or not jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating):
            return a
        return jnp.asarray(a).astype(jnp.float64)
    return jax.tree.map(leaf, tree)


def wp_tree(tree):
    """:func:`wp` on every floating leaf of a pytree (identity in double)."""
    if not sp():
        return tree
    import jax
    import jax.numpy as jnp

    def leaf(a):
        if a is None or not jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating):
            return a
        return jnp.asarray(a).astype(jnp.float32)
    return jax.tree.map(leaf, tree)


def compiler_options(precision: str):
    """Per-jit XLA options for a program of this precision: ``None`` in double (today's call)."""
    check(precision)
    if precision == "single" and os.environ.get("FESOM_SP_ALLOW_EXCESS") == "1":
        return None            # DIAGNOSTIC ONLY (speed A/B): XLA may then fold the roundings — never for results
    return {"xla_allow_excess_precision": False} if precision == "single" else None


def jit(fn, precision: str = "double", **kw):
    """``jax.jit`` with :func:`compiler_options`; in double exactly ``jax.jit(fn, **kw)``."""
    import jax
    opts = compiler_options(precision)
    return jax.jit(fn, **kw) if opts is None else jax.jit(fn, compiler_options=opts, **kw)


class Scoped:
    """A jitted function whose calls and lowering run inside :func:`scope` (single precision only; in double the
    jitted function itself is used, see :func:`scoped`)."""

    def __init__(self, fn, precision: str):
        self.fn, self.precision = fn, precision

    def __call__(self, *a, **kw):
        with scope(self.precision):
            return self.fn(*a, **kw)

    def lower(self, *a, **kw):
        with scope(self.precision):
            return self.fn.lower(*a, **kw)


def scoped(fn, precision: str):
    return fn if precision == "double" else Scoped(fn, precision)


# ---- host side --------------------------------------------------------------------------------------------------
def _cast_leaf(a, keep64: bool = False):
    import numpy as np
    import jax
    if a is None or isinstance(a, (bool, int)):
        return a
    if isinstance(a, float):
        return a if keep64 else float(np.float32(a))
    if isinstance(a, jax.Array):
        if not jax.numpy.issubdtype(a.dtype, jax.numpy.floating):
            return a
        target = np.float64 if keep64 else np.float32
        return a if a.dtype == target else a.astype(target)
    arr = np.asarray(a)
    if arr.dtype.kind != "f":
        return a
    return arr.astype(np.float64 if keep64 else np.float32)


def cast_tree(tree, keep64: bool = False):
    """Every floating leaf of a pytree (mesh, SSH operator, Params, forcing, ...) to float32 (or to float64 with
    ``keep64``). Integer and boolean leaves, ``None`` and the pytree structure are unchanged."""
    import jax
    return jax.tree.map(lambda a: _cast_leaf(a, keep64), tree)


def cast_state(state):
    """The State of a single-precision run: float32 except :data:`STATE_F64` and :data:`STATE_F64_RECOM` (float64);
    the carry zero when absent (a cold start, or a restart written in double precision). The BGC carry exists for
    full REcoM only (a State with ``bgc_alk_surf``; the passive mode has none)."""
    import numpy as np
    import jax
    vals = {}
    for f in dataclasses.fields(state):
        vals[f.name] = _cast_leaf(getattr(state, f.name), keep64=f.name in STATE_F64 + STATE_F64_RECOM)
    full_recom = getattr(state, "bgc_alk_surf", None) is not None
    for name in CARRY_FIELDS:
        if name == "bgc_carry" and not full_recom:
            vals[name] = None
            continue
        if vals.get(name) is None:
            like = state.bgc if name == "bgc_carry" else state.T
            if isinstance(like, jax.Array):
                vals[name] = jax.numpy.zeros_like(like, dtype=np.float32)
            else:
                vals[name] = np.zeros(np.shape(like), np.float32)
    return state.__class__(**vals)


def uncast_state(state):
    """A single-precision State widened to float64 (restart writing / handing to a double-precision consumer). The
    carry is kept (float64) — callers that want a pure double-precision State drop it (:func:`drop_carry`)."""
    return cast_tree(state, keep64=True)


def drop_carry(state):
    return dataclasses.replace(state, **{k: None for k in CARRY_FIELDS})
