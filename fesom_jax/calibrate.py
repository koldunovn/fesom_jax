"""The optimizer half of the ML-hook seam — differentiable parameter calibration.

This is the generic optimizer loop the paper's **calibration pillar** (§2) and **hybrid-ML
pillar** (§3) both drive: minimize a scalar ``loss_fn(tunables)`` over a **pytree of
tunable leaves** with an ``optax`` optimizer, where ``loss_fn`` closes over the
checkpointed differentiable :func:`fesom_jax.integrate.integrate`. The *exact same*
``optimize`` trains a single scalar (``{'k_gm': θ}``), a ``[nod2D]`` field leaf, or the
``tke_nn`` MLP weights — array leaves already differentiate, so widening the tunable
costs **zero** structural change (the design's "scalar calibration is the zeroth-order
NN" thesis).

Designed in Phase 7a (``docs/plans/20260607-fesom-jax-paramtune.md`` §1) and reused
verbatim here. Three entry points:

* :func:`optimize` — jit ``value_and_grad`` once, host-side loop (logging / early-stop;
  the work is the jitted forward+backward). Returns ``(final_tunables, history)``.
* :func:`grid_scan` — forward-only 1-D sweep: the misfit-bowl probe that **confirms the
  minimum sits at the injected value** before any backward is trusted.
* :func:`build_params` — map a tunable ``dict`` onto a full :class:`~fesom_jax.params.Params`
  (defaults for the unset leaves), the seam between the optimizer's flat dict and the
  model's structured parameter pytree.

House style: optax is optimizer-agnostic (Adam for scalars, ``optax.lbfgs`` later); the
loop is pure host Python so ``on_step`` / ``stop_fn`` stay flexible. Float64 throughout
(x64 enabled by :mod:`fesom_jax.config`).
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import optax

from .params import Params


def optimize(loss_fn: Callable, init, optimizer,
             *, n_iters: int, on_step: Callable | None = None,
             stop_fn: Callable | None = None, keep_params: bool = True):
    """Minimize ``loss_fn(tunables) -> scalar`` over the pytree ``init`` with ``optimizer``.

    ``init`` is any pytree of float64 leaves (a ``dict`` like ``{'k_gm': θ}``, a ``Params``,
    a ``tke_nn`` MLP). :func:`jax.value_and_grad` is jitted **once**; the host loop only
    logs / early-stops (all the compute is the jitted forward+backward of ``loss_fn``).

    * ``on_step(record)`` — called each iter with the latest history record (for live
      logging / checkpointing).
    * ``stop_fn(record) -> bool`` — early-stop predicate (e.g. ``|θ-target|/target < 2%``);
      checked *before* the update so the returned ``params`` is the one that triggered it.
    * ``keep_params`` — store ``device_get(params)`` in each record (default True, the
      Phase-7a scalar-demo behaviour). Set False for large leaves (NN/field) where a full
      per-iter snapshot is wasteful — use ``on_step`` to checkpoint instead.

    Returns ``(final_tunables, history)`` where ``history`` is a list of per-iter dicts
    ``{'it', 'loss', 'gnorm'[, 'params']}``.
    """
    vg = jax.jit(jax.value_and_grad(loss_fn))
    params, opt_state = init, optimizer.init(init)
    history: list[dict] = []
    for it in range(1, n_iters + 1):
        loss, grads = vg(params)
        gnorm = optax.tree.norm(grads)
        rec = {"it": it, "loss": float(loss), "gnorm": float(gnorm)}
        if keep_params:
            rec["params"] = jax.device_get(params)
        history.append(rec)
        if on_step is not None:
            on_step(rec)
        if stop_fn is not None and stop_fn(rec):
            break
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    return params, history


def grid_scan(loss_fn: Callable, base: dict, leaf: str, values: Iterable):
    """Forward-only 1-D sweep of ``base[leaf]`` over ``values`` → ``[(v, loss)]``.

    The misfit-bowl probe: CONFIRM ``argmin`` sits at the injected value before trusting
    the descent (well-posedness check, runnable on the login node — no backward). ``base``
    is a tunable ``dict``; each scan point overrides ``base[leaf] = v``. ``loss_fn`` is
    jitted once and re-used across the sweep.
    """
    f = jax.jit(loss_fn)
    out = []
    for v in values:
        point = {**base, leaf: jnp.asarray(v, jnp.float64)}
        out.append((float(v), float(f(point))))
    return out


# Fields of Params that are array/scalar tunables (everything registered as a leaf).
_PARAM_FIELDS = frozenset(f.name for f in dataclasses.fields(Params))


def build_params(tun: dict) -> Params:
    """Map a tunable ``dict`` onto a full :class:`~fesom_jax.params.Params`.

    Leaves named in ``tun`` are set from it; every other leaf takes its
    :meth:`Params.defaults` config-constant value. The seam between the optimizer's flat
    tunable dict and the model's structured parameter pytree — to add a tunable, add its
    key to ``tun`` (one line), nothing else changes.

    Scalar/array leaves (``k_gm``, ``k_ver``, …) are coerced to float64; a pytree leaf
    (the ``tke_nn`` MLP, once present) is passed through unchanged. Raises on an unknown
    key (a typo'd leaf name is a silent no-op otherwise).

    ⚠️ This is **generic** — it does NOT impose the C ``Redi_Kmax = K_GM_max`` auto-sync.
    A GM experiment couples them by passing both (``{'k_gm': θ, 'redi_kmax': θ}``), exactly
    as the grad-gate does; the namelist writer (:mod:`scripts.write_namelist`) enforces the
    sync on the Fortran-transfer side.
    """
    unknown = set(tun) - _PARAM_FIELDS
    if unknown:
        raise ValueError(
            f"build_params: unknown tunable leaf(s) {sorted(unknown)}; "
            f"valid Params fields are {sorted(_PARAM_FIELDS)}")
    repl = {}
    for k, v in tun.items():
        # tke_nn (a pytree leaf) passes through; scalars/fields → float64 arrays.
        repl[k] = v if k == "tke_nn" else jnp.asarray(v, jnp.float64)
    return dataclasses.replace(Params.defaults(), **repl)
