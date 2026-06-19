"""Checkpointed ``lax.scan`` time loop — Phase 3, Task 3.1 (the AD de-risking gate).

The Phase-2 :func:`fesom_jax.step.run` is a Python ``for`` loop over ``step_jit`` —
fine for a forward run, but reverse-mode AD through a Python loop unrolls the whole
trajectory into one giant graph (memory blows up with ``N``). Phase 3 replaces it
with a single :func:`jax.lax.scan` whose body is wrapped in :func:`jax.checkpoint`
(rematerialization): the backward pass stores only the per-step **carry** (the
:class:`~fesom_jax.state.State` pytree) and recomputes each step's internals on the
fly, so an ``N``-step backward pass fits device memory.

Two design points (both from the plan):

* **``is_first_step`` must not be a traced carry.** It only flips the AB2 ``ff_step``
  (1.0 vs 1.6) and is a compile-time constant. A ``lax.scan`` body must be uniform,
  so we run **step 1 eagerly** (``is_first_step=True``) *outside* the scan and scan
  steps ``2..N`` with ``is_first_step=False`` baked in. No traced bool, one body.

* **Loop-invariants are closed over, not carried.** ``mesh``/``op``/``stress_surf``/
  ``params`` are constant across steps; closing over them keeps the carry minimal
  (only ``State``). They are pytrees — fine to close over. ``params`` being closed
  over still differentiates correctly: ``scan`` hoists closed-over tracers as consts
  and accumulates their cotangents across all steps, so ``d(loss)/d(params)`` sums
  the per-step contributions (verified in ``test_gradient.py``).

For very long windows, swap the per-step ``jax.checkpoint`` for nested / policy-based
checkpointing (O(√N) memory); the simple per-step remat is enough for the Phase-3
N≤200 gate.
"""

from __future__ import annotations

import jax
from jax import lax

from .config import DT_DEFAULT
from .mesh import Mesh
from .params import Params
from .ssh import SSHOperator
from .state import State
from .step import step


def _run_steps(body, init, *, xs=None, length=None, checkpoint=True, segments=0):
    """Run ``length`` (or ``len(xs)``) steps of ``body`` via ``lax.scan``, optionally with
    **O(√N) two-level rematerialization** (``segments>0``).

    ``body`` is the RAW (un-checkpointed) step body ``(carry, x) -> (carry, None)``. Default
    (``segments=0``, ``checkpoint=True``) is the Phase-3 per-step checkpoint: the scan stores
    one carry per step (``N`` carries) — fine for short windows but that ``N×|State|`` stack is
    the buffer that OOMs long CORE2 adjoints. ``segments=S`` restructures the loop into S outer
    segments × M≈N/S inner steps with a checkpoint on BOTH the outer segment body and the inner
    per-step body: the backward then stores only **S + M ≈ 2√N** carries (recomputing each
    segment on the fly) instead of N — the standard √N-memory adjoint. Forward value is identical
    (checkpoint is forward-transparent); only the backward's storage/recompute changes. The
    remainder (``length mod S``) runs as a per-step-checkpointed tail."""
    if length is None:
        length = int(jax.tree.leaves(xs)[0].shape[0])
    step_fn = jax.checkpoint(body) if checkpoint else body
    if checkpoint and segments and segments > 1 and length >= 2 * segments:
        S = int(segments)
        M = length // S
        used = S * M

        def seg_body(carry, seg_xs):
            c, _ = (lax.scan(step_fn, carry, xs=None, length=M) if seg_xs is None
                    else lax.scan(step_fn, carry, seg_xs))
            return c, None

        seg_fn = jax.checkpoint(seg_body)
        if xs is None:
            init, _ = lax.scan(seg_fn, init, xs=None, length=S)
            if length - used:
                init, _ = lax.scan(step_fn, init, xs=None, length=length - used)
        else:
            main = jax.tree.map(lambda x: x[:used].reshape((S, M) + x.shape[1:]), xs)
            init, _ = lax.scan(seg_fn, init, main)
            if used < length:
                init, _ = lax.scan(step_fn, init, jax.tree.map(lambda x: x[used:], xs))
        return init
    if xs is None:
        init, _ = lax.scan(step_fn, init, xs=None, length=length)
    else:
        init, _ = lax.scan(step_fn, init, xs)
    return init


def integrate(state: State, mesh: Mesh, op: SSHOperator, stress_surf, n_steps: int,
              params: Params = None, *, dt: float = DT_DEFAULT,
              checkpoint: bool = True, step_forcings=None, forcing_static=None,
              ice_cfg=None, gm_cfg=None, kpp_cfg=None, tke_cfg=None,
              ale_cfg=None, visc_cfg=None, tracer_cfg=None,
              remat_blocks: bool = False, remat_segments: int = 0) -> State:
    """Integrate ``n_steps`` ocean timesteps from ``state`` via a checkpointed scan.

    Returns the final :class:`~fesom_jax.state.State`. Differentiable end-to-end
    (the whole point): ``jax.grad`` of a scalar loss of the returned state flows back
    through every substep — including the CG ``custom_linear_solve`` — for ``params``
    and/or the initial ``state``.

    ``n_steps`` / ``dt`` / ``checkpoint`` are **static** (Python values); under
    ``jax.jit`` pass them via ``static_argnames`` (see :data:`integrate_jit`). With
    ``checkpoint=False`` the body is not rematerialized — same forward value, but the
    backward pass stores all intermediates (the un-checkpointed baseline, for the
    memory comparison). ``params=None`` ⇒ the config-constant baseline.

    **CORE2 forcing.** Pass ``step_forcings`` (a
    :class:`~fesom_jax.core2_forcing.StepForcing` stacked with leading axis
    ``[n_steps]``) + ``forcing_static`` to drive the bulk/SSS/shortwave surface BCs:
    step 1 consumes ``step_forcings[0]`` (eager), the scan carries
    ``step_forcings[1:]`` as ``xs``. ``None`` ⇒ the pi analytical path (unchanged).

    **Long windows (``remat_segments``).** Per-step checkpointing stores N carries — the
    ``N×|State|`` stack that OOMs long CORE2 adjoints. ``remat_segments=-1`` switches to O(√N)
    two-level checkpointing (auto S≈√N segments; see :func:`_run_steps`) ⇒ only ~2√N carries, so
    a *continuous* backward over a much longer window fits one GPU. ``>1`` sets S explicitly;
    ``0`` (default) keeps the per-step path (byte-identical). Forward value is unchanged."""
    if params is None:
        params = Params.defaults()
    _seg = remat_segments
    if _seg == -1:
        import math
        _seg = max(2, int(round(math.sqrt(max(1, n_steps - 1)))))

    if step_forcings is None:
        # pi path (unchanged): static stress_surf, no per-step forcing, xs=None.
        state = step(state, mesh, op, stress_surf, params, dt=dt, is_first_step=True,
                     gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, tke_cfg=tke_cfg, ale_cfg=ale_cfg,
                     visc_cfg=visc_cfg, tracer_cfg=tracer_cfg, remat_blocks=remat_blocks)
        if n_steps <= 1:
            return state

        def body(carry, _):
            nxt = step(carry, mesh, op, stress_surf, params, dt=dt, is_first_step=False,
                       gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, tke_cfg=tke_cfg, ale_cfg=ale_cfg,
                       visc_cfg=visc_cfg, tracer_cfg=tracer_cfg, remat_blocks=remat_blocks)
            return nxt, None

        return _run_steps(body, state, length=n_steps - 1, checkpoint=checkpoint, segments=_seg)

    # CORE2 path: step 1 eager with step_forcings[0]; scan the rest as xs.
    # ``ice_cfg`` (an IceConfig) ⇒ the prognostic sea-ice step runs each step (Phase 6).
    sf0 = jax.tree.map(lambda x: x[0], step_forcings)
    state = step(state, mesh, op, stress_surf, params, dt=dt, is_first_step=True,
                 step_forcing=sf0, forcing_static=forcing_static, ice_cfg=ice_cfg,
                 gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, tke_cfg=tke_cfg, ale_cfg=ale_cfg,
                 visc_cfg=visc_cfg, tracer_cfg=tracer_cfg, remat_blocks=remat_blocks)
    if n_steps <= 1:
        return state

    rest = jax.tree.map(lambda x: x[1:], step_forcings)

    def body_core(carry, sf):
        nxt = step(carry, mesh, op, stress_surf, params, dt=dt, is_first_step=False,
                   step_forcing=sf, forcing_static=forcing_static, ice_cfg=ice_cfg,
                   gm_cfg=gm_cfg, kpp_cfg=kpp_cfg, tke_cfg=tke_cfg, ale_cfg=ale_cfg,
                   visc_cfg=visc_cfg, tracer_cfg=tracer_cfg, remat_blocks=remat_blocks)
        return nxt, None

    return _run_steps(body_core, state, xs=rest, checkpoint=checkpoint, segments=_seg)


# Jitted entry point (what the gradient harness and the memory-sanity gate call).
# ``state``/``mesh``/``op``/``stress_surf``/``params`` are pytree args; the scan
# length, dt and checkpoint flag are static.
integrate_jit = jax.jit(
    integrate,
    static_argnames=("n_steps", "dt", "checkpoint", "ice_cfg", "gm_cfg", "kpp_cfg",
                     "tke_cfg", "ale_cfg", "visc_cfg", "tracer_cfg",
                     "remat_blocks", "remat_segments"))
