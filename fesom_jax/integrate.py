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


def integrate(state: State, mesh: Mesh, op: SSHOperator, stress_surf, n_steps: int,
              params: Params = None, *, dt: float = DT_DEFAULT,
              checkpoint: bool = True, step_forcings=None, forcing_static=None,
              ice_cfg=None, gm_cfg=None) -> State:
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
    ``step_forcings[1:]`` as ``xs``. ``None`` ⇒ the pi analytical path (unchanged)."""
    if params is None:
        params = Params.defaults()

    if step_forcings is None:
        # pi path (unchanged): static stress_surf, no per-step forcing, xs=None.
        state = step(state, mesh, op, stress_surf, params, dt=dt, is_first_step=True,
                     gm_cfg=gm_cfg)
        if n_steps <= 1:
            return state

        def body(carry, _):
            nxt = step(carry, mesh, op, stress_surf, params, dt=dt, is_first_step=False,
                       gm_cfg=gm_cfg)
            return nxt, None

        body_fn = jax.checkpoint(body) if checkpoint else body
        state, _ = lax.scan(body_fn, state, xs=None, length=n_steps - 1)
        return state

    # CORE2 path: step 1 eager with step_forcings[0]; scan the rest as xs.
    # ``ice_cfg`` (an IceConfig) ⇒ the prognostic sea-ice step runs each step (Phase 6).
    sf0 = jax.tree.map(lambda x: x[0], step_forcings)
    state = step(state, mesh, op, stress_surf, params, dt=dt, is_first_step=True,
                 step_forcing=sf0, forcing_static=forcing_static, ice_cfg=ice_cfg,
                 gm_cfg=gm_cfg)
    if n_steps <= 1:
        return state

    rest = jax.tree.map(lambda x: x[1:], step_forcings)

    def body_core(carry, sf):
        nxt = step(carry, mesh, op, stress_surf, params, dt=dt, is_first_step=False,
                   step_forcing=sf, forcing_static=forcing_static, ice_cfg=ice_cfg,
                   gm_cfg=gm_cfg)
        return nxt, None

    body_fn = jax.checkpoint(body_core) if checkpoint else body_core
    state, _ = lax.scan(body_fn, state, xs=rest)
    return state


# Jitted entry point (what the gradient harness and the memory-sanity gate call).
# ``state``/``mesh``/``op``/``stress_surf``/``params`` are pytree args; the scan
# length, dt and checkpoint flag are static.
integrate_jit = jax.jit(integrate,
                        static_argnames=("n_steps", "dt", "checkpoint", "ice_cfg", "gm_cfg"))
