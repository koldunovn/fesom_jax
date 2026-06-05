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
              checkpoint: bool = True) -> State:
    """Integrate ``n_steps`` ocean timesteps from ``state`` via a checkpointed scan.

    Returns the final :class:`~fesom_jax.state.State`. Differentiable end-to-end
    (the whole point): ``jax.grad`` of a scalar loss of the returned state flows back
    through every substep — including the CG ``custom_linear_solve`` — for ``params``
    and/or the initial ``state``.

    ``n_steps`` / ``dt`` / ``checkpoint`` are **static** (Python values); under
    ``jax.jit`` pass them via ``static_argnames`` (see :data:`integrate_jit`). With
    ``checkpoint=False`` the body is not rematerialized — same forward value, but the
    backward pass stores all intermediates (the un-checkpointed baseline, for the
    memory comparison). ``params=None`` ⇒ the config-constant baseline."""
    if params is None:
        params = Params.defaults()

    # Step 1 eagerly (the AB2 first-step branch) so the scan body is uniform.
    state = step(state, mesh, op, stress_surf, params, dt=dt, is_first_step=True)
    if n_steps <= 1:
        return state

    # Scan steps 2..N with is_first_step=False baked in; close over the invariants.
    def body(carry, _):
        nxt = step(carry, mesh, op, stress_surf, params, dt=dt, is_first_step=False)
        return nxt, None

    body_fn = jax.checkpoint(body) if checkpoint else body
    state, _ = lax.scan(body_fn, state, xs=None, length=n_steps - 1)
    return state


# Jitted entry point (what the gradient harness and the memory-sanity gate call).
# ``state``/``mesh``/``op``/``stress_surf``/``params`` are pytree args; the scan
# length, dt and checkpoint flag are static.
integrate_jit = jax.jit(integrate, static_argnames=("n_steps", "dt", "checkpoint"))
