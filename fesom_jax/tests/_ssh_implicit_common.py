"""Shared pieces of the SSH-solve tangent-linear tests (``test_ssh_implicit*.py``).

* :func:`solve_ssh_reference` — the SSH solve as it was before the implicit-derivative
  rule: the early-stopped PCG wrapped in ``lax.custom_linear_solve``, whose tangent reuses
  the FORWARD solve and therefore its absolute stop threshold derived from the primal
  right-hand side. Kept here, not in the library, as the oracle for "forward bitwise" and
  "reverse gradient unchanged", and as the negative control of the dot test (it must FAIL
  the dot test at small amplitude).
* structured test directions on the mesh (smooth, masked, several wavelengths), so the dot
  test is not decided by one random draw.
* :func:`step_dot_test` — ``<y, M(a·x)>`` vs ``<Mᵀy, a·x>`` for one model step ``M``, at
  several amplitudes ``a``, with the tangent and the cotangent programs compiled once.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from fesom_jax import ssh
from fesom_jax.config import DT_DEFAULT, MAXITER, SOLTOL, SSH_ALPHA, SSH_THETA
from fesom_jax.reductions import global_dot, global_dot_pair


def solve_ssh_reference(op, ssh_rhs, *, x0=None, halo=None, mesh=None, hbar=None,
                        dt=DT_DEFAULT, alpha=SSH_ALPHA, theta=SSH_THETA,
                        forward_tol=SOLTOL, grad_tol=ssh.GRAD_SOLTOL, maxiter=MAXITER):
    """The pre-implicit-rule ``solve_ssh`` (custom_linear_solve with the early-stopped
    forward solve as the tangent solve), same signature as :func:`ssh.solve_ssh`."""
    n = op.n_nodes
    x0 = jnp.zeros((n,), ssh_rhs.dtype) if x0 is None else x0
    x0 = lax.stop_gradient(x0)
    if hbar is None:
        matvec = lambda x: ssh.ssh_matvec(op, x, halo)    # noqa: E731
    else:
        def matvec(x):
            return ssh.ssh_matvec(op, x, halo) + ssh.stiff_increment_matvec(
                mesh, hbar, x, dt=dt, alpha=alpha, theta=theta, halo=halo)
    precond = ssh.ssh_precond_for(op, halo)
    b_eff = ssh_rhs - matvec(x0)
    if halo is None:
        nrhs = ssh_rhs.shape[0]
        rtol_fwd = forward_tol * jnp.sqrt(jnp.sum(ssh_rhs * ssh_rhs) / nrhs)
        reduce_fn, reduce_pair_fn, n_global = None, None, None
    else:
        reduce_fn = lambda u, v: global_dot(u, v, halo.owned_mask, halo.axis_name)  # noqa: E731
        reduce_pair_fn = lambda r, z: global_dot_pair(                             # noqa: E731
            r, z, r, r, halo.owned_mask, halo.axis_name)
        n_global = halo.n_global
        rtol_fwd = forward_tol * jnp.sqrt(reduce_fn(ssh_rhs, ssh_rhs) / n_global)

    def solve(mv, b):
        return ssh._pcg(mv, precond, b, jnp.zeros_like(b), forward_tol, maxiter,
                        rtol_abs=rtol_fwd, reduce=reduce_fn, reduce_pair=reduce_pair_fn,
                        n_global=n_global)

    def transpose_solve(mv, b):
        return ssh._pcg(mv, precond, b, jnp.zeros_like(b), grad_tol, maxiter,
                        reduce=reduce_fn, reduce_pair=reduce_pair_fn, n_global=n_global)

    delta = jax.lax.custom_linear_solve(matvec, b_eff, solve, transpose_solve,
                                        symmetric=True)
    return x0 + delta


# --------------------------------------------------------------------------
# structured directions
# --------------------------------------------------------------------------
def node_pattern(mesh, kind: int = 0):
    """Smooth ``[nod2D]`` pattern of O(1) amplitude built from the rotated coordinates."""
    lon = np.asarray(mesh.coord_nod2D[:, 0])
    lat = np.asarray(mesh.coord_nod2D[:, 1])
    if kind == 0:
        return np.cos(lon) * np.sin(2.0 * lat) + 0.3 * np.sin(3.0 * lon + 0.5)
    return np.sin(2.0 * lon + 0.3) * np.cos(lat) - 0.4 * np.cos(5.0 * lat)


def elem_pattern(mesh, kind: int = 0):
    """Smooth ``[elem2D]`` pattern from the element centroids (rotated radians)."""
    x = np.asarray(mesh.elem_center_x)
    y = np.asarray(mesh.elem_center_y)
    if kind == 0:
        return np.sin(x + 0.2) * np.cos(2.0 * y) + 0.25 * np.cos(4.0 * x)
    return np.cos(3.0 * x) * np.sin(y + 0.1) + 0.3 * np.sin(2.0 * y)


def _levels(nl, kind):
    k = np.arange(nl) / max(nl - 1, 1)
    return np.cos(np.pi * k) if kind == 0 else np.exp(-3.0 * k)


def node3d(mesh, scale, kind=0):
    """``[nod2D, nl]`` masked structured field."""
    f = node_pattern(mesh, kind)[:, None] * _levels(mesh.nl, kind)[None, :]
    return jnp.asarray(np.where(np.asarray(mesh.node_layer_mask), scale * f, 0.0))


def node2d(mesh, scale, kind=0):
    return jnp.asarray(scale * node_pattern(mesh, kind))


def elem_uv(mesh, scale, kind=0):
    """``[elem2D, nl, 2]`` masked structured velocity-shaped field."""
    fu = elem_pattern(mesh, kind)[:, None] * _levels(mesh.nl, kind)[None, :]
    fv = elem_pattern(mesh, 1 - kind)[:, None] * _levels(mesh.nl, 1 - kind)[None, :]
    m = np.asarray(mesh.elem_layer_mask)
    return jnp.asarray(np.stack([np.where(m, scale * fu, 0.0),
                                 np.where(m, scale * fv, 0.0)], axis=-1))


# --------------------------------------------------------------------------
# dot test
# --------------------------------------------------------------------------
def _vdot(a, b):
    return sum(float(jnp.vdot(u, v)) for u, v in
               zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def _norm(a):
    return float(np.sqrt(_vdot(a, a)))


def step_dot_test(step_fn, state, x, y, amps=(1.0, 1e-4, 1e-8), consts=(),
                  return_ct=False):
    """Dot test of ``M = d step_fn / d state`` restricted to the fields named in ``x`` (in)
    and ``y`` (out).

    ``step_fn(state, *consts) -> State``; ``consts`` (mesh, operator, forcing, ...) are
    passed to the compiled programs as ARGUMENTS, not closed over, so XLA does not
    constant-fold the mesh scatters. ``x`` / ``y`` are dicts ``{field: array}``. Returns rows
    ``(amp, lhs, rhs, rel, rel_norm)`` with ``lhs = <y, M(amp·x)>``, ``rhs = <Mᵀy, amp·x>``,
    ``rel = |lhs − rhs| / max(|lhs|, |rhs|)`` and
    ``rel_norm = |lhs − rhs| / (‖M(amp·x)‖·‖y‖)``. ``return_ct`` also returns ``Mᵀy``
    (the one-step reverse gradient of ``<y, step>``)."""
    fin, fout = tuple(sorted(x)), tuple(sorted(y))

    def f(xin, st, cs):
        out = step_fn(dataclasses.replace(st, **xin), *cs)
        return {k: getattr(out, k) for k in fout}

    def x0_of(st):
        return {k: getattr(st, k) for k in fin}

    tl = jax.jit(lambda t, st, cs: jax.jvp(lambda xi: f(xi, st, cs), (x0_of(st),), (t,))[1])
    adj = jax.jit(lambda c, st, cs: jax.vjp(lambda xi: f(xi, st, cs), x0_of(st))[1](c)[0])
    ct = adj(y, state, tuple(consts))
    rows = []
    for a in amps:
        xa = {k: a * v for k, v in x.items()}
        mx = tl(xa, state, tuple(consts))
        lhs = _vdot(y, mx)
        rhs = _vdot(ct, xa)
        diff = abs(lhs - rhs)
        rows.append((a, lhs, rhs, diff / max(abs(lhs), abs(rhs), 1e-300),
                     diff / max(_norm(mx) * _norm(y), 1e-300)))
    return (rows, ct) if return_ct else rows


def format_rows(label, rows):
    out = [f"  dot test {label}:"]
    for a, lhs, rhs, rel, reln in rows:
        out.append(f"    amp={a:.0e}  <y,Mx>={lhs:+.15e}  <MTy,x>={rhs:+.15e}  "
                   f"rel={rel:.2e}  rel_norm={reln:.2e}")
    return "\n".join(out)


def parse_cg_iters(lines):
    """``[(iterations, rtol)]`` from ``FESOM_CG_ITERS`` debug lines
    (``[cg-iters] it=.. hit_cap=.. resid=.. rtol=..``)."""
    out = []
    for ln in lines:
        kv = dict(tok.split("=", 1) for tok in ln.split() if "=" in tok)
        out.append((int(kv["it"]), float(kv["rtol"])))
    return out
