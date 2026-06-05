"""ALE step (linfs) — substep 13 (Task 2.9).

Literal vectorized port of ``fesom_ale.c`` for the Phase-2 pi config (linfs ALE,
no cavity / no partial cells, single MPI rank):

* :func:`thickness_linfs` — ``hnode_new = hnode`` (``fesom_ale_thickness_linfs``,
  ``fesom_ale.c:10-16``). In **linfs** ``dh/dt = 0``, so the layer thickness is
  *static* for the whole run; the C routine is a ``memcpy``.

* :func:`compute_w` — vertical velocity ``w`` at interfaces
  (``fesom_ale_vert_vel_linfs``, ``fesom_ale.c:77``). The SAME antisymmetric
  edge→node ``(v·dx − u·dy)·helem`` transport-divergence scatter as
  :func:`~fesom_jax.ssh.compute_ssh_rhs` / :func:`~fesom_jax.ssh.compute_hbar`,
  but kept **per level** (not summed over the column) and driven by the **new**
  (post-:func:`~fesom_jax.momentum.update_vel`) velocity ``uv``. Then a reverse
  (bottom→top) cumulative sum over levels — the vertical integral of the
  horizontal divergence up from the no-flux bottom interface ``w[nzmax]=0`` — and
  finally ``÷ area(n, nz)`` → m/s.

  ⚠️ **Trap:** stage 3 divides by ``mesh.area`` (the *upper-edge scalar* CV area),
  **NOT** ``areasvol`` (which :func:`~fesom_jax.ssh.compute_hbar` used) — a
  different array. The C guards ``if (area > 0)``; we mirror it with a safe divide
  so masked / bottom lanes (where ``w = 0``) and the backward pass stay finite.
  Like ``hbar``, the ``÷area`` (``~1e9–1e12 m²``) divides the near-cancelling
  divergence's amplified absolute error back down, so ``w`` matches the dump
  tightly despite the loose per-level scatter floor.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from . import ops
from .mesh import Mesh


def thickness_linfs(hnode):
    """``hnode_new = hnode`` (substep 13, linfs; ``fesom_ale.c:10-16``).

    Linfs has ``dh/dt = 0`` so the per-vertex layer thickness never changes; the C
    ``fesom_ale_thickness_linfs`` is a ``memcpy(hnode_new, hnode)``. Returned as a
    distinct ``[nod2D, nl]`` array for the ``State.hnode_new`` slot (consumed by
    the tracer ALE reconstruction in Task 2.10). The substep-13 ``hnode_new`` node
    dump should therefore equal the static ``hnode``.
    """
    return jnp.asarray(hnode)


def compute_w(mesh: Mesh, uv, helem):
    """Vertical velocity ``w`` ``[nod2D, nl]`` at interfaces (substep 13).

    Mirror of ``fesom_ale_vert_vel_linfs`` (``fesom_ale.c:77-187``), three stages:

    1. **Per-level transport-divergence scatter.** For edge ``ed`` with adjacent
       cells ``(el1, el2)`` and cross vectors ``(dx1,dy1)`` / ``(dx2,dy2)``,
       ``c[nz] = (v[el1]·dx1 − u[el1]·dy1)·h[el1] − (v[el2]·dx2 − u[el2]·dy2)·h[el2]``
       per level ``nz``, scattered ``w[n1,nz] += c``, ``w[n2,nz] −= c``. This is
       :func:`~fesom_jax.ssh.compute_ssh_rhs`'s flux with ``alpha=1`` and no
       AB-velocity, **kept per level**. The per-element ``elem_layer_mask`` keeps
       each cell's contribution in ``[ulevels-1, nlevels-1)`` — ⊆ the node's range,
       so the node's bottom interface ``w[nzmax]`` receives nothing and stays 0
       (the no-flux BC).

    2. **Reverse (bottom→top) cumulative sum** over the level axis:
       ``w[nz] = Σ_{j≥nz} div[j]`` (the vertical integral of the divergence). Since
       the scatter is already 0 at and below ``nzmax``, a full suffix-sum equals
       the C loop ``for nz=nzmax-1..nzmin: w[nz] += w[nz+1]`` while preserving
       ``w[nzmax] = 0``.

    3. **``÷ area(n, nz)`` → m/s** (``mesh.area``, *not* ``areasvol``). The C's
       ``if (area > 0)`` guard becomes a safe divide ``where(area>0, area, 1)``:
       the only nonzero ``w`` lanes are ``[nzmin, nzmax)`` where ``area > 0``, so
       this is exact and AD-finite. A final ``node_iface_mask`` keeps ``w`` on the
       valid interface range ``[nzmin, nzmax]`` (zeroing the suffix-sum's spill
       above a cavity node's ``nzmin``; a no-op for non-cavity pi).
    """
    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s = jnp.where(has1, el1, 0)
    el2s = jnp.where(has2, el2, 0)
    cross = mesh.edge_cross_dxdy
    U, V = uv[:, :, 0], uv[:, :, 1]
    lm = mesh.elem_layer_mask

    def cterm(els, has, dxcol, dycol, sign):
        dx = cross[:, dxcol : dxcol + 1]                  # (edge,1)
        dy = cross[:, dycol : dycol + 1]
        u = U[els]; v = V[els]; h = helem[els]            # (edge,nl)
        m = lm[els] & has[:, None]
        term = jnp.where(m, (v * dx - u * dy) * h, 0.0)
        return sign * term                                # (edge,nl) — per level

    # Stage 1: per-level antisymmetric edge→node divergence scatter.
    c = cterm(el1s, has1, 0, 1, 1.0) + cterm(el2s, has2, 2, 3, -1.0)   # (edge,nl)
    vals = jnp.stack([c, -c], axis=1)                     # (edge,2,nl)
    div = ops.scatter_add(vals, mesh.edges, mesh.nod2D)   # (nod2D,nl)

    # Stage 2: reverse (bottom→top) cumulative sum over levels.
    w = lax.cumsum(div, axis=1, reverse=True)

    # Stage 3: ÷ area (NOT areasvol); safe divide mirrors the C `if (area > 0)`.
    safe_area = jnp.where(mesh.area > 0.0, mesh.area, 1.0)
    w = w / safe_area
    return ops.mask_below_bottom(w, mesh.node_iface_mask)
