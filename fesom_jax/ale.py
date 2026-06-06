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
from .config import DT_DEFAULT, USE_WSPLIT, WSPLIT_MAXCFL
from .mesh import Mesh


def _shift_down_zero(x):
    """``out[..., k] = x[..., k-1]``, **zero**-padded at ``k=0`` (no edge-replicate)."""
    return jnp.concatenate([jnp.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)


def thickness_linfs(hnode):
    """``hnode_new = hnode`` (substep 13, linfs; ``fesom_ale.c:10-16``).

    Linfs has ``dh/dt = 0`` so the per-vertex layer thickness never changes; the C
    ``fesom_ale_thickness_linfs`` is a ``memcpy(hnode_new, hnode)``. Returned as a
    distinct ``[nod2D, nl]`` array for the ``State.hnode_new`` slot (consumed by
    the tracer ALE reconstruction in Task 2.10). The substep-13 ``hnode_new`` node
    dump should therefore equal the static ``hnode``.
    """
    return jnp.asarray(hnode)


def commit_thickness(mesh: Mesh, hnode_new):
    """Commit the ALE thickness (substep 16, ``fesom_ale_commit_thickness``,
    ``fesom_ale.c:18``). Returns ``(hnode, helem)``:

    * ``hnode = hnode_new`` (a copy; linfs ⇒ unchanged).
    * ``helem`` = the per-element vertex mean ``(hnode[n0]+hnode[n1]+hnode[n2])/3``,
      masked to ``elem_layer_mask``. In linfs this reproduces the static reference
      thickness (all 3 vertices share the same full-cell ``hnode``).

    The substep-16 node dump records ``hnode`` (bit-for-bit the static thickness, as
    ``hnode_new`` was at substep 13)."""
    hnode = jnp.asarray(hnode_new)
    h3 = ops.gather_nodes_to_elem(hnode, mesh.elem_nodes)   # (elem2D, 3, nl)
    helem = (h3[:, 0] + h3[:, 1] + h3[:, 2]) / 3.0
    helem = ops.mask_below_bottom(helem, mesh.elem_layer_mask)
    return hnode, helem


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


def compute_cfl_z(mesh: Mesh, w, hnode_new, *, dt=DT_DEFAULT):
    """Vertical CFL number at interfaces ``cfl_z`` ``[nod2D, nl]`` (``fesom_ale_compute_cflz``,
    ``fesom_ale.c:204``). Each interface accumulates ``|w|·dt/h`` from the layers above and
    below it::

        cfl_z[i] = |w[i]|·dt·( [layer i  valid]/h[i]  +  [layer i-1 valid]/h[i-1] )

    so the surface/bottom interfaces get a single term and interior interfaces get both
    (matching the C ``+=`` accumulation over each layer's top/bottom faces). ``h`` is the
    layer thickness ``hnode_new``; the C skips ``h<=0`` layers, mirrored by the layer mask.

    Only consumed by :func:`compute_wvel_split` when ``use_wsplit`` is on (off for the pi
    reference config), but computed every step as in the C so ``State.cfl_z`` is populated."""
    h = hnode_new
    safe_h = jnp.where(h > 0.0, h, 1.0)
    inv_h = jnp.where(mesh.node_layer_mask & (h > 0.0), dt / safe_h, 0.0)  # layer field
    below = inv_h                       # layer i (below interface i)
    above = _shift_down_zero(inv_h)     # layer i-1 (above interface i); 0 at i=0
    cfl = jnp.abs(w) * (below + above)
    return ops.mask_below_bottom(cfl, mesh.node_iface_mask)


def compute_wvel_split(mesh: Mesh, w, cfl_z, *, use_wsplit=USE_WSPLIT,
                       maxcfl=WSPLIT_MAXCFL):
    """Split the vertical velocity ``w`` into explicit ``w_e`` and implicit ``w_i`` parts
    (``fesom_ale_compute_wvel_split``, ``fesom_ale.c:241``). Returns ``(w_e, w_i)``.

    Where the vertical CFL exceeds ``maxcfl`` the excess is moved to the implicit part so
    the explicit advection stays CFL-stable::

        dd  = max(cfl_z − maxcfl, 0) / maxcfl
        w_e = w / (1 + dd),   w_i = w · dd / (1 + dd)      (cfl_z > maxcfl)
        w_e = w,              w_i = 0                        (otherwise)

    **``use_wsplit`` is 0 in the pi (and CORE2 dt=1800) reference config**
    (``fesom_constants.h:56`` — the split seeded a Fortran day-92 blow-up), so this is the
    identity ``(w, 0)`` on the dump-matching path; the active branch is exercised by the
    synthetic high-CFL test. AD-safe: ``dd ≥ 0`` ⇒ ``1+dd ≥ 1`` (no zero divide), and the
    static ``maxcfl = 1.0`` floors the ``1/maxcfl``.

    ⚠️ ``w_i`` (the implicit part) feeds the ``impl_vert_visc`` advective tridiagonal terms,
    which the Phase-2 kernel drops under the ``w_i=0`` (``use_wsplit=0``) simplification —
    re-enabling those terms is a Phase-5/CORE2 item, needed only when ``use_wsplit`` is on."""
    if not use_wsplit:
        return jnp.asarray(w), jnp.zeros_like(w)
    inv_maxcfl = 1.0 / max(maxcfl, 1e-12)            # static (maxcfl is a Python float)
    dd = jnp.maximum(cfl_z - maxcfl, 0.0) * inv_maxcfl
    inv_1_dd = 1.0 / (1.0 + dd)                      # dd ≥ 0 ⇒ always finite
    split = cfl_z > maxcfl
    w_e = jnp.where(split, w * inv_1_dd, w)
    w_i = jnp.where(split, w * dd * inv_1_dd, 0.0)
    w_e = ops.mask_below_bottom(w_e, mesh.node_iface_mask)
    w_i = ops.mask_below_bottom(w_i, mesh.node_iface_mask)
    return w_e, w_i
