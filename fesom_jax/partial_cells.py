"""Partial bottom cells (FESOM2 ``use_partial_cell``, namelist ``&ale_def``).

With partial cells the bottom layer of every column ends at the topographic depth instead of the nearest
model interface. FESOM2 bounds the adjustment: the bottom interface of an element moves at most to the
mid-depth of the layer below (deeper) or to the mid-depth of its own bottom layer (shallower). Everything
else in the model reads the result through the bottom entries of the layer geometry: the element bottom
``zbar_e_bot``, the node bottom ``zbar_n_bot``, the bottom thicknesses and the bottom entries of
``zbar_3d_n``/``Z_3d_n``/``hnode``/``helem``.

This module is a transcription of FESOM2 ``oce_ale.F90`` (upstream dbf28153):

* :func:`elem_bottom_depth` — ``init_bottom_elem_thickness`` (``:244-359``, ``use_depthonelem=.false.``);
* :func:`node_bottom_depth` — ``init_bottom_node_thickness`` (``:363-460``): the deepest element bottom
  around the node;
* :func:`apply` — both, plus the depth clamp of the mesh read (``oce_mesh.F90:619``) and the
  ``zbar_3d_n(nlevels_nod2D) = zbar_n_bot`` slot of ``init_ale`` (``:222``). It runs once, on the host,
  right after ``load_mesh``, and returns a :class:`~fesom_jax.mesh.Mesh` whose optional ``zbar_e_bot`` /
  ``zbar_n_bot`` fields are set. ``None`` there means full cells, so a full-cell mesh traces exactly the
  program it traced before;
* traceable helpers for the kernels: :func:`bottom_elem_thickness`, :func:`bottom_node_thickness`,
  :func:`static_node_geometry`.

The level counts ``nlevels``/``nlevels_nod2D`` do not depend on partial cells (the partitioner's
``find_levels`` compares the depth with the mid-depths ``Z``, ``mod_mesh_utils.F90:530``), so an existing
mesh export stays valid.

Not covered (refused or not representable): cavity partial cells (``use_cavity_partial_cell``), cavities
together with partial bottom cells, ``use_depthonelem`` (depth given on elements).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from .mesh import Mesh

# FESOM2 gen_modules_config.F90:102 ``thers_zbar_lev=5``: the mesh read clamps every node depth to at
# least zbar(thers_zbar_lev) (oce_mesh.F90:619, 1-based level 5).
THERS_ZBAR_LEV = 5


# --------------------------------------------------------------------------
# Host-side construction (numpy, once at setup)
# --------------------------------------------------------------------------
def clamp_depth(depth, zbar, thers_zbar_lev: int = THERS_ZBAR_LEV):
    """Node depth as FESOM2's mesh read leaves it (``oce_mesh.F90:617-619``): negative, and no shallower
    than ``zbar(thers_zbar_lev)``."""
    x = np.asarray(depth, np.float64)
    x = np.where(x > 0.0, -x, x)
    lim = float(np.asarray(zbar, np.float64)[thers_zbar_lev - 1])
    return np.where(x > lim, lim, x)


def elem_bottom_depth(dd, nlevels, zbar, Z, thresh: float = 0.0):
    """``zbar_e_bot`` of ``init_bottom_elem_thickness`` (``oce_ale.F90:258-345``) for element depths ``dd``.

    ``nlevels`` is FESOM2's 1-based ``nle`` (the element's bottom interface), ``zbar`` the ``nl`` interface
    depths and ``Z`` the ``nl-1`` mid-depths (both ≤ 0). Per element, in the Fortran's order:

    * ``zbar(nle-1) - zbar(nle) <= thresh`` ⇒ full cell, ``zbar(nle)``;
    * ``dd < zbar(nle)`` (deeper than the full cell) ⇒ ``max(dd, zbar(nle)+(zbar(nle)-Z(nle-1)))`` when
      ``nle == nl``, else ``max(Z(nle), dd)``;
    * otherwise (shallower) ⇒ ``min(Z(nle-1), dd)``.
    """
    dd = np.asarray(dd, np.float64)
    zbar = np.asarray(zbar, np.float64)
    Z = np.asarray(Z, np.float64)
    nl = zbar.size
    nle = np.asarray(nlevels, np.int64)
    zb_bot = zbar[nle - 1]                                   # zbar(nle)
    zb_up = zbar[nle - 2]                                    # zbar(nle-1)
    Z_up = Z[nle - 2]                                        # Z(nle-1)
    Z_dn = Z[np.minimum(nle - 1, nl - 2)]                    # Z(nle); only read where nle < nl
    full = (zb_up - zb_bot) <= thresh
    deeper = dd < zb_bot
    deep_last = np.maximum(dd, zb_bot + (zb_bot - Z_up))     # nle == nl
    deep_mid = np.maximum(Z_dn, dd)                          # max(Z(nle), dd)
    deep = np.where(nle == nl, deep_last, deep_mid)
    shallow = np.minimum(Z_up, dd)                           # min(Z(nle-1), dd)
    return np.where(full, zb_bot, np.where(deeper, deep, shallow))


def node_bottom_depth(zbar_e_bot, nod_in_elem2D, offsets):
    """``zbar_n_bot`` of ``init_bottom_node_thickness`` (``oce_ale.F90:439-445``): the minimum (deepest)
    ``zbar_e_bot`` over the node's elements, from the CSR node→element lists."""
    zbar_e_bot = np.asarray(zbar_e_bot, np.float64)
    flat = np.asarray(nod_in_elem2D, np.int64)
    off = np.asarray(offsets, np.int64)
    if np.any(off[1:] <= off[:-1]):
        raise ValueError("partial cells: a node without elements in nod_in_elem2D")
    return np.minimum.reduceat(zbar_e_bot[flat], off[:-1])


def apply(mesh: Mesh, *, thresh: float = 0.0) -> Mesh:
    """``mesh`` with partial bottom cells (FESOM2 ``use_partial_cell=.true.``, ``partial_cell_thresh=thresh``).

    Sets ``zbar_e_bot (elem2D,)``, ``zbar_n_bot (nod2D,)`` and the bottom slot
    ``zbar_3d_n[n, nlevels_nod2D[n]-1] = zbar_n_bot[n]`` (``init_ale``, ``oce_ale.F90:222``). The element
    depth is the mean of the three (clamped) node depths, summed in the order of ``mesh.elem_nodes``
    (``sum(depth(elnodes))/3``). Runs on the host; call it once, right after ``load_mesh`` and before the
    partition, the sharded mesh, the SSH operator and the cold state are built.
    """
    if mesh.zbar_e_bot is not None or mesh.zbar_n_bot is not None:
        raise ValueError("partial_cells.apply: the mesh already carries partial cells")
    if np.any(np.asarray(mesh.ulevels) > 1) or np.any(np.asarray(mesh.ulevels_nod2D) > 1):
        raise NotImplementedError(
            "partial cells with ice-shelf cavities are not ported: FESOM2 then also needs the surface "
            "partial cells (use_cavity_partial_cell, init_surface_elem_depth) and the cavity PGF branches")
    zbar = np.asarray(mesh.zbar, np.float64)
    Z = np.asarray(mesh.Z, np.float64)
    depth = clamp_depth(mesh.depth, zbar)
    en = np.asarray(mesh.elem_nodes)
    dd = (depth[en[:, 0]] + depth[en[:, 1]] + depth[en[:, 2]]) / 3.0
    zbar_e_bot = elem_bottom_depth(dd, mesh.nlevels, zbar, Z, thresh)
    zbar_n_bot = node_bottom_depth(zbar_e_bot, mesh.nod_in_elem2D, mesh.nod_in_elem2D_offsets)

    nln = np.asarray(mesh.nlevels_nod2D, np.int64)
    z3 = np.array(mesh.zbar_3d_n, np.float64, copy=True)
    z3[np.arange(z3.shape[0]), nln - 1] = zbar_n_bot
    return dataclasses.replace(mesh, zbar_e_bot=jnp.asarray(zbar_e_bot), zbar_n_bot=jnp.asarray(zbar_n_bot),
                               zbar_3d_n=jnp.asarray(z3))


# --------------------------------------------------------------------------
# Traceable helpers (dense or sharded mesh; pad lanes stay finite)
# --------------------------------------------------------------------------
def _level_index(nlev, shift: int, nl: int):
    """0-based level index ``nlev - shift``, clipped into ``[0, nl-1]`` so pad lanes (level count 0) and
    any other out-of-range lane read a real entry instead of wrapping."""
    return jnp.clip(jnp.asarray(nlev) - shift, 0, nl - 1)


def bottom_elem_thickness(mesh: Mesh):
    """``bottom_elem_thickness = zbar(nle-1) - zbar_e_bot`` (``oce_ale.F90:345``), ``(elem2D,)``."""
    nl = mesh.zbar.shape[0]
    return mesh.zbar[_level_index(mesh.nlevels, 2, nl)] - mesh.zbar_e_bot


def bottom_node_thickness(mesh: Mesh):
    """``bottom_node_thickness = zbar(nln-1) - zbar_n_bot`` (``oce_ale.F90:445``), ``(nod2D,)``."""
    nl = mesh.zbar.shape[0]
    return mesh.zbar[_level_index(mesh.nlevels_nod2D, 2, nl)] - mesh.zbar_n_bot


def static_node_geometry(mesh: Mesh):
    """The per-node interface and mid-layer depths ``(zbar_3d_n, Z_3d_n)``, each ``(nod2D, nl)``, that
    ``init_ale`` builds under partial cells (``oce_ale.F90:202-237``, no cavity):

    * ``zbar_3d_n`` = ``zbar`` with the bottom interface ``nlevels_nod2D`` (1-based) at ``zbar_n_bot``;
    * ``Z_3d_n`` = ``Z`` with the bottom layer ``nlevels_nod2D-1`` (1-based) at
      ``zbar(nln-1) + (zbar_n_bot - zbar(nln-1))/2``.

    Lanes below the bottom keep the nominal ``zbar``/``Z`` (strictly decreasing, so every spacing stays
    positive for the backward pass; FESOM2 leaves them 0 and never reads them). The last ``Z_3d_n`` lane
    repeats lane ``nl-2``, as the full-cell padded ``Z`` does. ``zbar_n_bot`` lies in ``[Z(nln), Z(nln-1)]``
    (or below ``zbar(nl)`` when ``nln == nl``), so the bottom interface stays above the nominal one below it.
    """
    zbar = mesh.zbar
    nl = zbar.shape[0]
    n = mesh.zbar_n_bot.shape[0]
    k = jnp.arange(nl)[None, :]
    i_bot = _level_index(mesh.nlevels_nod2D, 1, nl)[:, None]      # 0-based bottom interface
    i_lay = _level_index(mesh.nlevels_nod2D, 2, nl)[:, None]      # 0-based bottom layer
    zbot = mesh.zbar_n_bot[:, None]
    zbar3 = jnp.where(k == i_bot, zbot, jnp.broadcast_to(zbar[None, :], (n, nl)))
    z_up = zbar[i_lay]                                             # zbar(nln-1), (n,1)
    Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])
    Z3 = jnp.where(k == i_lay, z_up + (zbot - z_up) / 2.0, jnp.broadcast_to(Zp[None, :], (n, nl)))
    Z3 = jnp.concatenate([Z3[:, :-1], Z3[:, -2:-1]], axis=1)       # pad lane = lane nl-2
    return zbar3, Z3
