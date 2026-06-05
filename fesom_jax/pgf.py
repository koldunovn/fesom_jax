"""Pressure-gradient force at elements (substep 3 / Task 2.2).

Literal vectorized port of ``fesom_pressure_force_linfs_fullcell``
(``fesom_eos.c:285-313``, driven from ``fesom_step.c:113``) for the linfs
full-cell config. Per element ``e`` and layer ``nz``:

    pgf_x[e,nz] = (Σ_i ∂N_i/∂x · hpressure[V_i(e), nz]) / ρ0
    pgf_y[e,nz] = (Σ_i ∂N_i/∂y · hpressure[V_i(e), nz]) / ρ0

with ``∂N_i/∂x = gradient_sca[e, 0:3]``, ``∂N_i/∂y = gradient_sca[e, 3:6]``. A clean
gather (node→element via ``elem_nodes``) + linear shape-function contraction →
map/gather class (~1e-15). Output is an element **layer** field
(``elem_layer_mask``). The 3-term sum is written in the C's association order.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ops
from .config import DENSITY_0
from .mesh import Mesh


def pressure_force_linfs(mesh: Mesh, hpressure):
    """``(pgf_x, pgf_y)`` element layer fields ``[elem2D, nl]`` from the node
    hydrostatic pressure ``hpressure`` ``[nod2D, nl]``."""
    inv_r = 1.0 / DENSITY_0
    hp = ops.gather_nodes_to_elem(hpressure, mesh.elem_nodes)   # (elem2D, 3, nl)
    g = mesh.gradient_sca                                       # (elem2D, 6)
    hp0, hp1, hp2 = hp[:, 0], hp[:, 1], hp[:, 2]                # each (elem2D, nl)

    # keep the C's left-to-right association: ((g0·hp0 + g1·hp1) + g2·hp2)·inv_r
    pgf_x = (g[:, 0:1] * hp0 + g[:, 1:2] * hp1 + g[:, 2:3] * hp2) * inv_r
    pgf_y = (g[:, 3:4] * hp0 + g[:, 4:5] * hp1 + g[:, 5:6] * hp2) * inv_r

    pgf_x = ops.mask_below_bottom(pgf_x, mesh.elem_layer_mask)
    pgf_y = ops.mask_below_bottom(pgf_y, mesh.elem_layer_mask)
    return pgf_x, pgf_y
